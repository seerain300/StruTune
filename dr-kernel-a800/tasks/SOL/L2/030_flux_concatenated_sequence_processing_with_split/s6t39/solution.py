import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seq_kernel(
    A_ptr,        # *f32, [B, M, H]
    B_ptr,        # *f32, [B, N, H]
    Out_ptr,      # *f32, [B, C, H], C = M + N
    B: tl.constexpr,    # batch size (constexpr for specialization)
    M: tl.constexpr,    # text_seq_len (constexpr)
    N: tl.constexpr,    # img_seq_len (constexpr)
    H: tl.constexpr,    # hidden_dim (constexpr)
    stride_ab,    # int: stride along batch for A
    stride_am,    # int: stride along seq for A
    stride_ah,    # int: stride along hidden for A
    stride_bb,    # int: stride along batch for B
    stride_bn,    # int: stride along seq for B
    stride_bh,    # int: stride along hidden for B
    stride_ob,    # int: stride along batch for Out
    stride_oc,    # int: stride along seq for Out
    stride_oh,    # int: stride along hidden for Out
):
    # Grid: (B, 1) — one program per batch, handles both halves via masks
    b = tl.program_id(0)

    # Offsets for sequence positions
    m_offsets = tl.arange(0, M)               # [M]
    n_offsets = tl.arange(0, N)               # [N]
    h_offsets = tl.arange(0, H)               # [H]

    # Masks
    mask_m = m_offsets < M
    mask_n = n_offsets < N
    mask_h = h_offsets < H

    # Compute pointers for A and B, and store into Out at disjoint regions
    # A[b, m, h] -> Out[b, m, h]
    a_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + h_offsets[None, :] * stride_ah  # shape [M, H]
    out_a_ptrs = Out_ptr + b * stride_ob + m_offsets[:, None] * stride_oc + h_offsets[None, :] * stride_oh
    a_mask = mask_m[:, None] & mask_h[None, :]
    tl.store(out_a_ptrs, tl.load(a_ptrs, mask=a_mask, other=0.0))

    # B[b, n, h] -> Out[b, M + n, h]
    b_ptrs = B_ptr + b * stride_bb + n_offsets[:, None] * stride_bn + h_offsets[None, :] * stride_bh  # shape [N, H]
    out_b_ptrs = Out_ptr + b * stride_ob + (M + n_offsets)[:, None] * stride_oc + h_offsets[None, :] * stride_oh
    b_mask = mask_n[:, None] & mask_h[None, :]
    tl.store(out_b_ptrs, tl.load(b_ptrs, mask=b_mask, other=0.0))


@triton.jit
def _batched_matmul_kernel(
    X_ptr,   # *f32, [B, C, K] where C = M + N
    W_ptr,   # *f32, [K, K] (process_weight)
    P_ptr,   # *f32, [B, C, K]
    B: tl.constexpr,      # batch size
    C: tl.constexpr,      # sequence length (M + N)
    K: tl.constexpr,      # hidden dim
    # Strides for X: [B, C, K]
    stride_xb,  # int
    stride_xc,  # int
    stride_xk,  # int
    # Strides for W: [K, K]
    stride_w0,  # int
    stride_w1,  # int
    # Strides for P: [B, C, K]
    stride_pb,  # int
    stride_pc,  # int
    stride_pk,  # int
    BLOCK_M: tl.constexpr,  # tile over C
    BLOCK_N: tl.constexpr,  # tile over K
    BLOCK_K: tl.constexpr,  # tile over reduction dim K
):
    # Grid: (B, ceil_div(C, BLOCK_M))
    b = tl.program_id(0)
    c_block = tl.program_id(1)

    # tile offsets along sequence and hidden
    m_offsets = c_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M], represent rows in C
    n_offsets = tl.arange(0, BLOCK_N)                      # [BLOCK_N], represent columns in K (hidden)
    k_offsets = tl.arange(0, BLOCK_K)                      # [BLOCK_K], reduction dimension

    # masks
    mask_m = m_offsets < C
    mask_n = n_offsets < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks
    for k0 in range(0, K, BLOCK_K):
        # Load X tile: [BLOCK_M, BLOCK_K] = X[b, m, k]
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + (k0 + k_offsets)[None, :] * stride_xk  # [BLOCK_M, BLOCK_K]
        x_mask = mask_m[:, None] & ((k0 + k_offsets)[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load W tile: [BLOCK_K, BLOCK_N] = W[k, n]
        w_ptrs = W_ptr + (k0 + k_offsets)[:, None] * stride_w0 + n_offsets[None, :] * stride_w1  # [BLOCK_K, BLOCK_N]
        w_mask = ((k0 + k_offsets)[:, None] < K) & (mask_n[None, :])
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate: acc += x_tile @ w_tile
        # x_tile: [BLOCK_M, BLOCK_K], w_tile: [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(x_tile, w_tile)

    # Store results to P[b, m, n]
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(p_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original:
        - Concatenate encoder_hidden_states and hidden_states along sequence.
        - Apply linear projection using Triton GEMM.
        - Split outputs back into encoder and hidden streams.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be CUDA for Triton."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors for deterministic results."

        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]  # text_seq_len
        N = hidden_states.shape[1]          # img_seq_len
        H = hidden_states.shape[2]          # hidden_dim
        C = M + N

        # Ensure contiguity
        A = encoder_hidden_states.contiguous()
        Bt = hidden_states.contiguous()
        W = process_weight.contiguous()  # [H, H]

        # 1) Build concatenated X [B, C, H] using Triton
        X_out = torch.empty((B, C, H), device=hidden_states.device, dtype=torch.float32)

        _concat_seq_kernel[(B,)](
            A, Bt, X_out,
            B, M, N, H,
            A.stride(0), A.stride(1), A.stride(2),
            Bt.stride(0), Bt.stride(1), Bt.stride(2),
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            num_warps=1, num_stages=1
        )

        # 2) Compute P = X @ W^T using Triton
        P = torch.empty((B, C, H), device=hidden_states.device, dtype=torch.float32)

        # Grid: (B, tiles along C)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (B, triton.cdiv(C, BLOCK_M))

        _batched_matmul_kernel[grid](
            X_out, W, P,
            B, C, H,
            X_out.stride(0), X_out.stride(1), X_out.stride(2),
            W.stride(0), W.stride(1),
            P.stride(0), P.stride(1), P.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 3) Split into encoder and hidden outputs along sequence dim
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
