import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seq_kernel(
    A_ptr,        # *f32, [B, M, H]
    B_ptr,        # *f32, [B, N, H]
    Out_ptr,      # *f32, [B, C, H], C = M + N
    B: tl.constexpr,    # batch size
    M: tl.constexpr,    # text_seq_len
    N: tl.constexpr,    # img_seq_len
    H: tl.constexpr,    # hidden_dim
    stride_ab,    # int: stride along batch for A
    stride_am,    # int: stride along seq for A
    stride_ah,    # int: stride along hidden for A
    stride_bb,    # int: stride along batch for B
    stride_bn,    # int: stride along seq for B
    stride_bh,    # int: stride along hidden for B
    stride_ob,    # int: stride along batch for Out
    stride_oc,    # int: stride along seq for Out
    stride_oh,    # int: stride along hidden for Out
    BLOCK_M: tl.constexpr,  # tile over sequences
):
    # Grid: (B, ceil_div(M + N, BLOCK_M))
    b = tl.program_id(0)
    m_block = tl.program_id(1)

    # Offsets within C (sequence dimension)
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    C = M + N
    store_mask = m_offsets < C

    # For each position, decide source: A if m_offsets < M, else B
    from_A = m_offsets < M

    # Loop over hidden dimension H
    for h in range(0, H):
        # A: Out[b, m, h] = A[b, m, h] for m < M
        a_ptrs = A_ptr + b * stride_ab + m_offsets * stride_am + h * stride_ah
        a_mask = store_mask & from_A
        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)  # shape [BLOCK_M]

        # B: Out[b, m, h] = B[b, m - M, h] for m >= M
        b_ptrs = B_ptr + b * stride_bb + (m_offsets - M) * stride_bn + h * stride_bh
        b_mask = store_mask & (~from_A)
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)  # shape [BLOCK_M]

        # Select
        out_vals = tl.where(from_A, a_vals, b_vals)  # [BLOCK_M]

        # Store to Out
        out_ptrs = Out_ptr + b * stride_ob + m_offsets * stride_oc + h * stride_oh
        tl.store(out_ptrs, out_vals, mask=store_mask)


@triton.jit
def _batched_gemm_kernel_outer(
    X_ptr,      # *f32, [B, C, K]
    W_ptr,      # *f32, [K, K]
    P_ptr,      # *f32, [B, C, K]
    B: tl.constexpr,      # batch size
    C: tl.constexpr,      # sequence length
    K: tl.constexpr,      # hidden_dim (constexpr)
    stride_xb,  # int: stride for batch in X
    stride_xc,  # int: stride for seq in X
    stride_xk,  # int: stride for hidden in X
    stride_w0,  # int: stride along dim 0 (rows) for W
    stride_w1,  # int: stride along dim 1 (cols) for W
    stride_pb,  # int: stride along batch for P
    stride_pc,  # int: stride along seq for P
    stride_pk,  # int: stride along hidden for P
    BLOCK_M: tl.constexpr,  # tile over C (sequences)
    BLOCK_N: tl.constexpr,  # tile over K (output features)
    BLOCK_K: tl.constexpr,  # tile over reduction dim (input features)
):
    # 3D grid: (B, ceil_div(C, BLOCK_M), ceil_div(K, BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_m = m_offsets < C
    mask_n = n_offsets < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_ids < K

        # Load X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + k_ids[None, :] * stride_xk
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load W tile: [BLOCK_K, BLOCK_N], W is [K, K]
        w_ptrs = W_ptr + k_ids[:, None] * stride_w0 + n_offsets[None, :] * stride_w1
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Explicit outer-product accumulation over kk in this chunk
        for kk in range(0, BLOCK_K):
            # Vector x_col: [BLOCK_M]
            x_col = x_tile[:, kk]  # masked by x_mask
            # Vector w_row: [BLOCK_N]
            w_row = w_tile[kk, :]  # masked by w_mask

            # Outer product contribution: [BLOCK_M, BLOCK_N]
            contrib = x_col[:, None] * w_row[None, :]
            acc += contrib

    # Store acc to P
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    p_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(p_ptrs, acc, mask=p_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenates encoder_hidden_states and hidden_states along sequence dim in Triton.
        - Performs batched GEMM X @ process_weight.T in Triton.
        - Splits result back into (processed_encoder, processed_hidden).
        """
        # Ensure Triton inputs: CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton kernels."
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3 and process_weight.dim() == 2
        B = hidden_states.shape[0]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "Hidden dimensions must match."
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]."

        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        C = M + N

        # 1) Concatenate along sequence dim into Out [B, C, H]
        Out = torch.empty((B, C, H), device=hidden_states.device, dtype=torch.float32)

        grid_concat = (B, triton.cdiv(C, 64))  # tile along C by 64
        _concat_seq_kernel[grid_concat](
            encoder_hidden_states, hidden_states, Out,
            B, M, N, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            Out.stride(0), Out.stride(1), Out.stride(2),
            BLOCK_M=64,
            num_warps=2,
        )

        # 2) GEMM: P = Out @ process_weight.T, P [B, C, H]
        P = torch.empty((B, C, H), device=hidden_states.device, dtype=torch.float32)

        grid_gemm = (B, triton.cdiv(C, 64), triton.cdiv(H, 64))
        _batched_gemm_kernel_outer[grid_gemm](
            Out, process_weight, P,
            B, C, H,
            Out.stride(0), Out.stride(1), Out.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            P.stride(0), P.stride(1), P.stride(2),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
            num_warps=4,
        )

        # 3) Split along sequence dim
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
