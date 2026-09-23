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
    BLOCK_M: tl.constexpr,  # tile over sequence (M + N)
    BLOCK_N: tl.constexpr,  # tile over hidden
):
    # Grid: (B, ceil_div(M + N, BLOCK_M))
    b = tl.program_id(0)
    m_block = tl.program_id(1)

    # Sequence offsets this program handles
    seq_start = m_block * BLOCK_M
    m_offsets = seq_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_seq = m_offsets < (M + N)

    # Hidden offsets
    h_offsets = tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_h = h_offsets < H

    # Build 2D pointers for Out
    out_ptrs = Out_ptr + b * stride_ob + m_offsets[:, None] * stride_oc + h_offsets[None, :] * stride_oh  # [BLOCK_M, BLOCK_N]

    # Source selection: which rows come from A vs B
    from_A = m_offsets[:, None] < M  # [BLOCK_M, 1] -> broadcast
    from_B = ~from_A

    # Load from A: A[b, m, h]
    a_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + h_offsets[None, :] * stride_ah
    a_mask = mask_seq[:, None] & mask_h[None, :]
    a_vals = tl.load(a_ptrs, mask=a_mask & from_A, other=0.0)

    # Load from B: B[b, m - M, h]
    b_ptrs = B_ptr + b * stride_bb + (m_offsets[:, None] - M) * stride_bn + h_offsets[None, :] * stride_bh
    b_mask = mask_seq[:, None] & mask_h[None, :]
    b_vals = tl.load(b_ptrs, mask=b_mask & from_B, other=0.0)

    # Select and store
    vals = tl.where(from_A, a_vals, b_vals)
    tl.store(out_ptrs, vals, mask=mask_seq[:, None] & mask_h[None, :])


@triton.jit
def _batched_gemm_3d_kernel(
    X_ptr,  # *f32, [B, C, K]
    W_ptr,  # *f32, [K, K] (process_weight)
    P_ptr,  # *f32, [B, C, K]
    B: tl.constexpr,          # batch size (constexpr for specialization)
    C: tl.constexpr,          # sequence length (M + N)
    K: tl.constexpr,          # hidden dim (constexpr)
    stride_xb,    # int: stride along batch for X
    stride_xc,    # int: stride along seq for X
    stride_xk,    # int: stride along hidden for X
    stride_w0,    # int: stride along rows for W (K)
    stride_w1,    # int: stride along cols for W (K)
    stride_pb,    # int: stride along batch for P
    stride_pc,    # int: stride along seq for P
    stride_pk,    # int: stride along hidden for P
    BLOCK_M: tl.constexpr,  # tile over C (sequence)
    BLOCK_N: tl.constexpr,  # tile over K (output features)
    BLOCK_K: tl.constexpr,  # tile over reduction K
):
    # Grid: (B, ceil_div(C, BLOCK_M), ceil_div(K, BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    mask_m = m_offsets < C
    mask_n = n_offsets < K

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < K

        # Load X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + k_offsets[None, :] * stride_xk
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load W tile as (K, N_TILE): [BLOCK_K, BLOCK_N]
        # W is [K, K], we want W[k, n] to form B[k, n]
        w_ptrs = W_ptr + k_offsets[:, None] * stride_w0 + n_offsets[None, :] * stride_w1
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate: acc += x_tile @ w_tile
        acc += tl.dot(x_tile, w_tile)  # [BLOCK_M, BLOCK_N]

    # Store results
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    p_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(p_ptrs, acc, mask=p_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version that:
          - Concatenates encoder_hidden_states and hidden_states along the sequence dimension using Triton.
          - Applies the linear projection with a Triton GEMM (to ensure heavy computation is in Triton).
          - Splits the result back into (processed_encoder, processed_hidden).
        """
        # Ensure CUDA and dtype
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA for Triton."
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        H = hidden_states.shape[2]
        # Make inputs contiguous (use float32)
        A = encoder_hidden_states.contiguous()  # [B, M, H]
        Bseq = hidden_states.contiguous()       # [B, N, H]

        # Allocate output X [B, C, H]
        C = M + N
        X = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton concatenation kernel
        BLOCK_M = 128
        BLOCK_N = 128
        grid_concat = (B, triton.cdiv(C, BLOCK_M))
        _concat_seq_kernel[grid_concat](
            A, Bseq, X,
            B, M, N, H,
            A.stride(0), A.stride(1), A.stride(2),
            Bseq.stride(0), Bseq.stride(1), Bseq.stride(2),
            X.stride(0), X.stride(1), X.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # Prepare output P [B, C, H]
        P = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

        # Strides for X [B, C, H]
        stride_xb, stride_xc, stride_xk = X.stride(0), X.stride(1), X.stride(2)
        # Strides for W [H, H]
        stride_w0, stride_w1 = process_weight.stride(0), process_weight.stride(1)
        # Strides for P [B, C, H]
        stride_pb, stride_pc, stride_pk = P.stride(0), P.stride(1), P.stride(2)

        # Launch Triton GEMM kernel: P = X @ W^T
        BLOCK_Mg = 64
        BLOCK_Ng = 64
        BLOCK_Kg = 64
        grid_gemm = (B, triton.cdiv(C, BLOCK_Mg), triton.cdiv(H, BLOCK_Ng))
        _batched_gemm_3d_kernel[grid_gemm](
            X, process_weight, P,
            B, C, H,
            stride_xb, stride_xc, stride_xk,
            stride_w0, stride_w1,
            stride_pb, stride_pc, stride_pk,
            BLOCK_M=BLOCK_Mg, BLOCK_N=BLOCK_Ng, BLOCK_K=BLOCK_Kg,
            num_warps=4, num_stages=2,
        )

        # Split along sequence dimension
        processed_encoder = P[:, :M, :]  # [B, M, H]
        processed_hidden = P[:, M:, :]   # [B, N, H]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
