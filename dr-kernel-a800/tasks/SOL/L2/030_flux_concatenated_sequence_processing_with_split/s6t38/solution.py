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
    BLOCK_M: tl.constexpr,  # tile size for M
    BLOCK_N: tl.constexpr,  # tile size for N (here H)
):
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    # Offsets in sequence and hidden dims
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Masks
    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # Pointers for A: Out[b, m, h] where m in [0, M), h in [0, H)
    a_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + n_offsets[None, :] * stride_ah
    a_vals = tl.load(a_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    # Pointers for B: Out[b, m, h] where m in [M, M+N), h in [0, H)
    # For Out, seq index is m_offsets + M for B part
    b_ptrs = B_ptr + b * stride_bb + (m_offsets[:, None] - M) * stride_bn + n_offsets[None, :] * stride_bh
    b_vals = tl.load(b_ptrs, mask=(m_offsets[:, None] >= M) & mask_m[:, None] & mask_n[None, :], other=0.0)

    # Select A for m<M, B for m>=M
    from_A = m_offsets[:, None] < M
    vals = tl.where(from_A, a_vals, b_vals)  # [BLOCK_M, BLOCK_N]

    # Store to Out at [b, m, h]
    out_ptrs = Out_ptr + b * stride_ob + m_offsets[:, None] * stride_oc + n_offsets[None, :] * stride_oh
    tl.store(out_ptrs, vals, mask=(mask_m[:, None] & mask_n[None, :]))


@triton.jit
def _batched_gemm_kernel(
    X_ptr,        # *f32, [B, C, K] where C = M + N
    W_ptr,        # *f32, [K, K] (process_weight)
    P_ptr,        # *f32, [B, C, K]
    B: tl.constexpr,      # batch size
    C: tl.constexpr,      # sequence length
    K: tl.constexpr,      # hidden dim (constexpr)
    # Strides for X: [B, C, K]
    stride_xb,    # int
    stride_xc,    # int
    stride_xk,    # int
    # Strides for W: [K, K]
    stride_w0,    # int: stride along rows
    stride_w1,    # int: stride along cols
    # Strides for P: [B, C, K]
    stride_pb,    # int
    stride_pc,    # int
    stride_pk,    # int
    BLOCK_M: tl.constexpr,  # tile over C (sequences)
    BLOCK_N: tl.constexpr,  # tile over K (hidden dim)
    BLOCK_K: tl.constexpr,  # tile over reduction dim K
):
    # Grid: (B, ceil_div(C, BLOCK_M))
    b = tl.program_id(0)
    m_block = tl.program_id(1)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)          # [BLOCK_M]
    n_offsets = tl.arange(0, BLOCK_N)                              # [BLOCK_N] full K tile
    k_offsets = tl.arange(0, BLOCK_K)                              # [BLOCK_K] reduction tile

    # Masks for boundaries
    mask_m = m_offsets < C
    mask_n = n_offsets < K

    # Initialize accumulator [BLOCK_M, BLOCK_N] in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + k_offsets  # [BLOCK_K]
        mask_k = k_idx < K

        # Load X tile: X[b, m, k] -> [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + k_idx[None, :] * stride_xk
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W tile: W[k, n] -> [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_idx[:, None] * stride_w0 + n_offsets[None, :] * stride_w1
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate: acc += x_tile @ w_tile
        # x_tile: [BM, BK], w_tile: [BK, BN] -> [BM, BN]
        acc += tl.dot(x_tile, w_tile)

    # Store result to P[b, m, n] where n = n_offsets
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(p_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton implementation:
        1) Concatenate encoder_hidden_states and hidden_states along sequence into X [B, C, H].
        2) Compute P = X @ process_weight.T using a Triton GEMM kernel.
        3) Split P into processed_encoder [B, M, H] and processed_hidden [B, N, H].
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton."
        # Ensure dtype float32 for predictable behavior
        dtype = torch.float32
        A = encoder_hidden_states.contiguous().to(dtype)
        B = hidden_states.contiguous().to(dtype)
        W = process_weight.contiguous().to(dtype)

        Bsz = A.shape[0]
        M = A.shape[1]
        N = B.shape[1]
        H = A.shape[2]
        assert W.shape == (H, H), "process_weight must have shape [hidden_dim, hidden_dim]."

        # 1) Concatenate along sequence dimension: X [B, C, H], C = M + N
        C = M + N
        X_out = torch.empty((Bsz, C, H), dtype=dtype, device=A.device)

        # Launch concat kernel
        # Grid over (batch, tiles along M, tiles along N)
        # Choose BLOCK sizes to cover typical dims; masks will handle edges
        BLOCK_M = 64
        BLOCK_N = 64
        grid_concat = (Bsz, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _concat_seq_kernel[grid_concat](
            A, B, X_out,
            B=Bsz, M=M, N=N, H=H,
            stride_ab=A.stride(0), stride_am=A.stride(1), stride_ah=A.stride(2),
            stride_bb=B.stride(0), stride_bn=B.stride(1), stride_bh=B.stride(2),
            stride_ob=X_out.stride(0), stride_oc=X_out.stride(1), stride_oh=X_out.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # 2) GEMM: P = X @ W^T, output [B, C, H]
        P_out = torch.empty((Bsz, C, H), dtype=dtype, device=A.device)

        # Strides for X [B, C, K]
        stride_xb = X_out.stride(0)
        stride_xc = X_out.stride(1)
        stride_xk = X_out.stride(2)

        # Strides for W [K, K]
        stride_w0 = W.stride(0)
        stride_w1 = W.stride(1)

        # Strides for P [B, C, K]
        stride_pb = P_out.stride(0)
        stride_pc = P_out.stride(1)
        stride_pk = P_out.stride(2)

        # Tiling parameters for GEMM
        BLOCK_M_GEMM = 64
        BLOCK_N_GEMM = 64
        BLOCK_K_GEMM = 64

        # Launch GEMM kernel: grid over (batch, tiles along C)
        grid_gemm = (Bsz, triton.cdiv(C, BLOCK_M_GEMM))
        _batched_gemm_kernel[grid_gemm](
            X_out, W, P_out,
            B=Bsz, C=C, K=H,
            stride_xb=stride_xb, stride_xc=stride_xc, stride_xk=stride_xk,
            stride_w0=stride_w0, stride_w1=stride_w1,
            stride_pb=stride_pb, stride_pc=stride_pc, stride_pk=stride_pk,
            BLOCK_M=BLOCK_M_GEMM, BLOCK_N=BLOCK_N_GEMM, BLOCK_K=BLOCK_K_GEMM,
            num_warps=4, num_stages=2,
        )

        # 3) Split outputs
        processed_encoder = P_out[:, :M, :]  # [B, M, H]
        processed_hidden = P_out[:, M:, :]   # [B, N, H]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
