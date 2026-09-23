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
    BLOCK_M: tl.constexpr,  # tile size over sequences per batch
    BLOCK_N: tl.constexpr,  # tile size over hidden dim per batch
):
    # 2D grid: (B, ceil_div(C, BLOCK_M)), where C = M + N
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    C = M + N

    # offsets for m dimension
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_m = m_offsets < C
    from_A = m_offsets < M

    # loop over hidden dimension H; H is constexpr, Triton will unroll
    for h in range(0, H):
        # pointers for A and B at this hidden h
        a_ptrs = A_ptr + b * stride_ab + m_offsets * stride_am + h * stride_ah
        b_ptrs = B_ptr + b * stride_bb + (m_offsets - M) * stride_bn + h * stride_bh

        # masks for A and B based on from_A
        mask_a = mask_m & from_A
        mask_b = mask_m & (~from_A)

        # load values
        a_vals = tl.load(a_ptrs, mask=mask_a, other=0.0)  # [BLOCK_M]
        b_vals = tl.load(b_ptrs, mask=mask_b, other=0.0)  # [BLOCK_M]

        # select based on from_A
        vals = tl.where(from_A, a_vals, b_vals)  # [BLOCK_M]

        # store to Out[b, m_offsets, h]
        out_ptrs = Out_ptr + b * stride_ob + m_offsets * stride_oc + h * stride_oh
        tl.store(out_ptrs, vals, mask=mask_m)


@triton.jit
def _batched_gemm_kernel_2d(
    X_ptr,   # *f32, [B, C, K]
    W_ptr,   # *f32, [K, K]
    P_ptr,   # *f32, [B, C, K]
    B: tl.constexpr,      # batch size
    C: tl.constexpr,      # sequence length (M + N)
    K: tl.constexpr,      # hidden dim
    # Strides for X: (B, C, K)
    stride_xb,  # int
    stride_xc,  # int
    stride_xk,  # int
    # Strides for W: [K, K]
    stride_w0,  # int: row stride (dim 0)
    stride_w1,  # int: col stride (dim 1)
    # Strides for P: [B, C, K]
    stride_pb,  # int
    stride_pc,  # int
    stride_pk,  # int
    BLOCK_M: tl.constexpr,  # tile size over C
    BLOCK_N: tl.constexpr,  # tile size over K
    BLOCK_K: tl.constexpr,  # reduction tile over K
):
    # 2D grid: (B, ceil_div(C, BLOCK_M))
    b = tl.program_id(0)
    m_block = tl.program_id(1)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)      # [BLOCK_M], output rows
    n_offsets = tl.arange(0, BLOCK_N)                          # [BLOCK_N], output cols

    # masks for output tile
    mask_m = m_offsets < C
    mask_n = n_offsets < K

    # initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # reduction over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)                  # [BLOCK_K]

        # load X tile: [BLOCK_M, BLOCK_K] = X[b, m_offsets, k_offsets]
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + k_offsets[None, :] * stride_xk
        x_mask = mask_m[:, None] & (k_offsets[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # load W^T tile: we want W[k_offsets, n_offsets] as [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_offsets[:, None] * stride_w0 + n_offsets[None, :] * stride_w1
        w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # accumulate
        acc += tl.dot(x_tile, w_tile)

    # store acc to P[b, m_offsets, n_offsets]
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(p_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
          - Concatenates encoder_hidden_states and hidden_states along sequence dim in a Triton kernel.
          - Computes processed = concatenated @ process_weight.T in a Triton GEMM kernel.
          - Returns split streams.
        Requirements:
          - All numeric computation is in Triton kernels (no torch.matmul, torch.cat, etc. in forward).
          - Inputs must be CUDA tensors; we enforce .contiguous() for predictable strides.
        """
        # Ensure CUDA tensors and contiguity
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All inputs must be CUDA tensors for Triton kernels."
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape == (B, M, H), "encoder_hidden_states shape must be [B, M, H]"
        assert hidden_states.shape == (B, N, H), "hidden_states shape must be [B, N, H]"
        assert process_weight.shape == (H, H), "process_weight must be [H, H]"

        # Make tensors contiguous and ensure dtype is float32
        A = encoder_hidden_states.contiguous()         # [B, M, H]
        B_img = hidden_states.contiguous()             # [B, N, H]
        W = process_weight.contiguous()                # [H, H]
        if A.dtype != torch.float32:
            A = A.float()
        if B_img.dtype != torch.float32:
            B_img = B_img.float()
        if W.dtype != torch.float32:
            W = W.float()

        device = A.device

        # Allocate concatenated X [B, C, H], C = M + N
        C = M + N
        Out = torch.empty((B, C, H), dtype=torch.float32, device=device)

        # Launch concatenation Triton kernel with conservative tile sizes
        BLOCK_M = 64
        BLOCK_N = 64
        grid_concat = (B, triton.cdiv(C, BLOCK_M))
        _concat_seq_kernel[grid_concat](
            A, B_img, Out,
            B, M, N, H,
            A.stride(0), A.stride(1), A.stride(2),
            B_img.stride(0), B_img.stride(1), B_img.stride(2),
            Out.stride(0), Out.stride(1), Out.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        # Allocate output P [B, C, H]
        P = torch.empty((B, C, H), dtype=torch.float32, device=device)

        # Launch GEMM Triton kernel: P = Out @ W^T
        # Pass B, C, K as constexpr specializations
        grid_gemm = (B, triton.cdiv(C, BLOCK_M))
        _batched_gemm_kernel_2d[grid_gemm](
            Out, W, P,
            B, C, H,  # constexpr specializations
            Out.stride(0), Out.stride(1), Out.stride(2),     # X strides
            W.stride(0), W.stride(1),                         # W strides
            P.stride(0), P.stride(1), P.stride(2),           # P strides
            BLOCK_M=BLOCK_M, BLOCK_N=64, BLOCK_K=64,
            num_warps=4, num_stages=3
        )

        # Split outputs: first M rows are encoder, next N rows are image
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
