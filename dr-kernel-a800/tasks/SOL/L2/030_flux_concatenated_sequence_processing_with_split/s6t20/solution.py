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
    BLOCK_M: tl.constexpr,  # tile size along seq (A)
    BLOCK_N: tl.constexpr,  # tile size along seq (B)
):
    # Grid: (B, ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    # Compute sequence offsets for A and B
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Masks for boundaries
    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # Hidden dimension indices
    h = tl.arange(0, H)  # [H]

    # Load from A: A[b, m_offsets, h]
    a_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + h[None, :] * stride_ah
    a_mask = mask_m[:, None] & (h[None, :] < H)
    a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, H]

    # Load from B: B[b, n_offsets, h]
    b_ptrs = B_ptr + b * stride_bb + n_offsets[:, None] * stride_bn + h[None, :] * stride_bh
    b_mask = mask_n[:, None] & (h[None, :] < H)
    b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_N, H]

    # Compute output positions: Out[b, m_offsets, h] and Out[b, M + n_offsets, h]
    # For A part (first M sequences)
    out_a_ptrs = Out_ptr + b * stride_ob + m_offsets[:, None] * stride_oc + h[None, :] * stride_oh
    # For B part (next N sequences)
    out_b_ptrs = Out_ptr + b * stride_ob + (M + n_offsets)[:, None] * stride_oc + h[None, :] * stride_oh

    # Store from A and B
    tl.store(out_a_ptrs, a_vals, mask=(mask_m[:, None] & (h[None, :] < H)))
    tl.store(out_b_ptrs, b_vals, mask=(mask_n[:, None] & (h[None, :] < H)))


def _triton_concatenation(
    encoder_hidden_states: torch.Tensor,  # [B, M, H]
    hidden_states: torch.Tensor,         # [B, N, H]
) -> torch.Tensor:
    """
    Triton-orchestrated concatenation along sequence dimension without torch.cat.
    Returns Out [B, M + N, H].
    """
    assert encoder_hidden_states.is_cuda and hidden_states.is_cuda, "Tensors must be CUDA for Triton."
    assert encoder_hidden_states.dtype == torch.float32 and hidden_states.dtype == torch.float32, "Expected float32 tensors."
    assert encoder_hidden_states.is_contiguous() and hidden_states.is_contiguous(), "Expected contiguous tensors."

    B, M, H = encoder_hidden_states.shape
    B2, N, H2 = hidden_states.shape
    assert B == B2 and H == H2, "Batch and hidden_dim must match."

    C = M + N
    Out = torch.empty((B, C, H), device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)

    # Strides
    stride_ab, stride_am, stride_ah = encoder_hidden_states.stride()
    stride_bb, stride_bn, stride_bh = hidden_states.stride()
    stride_ob, stride_oc, stride_oh = Out.stride()

    # Choose tile sizes; masks handle tails
    BLOCK_M = 64
    BLOCK_N = 64

    grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _concat_seq_kernel[grid](
        encoder_hidden_states,
        hidden_states,
        Out,
        B, M, N, H,
        stride_ab, stride_am, stride_ah,
        stride_bb, stride_bn, stride_bh,
        stride_ob, stride_oc, stride_oh,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4,
    )
    return Out


@triton.jit
def _batched_gemm_kernel_3d(
    X_ptr,   # *f32, [B, C, K] where C = M + N, K = H
    W_ptr,   # *f32, [H, H] (process_weight)
    P_ptr,   # *f32, [B, C, K]
    B: tl.constexpr,      # batch size
    C: tl.constexpr,      # sequence length (M + N)
    K: tl.constexpr,      # hidden dim
    stride_xb,  # int: stride along batch for X
    stride_xc,  # int: stride along seq for X
    stride_xk,  # int: stride along hidden for X (K dim)
    stride_w0,  # int: stride along dim 0 of W (rows)
    stride_w1,  # int: stride along dim 1 of W (cols)
    stride_pb,  # int: stride along batch for P
    stride_pc,  # int: stride along seq for P
    stride_pk,  # int: stride along hidden for P
    BLOCK_M: tl.constexpr,  # tile size along seq (C)
    BLOCK_N: tl.constexpr,  # tile size along hidden (K)
    BLOCK_K: tl.constexpr,  # tile size along reduction (K)
):
    # 3D grid: (B, ceil_div(C, BLOCK_M), ceil_div(K, BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    # tile offsets
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    k_offsets = tl.arange(0, BLOCK_K)                     # [BLOCK_K]

    # masks for boundaries
    mask_m = m_offsets < C
    mask_n = n_offsets < K

    # initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + k_offsets  # [BLOCK_K]
        mask_k = k_offsets < K

        # Load X tile: X[b, m, k] -> shape [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + k_offsets[None, :] * stride_xk
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W^T tile: we need W[k, n] to act as W^T[n, k]. W is [H, H], strides (stride_w0=H, stride_w1=1).
        # For W^T[n, k] we access W[k, n] with strides (row=k, col=n).
        w_ptrs = W_ptr + k_offsets[:, None] * stride_w0 + n_offsets[None, :] * stride_w1
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(x_tile, w_tile)

    # Store results to P[b, m, n]
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(p_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version that:
          - Uses Triton to concatenate along sequence dim.
          - Uses Triton matmul for the projection to match PyTorch numerics.
          - Returns (processed_encoder, processed_hidden).
        """
        # Ensure tensors are on CUDA; Triton requires it
        if not hidden_states.is_cuda or not encoder_hidden_states.is_cuda or not process_weight.is_cuda:
            if torch.cuda.is_available():
                hidden_states = hidden_states.cuda(non_blocking=True)
                encoder_hidden_states = encoder_hidden_states.cuda(non_blocking=True)
                process_weight = process_weight.cuda(non_blocking=True)

        # Triton concatenation: Out [B, C, H], C = M + N
        X = _triton_concatenation(encoder_hidden_states, hidden_states)  # [B, C, H]

        # Shapes
        B, C, H = X.shape
        assert process_weight.shape == (H, H), "process_weight must be [H, H] matching hidden_dim."

        # Allocate output for GEMM
        P = torch.empty((B, C, H), device=X.device, dtype=X.dtype)

        # Strides
        stride_xb, stride_xc, stride_xk = X.stride()        # X: [B, C, H]
        stride_pb, stride_pc, stride_pk = P.stride()        # P: [B, C, H]
        # W is [H, H]; in original, W is process_weight [H, H]; we use W^T in kernel.
        stride_w0 = process_weight.stride(0)                # stride along dim 0 (rows) for W
        stride_w1 = process_weight.stride(1)                # stride along dim 1 (cols) for W

        # Tile sizes (conservative for correctness, can be tuned for speed)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        grid = (B, triton.cdiv(C, BLOCK_M), triton.cdiv(H, BLOCK_N))
        _batched_gemm_kernel_3d[grid](
            X, process_weight, P,
            B, C, H,
            stride_xb, stride_xc, stride_xk,
            stride_w0, stride_w1,
            stride_pb, stride_pc, stride_pk,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Split along sequence dimension
        processed_encoder = P[:, :encoder_hidden_states.shape[1], :]  # [B, M, H]
        processed_hidden = P[:, encoder_hidden_states.shape[1]:, :]   # [B, N, H]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
