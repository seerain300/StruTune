import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_seq_kernel(
    A_ptr,         # *f32, [B, M, H]
    B_ptr,         # *f32, [B, N, H]
    Out_ptr,       # *f32, [B, C, H], C = M + N
    B: tl.constexpr,    # batch size
    M: tl.constexpr,    # text_seq_len
    N: tl.constexpr,    # img_seq_len
    H: tl.constexpr,    # hidden_dim
    stride_ab,     # int: stride for batch in A
    stride_am,     # int: stride for seq in A
    stride_ah,     # int: stride for hidden in A
    stride_bb,     # int: stride for batch in B
    stride_bn,     # int: stride for seq in B
    stride_bh,     # int: stride for hidden in B
    stride_ob,     # int: stride for batch in Out
    stride_oc,     # int: stride for seq in Out
    stride_oh,     # int: stride for hidden in Out
    BLOCK_M: tl.constexpr,  # tile size along sequence for A/B copies
):
    # 2D grid: (B, ceil_div(C, BLOCK_M)), each program copies a block of rows for one batch
    b = tl.program_id(0)
    block_m = tl.program_id(1)

    # We split into two regions in Out: [0:M) from A and [M:M+N) from B
    # For A: copy A[b, m, :] into Out[b, m, :]
    m_offsets = block_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_m = m_offsets < M

    # Loop over each m offset and copy the entire hidden dim H
    for i in range(0, BLOCK_M):
        m_idx = m_offsets[i]
        if mask_m[i]:
            a_row = A_ptr + b * stride_ab + m_idx * stride_am
            a_col = tl.arange(0, H) * stride_ah
            a_ptrs = a_row + a_col
            a_vals = tl.load(a_ptrs)  # load full row

            out_row = Out_ptr + b * stride_ob + m_idx * stride_oc
            out_col = tl.arange(0, H) * stride_oh
            out_ptrs = out_row + out_col
            tl.store(out_ptrs, a_vals)

    # For B: copy B[b, n, :] into Out[b, M + n, :]
    n_offsets = block_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_n = n_offsets < N

    # Map n_offsets to Out rows at index M + n_offsets
    for i in range(0, BLOCK_M):
        n_idx = n_offsets[i]
        if mask_n[i]:
            b_row = B_ptr + b * stride_bb + n_idx * stride_bn
            b_col = tl.arange(0, H) * stride_bh
            b_ptrs = b_row + b_col
            b_vals = tl.load(b_ptrs)

            out_row = Out_ptr + b * stride_ob + (M + n_idx) * stride_oc
            out_col = tl.arange(0, H) * stride_oh
            out_ptrs = out_row + out_col
            tl.store(out_ptrs, b_vals)


@triton.jit
def _batched_matmul_kernel(
    X_ptr,  # *f32, [B, C, K] where C = M + N and K = H
    W_ptr,  # *f32, [K, K] (process_weight)
    P_ptr,  # *f32, [B, C, K]
    B: tl.constexpr,    # batch size
    C: tl.constexpr,    # sequence length (M + N)
    K: tl.constexpr,    # hidden dim (H)
    stride_xb,  # int
    stride_xc,  # int
    stride_xk,  # int
    stride_w0,  # int: stride along dim 0 of W (rows)
    stride_w1,  # int: stride along dim 1 of W (cols)
    stride_pb,  # int
    stride_pc,  # int
    stride_pk,  # int
    BLOCK_M: tl.constexpr,  # tile along C
    BLOCK_N: tl.constexpr,  # tile along K (output feature)
    BLOCK_K: tl.constexpr,  # reduction tile along K (hidden)
):
    # Grid: (B, ceil_div(C, BLOCK_M))
    b = tl.program_id(0)
    c_block = tl.program_id(1)

    m_offsets = c_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = tl.arange(0, BLOCK_N)                      # [BLOCK_N]

    mask_m = m_offsets < C
    mask_n = n_offsets < K

    # Accumulator [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < K

        # Load X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + k_offsets[None, :] * stride_xk
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K], float32

        # Load W tile (we need W[k, n]): [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_offsets[:, None] * stride_w0 + n_offsets[None, :] * stride_w1
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N], float32

        # Accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(x_tile, w_tile)

    # Store results to P: P[b, m, n]
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    p_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(p_ptrs, acc, mask=p_mask)


def triton_run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-optimized version of the original run function.
    Returns (processed_encoder, processed_hidden).
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA device."
    B = hidden_states.shape[0]
    M = encoder_hidden_states.shape[1]
    N = hidden_states.shape[1]
    H = hidden_states.shape[2]
    assert encoder_hidden_states.shape[2] == H, "Hidden dims must match between encoder_hidden_states and hidden_states."
    assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]."

    # Ensure contiguous and float32
    A = encoder_hidden_states.contiguous().to(torch.float32)
    Bn = hidden_states.contiguous().to(torch.float32)
    W = process_weight.contiguous().to(torch.float32)

    # Allocate output for concatenation
    C = M + N
    Out = torch.empty((B, C, H), device=A.device, dtype=torch.float32)

    # Launch concatenation kernel
    BLOCK_M = 64  # tile along sequence to copy
    grid_concat = (B, triton.cdiv(C, BLOCK_M))
    _concatenate_seq_kernel[grid_concat](
        A, Bn, Out,
        B=B, M=M, N=N, H=H,
        stride_ab=A.stride(0), stride_am=A.stride(1), stride_ah=A.stride(2),
        stride_bb=Bn.stride(0), stride_bn=Bn.stride(1), stride_bh=Bn.stride(2),
        stride_ob=Out.stride(0), stride_oc=Out.stride(1), stride_oh=Out.stride(2),
        BLOCK_M=BLOCK_M,
        num_warps=1, num_stages=1,
    )

    # Allocate result P [B, C, H]
    P = torch.empty((B, C, H), device=A.device, dtype=torch.float32)

    # Launch GEMM kernel
    BLOCK_M = 64  # tile over C (sequence)
    BLOCK_N = 64  # tile over output hidden (same H)
    BLOCK_K = 64  # reduction tile over K (hidden)
    grid_gemm = (B, triton.cdiv(C, BLOCK_M))
    _batched_matmul_kernel[grid_gemm](
        Out, W, P,
        B=B, C=C, K=H,
        stride_xb=Out.stride(0), stride_xc=Out.stride(1), stride_xk=Out.stride(2),
        stride_w0=W.stride(0), stride_w1=W.stride(1),
        stride_pb=P.stride(0), stride_pc=P.stride(1), stride_pk=P.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    # Split along sequence dimension
    processed_encoder = P[:, :M, :]
    processed_hidden = P[:, M:, :]
    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # Original Model.forward signature: run(hidden_states, encoder_hidden_states, process_weight)
        # So we map:
        # args[0] -> encoder_hidden_states [B, M, H]
        # args[1] -> hidden_states [B, N, H]
        # args[2] -> process_weight [H, H]
        encoder_hidden_states = args[0]
        hidden_states = args[1]
        process_weight = args[2]
        return triton_run(hidden_states, encoder_hidden_states, process_weight)


def run(*args):
    return ModelNew()(*args)
