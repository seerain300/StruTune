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
    BLOCK_M: tl.constexpr,  # tile along sequence dimension (C)
):
    # grid over batch and tiles over C
    b = tl.program_id(0)
    m_block = tl.program_id(1)

    # offsets in sequence dimension
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    C = M + N
    mask_m = m_offsets < C

    # For each sequence position m, determine if it comes from A or B
    from_A = m_offsets < M  # boolean [BLOCK_M]

    # Loop over hidden dimension H (columns)
    for h in range(0, H):
        # Compute source pointers
        # A[b, m, h]
        a_ptrs = A_ptr + b * stride_ab + m_offsets * stride_am + h * stride_ah
        # B[b, m - M, h]
        b_ptrs = B_ptr + b * stride_bb + (m_offsets - M) * stride_bn + h * stride_bh

        # Load values with masks
        a_vals = tl.load(a_ptrs, mask=mask_m & from_A, other=0.0)
        b_vals = tl.load(b_ptrs, mask=mask_m & (~from_A), other=0.0)
        vals = a_vals + b_vals  # combine from A/B

        # Store to Out[b, m, h]
        out_ptrs = Out_ptr + b * stride_ob + m_offsets * stride_oc + h * stride_oh
        tl.store(out_ptrs, vals, mask=mask_m)


@triton.jit
def _batched_matmul_kernel_blocked(
    X_ptr,      # *f32, [B, C, K] where C = M + N
    W_ptr,      # *f32, [K, K] (process_weight)
    P_ptr,      # *f32, [B, C, K]
    B: tl.constexpr,    # batch size
    C: tl.constexpr,    # sequence length
    K: tl.constexpr,    # hidden dim
    stride_xb,   # int: stride along batch for X
    stride_xc,   # int: stride along seq for X
    stride_xk,   # int: stride along hidden for X
    stride_w0,   # int: stride along dim 0 for W (rows)
    stride_w1,   # int: stride along dim 1 for W (cols)
    stride_pb,   # int: stride along batch for P
    stride_pc,   # int: stride along seq for P
    stride_pk,   # int: stride along hidden for P
    BLOCK_M: tl.constexpr,  # tile over C
    BLOCK_N: tl.constexpr,  # tile over K
    BLOCK_K: tl.constexpr,  # tile over reduction dim
):
    # 3D grid: (batch, tiles over C, tiles over K)
    b = tl.program_id(0)
    c_block = tl.program_id(1)
    k_block = tl.program_id(2)

    # tile offsets
    c_offsets = c_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = k_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # masks for boundaries
    mask_c = c_offsets < C
    mask_k = k_offsets < K

    # Initialize accumulator [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        # Load X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + b * stride_xb + c_offsets[:, None] * stride_xc + (k0 + tl.arange(0, BLOCK_K))[None, :] * stride_xk
        k_range = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_x = (c_offsets[:, None] < C) & (k_range[None, :] < K)
        x_vals = tl.load(x_ptrs, mask=mask_x, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load W tile as [BLOCK_K, BLOCK_N]: W[k, n]
        w_ptrs = W_ptr + (k0 + tl.arange(0, BLOCK_K))[:, None] * stride_w0 + k_offsets[None, :] * stride_w1
        mask_w = (k_range[:, None] < K) & (k_offsets[None, :] < K)
        w_vals = tl.load(w_ptrs, mask=mask_w, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(x_vals, w_vals)  # [BLOCK_M, BLOCK_N]

    # Store acc to P[b, c_offsets, k_offsets]
    p_ptrs = P_ptr + b * stride_pb + c_offsets[:, None] * stride_pc + k_offsets[None, :] * stride_pk
    mask_p = (c_offsets[:, None] < C) & (k_offsets[None, :] < K)
    tl.store(p_ptrs, acc, mask=mask_p)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure CUDA and float32, contiguous
        device = hidden_states.device
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]  # text_seq_len
        N = hidden_states.shape[1]          # img_seq_len
        H = encoder_hidden_states.shape[2]  # hidden_dim (same as hidden_states.shape[2])
        C = M + N

        # Allocate output for concatenation
        Out = torch.empty((B, C, H), dtype=torch.float32, device=device)

        # Launch Triton concatenation kernel
        # Use a moderate BLOCK_M to cover sequence tiles; masks handle tails.
        BLOCK_M = 128
        grid = (B, triton.cdiv(C, BLOCK_M))
        _concat_seq_kernel[grid](
            encoder_hidden_states, hidden_states, Out,
            B=B, M=M, N=N, H=H,
            stride_ab=encoder_hidden_states.stride(0), stride_am=encoder_hidden_states.stride(1), stride_ah=encoder_hidden_states.stride(2),
            stride_bb=hidden_states.stride(0), stride_bn=hidden_states.stride(1), stride_bh=hidden_states.stride(2),
            stride_ob=Out.stride(0), stride_oc=Out.stride(1), stride_oh=Out.stride(2),
            BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=2
        )

        # Prepare output for GEMM
        P = torch.empty((B, C, H), dtype=torch.float32, device=device)

        # Launch Triton GEMM kernel: P = Out @ W^T
        # Use conservative block sizes for robustness across H up to 4096.
        BLOCK_M_GEMM = 64
        BLOCK_N_GEMM = 64
        BLOCK_K_GEMM = 64
        grid_gemm = (B, triton.cdiv(C, BLOCK_M_GEMM), triton.cdiv(H, BLOCK_N_GEMM))
        _batched_matmul_kernel_blocked[grid_gemm](
            Out, process_weight, P,
            B=B, C=C, K=H,
            stride_xb=Out.stride(0), stride_xc=Out.stride(1), stride_xk=Out.stride(2),
            stride_w0=process_weight.stride(0), stride_w1=process_weight.stride(1),
            stride_pb=P.stride(0), stride_pc=P.stride(1), stride_pk=P.stride(2),
            BLOCK_M=BLOCK_M_GEMM, BLOCK_N=BLOCK_N_GEMM, BLOCK_K=BLOCK_K_GEMM,
            num_warps=4, num_stages=2
        )

        # Split back into encoder and image streams
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
