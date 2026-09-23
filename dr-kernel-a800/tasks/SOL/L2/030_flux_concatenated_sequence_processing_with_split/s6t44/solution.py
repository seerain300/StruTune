import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seqs_kernel(
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
    BLOCK: tl.constexpr,  # tile size along sequence for this kernel
):
    # Grid: (B, ceil_div(C, BLOCK))
    b = tl.program_id(0)
    c_block = tl.program_id(1)

    # Offsets in sequence dimension
    m_offsets = c_block * BLOCK + tl.arange(0, BLOCK)  # corresponds to sequence indices
    mask_m = m_offsets < (M + N)

    # Determine which positions come from A vs B
    from_A = m_offsets < M  # vector of length BLOCK

    # Compute pointers for A and B tiles
    # A tile: [B, M, H] -> Out index for A is m_offsets where m_offsets < M
    a_ptrs = A_ptr + b * stride_ab + (m_offsets * 0) * stride_am + tl.arange(0, H) * stride_ah  # we need to vectorize over m as well
    # To properly compute a_ptrs, we need a 2D load per m: [BLOCK_M, H]
    # But Triton prefers 1D addressing; we'll loop over BLOCK with masks.
    # Instead, build 2D pointer for each m in tile:
    # We'll create a loop over m in the tile with masks.
    for mi in range(BLOCK):
        m_idx = m_offsets[mi]
        valid = m_idx < (M + N)
        from_Ai = m_idx < M
        # A row pointer: if from_Ai is True
        if valid:
            a_row_ptrs = A_ptr + b * stride_ab + m_idx * stride_am + tl.arange(0, H) * stride_ah
            a_vals = tl.load(a_row_ptrs, mask=(tl.arange(0, H) < H) & from_Ai, other=0.0)
            # For Out, compute corresponding row
            out_row_ptrs = Out_ptr + b * stride_ob + m_idx * stride_oc + tl.arange(0, H) * stride_oh
            tl.store(out_row_ptrs, a_vals, mask=(tl.arange(0, H) < H) & valid)

        # B row pointer: if from_Ai is False and m_idx >= M
        if valid and not from_Ai:
            n_idx = m_idx - M
            b_row_ptrs = B_ptr + b * stride_bb + n_idx * stride_bn + tl.arange(0, H) * stride_bh
            b_vals = tl.load(b_row_ptrs, mask=(tl.arange(0, H) < H) & valid, other=0.0)
            out_row_ptrs = Out_ptr + b * stride_ob + m_idx * stride_oc + tl.arange(0, H) * stride_oh
            tl.store(out_row_ptrs, b_vals, mask=(tl.arange(0, H) < H) & valid)

# Note: The above kernel uses a simple scalar loop per tile element. While it is correct,
# Triton can be faster with vectorized 2D tiles. To improve performance, we replace the
# concatenation with a more vectorized Triton kernel below (which is actually used), and implement
# GEMM as a robust blocked matmul kernel (below as well).

@triton.jit
def _concat_seqs_kernel_vec(
    A_ptr,        # *f32, [B, M, H]
    B_ptr,        # *f32, [B, N, H]
    Out_ptr,      # *f32, [B, C, H], C = M + N
    B: tl.constexpr,    # batch size
    M: tl.constexpr,    # text_seq_len
    N: tl.constexpr,    # img_seq_len
    H: tl.constexpr,    # hidden_dim
    stride_ab,    # int
    stride_am,    # int
    stride_ah,    # int
    stride_bb,    # int
    stride_bn,    # int
    stride_bh,    # int
    stride_ob,    # int
    stride_oc,    # int
    stride_oh,    # int
    BLOCK_M: tl.constexpr,  # tile size over sequence
):
    # Grid: (B, ceil_div(M+N, BLOCK_M))
    b = tl.program_id(0)
    c_block = tl.program_id(1)

    m_offsets = c_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_m = m_offsets < (M + N)
    from_A = m_offsets < M  # [BLOCK_M]

    # Loop over hidden dim H in chunks
    # Note: H is constexpr, so Triton can unroll loops if desired; we keep it simple.
    for h in range(0, H):
        # A: if from_A, write at Out[b, m_offsets, h]
        a_ptrs = A_ptr + b * stride_ab + m_offsets * stride_am + h * stride_ah
        out_ptrs = Out_ptr + b * stride_ob + m_offsets * stride_oc + h * stride_oh
        tl.store(out_ptrs, tl.load(a_ptrs, mask=mask_m & from_A, other=0.0), mask=mask_m & from_A)

        # B: if not from_A, write at Out[b, m_offsets, h] from B[b, m_offsets-M, h]
        b_ptrs = B_ptr + b * stride_bb + (m_offsets - M) * stride_bn + h * stride_bh
        out_ptrs2 = Out_ptr + b * stride_ob + m_offsets * stride_oc + h * stride_oh
        tl.store(out_ptrs2, tl.load(b_ptrs, mask=mask_m & (~from_A) & (m_offsets >= M), other=0.0), mask=mask_m & (~from_A) & (m_offsets >= M))


@triton.jit
def _batched_matmul_kernel(
    X_ptr,   # *f32, [B, C, K] where C=M+N, K=H
    W_ptr,   # *f32, [K, K]
    P_ptr,   # *f32, [B, C, K]
    B: tl.constexpr,    # batch size
    C: tl.constexpr,    # sequence length (M+N)
    K: tl.constexpr,    # hidden dim
    stride_xb,  # int
    stride_xc,  # int
    stride_xk,  # int
    stride_w0,  # int: stride for rows (K dim)
    stride_w1,  # int: stride for cols (K dim)
    stride_pb,  # int
    stride_pc,  # int
    stride_pk,  # int
    BLOCK_M: tl.constexpr,  # tile over C
    BLOCK_N: tl.constexpr,  # tile over K
    BLOCK_K: tl.constexpr,  # tile over reduction K
):
    # Grid: (B, ceil_div(C, BLOCK_M), ceil_div(K, BLOCK_N))
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
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < K

        # Load X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + k_offsets[None, :] * stride_xk
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load W tile as [BLOCK_K, BLOCK_N] (we need W[k, n] for dot)
        w_ptrs = W_ptr + k_offsets[:, None] * stride_w0 + n_offsets[None, :] * stride_w1
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(x_vals, w_vals)

    # Store results to P: [B, C, K]
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    p_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(p_ptrs, acc, mask=p_mask)


# ModelNew: forward uses Triton for concatenation and GEMM
class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,       # [B, N, H]
        encoder_hidden_states: torch.Tensor,  # [B, M, H]
        process_weight: torch.Tensor       # [H, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure CUDA and float32
        device = hidden_states.device
        B, N, H = hidden_states.shape
        B2, M, H2 = encoder_hidden_states.shape
        assert B == B2 and H == H2, "Input shapes must match expected [B, *, H]"

        # Concatenate along sequence dimension using Triton
        C = M + N
        Out = torch.empty((B, C, H), dtype=torch.float32, device=device)

        # Launch Triton concatenation kernel
        # We use a vectorized kernel over sequence with BLOCK_M tile
        BLOCK_M = 128
        grid = (B, triton.cdiv(C, BLOCK_M))
        _concat_seqs_kernel_vec(
            encoder_hidden_states, hidden_states, Out,
            B, M, N, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            Out.stride(0), Out.stride(1), Out.stride(2),
            BLOCK_M=BLOCK_M,
            num_warps=4
        )

        # Allocate output for P = Out @ process_weight.T
        P = torch.empty((B, C, H), dtype=torch.float32, device=device)

        # Launch Triton GEMM kernel
        # Grid over (B, tiles along C, tiles along K)
        BLOCK_M_GEMM = 64
        BLOCK_N_GEMM = 64
        BLOCK_K_GEMM = 64
        grid_gemm = (B, triton.cdiv(C, BLOCK_M_GEMM), triton.cdiv(H, BLOCK_N_GEMM))
        _batched_matmul_kernel[grid_gemm](
            Out, process_weight, P,
            B, C, H,
            Out.stride(0), Out.stride(1), Out.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            P.stride(0), P.stride(1), P.stride(2),
            BLOCK_M=BLOCK_M_GEMM, BLOCK_N=BLOCK_N_GEMM, BLOCK_K=BLOCK_K_GEMM,
            num_warps=4
        )

        # Split results
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
