import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seqs_kernel(
    A_ptr,  # *f32, [B, M, H]
    B_ptr,  # *f32, [B, N, H]
    Out_ptr,  # *f32, [B, C, H]
    B: tl.constexpr,   # batch size
    M: tl.constexpr,   # text_seq_len
    N: tl.constexpr,   # img_seq_len
    H: tl.constexpr,   # hidden_dim
    stride_ab,  # int: stride for batch in A
    stride_am,  # int: stride for seq in A
    stride_ah,  # int: stride for hidden in A
    stride_bb,  # int: stride for batch in B
    stride_bn,  # int: stride for seq in B
    stride_bh,  # int: stride for hidden in B
    stride_ob,  # int: stride for batch in Out
    stride_oc,  # int: stride for seq in Out
    stride_oh,  # int: stride for hidden in Out
    C: tl.constexpr,   # total seq length = M + N
    BLOCK_M: tl.constexpr,  # tile over seq dimension
):
    # Grid is (B, ceil_div(C, BLOCK_M))
    b = tl.program_id(0)
    c_block = tl.program_id(1)

    # Compute sequence indices this program handles
    m_offsets = c_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_c = m_offsets < C

    # Determine which positions come from A and which from B
    from_A = m_offsets < M
    m_from_B = m_offsets - M  # valid only when from_A is False

    # Loop over hidden dimension H and write each column
    # We use a simple loop over h to ensure correctness; H is constexpr so it unrolls.
    for h in range(0, H):
        # Compute pointers for A and B sources
        a_ptrs = A_ptr + b * stride_ab + m_offsets * stride_am + h * stride_ah  # shape [BLOCK_M]
        b_ptrs = B_ptr + b * stride_bb + m_from_B * stride_bn + h * stride_bh  # shape [BLOCK_M]

        # Load values with masks: from_A mask for A, (~from_A) & mask_c for B
        a_vals = tl.load(a_ptrs, mask=mask_c & from_A, other=0.0)  # [BLOCK_M]
        b_vals = tl.load(b_ptrs, mask=mask_c & (~from_A), other=0.0)  # [BLOCK_M]

        # Select source based on from_A
        vals = tl.where(from_A, a_vals, b_vals)  # [BLOCK_M]

        # Store to Out[b, m, h]
        out_ptrs = Out_ptr + b * stride_ob + m_offsets * stride_oc + h * stride_oh
        tl.store(out_ptrs, vals, mask=mask_c)


@triton.jit
def _strided_gemm_bc_ck_kernel(
    X_ptr,  # *f32, [B, C, K] where C = M + N
    W_ptr,  # *f32, [K, K]
    P_ptr,  # *f32, [B, C, K]
    B: tl.constexpr,   # batch size
    C: tl.constexpr,   # sequence length (M + N)
    K: tl.constexpr,   # hidden dim
    # Strides for X: [B, C, K]
    stride_xb,  # int
    stride_xc,  # int
    stride_xk,  # int
    # Strides for W: [K, K]
    stride_w0,  # int: stride along dim 0 (rows)
    stride_w1,  # int: stride along dim 1 (cols)
    # Strides for P: [B, C, K]
    stride_pb,  # int
    stride_pc,  # int
    stride_pk,  # int
    BLOCK_M: tl.constexpr,  # tile over C (sequences)
    BLOCK_N: tl.constexpr,  # tile over K (hidden dim)
    BLOCK_K: tl.constexpr,  # tile over reduction dim K
):
    # 2D grid: (batch, tiles along C)
    b = tl.program_id(0)
    m_block = tl.program_id(1)

    # Tile offsets
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = tl.arange(0, BLOCK_N)  # [BLOCK_N] columns (hidden dim)
    k_offsets = tl.arange(0, BLOCK_K)  # [BLOCK_K] reduction dim

    # Masks for boundaries
    mask_m = m_offsets < C
    mask_n = n_offsets < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        # Compute X tile: [BLOCK_M, BLOCK_K] for X[b, m, k]
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + (k0 + k_offsets[None, :]) * stride_xk
        x_mask = mask_m[:, None] & ( (k0 + k_offsets[None, :]) < K )
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Compute W tile: [BLOCK_K, BLOCK_N] for W[k, n]
        w_ptrs = W_ptr + (k0 + k_offsets[:, None]) * stride_w0 + n_offsets[None, :] * stride_w1
        w_mask = ((k0 + k_offsets[:, None]) < K) & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate: acc += x_tile @ w_tile
        acc += tl.dot(x_tile, w_tile)  # [BLOCK_M, BLOCK_N]

    # Store results to P[b, m, n] with masks
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(p_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [batch, img_seq_len, hidden_dim]
        encoder_hidden_states: [batch, text_seq_len, hidden_dim]
        process_weight: [hidden_dim, hidden_dim]
        Returns (processed_encoder_hidden_states, processed_hidden_states) both [batch, stream_len, hidden_dim]
        """
        # Ensure CUDA tensors and dtype float32 for consistent behavior
        A = encoder_hidden_states.contiguous().to(torch.float32).cuda()
        Bmat = hidden_states.contiguous().to(torch.float32).cuda()
        W = process_weight.contiguous().to(torch.float32).cuda()

        Bsz = A.shape[0]
        M = A.shape[1]
        N = Bmat.shape[1]
        H = A.shape[2]
        device = A.device

        # Allocate output for concatenation: [B, C, H], C = M + N
        C = M + N
        Out = torch.empty((Bsz, C, H), device=device, dtype=torch.float32)

        # Launch Triton concatenation kernel
        BLOCK_M = 128  # tile size over sequence; works for small to large C
        grid_concat = (Bsz, triton.cdiv(C, BLOCK_M))
        _concat_seqs_kernel[grid_concat](
            A, Bmat, Out,
            B=Bsz, M=M, N=N, H=H,
            stride_ab=A.stride(0), stride_am=A.stride(1), stride_ah=A.stride(2),
            stride_bb=Bmat.stride(0), stride_bn=Bmat.stride(1), stride_bh=Bmat.stride(2),
            stride_ob=Out.stride(0), stride_oc=Out.stride(1), stride_oh=Out.stride(2),
            C=C, BLOCK_M=BLOCK_M,
            num_warps=4,
        )

        # Launch Triton GEMM: P = Out @ W^T, shapes: Out[B, C, K], W[K, K], P[B, C, K]
        K = H
        # Choose tile sizes; for H up to 4096, 64x64x64 is a safe default
        BLOCK_M_G = 64
        BLOCK_N_G = 64
        BLOCK_K_G = 64
        grid_gemm = (Bsz, triton.cdiv(C, BLOCK_M_G))
        _strided_gemm_bc_ck_kernel[grid_gemm](
            Out, W, Out,  # write back into same buffer (noalias is fine)
            B=Bsz, C=C, K=K,
            stride_xb=Out.stride(0), stride_xc=Out.stride(1), stride_xk=Out.stride(2),
            stride_w0=W.stride(0), stride_w1=W.stride(1),
            stride_pb=Out.stride(0), stride_pc=Out.stride(1), stride_pk=Out.stride(2),
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
            num_warps=4, num_stages=2,
        )

        # Split back into two streams
        processed_encoder = Out[:, :M, :]
        processed_hidden = Out[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
