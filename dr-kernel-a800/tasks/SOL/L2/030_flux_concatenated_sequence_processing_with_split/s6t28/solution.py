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
    BLOCK_M: tl.constexpr,  # tile along sequence (C dimension)
):
    # Grid: (B, ceil_div(C, BLOCK_M))
    b = tl.program_id(0)
    c_block = tl.program_id(1)

    # Sequence offsets for this tile
    m_offsets = c_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M], corresponds to Out[:, m_offsets, :]
    # Hidden dimension
    h = tl.arange(0, H)  # [H]

    # Determine which positions come from A (encoder) vs B (image)
    from_A = m_offsets < M  # boolean vector [BLOCK_M]

    # Build pointers for A and B
    # A[b, m, h] with m in [0, M)
    a_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + h[None, :] * stride_ah
    a_mask = (m_offsets[:, None] < M) & (h[None, :] < H)
    a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)

    # B[b, n, h] with n in [0, N), Out index m = M + n for m >= M
    b_ptrs = B_ptr + b * stride_bb + (m_offsets - M)[:, None] * stride_bn + h[None, :] * stride_bh
    b_mask = (~from_A[:, None]) & (h[None, :] < H)
    b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)

    vals = tl.where(from_A[:, None], a_vals, b_vals)  # [BLOCK_M, H]

    # Store into Out[b, m_offsets, h]
    out_ptrs = Out_ptr + b * stride_ob + m_offsets[:, None] * stride_oc + h[None, :] * stride_oh
    # We only store valid m_offsets < C; since from_A selects m < M, and (~from_A) selects m >= M, overall m_offsets must be < C
    out_mask = (m_offsets[:, None] < C) & (h[None, :] < H)
    tl.store(out_ptrs, vals, mask=out_mask)


@triton.jit
def _batched_matmul_kernel(
    X_ptr,      # *f32, [B, C, K] where K = H
    W_ptr,      # *f32, [K, K] (process_weight)
    P_ptr,      # *f32, [B, C, K]
    B: tl.constexpr,            # batch size (not used directly; we only need C, K as constexpr here)
    C: tl.constexpr,            # sequence length (M + N)
    K: tl.constexpr,            # hidden dim (H)
    stride_xb,  # int: stride for batch in X
    stride_xc,  # int: stride for seq in X
    stride_xk,  # int: stride for hidden in X
    stride_w0,  # int: stride for dim 0 (rows) in W
    stride_w1,  # int: stride for dim 1 (cols) in W
    stride_pb,  # int: stride for batch in P
    stride_pc,  # int: stride for seq in P
    stride_pk,  # int: stride for hidden in P
    BLOCK_M: tl.constexpr,      # tile along C (sequences)
    BLOCK_N: tl.constexpr,      # tile along K (hidden)
    BLOCK_K: tl.constexpr,      # tile along reduction dim
):
    # Grid: (batch, tiles along hidden dim)
    b = tl.program_id(0)
    n_block = tl.program_id(1)

    # Tile offsets for hidden and output sequence
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N], output hidden indices
    # We will compute acc of shape [BLOCK_M, BLOCK_N] and fill all sequence positions
    # For each m in [0, C), compute X[b, m, :] @ W^T and store into P[b, m, :]
    # Loop over hidden tiles to build X tiles and accumulate
    # However, Triton does not support loops with runtime bounds; we can iterate using range(0, K, BLOCK_K)
    # and in each iteration accumulate for all m. To do that, we create masks and pointers accordingly.
    # Simpler: for each k chunk, load X rows and W rows, and accumulate into a vector of size BLOCK_N.
    # But since we need [BLOCK_M, BLOCK_N], we loop over m in chunks too. We’ll use a nested approach.

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # We need to compute acc += sum_{k=k0..k0+BLOCK_K} X[b, m, k] * W[k, :]
        # For each m in tile, load X row and corresponding W chunk, then outer product and add to acc.
        # Since BLOCK_M can be larger than C, we must guard by m < C. But here we only store m < C.
        # We can compute per m and accumulate.

        # For each m in tile, compute:
        m0 = 0
        while m0 < C:
            m_offsets = m0 + tl.arange(0, BLOCK_M)  # [BLOCK_M]
            mask_m = m_offsets < C

            # Load X tile: X[b, m, k] for m in m_offsets, k in k_offsets -> shape [BLOCK_M, BLOCK_K]
            x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + k_offsets[None, :] * stride_xk
            x_mask = mask_m[:, None] & (k_offsets[None, :] < K)
            x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

            # Load W chunk: W[k, n_offsets] -> shape [BLOCK_K, BLOCK_N]
            w_ptrs = W_ptr + k_offsets[:, None] * stride_w0 + n_offsets[None, :] * stride_w1
            w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < K)
            w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

            # Accumulate: acc[m, n] += sum_k x_vals[m, k] * w_vals[k, n]
            # x_vals: [BM, BK], w_vals: [BK, BN] -> tl.dot(x_vals, w_vals) -> [BM, BN]
            acc += tl.dot(x_vals, w_vals)

            m0 += BLOCK_M

    # Now store acc into P[b, :, :]
    # We need to write acc for all sequence m and hidden n_offsets
    # Loop over m again to store since we computed acc in chunks of m.
    m0 = 0
    while m0 < C:
        m_offsets = m0 + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        mask_m = m_offsets < C

        # Construct P pointers for [BLOCK_M, BLOCK_N]
        p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
        store_mask = mask_m[:, None] & (n_offsets[None, :] < K)
        tl.store(p_ptrs, acc, mask=store_mask)

        m0 += BLOCK_M


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton implementation:
          concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, C, H], C = M + N
          processed = concatenated @ process_weight.T                             # [B, C, H]
          processed_encoder = processed[:, :M, :]
          processed_hidden = processed[:, M:, :]
        We perform concatenation in Triton and GEMM in Triton. No torch ops in forward.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."

        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        H = hidden_states.shape[2]  # hidden_dim
        C = M + N

        # Ensure contiguous
        A = encoder_hidden_states.contiguous()  # [B, M, H]
        Bt = hidden_states.contiguous()        # [B, N, H]
        W = process_weight.contiguous()        # [H, H]

        # Allocate output for concatenation
        Out = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton concatenation kernel
        BLOCK_M = 128  # tile along C dimension
        grid_cat = (B, triton.cdiv(C, BLOCK_M))
        _concat_seq_kernel[grid_cat](
            A, Bt, Out,
            B=B, M=M, N=N, H=H,
            stride_ab=A.stride(0), stride_am=A.stride(1), stride_ah=A.stride(2),
            stride_bb=Bt.stride(0), stride_bn=Bt.stride(1), stride_bh=Bt.stride(2),
            stride_ob=Out.stride(0), stride_oc=Out.stride(1), stride_oh=Out.stride(2),
            BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=2
        )

        # Prepare inputs for Triton GEMM: X = Out [B, C, H], W [H, H], output P [B, C, H]
        P = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton GEMM kernel: grid over (batch, tiles over hidden dim)
        BLOCK_M_gemm = 64   # tile along sequence (C) per program
        BLOCK_N_gemm = 64   # tile along hidden (K) per store
        BLOCK_K_gemm = 64   # reduction tile
        grid_gemm = (B, triton.cdiv(H, BLOCK_N_gemm))

        _batched_matmul_kernel[grid_gemm](
            Out, W, P,
            C=C, K=H,
            stride_xb=Out.stride(0), stride_xc=Out.stride(1), stride_xk=Out.stride(2),
            stride_w0=W.stride(0), stride_w1=W.stride(1),
            stride_pb=P.stride(0), stride_pc=P.stride(1), stride_pk=P.stride(2),
            BLOCK_M=BLOCK_M_gemm, BLOCK_N=BLOCK_N_gemm, BLOCK_K=BLOCK_K_gemm,
            num_warps=4, num_stages=2
        )

        # Split back into two streams
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
