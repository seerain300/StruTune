import torch

# Triton is required for the numerical computation
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel 1: concatenate [B, M, K] and [B, N, K] along sequence to [B, M+N, K]
# Each program handles a block of sequences for one batch.
@triton.jit
def _concat_sequences_kernel(
    A_ptr,      # pointer to A [B, M, K]
    B_ptr,      # pointer to B [B, N, K]
    Out_ptr,    # pointer to Out [B, C, K], where C = M + N
    B,          # int: batch size
    M,          # int: text_seq_len
    N,          # int: img_seq_len
    K,          # int: hidden_dim
    stride_ab,  # int: stride for batch in A
    stride_am,  # int: stride for seq in A
    stride_ak,  # int: stride for hidden in A
    stride_bb,  # int: stride for batch in B
    stride_bn,  # int: stride for seq in B
    stride_bk,  # int: stride for hidden in B
    stride_ob,  # int: stride for batch in Out
    stride_oc,  # int: stride for seq in Out
    stride_ok,  # int: stride for hidden in Out
    C,          # int: total sequence length (M + N)
    BLOCK_M: tl.constexpr,  # tile size along sequence
):
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = m_offsets < C  # valid positions in the output

    # Determine which positions come from A and which from B
    from_A = m_offsets < M

    # Compute output offsets for each hidden dimension (K) and sequence m
    out_offsets = b * stride_ob + m_offsets[:, None] * stride_oc + tl.arange(0, K)[None, :] * stride_ok  # shape [BLOCK_M, K] by broadcasting

    # Masks for each source
    mask_m = (m_offsets[:, None] < M)
    mask_n = (m_offsets[:, None] >= M)

    # Load from A where applicable
    a_batch_off = b * stride_ab
    a_row_off = m_offsets * stride_am  # shape [BLOCK_M]
    a_col_off = tl.arange(0, K) * stride_ak  # shape [K]
    a_ptrs = A_ptr + a_batch_off + a_row_off[:, None] + a_col_off[None, :]
    a_mask = mask_m & (tl.arange(0, K)[None, :] < K)
    a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, K]

    # Load from B where applicable (shift by M)
    b_row_off = (m_offsets - M) * stride_bn  # valid where from_A is False
    b_col_off = tl.arange(0, K) * stride_bk
    b_ptrs = B_ptr + b * stride_bb + b_row_off[:, None] + b_col_off[None, :]
    b_mask = mask_n & (tl.arange(0, K)[None, :] < K)
    b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_M, K]

    # Select based on from_A
    vals = tl.where(from_A[:, None], a_vals, b_vals)  # [BLOCK_M, K]

    # Store to Out, using mask across m and k
    out_ptrs = Out_ptr + out_offsets
    tl.store(out_ptrs, vals, mask=(mask[:, None] & (tl.arange(0, K)[None, :] < K)))


# Triton kernel 2: batched matmul X[B, C, K] @ W[K, K] -> P[B, C, K]
# Each program handles a [BLOCK_M x BLOCK_N] tile for one batch, accumulating explicitly over K.
@triton.jit
def _batched_gemm_kernel_explicit(
    X_ptr,      # pointer to X [B, C, K] (float32)
    W_ptr,      # pointer to W [K, K] (float32)
    P_ptr,      # pointer to P [B, C, K] (float32)
    B,          # int: batch size
    C,          # int: sequence length (M + N)
    K: tl.constexpr,            # int: hidden dim (constexpr for Triton)
    stride_xb,  # int: stride for batch in X
    stride_xc,  # int: stride for seq in X
    stride_xk,  # int: stride for hidden in X
    stride_w0,  # int: stride for row in W (K, K) row stride
    stride_w1,  # int: stride for col in W (K, K) col stride
    stride_pb,  # int: stride for batch in P
    stride_pc,  # int: stride for seq in P
    stride_pk,  # int: stride for hidden in P
    BLOCK_M: tl.constexpr,  # tile along seq (C)
    BLOCK_N: tl.constexpr,  # tile along hidden (K)
):
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Explicit accumulation over K
    for k in range(0, K):
        # Load X[b, m_offsets, k] -> [BLOCK_M]
        x_ptrs = X_ptr + b * stride_xb + m_offsets * stride_xc + k * stride_xk
        x_mask = m_offsets < C
        x_vec = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M], float32

        # Load W[k, n_offsets] -> [BLOCK_N]
        w_ptrs = W_ptr + k * stride_w0 + n_offsets * stride_w1
        w_mask = n_offsets < K
        w_vec = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_N], float32

        # Outer product and accumulate
        acc += x_vec[:, None] * w_vec[None, :]

    # Store acc to P at (b, m_offsets, n_offsets)
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    p_mask = (m_offsets[:, None] < C) & (n_offsets[None, :] < K)
    tl.store(p_ptrs, acc, mask=p_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run function.
        - Concatenation is done by a Triton kernel.
        - GEMM is done by a Triton kernel with explicit accumulation over K (robust numerics).
        - Outputs are split back into encoder and hidden streams.
        """
        # Ensure inputs are on CUDA and contiguous
        device = hidden_states.device
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors."
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "hidden_states and encoder_hidden_states must be 3D [B, seq_len, H]"
        assert process_weight.dim() == 2, "process_weight must be 2D [H, H]"
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        K = hidden_states.shape[2]

        # Make inputs contiguous and ensure dtype is float32
        A = encoder_hidden_states.contiguous().to(torch.float32)   # [B, M, K]
        Bt = hidden_states.contiguous().to(torch.float32)          # [B, N, K]
        W = process_weight.contiguous().to(torch.float32)          # [K, K]

        # 1) Concatenate A and Bt into X [B, C, K], with Triton
        C = M + N
        X = torch.empty((B, C, K), device=device, dtype=torch.float32)

        # Strides
        stride_ab = A.stride(0)
        stride_am = A.stride(1)
        stride_ak = A.stride(2)
        stride_bb = Bt.stride(0)
        stride_bn = Bt.stride(1)
        stride_bk = Bt.stride(2)
        stride_ob = X.stride(0)
        stride_oc = X.stride(1)
        stride_ok = X.stride(2)

        # Choose tiling for concatenation: along seq dimension
        BLOCK_M = 128  # tile across sequence; masks handle remainder
        grid_concat = (B, triton.cdiv(C, BLOCK_M))
        _concat_sequences_kernel[grid_concat](
            A, Bt, X,
            B, M, N, K,
            stride_ab, stride_am, stride_ak,
            stride_bb, stride_bn, stride_bk,
            stride_ob, stride_oc, stride_ok,
            C,
            BLOCK_M=BLOCK_M,
            num_warps=4,
            num_stages=2,
        )

        # 2) Batched GEMM: X @ W -> P [B, C, K], with Triton (explicit accumulation)
        P = torch.empty((B, C, K), device=device, dtype=torch.float32)

        # Strides for X, W, P
        stride_xb = X.stride(0)
        stride_xc = X.stride(1)
        stride_xk = X.stride(2)
        stride_w0 = W.stride(0)
        stride_w1 = W.stride(1)
        stride_pb = P.stride(0)
        stride_pc = P.stride(1)
        stride_pk = P.stride(2)

        # GEMM tiling parameters
        BLOCK_M_gemm = 64
        BLOCK_N_gemm = 64

        # 3D grid: (B, tiles along C, tiles along K)
        grid_gemm = (B, triton.cdiv(C, BLOCK_M_gemm), triton.cdiv(K, BLOCK_N_gemm))
        _batched_gemm_kernel_explicit[grid_gemm](
            X, W, P,
            B, C,
            K,                       # constexpr for Triton
            stride_xb, stride_xc, stride_xk,
            stride_w0, stride_w1,
            stride_pb, stride_pc, stride_pk,
            BLOCK_M=BLOCK_M_gemm,
            BLOCK_N=BLOCK_N_gemm,
            num_warps=4,
            num_stages=2,
        )

        # 3) Split outputs back
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
