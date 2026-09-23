import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_kernel(
    out_ptr,            # *fp32, output A: [B, M, K]
    x1_ptr,             # *fp32, encoder_hidden_states: [B, T, K]
    x2_ptr,             # *fp32, hidden_states: [B, I, K]
    B: tl.int32,
    T: tl.int32,        # text_seq_len
    I: tl.int32,        # img_seq_len
    K: tl.int32,        # hidden_dim
    out_stride_b: tl.int32, out_stride_m: tl.int32, out_stride_k: tl.int32,
    x1_stride_b: tl.int32, x1_stride_t: tl.int32, x1_stride_k: tl.int32,
    x2_stride_b: tl.int32, x2_stride_i: tl.int32, x2_stride_k: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Grid: (B, ceil(M / BLOCK_M), ceil(K / BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    k_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # along sequence dimension
    k_offsets = k_block * BLOCK_N + tl.arange(0, BLOCK_N)  # along hidden_dim

    # 2D tile indices
    m = m_offsets[:, None]  # shape [BLOCK_M, 1]
    n = k_offsets[None, :]  # shape [1, BLOCK_N]

    # Mask for valid m and n
    mask_m = m < (T + I)
    mask_n = n < K
    mask = mask_m & mask_n  # ensure valid stores

    # Determine source (encoder vs image) for each m
    is_encoder = m < T  # shape [BLOCK_M, 1]

    # Compute pointers for loads
    # For encoder: x1[b, m, k]
    x1_ptrs = x1_ptr + b * x1_stride_b + m * x1_stride_t + n * x1_stride_k
    # For image: x2[b, m-T, k]
    x2_ptrs = x2_ptr + b * x2_stride_b + (m - T) * x2_stride_i + n * x2_stride_k

    # Load with masks; out-of-range rows get 0
    x1_vals = tl.load(x1_ptrs, mask=(m < T) & mask, other=0.0)
    x2_vals = tl.load(x2_ptrs, mask=(m >= T) & mask, other=0.0)
    # Select based on is_encoder
    vals = tl.where(is_encoder, x1_vals, x2_vals)

    # Store to out[b, m, k]
    out_ptrs = out_ptr + b * out_stride_b + m * out_stride_m + n * out_stride_k
    tl.store(out_ptrs, vals, mask=mask)


@triton.jit
def batched_matmul_kernel(
    C_ptr, A_ptr, W_ptr,
    B: tl.int32, M: tl.int32, N: tl.int32, K: tl.int32,
    A_stride_b: tl.int32, A_stride_m: tl.int32, A_stride_k: tl.int32,
    W_stride_k: tl.int32, W_stride_n: tl.int32,
    C_stride_b: tl.int32, C_stride_m: tl.int32, C_stride_n: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, ceil(M / BLOCK_M), ceil(N / BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of C
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # cols of C

    m = m_offsets[:, None]  # [BLOCK_M, 1]
    n = n_offsets[None, :]  # [1, BLOCK_N]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k = k_offsets[None, :]                  # [1, BLOCK_K]

        # Pointers for A[b, m, k] and W[k, n]
        A_ptrs = A_ptr + b * A_stride_b + m * A_stride_m + k * A_stride_k        # [BM, BK]
        W_ptrs = W_ptr + k * W_stride_k + n * W_stride_n                          # [BK, BN]

        # Masks for valid loads
        A_mask = (m < M) & (k < K)
        W_mask = (k < K) & (n < N)
        mask = A_mask & W_mask

        # Load tiles
        A_tile = tl.load(A_ptrs, mask=mask, other=0.0)  # [BM, BK]
        W_tile = tl.load(W_ptrs, mask=mask.T, other=0.0)  # [BK, BN], .T to align dims

        # Accumulate
        acc += tl.dot(A_tile, W_tile)  # [BM, BN]

    # Store results to C
    C_ptrs = C_ptr + b * C_stride_b + m * C_stride_m + n * C_stride_n
    C_mask = (m < M) & (n < N)
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Performs:
          1) Concatenation of encoder_hidden_states and hidden_states along sequence dim.
          2) Linear projection via matmul with process_weight.
          3) Splits back into encoder and image outputs.
        """
        # Ensure inputs are float32 for Triton kernels; keep device consistent
        device = hidden_states.device
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]
        M = T + I

        # Make inputs contiguous and cast to fp32 for compute
        x1 = encoder_hidden_states.contiguous().to(torch.float32)
        x2 = hidden_states.contiguous().to(torch.float32)
        W = process_weight.contiguous().to(torch.float32)

        # Allocate output A for concatenation: [B, M, K]
        A = torch.empty((B, M, K), device=device, dtype=torch.float32)

        # Launch concat kernel
        BLOCK_M = 128
        BLOCK_N = 64
        grid_concat = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        concat_seq_kernel[grid_concat](
            A, x1, x2,
            B, T, I, K,
            A.stride(0), A.stride(1), A.stride(2),
            x1.stride(0), x1.stride(1), x1.stride(2),
            x2.stride(0), x2.stride(1), x2.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        # Allocate output C for matmul: [B, M, K]
        C = torch.empty((B, M, K), device=device, dtype=torch.float32)

        # Launch batched matmul kernel
        BLOCK_M_G = 64
        BLOCK_N_G = 64
        BLOCK_K_G = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M_G), triton.cdiv(K, BLOCK_N_G))
        batched_matmul_kernel[grid_gemm](
            C, A, W,
            B, M, K, K,
            A.stride(0), A.stride(1), A.stride(2),   # A is [B, M, K]
            W.stride(0), W.stride(1),                # W is [K, K]
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
            num_warps=4, num_stages=2
        )

        # Split outputs
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast back to original dtype if needed
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden