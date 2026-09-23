import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_kernel(
    out_ptr,        # *T, output tensor: [B, M, K], M = T + I
    x1_ptr,         # *T, encoder_hidden_states: [B, T, K]
    x2_ptr,         # *T, hidden_states: [B, I, K]
    B: tl.constexpr,
    T: tl.constexpr,   # text_seq_len
    I: tl.constexpr,   # img_seq_len
    K: tl.constexpr,   # hidden_dim
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
):
    # 3D grid: (B, tiles over M, tiles over N=K)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Build 2D tile indices
    m = m_offsets[:, None]  # [BLOCK_M, 1]
    n = n_offsets[None, :]  # [1, BLOCK_N]

    # Masks for valid rows and columns
    mask_m = m < (T + I)
    mask_n = n < K
    mask = mask_m & mask_n

    # Compute destination offsets for out[b, m, n]
    out_off = b * out.stride(0) + m * out.stride(1) + n * out.stride(2)
    # For source, need to decide x1 or x2 based on m < T
    mask_x1 = mask & (m < T)
    mask_x2 = mask & (m >= T)

    # Load from x1: x1[b, m, n]
    x1_off = b * x1.stride(0) + m * x1.stride(1) + n * x1.stride(2)
    val1 = tl.load(x1_ptr + x1_off, mask=mask_x1, other=0)

    # Load from x2: x2[b, m - T, n]
    m2 = m - T  # for encoder part, m2 will be negative for m < T (already masked)
    x2_off = b * x2.stride(0) + m2 * x2.stride(1) + n * x2.stride(2)
    val2 = tl.load(x2_ptr + x2_off, mask=mask_x2, other=0)

    # Select between val1 and val2
    val = tl.where(mask_x1, val1, tl.where(mask_x2, val2, 0))

    # Store to out
    tl.store(out_ptr + out_off, val, mask=mask)


@triton.jit
def batched_matmul_kernel(
    C_ptr,          # *T, output: [B, M, K]
    A_ptr,          # *T, input: [B, M, K] (concatenated)
    W_ptr,          # *T, weight: [K, K]
    B: tl.constexpr,    # batch size
    M: tl.constexpr,    # sequence length (T + I)
    K: tl.constexpr,    # hidden_dim
    BLOCK_M: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 64,
):
    # Grid: (B, tiles over M, tiles over N)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    m = m_offsets[:, None]   # [BLOCK_M, 1]
    n = n_offsets[None, :]   # [1, BLOCK_N]

    # Masks
    mask_m = m < M
    mask_n = n < K
    mask = mask_m & mask_n

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k = k_offsets[None, :]                  # [1, BLOCK_K]

        # Load A tile: A[b, m, k] -> shape [BLOCK_M, BLOCK_K]
        A_off = b * A.stride(0) + m * A.stride(1) + k * A.stride(2)
        A_mask = mask_m & (k < K)
        A_tile = tl.load(A_ptr + A_off, mask=A_mask[:, None], other=0.0)

        # Load W tile: W[k, n] -> shape [BLOCK_K, BLOCK_N]
        W_off = k * W.stride(0) + n * W.stride(1)
        W_mask = (k < K) & mask_n
        W_tile = tl.load(W_ptr + W_off, mask=W_mask[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, W_tile)

    # Store results
    C_off = b * C.stride(0) + m * C.stride(1) + n * C.stride(2)
    tl.store(C_ptr + C_off, acc, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.

        - Concatenates encoder_hidden_states and hidden_states along sequence dimension in Triton.
        - Applies linear projection via batched GEMM in Triton: C = A @ process_weight.
        - Splits the result back into encoder and hidden streams.

        Args:
            hidden_states: [batch, img_seq_len, hidden_dim]
            encoder_hidden_states: [batch, text_seq_len, hidden_dim]
            process_weight: [hidden_dim, hidden_dim] (no bias)

        Returns:
            Tuple (processed_encoder_hidden_states, processed_hidden_states)
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "Inputs must be CUDA tensors for Triton execution."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]
        M = T + I

        # Ensure tensors are contiguous to simplify stride handling in Triton
        x1 = encoder_hidden_states.contiguous()
        x2 = hidden_states.contiguous()
        W = process_weight.contiguous()

        # 1) Concatenate along sequence dimension using Triton
        A = torch.empty((B, M, K), device=hidden_states.device, dtype=hidden_states.dtype)
        BLOCK_M_C = 64
        BLOCK_N_C = 64
        grid_concat = (B, triton.cdiv(M, BLOCK_M_C), triton.cdiv(K, BLOCK_N_C))
        concat_seq_kernel[grid_concat](
            A, x1, x2, B, T, I, K, BLOCK_M=BLOCK_M_C, BLOCK_N=BLOCK_N_C
        )

        # 2) Batched GEMM: C = A @ W
        C = torch.empty((B, M, K), device=hidden_states.device, dtype=hidden_states.dtype)
        BLOCK_M_G = 64
        BLOCK_N_G = 64
        BLOCK_K_G = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M_G), triton.cdiv(K, BLOCK_N_G))
        batched_matmul_kernel[grid_gemm](
            C, A, W, B, M, K,
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
            num_warps=4, num_stages=2
        )

        # 3) Split back into two streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]
        return processed_encoder, processed_hidden