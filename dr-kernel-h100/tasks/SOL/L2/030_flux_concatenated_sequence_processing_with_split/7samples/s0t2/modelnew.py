import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_kernel(
    out_ptr,         # *float32
    x1_ptr,          # *float32, encoder_hidden_states: [B, T, K]
    x2_ptr,          # *float32, hidden_states: [B, I, K]
    B: tl.constexpr, # batch size (for grid only)
    T: tl.constexpr, # text_seq_len
    I: tl.constexpr, # img_seq_len
    K: tl.constexpr, # hidden_dim
):
    # Grid: (B, ceil((T+I)/BLOCK_M), ceil(K/BLOCK_N))
    # Each program handles one batch b and a tile along the concatenated sequence dimension.
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    BLOCK_M = 64
    BLOCK_N = 64

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < (T + I)
    mask_n = n_offsets < K

    # For each m in [0, T+I), decide source: from x1 if m < T, else from x2
    is_x1 = m_offsets[:, None] < T  # shape [BLOCK_M, 1], broadcast over N

    # Base offsets for x1 and x2: both have stride 1 over K, but we need correct b
    # x1[b, m, n] -> ((b * T + m) * K) + n
    # x2[b, m, n] -> ((b * I + m) * K) + n
    # out[b, m, n] -> (((b * (T+I)) + m) * K) + n

    # We'll load with masking: load from x1 where is_x1, else load from x2 where ~is_x1
    # Create pointers for x1 and x2 loads
    x1_ptrs = x1_ptr + (b * T + m_offsets[:, None]) * K + n_offsets[None, :]
    x2_ptrs = x2_ptr + (b * I + m_offsets[:, None]) * K + n_offsets[None, :]

    # Masks for loads
    mask_x1 = mask_m[:, None] & mask_n[None, :] & is_x1
    mask_x2 = mask_m[:, None] & mask_n[None, :] & (~is_x1)

    # Load with defaults; Triton will use masked loads
    vals_x1 = tl.load(x1_ptrs, mask=mask_x1, other=0.0)
    vals_x2 = tl.load(x2_ptrs, mask=mask_x2, other=0.0)

    # Select value for each m: vals_x1 if is_x1, else vals_x2
    # vals_x2 has zeros where mask_x2 is False (i.e., where is_x1 is True).
    vals = tl.where(is_x1, vals_x1, vals_x2)

    # Store into out
    out_ptrs = out_ptr + (b * (T + I) + m_offsets[:, None]) * K + n_offsets[None, :]
    tl.store(out_ptrs, vals, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def batched_matmul_kernel(
    C_ptr,           # *float32, output: [B, M, K]
    A_ptr,           # *float32, input: [B, M, K]
    W_ptr,           # *float32, weight: [K, K]
    B: tl.constexpr, # batch size
    M: tl.constexpr, # sequence length (T+I)
    N: tl.constexpr, # K (hidden_dim)
    K: tl.constexpr, # K (hidden_dim)
    # tiling parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, ceil(M/BLOCK_M), ceil(N/BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)

        # A tiles: A[b, m, k]
        a_ptrs = A_ptr + (b * M + m[:, None]) * K + k[None, :]
        a_mask = (m[:, None] < M) & (k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # W tiles: W[k, n]
        w_ptrs = W_ptr + k[:, None] * N + n[None, :]
        w_mask = (k[:, None] < K) & (n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, w)

    # Store result
    c_ptrs = C_ptr + (b * M + m[:, None]) * N + n[None, :]
    c_mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized forward:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dimension using a Triton kernel.
        2) Apply linear projection using a Triton GEMM: C = A @ process_weight.
        3) Split C back into processed_encoder_hidden_states and processed_hidden_states.

        Args:
            hidden_states: [B, I, K]
            encoder_hidden_states: [B, T, K]
            process_weight: [K, K]
        Returns:
            (processed_encoder_hidden_states: [B, T, K], processed_hidden_states: [B, I, K])
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors."
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D tensors [B, dim, K]."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == K and process_weight.shape[0] == K and process_weight.shape[1] == K, "Hidden dim mismatch."

        # Allocate concatenated A: [B, M, K]
        M = T + I
        A = torch.empty((B, M, K), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch concat kernel
        BLOCK_M_C = 64
        BLOCK_N_C = 64
        grid_concat = (B, triton.cdiv(M, BLOCK_M_C), triton.cdiv(K, BLOCK_N_C))
        concat_seq_kernel[grid_concat](
            A, encoder_hidden_states, hidden_states,
            B, T, I, K
        )

        # Allocate output C: [B, M, K]
        C = torch.empty((B, M, K), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch GEMM kernel: C = A @ process_weight
        # We use autotune-friendly static block sizes; matmul is the heavy compute.
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        batched_matmul_kernel[grid_gemm](
            C, A, process_weight,
            B, M, K, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Split back
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]
        return processed_encoder, processed_hidden