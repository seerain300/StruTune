import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_dim1_kernel(
    out_ptr,         # *float32, output A: [B, M, K]
    x1_ptr,          # *float32, encoder_hidden_states: [B, T, K]
    x2_ptr,          # *float32, hidden_states: [B, I, K]
    B: tl.constexpr, # batch size
    T: tl.constexpr, # text_seq_len
    I: tl.constexpr, # img_seq_len
    K: tl.constexpr, # hidden_dim
    BLOCK_M: tl.constexpr,  # tile along M (sequence)
    BLOCK_K: tl.constexpr,  # tile along K (hidden)
):
    # program ids
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    k_block = tl.program_id(2)

    # offsets within tiles
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # along sequence
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)  # along hidden

    # masks for bounds
    mask_m = m_offsets < (T + I)
    mask_k = k_offsets < K

    # decide source: x1 for m < T, x2 otherwise
    use_x1 = m_offsets[:, None] < T
    # pointers for x1 and x2 loads
    x1_ptrs = x1_ptr + b * x1_ptr.stride(0) + m_offsets[:, None] * x1_ptr.stride(1) + k_offsets[None, :] * x1_ptr.stride(2)
    x2_ptrs = x2_ptr + b * x2_ptr.stride(0) + (m_offsets[:, None] - T) * x2_ptr.stride(1) + k_offsets[None, :] * x2_ptr.stride(2)

    # load with mask
    x1_vals = tl.load(x1_ptrs, mask=mask_m[:, None] & mask_k[None, :] & use_x1, other=0.0)
    x2_vals = tl.load(x2_ptrs, mask=mask_m[:, None] & mask_k[None, :] & (~use_x1), other=0.0)

    # select based on use_x1
    A_tile = tl.where(use_x1, x1_vals, x2_vals)

    # store into out (A) with stride
    out_ptrs = out_ptr + b * out_ptr.stride(0) + m_offsets[:, None] * out_ptr.stride(1) + k_offsets[None, :] * out_ptr.stride(2)
    tl.store(out_ptrs, A_tile, mask=mask_m[:, None] & mask_k[None, :])


@triton.jit
def batched_matmul_kernel(
    C_ptr, A_ptr, W_ptr,
    B, M, K, N,
    A_s0, A_s1, A_s2,
    W_s0, W_s1,
    C_s0, C_s1, C_s2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # program ids over batch, M tiles, N tiles
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    # tile offsets
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # along sequence
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # along hidden

    # masks
    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # pointers for A[b, m, k] and W[k, n]
        A_ptrs = A_ptr + b * A_s0 + m_offsets[:, None] * A_s1 + k_offsets[None, :] * A_s2
        W_ptrs = W_ptr + k_offsets[:, None] * W_s0 + n_offsets[None, :] * W_s1

        # loads with masks
        A_tile = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        W_tile = tl.load(W_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # accumulate
        acc += tl.dot(A_tile, W_tile)

    # store results
    C_ptrs = C_ptr + b * C_s0 + m_offsets[:, None] * C_s1 + n_offsets[None, :] * C_s2
    tl.store(C_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenate encoder_hidden_states and hidden_states along sequence dim using Triton.
        - Apply linear projection with process_weight using Triton GEMM.
        - Split and return the two streams.
        """
        # Shapes
        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = encoder_hidden_states.shape[2]

        # Allocate A (concatenated) as float32 for computation; output C also float32
        A = torch.empty((B, T + I, K), device=hidden_states.device, dtype=torch.float32)
        C = torch.empty((B, T + I, K), device=hidden_states.device, dtype=torch.float32)

        # Ensure inputs are float32 for Triton kernels; keep original dtypes for return
        x1 = encoder_hidden_states.to(torch.float32)
        x2 = hidden_states.to(torch.float32)
        W = process_weight.to(torch.float32)

        # Launch concatenation kernel
        BLOCK_M_C = 128
        BLOCK_K_C = 64
        grid_concat = (B, triton.cdiv(T + I, BLOCK_M_C), triton.cdiv(K, BLOCK_K_C))
        concat_seq_dim1_kernel[grid_concat](
            A, x1, x2,
            B=B, T=T, I=I, K=K,
            BLOCK_M=BLOCK_M_C, BLOCK_K=BLOCK_K_C,
            num_warps=4, num_stages=2
        )

        # Launch GEMM kernel: C = A @ W
        # N == K
        BLOCK_M_G = 64
        BLOCK_N_G = 64
        BLOCK_K_G = 64
        grid_gemm = (B, triton.cdiv(T + I, BLOCK_M_G), triton.cdiv(K, BLOCK_N_G))
        batched_matmul_kernel[grid_gemm](
            C, A, W,
            B=B, M=T+I, K=K, N=K,
            A_s0=A.stride(0), A_s1=A.stride(1), A_s2=A.stride(2),
            W_s0=W.stride(0), W_s1=W.stride(1),
            C_s0=C.stride(0), C_s1=C.stride(1), C_s2=C.stride(2),
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
            num_warps=4, num_stages=2
        )

        # Split back into two streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast back to original dtypes to match the original Model's behavior
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden