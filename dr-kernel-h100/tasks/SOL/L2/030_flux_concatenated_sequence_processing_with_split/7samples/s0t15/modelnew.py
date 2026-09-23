import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    C_ptr, A_ptr, W_ptr,
    B: tl.constexpr, M: tl.constexpr, K: tl.constexpr,
    A_s0, A_s1, A_s2,       # strides for A: [B, M, K]
    W_s0, W_s1,             # strides for W: [K, K]
    C_s0, C_s1, C_s2,       # strides for C: [B, M, K]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 3D launch grid: (batch, tiles along M, tiles along N)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of A
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # output columns

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Pointers for A[b, m, k]
        a_ptrs = A_ptr + b * A_s0 + m_offsets[:, None] * A_s1 + k_offsets[None, :] * A_s2
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Pointers for W[k, n]
        w_ptrs = W_ptr + k_offsets[:, None] * W_s0 + n_offsets[None, :] * W_s1
        w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, w)

    # Store result to C[b, m, n]
    c_ptrs = C_ptr + b * C_s0 + m_offsets[:, None] * C_s1 + n_offsets[None, :] * C_s2
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < K)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        1) Concatenate along sequence dimension: A = [encoder_hidden_states, hidden_states] in [B, T+I, K]
        2) Apply linear projection with Triton: C = A @ process_weight.T
        3) Split back into two streams.
        """
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]  # text_seq_len
        I = hidden_states.shape[1]          # img_seq_len
        K = hidden_states.shape[2]          # hidden_dim

        # 1) Concatenate along sequence dimension (dim=1)
        # Ensure dtypes match
        assert encoder_hidden_states.shape[2] == K and hidden_states.shape[2] == K, "Mismatched hidden_dim"
        assert encoder_hidden_states.shape[0] == B and hidden_states.shape[0] == B, "Mismatched batch size"
        A = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, K]

        # 2) Linear projection via Triton: C = A @ process_weight.T
        # Ensure process_weight is [K, K] for this operation (original uses [K, K])
        W = process_weight  # [K, K], typically no bias
        # Allocate output in float32 for numeric stability
        C = torch.empty((B, T + I, K), device=A.device, dtype=torch.float32)

        # Cast A and W to float32 for kernel (compute in fp32)
        A_fp32 = A.to(torch.float32)
        W_fp32 = W.to(torch.float32)

        # Grid: (B, tiles along M, tiles along N)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (B, triton.cdiv(T + I, BLOCK_M), triton.cdiv(K, BLOCK_N))

        batched_matmul_kernel[grid](
            C, A_fp32, W_fp32,
            B, T + I, K,
            A_fp32.stride(0), A_fp32.stride(1), A_fp32.stride(2),
            W_fp32.stride(0), W_fp32.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 3) Split back into two streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast outputs back to original input dtypes (match reference)
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden