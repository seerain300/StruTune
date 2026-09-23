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

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of A (sequence positions)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # output columns (hidden_dim)

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A[b, m, k] tile: shape (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + b * A_s0 + m_offsets[:, None] * A_s1 + k_offsets[None, :] * A_s2
        a_mask = (m_offsets[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W[k, n] tile: shape (BLOCK_K, BLOCK_N)
        w_ptrs = W_ptr + k_offsets[:, None] * W_s0 + n_offsets[None, :] * W_s1
        w_mask = mask_k[:, None] & (n_offsets[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        acc += tl.dot(a, w)

    # Store results to C[b, m, n]
    c_ptrs = C_ptr + b * C_s0 + m_offsets[:, None] * C_s1 + n_offsets[None, :] * C_s2
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < K)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,             # [B, I, K]
        encoder_hidden_states: torch.Tensor,     # [B, T, K]
        process_weight: torch.Tensor,            # [K, K]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1) Concatenate along the sequence dimension to get A: [B, M, K], M = T + I
        # Using torch.cat here is safe and matches the reference.
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]
        A = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, K]

        # 2) Prepare inputs for Triton: cast to float32 and ensure contiguous
        A_fp32 = A.to(torch.float32)
        W_fp32 = process_weight.to(torch.float32)

        # Allocate output C: [B, M, K] in float32
        B = A_fp32.shape[0]
        M = A_fp32.shape[1]
        C = torch.empty((B, M, K), dtype=torch.float32, device=A_fp32.device)

        # Choose block sizes; can be tuned later
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        # Launch Triton kernel
        grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        batched_matmul_kernel[grid](
            C, A_fp32, W_fp32,
            B, M, K,
            A_fp32.stride(0), A_fp32.stride(1), A_fp32.stride(2),
            W_fp32.stride(0), W_fp32.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split back into two streams: processed_encoder [B, T, K], processed_hidden [B, I, K]
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast outputs back to original dtype (match reference)
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden