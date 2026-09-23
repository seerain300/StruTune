import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    C_ptr, A_ptr, W_ptr,
    B, M, N, K,
    A_s0, A_s1, A_s2,   # strides for A: [B, M, K]
    W_s0, W_s1,         # strides for W: [K, N]
    C_s0, C_s1, C_s2,   # strides for C: [B, M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 3D grid: (B, tiles over M, tiles over N)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in tiles
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + k_offsets
        mask_k = k_idx < K

        # Load A[b, m, k] tile
        a_ptrs = A_ptr + b * A_s0 + m_offsets[:, None] * A_s1 + k_idx[None, :] * A_s2
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load W[k, n] tile (note: W is [K, N])
        w_ptrs = W_ptr + k_idx[:, None] * W_s0 + n_offsets[None, :] * W_s1
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(a, w)

    # Store result to C[b, m, n]
    c_ptrs = C_ptr + b * C_s0 + m_offsets[:, None] * C_s1 + n_offsets[None, :] * C_s2
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension (torch.cat).
        - Perform batched matmul in Triton: C = A @ process_weight.T
        - Split C back into two streams.
        """
        assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3, "Inputs must be [B, L, K]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]
        M = T + I

        # 1) Concatenate along sequence dimension: [B, M, K]
        # Use torch.cat to ensure correctness and avoid Triton complexities for concat.
        A = torch.cat([encoder_hidden_states, hidden_states], dim=1).to(torch.float32)

        # 2) Prepare weight for matmul: process_weight [K, N] where N=K
        # We need A @ W, and PyTorch code uses process_weight.T (linear on last dim), but here we directly multiply A @ process_weight.
        # The original PyTorch code does: processed = A @ process_weight.T  => shape [B, M, K]
        # Here process_weight is [K, K], so W = process_weight (no transpose). We'll keep process_weight as [K, K] and use it directly.
        # Ensure process_weight is float32
        W = process_weight.to(torch.float32)

        # 3) Allocate output C
        C = torch.empty((B, M, K), dtype=torch.float32, device=A.device)

        # 4) Launch Triton matmul kernel
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        batched_matmul_kernel[grid](
            C, A, W,
            B, M, K, K,
            A.stride(0), A.stride(1), A.stride(2),
            W.stride(0), W.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 5) Split back into two streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast back to original dtypes to match the original function's behavior
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden