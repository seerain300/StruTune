import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    C_ptr, A_ptr, W_ptr,
    B, M, K, N,  # N == K for this task
    A_s0, A_s1, A_s2,
    W_s0, W_s1,
    C_s0, C_s1, C_s2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (B, tiles over M, tiles over N)
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A[b, m, k] and W[k, n]
        A_ptrs = A_ptr + b * A_s0 + m_offsets[:, None] * A_s1 + k_offsets[None, :] * A_s2
        W_ptrs = W_ptr + k_offsets[:, None] * W_s0 + n_offsets[None, :] * W_s1

        # Masks for bounds
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        W_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load tiles; cast to float32 for accumulation
        a = tl.load(A_ptrs, mask=A_mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptrs, mask=W_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, w)

    # Store results to C[b, m, n]
    C_ptrs = C_ptr + b * C_s0 + m_offsets[:, None] * C_s1 + n_offsets[None, :] * C_s2
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,       # [B, I, K]
        encoder_hidden_states: torch.Tensor,  # [B, T, K]
        process_weight: torch.Tensor      # [K, K]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]

        # Concatenate along sequence dimension to form A: [B, M, K], M = T + I
        # Using torch.cat to avoid complex stride handling in Triton kernel.
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, M, K]

        # Ensure inputs for Triton are on CUDA and float32
        device = concatenated.device
        # A (concatenated) might not be contiguous; we make it contiguous for simpler strides
        A = concatenated.contiguous()

        # Output C: [B, M, K], float32 for compute
        C = torch.empty((B, T + I, K), device=device, dtype=torch.float32)

        # Process weight as float32
        W = process_weight.to(torch.float32).contiguous()

        # Launch Triton batched matmul: C = A @ W
        M = T + I
        N = K  # Since W is [K, K], N == K

        # Choose block sizes; these work well for typical dims up to 2048
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        batched_matmul_kernel[grid](
            C, A, W,
            B, M, K, N,
            A.stride(0), A.stride(1), A.stride(2),
            W.stride(0), W.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Split outputs along the sequence dimension
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast back to original dtypes (match inputs)
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
