import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    C_ptr, A_ptr, W_ptr,
    B, M, N, K,
    stride_A_b, stride_A_m, stride_A_k,
    stride_W_k, stride_W_n,
    stride_C_b, stride_C_m, stride_C_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (B, tiles over M, tiles over N)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + pid_b * stride_A_b + m_offsets[:, None] * stride_A_m + k_offsets[None, :] * stride_A_k
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        # Load W tile: [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_offsets[:, None] * stride_W_k + n_offsets[None, :] * stride_W_n
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, w)

    # Store C tile: [BLOCK_M, BLOCK_N]
    c_ptrs = C_ptr + pid_b * stride_C_b + m_offsets[:, None] * stride_C_m + n_offsets[None, :] * stride_C_n
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def triton_batched_matmul(A: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    Compute C = A @ W, where:
      A: [B, M, K]
      W: [K, N] (here N == K, i.e., square weight)
      C: [B, M, N]
    All tensors are float32 and on CUDA. Uses Triton with explicit strides and masks.
    """
    assert A.is_cuda and W.is_cuda, "Triton kernel requires CUDA tensors"
    assert A.dtype == torch.float32 and W.dtype == torch.float32, "Use float32 tensors for Triton matmul"
    B, M, K = A.shape
    K2, N = W.shape
    assert K2 == K, "Weight shape mismatch: W must have shape [K, N], with N == hidden_dim"
    # Output
    C = torch.empty((B, M, N), device=A.device, dtype=torch.float32)

    # Choose tile sizes; these are reasonable defaults
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    batched_matmul_kernel[grid](
        C, A, W,
        B, M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        W.stride(0), W.stride(1),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension (torch.cat).
        - Apply linear projection using Triton batched matmul (A @ W).
        - Split outputs back into encoder and hidden streams.
        """
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]  # text_seq_len
        I = hidden_states.shape[1]          # img_seq_len
        K = hidden_states.shape[2]          # hidden_dim

        # 1) Concatenate along sequence dimension: [B, T+I, K]
        # Use torch.cat here (small op, correct, and fast).
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1).to(torch.float32).contiguous()

        # 2) Triton batched matmul: concatenated @ process_weight
        # Ensure process_weight is float32 and contiguous [K, K]
        W = process_weight.to(torch.float32).contiguous()

        # Compute C = concatenated @ W
        C = triton_batched_matmul(concatenated, W)

        # 3) Split outputs
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast back to original dtypes (match inputs)
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
