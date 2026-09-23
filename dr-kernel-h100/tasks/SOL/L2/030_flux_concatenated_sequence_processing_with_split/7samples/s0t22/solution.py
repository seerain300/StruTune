import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel_3d(
    C_ptr,     # *fp32, output [B, M, K]
    A_ptr,     # *fp32, input [B, M, K]
    W_ptr,     # *fp32, weight [K, K]
    B: tl.constexpr,   # batch size
    M: tl.constexpr,   # total sequence length after concat
    K: tl.constexpr,   # hidden_dim
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    A_stride_b: tl.constexpr,
    A_stride_m: tl.constexpr,
    A_stride_k: tl.constexpr,
    W_stride0: tl.constexpr,
    W_stride1: tl.constexpr,
    C_stride_b: tl.constexpr,
    C_stride_m: tl.constexpr,
    C_stride_k: tl.constexpr,
):
    # 3D grid: (B, tiles along M, tiles along N)
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for boundaries
    mask_m = m_offsets < M
    mask_n = n_offsets < K

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A[b, m, k] tile
        A_ptrs = A_ptr + b * A_stride_b + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        A_tile = tl.load(A_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)

        # Load W[k, n] tile
        W_ptrs = W_ptr + k_offsets[:, None] * W_stride0 + n_offsets[None, :] * W_stride1
        W_tile = tl.load(W_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, W_tile)

    # Store result to C[b, m, n]
    C_ptrs = C_ptr + b * C_stride_b + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_k
    tl.store(C_ptrs, acc, mask=(mask_m[:, None] & mask_n[None, :]))


@triton.jit
def concat_seq_dim1_torch_only(encoder_hidden_states, hidden_states, out):
    # This is a torch-only helper for concatenation to ensure correctness.
    # We'll call it from ModelNew.forward before launching Triton matmul.
    # Note: Since Triton doesn't support per-lane branching for source selection cleanly here,
    # using torch.cat ensures correctness and avoids illegal memory access in Triton kernels.
    # out = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    pass  # Placeholder, not used in forward


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenate along sequence dimension (dim=1) using torch for robustness.
        - Perform batched matmul in Triton: C = A @ process_weight.T
        - Split outputs back and return.
        """
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]
        M = T + I

        # 1) Concatenate along sequence dimension
        # out shape: [B, M, K]
        A = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # Ensure A and process_weight are float32 for Triton (Triton kernels typically use fp32)
        A = A.to(torch.float32)
        process_weight = process_weight.to(torch.float32)

        # 2) Prepare output C [B, M, K]
        C = torch.empty((B, M, K), device=A.device, dtype=torch.float32)

        # 3) Launch Triton batched matmul kernel: C = A @ process_weight.T
        # Note: process_weight is [K, K], A is [B, M, K], so matmul: [B, M, K] @ [K, K] -> [B, M, K]
        # Grid: (B, tiles along M, tiles along N=K)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        batched_matmul_kernel_3d[grid](
            C, A, process_weight,
            B, M, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            A_stride_b=A.stride(0), A_stride_m=A.stride(1), A_stride_k=A.stride(2),
            W_stride0=process_weight.stride(0), W_stride1=process_weight.stride(1),
            C_stride_b=C.stride(0), C_stride_m=C.stride(1), C_stride_k=C.stride(2),
            num_warps=4, num_stages=2
        )

        # 4) Split outputs back
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Cast back to original dtypes (optional: keep fp32 if that's the model's dtype)
        # Return as original input dtypes
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
