import torch
import triton
import triton.language as tl

# 2D tiled matmul kernel computing C = A @ B_T, where B_T is B transposed.
# A: (M, K), B_T: (K, N), C: (M, N)
@triton.jit
def matmul_at_bT_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_btk, stride_btn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program ids for the 2D grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col offsets for this C tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Create pointers to the start of this output tile
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)

    # Initialize accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A tile: shape (BLOCK_M, BLOCK_K)
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        # Pointers for B_T tile: shape (BLOCK_K, BLOCK_N)
        BT_ptrs = BT_ptr + (offs_k[:, None] * stride_btk + offs_n[None, :] * stride_btn)

        # Masks for boundary conditions
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        bt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles; Triton will infer dtype from tensor arguments, here we cast to float32 for accumulation
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)
        BT_tile = tl.load(BT_ptrs, mask=bt_mask, other=0.0)

        # Accumulate: A_tile (M,K) @ BT_tile (K,N) -> (M,N)
        # Explicitly cast to float32 before multiplication to ensure stability
        acc += tl.dot(A_tile.to(tl.float32), BT_tile.to(tl.float32))

    # Store results back, casting to output dtype (assume float16 input/output here)
    # We will store as float32; if output is float16, Triton will cast on store automatically.
    # However, to be explicit, let's cast to float16 in case output tensor is float16.
    C_out = acc  # keep float32 for precision
    # If C tensor is float16, Triton can cast on store; we ensure pointer type via tensor creation in host.
    tl.store(C_ptrs, C_out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute C = A @ B.T using Triton. A: (M, K), B: (N, K), returns C: (M, N).
        """
        # Ensure CUDA tensors
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton."
        # Shapes
        M, K = A.shape
        N, K2 = B.shape
        assert K == K2, "B's second dimension must match A's second dimension (K)."

        # Make B.T contiguous for efficient access
        BT = B.transpose(0, 1).contiguous()  # BT: (K, N)

        # Output tensor: match A's dtype (float16 in provided get_inputs)
        out = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Choose tile sizes; tuned for balance of performance and shared memory
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        # Grid: one program per output tile
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch kernel; pass strides in elements (PyTorch strides are in elements already)
        matmul_at_bT_kernel[grid](
            A, BT, out,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=4,
        )
        return out


def run(*args):
    return ModelNew()(*args)
