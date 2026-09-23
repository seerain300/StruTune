import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Tiny tiles for very small M/N
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32},  num_warps=2,  num_stages=2),

        # Balanced tiles
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),

        # Larger tiles for big N/K
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=16, num_stages=5),
    ],
    key=['M', 'N', 'K'],  # tune per problem size
)
@triton.jit
def matmul_bT_kernel(
    A_ptr,  # *fp16, [M, K]
    B_ptr,  # *fp16, [K, N] (we access as B.T via strides)
    C_ptr,  # *fp16, [M, N]
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids for 2D grid over tiles of M and N
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K (compile-time known via static_range)
    for k in tl.static_range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * K) + offs_k[None, :]
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B tile accessed as B.T: we want B[k, n] but we pass B as [K, N], so indexing as B[k, n] with strides
        # B's strides: stride0 = B.stride(0) = N, stride1 = B.stride(1) = 1 for contiguous [K, N]
        # We read B[k, offs_n] for each k in offs_k
        b_ptrs = B_ptr + offs_k[:, None] * N + offs_n[None, :]
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Write back to C (fp16). Mask for boundaries.
    c_ptrs = C_ptr + (offs_m[:, None] * N) + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)  # Triton will cast fp32 to fp16 on store if C_ptr is fp16


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous and on CUDA (Triton requires CUDA)
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton kernel."
        A = A.contiguous()
        B = B.contiguous()

        # Shapes: A [M, K], B [K, N]
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Incompatible shapes: A is [{M}, {K}], B is [{Kb}, {N}]."

        # Output tensor (fp16 to match the original model's dtype in harness)
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Grid: number of tiles in M and N dimensions
        # BLOCK sizes are selected by autotune; we compute grid using assumed max block, but Triton uses meta
        # We launch with a 2D grid; Triton will substitute BLOCK_M/N from chosen config.
        # We need to provide a grid function that depends on meta parameters.
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        matmul_bT_kernel[grid](A, B, C, M, N, K)
        return C


def run(*args):
    return ModelNew()(*args)
