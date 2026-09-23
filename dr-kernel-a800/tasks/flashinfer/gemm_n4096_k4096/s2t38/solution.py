import torch
import triton
import triton.language as tl


@triton.jit
def matmul_bT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program ids for 2D tiling over (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col offsets for this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A and B tiles
        # A is [M, K], B is [K, N]; we index B as B[k, n] with strides (stride_bk, stride_bn)
        A_tile_ptr = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        B_tile_ptr = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        # Masks for bounds
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load and cast to fp32 for accumulation
        A_tile = tl.load(A_tile_ptr, mask=a_mask, other=0.0)
        B_tile = tl.load(B_tile_ptr, mask=b_mask, other=0.0)
        A_tile = A_tile.to(tl.float32)
        B_tile = B_tile.to(tl.float32)

        # Accumulate
        acc += tl.dot(A_tile, B_tile)

    # Write back to C (cast handled by store if C_ptr dtype differs)
    C_tile_ptr = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_tile_ptr, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Validate inputs
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Incompatible shapes: A {A.shape}, B {B.shape}"

        # Ensure contiguous for coalesced access
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Output tensor (same dtype as inputs)
        C = torch.empty((M, N), dtype=A.dtype, device=A.device)

        # Choose tile sizes based on problem dimensions (deterministic, no autotune)
        # Favor moderate tiles to handle varied M while keeping good performance for N up to 4096.
        if M <= 32:
            BLOCK_M = 16
        elif M <= 64:
            BLOCK_M = 32
        else:
            BLOCK_M = 64

        if N >= 1024:
            BLOCK_N = 128
        elif N >= 512:
            BLOCK_N = 64
        else:
            BLOCK_N = 32

        BLOCK_K = 64  # good default for K=4096

        num_warps = 4
        num_stages = 3

        # Grid: number of programs along M and N
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel
        matmul_bT_kernel[grid](
            A_c, B_c, C,
            M, N, K,
            A_c.stride(0), A_c.stride(1),
            B_c.stride(0), B_c.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=num_stages,
        )

        return C


def run(*args):
    return ModelNew()(*args)
