import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,       # A strides: row (M), col (K)
    stride_bk, stride_bn,       # B strides: row (K), col (N) -> emulate B.T via strides
    stride_cm, stride_cn,       # C strides: row (M), col (N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program id: each program handles a tile [BLOCK_M, BLOCK_N] of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        # A tile: [BLOCK_M, BLOCK_K]
        A_tile_ptrs = A_ptr + offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak
        # B tile: [BLOCK_K, BLOCK_N], using B's original strides to emulate B.T
        B_tile_ptrs = B_ptr + (k0 + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

        # Masks for boundary conditions
        a_mask = (offs_m[:, None] < M) & ((k0 + offs_k[None, :]) < K)
        b_mask = ((k0 + offs_k[:, None]) < K) & (offs_n[None, :] < N)

        # Load tiles and cast to fp32 for accumulation
        a = tl.load(A_tile_ptrs, mask=a_mask, other=0.0)
        b = tl.load(B_tile_ptrs, mask=b_mask, other=0.0)
        a = a.to(tl.float32)
        b = b.to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Write back results to C
    C_tile_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_tile_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def __init__(self, block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=2):
        super().__init__()
        # Default tile sizes and launch params
        self.default_block_m = block_m
        self.default_block_n = block_n
        self.default_block_k = block_k
        self.default_num_warps = num_warps
        self.default_num_stages = num_stages

    def _choose_tiling(self, N):
        # Simple heuristic based on N to pick tile sizes
        if N >= 4096:
            return dict(block_m=128, block_n=128, block_k=64, num_warps=8, num_stages=3)
        elif N >= 1024:
            return dict(block_m=128, block_n=64, block_k=64, num_warps=8, num_stages=3)
        else:
            return dict(block_m=64, block_n=64, block_k=32, num_warps=4, num_stages=2)

    def forward(self, A, B):
        # Ensure CUDA tensors
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton kernel."

        # Shapes
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, f"Incompatible shapes: A is {A.shape}, B is {B.shape}"

        # Ensure dtypes match and are supported
        assert A.dtype == B.dtype, "A and B must have the same dtype."
        assert A.dtype in (torch.float16, torch.bfloat16, torch.float32), "Unsupported dtype."

        # Make inputs contiguous
        A = A.contiguous()
        B = B.contiguous()

        # Output tensor (same dtype as input)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Choose tile sizes based on N
        tiling = self._choose_tiling(N)
        block_m = tiling["block_m"]
        block_n = tiling["block_n"]
        block_k = tiling["block_k"]
        num_warps = tiling["num_warps"]
        num_stages = tiling["num_stages"]

        # Compute grid size: number of tiles along M and N
        grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

        # Launch Triton kernel
        _matmul_bt_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),     # A strides
            B.stride(1), B.stride(0),     # B strides to emulate B.T
            C.stride(0), C.stride(1),     # C strides
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            num_warps=num_warps, num_stages=num_stages,
        )

        return C


def run(*args):
    return ModelNew()(*args)
