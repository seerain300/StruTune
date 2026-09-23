import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel_2d(
    A_ptr, BT_ptr, C_ptr,
    stride_am, stride_ak,        # A strides: (M, K)
    stride_bk, stride_bn,        # BT strides: (K, N)
    stride_cm, stride_cn,        # C strides: (M, N)
    M: tl.constexpr,             # constexpr sizes for static unrolling
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    # 2D launch grid over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col indices for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # FP32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks; K is constexpr, so tl.static_range is valid
    for k0 in tl.static_range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # K indices for this chunk

        # Load A tile: shape (BLOCK_M, BLOCK_K)
        A_tile_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(A_tile_ptrs, mask=a_mask, other=0.0)

        # Load BT tile: shape (BLOCK_K, BLOCK_N)
        BT_tile_ptrs = BT_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        bt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(BT_tile_ptrs, mask=bt_mask, other=0.0)

        # Accumulate in FP32
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

    # Store result to C (output dtype: float16)
    C_tile_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_tile_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Triton requires CUDA tensors
        assert A.is_cuda and B.is_cuda, "ModelNew.forward: inputs must be CUDA tensors"
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        N, K2 = B.shape
        assert K == K2, f"Incompatible shapes: A is (*, {K}), B is ({N}, {K2})"

        # BT = B.T contiguous as (K, N)
        BT = B.transpose(0, 1).contiguous()

        # Output tensor: float16 to match original
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bk, stride_bn = BT.stride(0), BT.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Tile and launch configuration
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Performance-oriented launch params
        num_warps = 4
        num_stages = 2

        # IMPORTANT: pass M, N, K as constexpr meta-parameters so tl.static_range works
        matmul_at_bT_kernel_2d[grid](
            A, BT, C,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            M=M, N=N, K=K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=num_stages,
        )

        return C


def run(*args):
    return ModelNew()(*args)
