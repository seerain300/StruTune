import torch
import triton
import triton.language as tl

# Generic GEMM Triton kernel: C[M, N] = A[M, K] @ B_T[K, N], where B_T[n, k] = B[k, n]
@triton.jit
def _matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # strides for A (row-major): A[i, j] = A_ptr + i*stride_am + j*stride_ak
    stride_bk, stride_bn,   # strides for B (row-major): B[i, j] = B_ptr + i*stride_bk + j*stride_bn
    stride_cm, stride_cn,   # strides for C (row-major): C[i, j] = C_ptr + i*stride_cm + j*stride_cn
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch: each program computes a BLOCK_M x BLOCK_N tile of C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Pointers for A tile: shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)  # f16

        # Pointers for B tile as B_T: BT[n, k] = B[k, n], shape [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b_mask = (n_offsets[None, :] < N) & (k_offsets[:, None] < K)
        b_tile = tl.load(B_ptrs, mask=b_mask, other=0.0)  # f16

        # Accumulate in fp32
        acc += tl.dot(a_tile.to(tl.float32), b_tile.to(tl.float32))

    # Store results to C[m, n] in float16
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")

        # Ensure dtype float16
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        # Validate shapes: A [M, K], B [K, N]
        M, K = A.shape
        N, Kb = B.shape
        if K != Kb:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (N={N}, K={Kb}). B's K must equal A's K.")

        # Enforce contiguity for robust stride handling
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Allocate output C as contiguous [M, N], float16
        C = torch.empty((M, N), dtype=torch.float16, device=A.device).contiguous()

        # Choose block sizes (good defaults for fp16 and large K/N)
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64

        # Launch grid over tiles in M and N
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_bt_kernel[grid](
            A_c, B_c, C,
            M, N, K,
            A_c.stride(0), A_c.stride(1),
            B_c.stride(0), B_c.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        return C


def run(*args):
    return ModelNew()(*args)
