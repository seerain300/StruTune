import torch
import triton
import triton.language as tl


@triton.jit
def matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program IDs for 2D tiling over output matrix C
    pid_m = tl.program_id(0)  # tile along M
    pid_n = tl.program_id(1)  # tile along N

    # Compute offsets for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Masks for boundaries
        m_mask = m_offsets < M
        n_mask = n_offsets < N
        k_mask = k_offsets < K

        # Load A[m, k] tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_tile = tl.load(
            a_ptrs,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0
        ).to(tl.float32)  # cast to fp32 for accumulation

        # Load B[k, n] tile as if B is [K, N]: shape [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        b_tile = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0
        ).to(tl.float32)  # cast to fp32 for accumulation

        # Outer-product accumulation across the BLOCK_K dimension
        # For each kk in [0, BLOCK_K), multiply a_col[:, kk] with b_row[kk, :] and add to acc
        # This avoids tl.dot and ensures correctness.
        for kk in range(0, BLOCK_K):
            # Safe: k_mask ensures we skip contributions when k_offsets[kk] >= K
            a_col = a_tile[:, kk]                     # [BLOCK_M]
            b_row = b_tile[kk, :]                    # [BLOCK_N]
            acc += a_col[:, None] * b_row[None, :]  # [BLOCK_M, BLOCK_N]

    # Store results to C in fp16 (C is fp16 tensor from host)
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure tensors are on CUDA for Triton
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton kernel."
        # Make inputs contiguous for coalesced access
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"Incompatible shapes: A is [{M},{K}], B is [{K2},{N}]"

        # Allocate output tensor (same dtype as inputs, typically fp16)
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Choose fixed tile sizes for robustness (avoid autotune edge issues)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        matmul_bt_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3
        )
        return C


def run(*args):
    return ModelNew()(*args)
