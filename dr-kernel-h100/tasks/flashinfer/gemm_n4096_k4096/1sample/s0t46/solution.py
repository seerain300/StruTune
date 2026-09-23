import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_at_bt_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTk, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Single program dimension over N
    pid_n = tl.program_id(0)
    n0 = pid_n * BLOCK_N
    cols = n0 + tl.arange(0, BLOCK_N)
    mask_n = cols < N

    # Row index is 0 for M==1; for safety, guard with M
    row = 0
    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K, with proper masks
    for k0 in range(0, K, BLOCK_K):
        k_vec = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_vec < K

        # Load A[0, k] vector (masked on K)
        a_ptrs = A + row * stride_am + k_vec * stride_ak
        a = tl.load(a_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        # Load BT[k, cols] tile with 2D mask over K and N
        bt_ptrs = BT + k_vec[:, None] * stride_bTk + cols[None, :] * stride_bTn
        bt_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(bt_ptrs, mask=bt_mask, other=0.0).to(tl.float32)

        # Accumulate: sum over K chunk
        acc += tl.sum(b * a[None, :], axis=0)

    # Store result to C[0, cols]
    c_ptrs = C + row * stride_cm + cols * stride_cn
    tl.store(c_ptrs, acc, mask=mask_n)


@triton.jit
def _matmul_generic_at_bt_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTk, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiled GEMM: C[m, n] = sum_k A[m, k] * BT[k, n]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    m = m0 + tl.arange(0, BLOCK_M)
    n = n0 + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)

        a_ptrs = A + m[:, None] * stride_am + k[None, :] * stride_ak
        b_ptrs = BT + k[:, None] * stride_bTk + n[None, :] * stride_bTn

        a = tl.load(a_ptrs, mask=(m[:, None] < M) & (k[None, :] < K), other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=(k[:, None] < K) & (n[None, :] < N), other=0.0).to(tl.float32)

        acc += tl.dot(a, b)

    c_ptrs = C + m[:, None] * stride_cm + n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(m[:, None] < M) & (n[None, :] < N))


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure device and contiguity for predictable strides
        A = A.contiguous()
        B = B.contiguous()

        # Explicit transpose (view) and make it contiguous for predictable strides
        BT = B.transpose(0, 1).contiguous()  # BT: [K, N]

        M, K = A.shape
        N = BT.shape[1]  # since BT is [K, N]

        # Output buffer in fp32 for accumulation; will cast back to A.dtype at the end
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        if M == 1:
            # Fast row-wise kernel for M == 1
            BLOCK_N = 256
            BLOCK_K = 256
            grid = (triton.cdiv(N, BLOCK_N),)
            _rowwise_at_bt_kernel[grid](
                A, BT, C,
                M, N, K,
                A.stride(0), A.stride(1),
                BT.stride(0), BT.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=8, num_stages=4,
            )
        else:
            # Generic fallback for M > 1
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _matmul_generic_at_bt_kernel[grid](
                A, BT, C,
                M, N, K,
                A.stride(0), A.stride(1),
                BT.stride(0), BT.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )

        # Cast to original dtype to match torch.matmul behavior
        return C.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
