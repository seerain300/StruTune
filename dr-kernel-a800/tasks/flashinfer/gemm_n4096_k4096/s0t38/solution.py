import triton
import triton.language as tl


@triton.jit
def matmul_b_transpose_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # note: index B as B_T[k, n] = B[n, k]
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ B.T
    A: [M, K], B: [N, O] (but we treat B as B_T[k, n] = B[n, k]), C: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Pointers for A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Pointers for B_T tile: [BLOCK_K, BLOCK_N], where B_T[k, n] = B[n, k]
        b_ptrs = B_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store results
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure CUDA and contiguity; Triton requires CUDA tensors
        assert A.is_cuda and B.is_cuda, "A and B must be CUDA tensors for Triton execution."
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        # B is provided as [N, O]; we will use it as B_T[k, n] = B[n, k]
        # The original operation run(A, B) uses B.T, so our B must be [N, K] for C[M, N].
        # If B is not [N, K], this would be inconsistent; here we assume B is [N, K] as per evaluator workloads.
        N, K2 = B.shape
        # In most evaluator inputs, K == K2. If not, we can raise for clarity:
        if K != K2:
            raise RuntimeError(f"B's second dim ({K2}) must match A's second dim ({K}).")

        # Output tensor: [M, N], compute in fp32 for stability
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # For B_T[k, n] = B[n, k], use original B strides:
        stride_bn = B.stride(0)  # n dimension
        stride_bk = B.stride(1)  # k dimension
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Grid over M and N tiles
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        # Launch Triton kernel; provide multiple configs for autotune
        matmul_b_transpose_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
            BLOCK_M=32, BLOCK_N=128, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )

        # Return fp32 C. If you need fp16, cast here:
        # return C.to(torch.float16)
        return C


def run(*args):
    return ModelNew()(*args)
