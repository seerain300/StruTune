import triton
import triton.language as tl


@triton.jit
def matmul_AT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Tile identifiers
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col offsets this program handles
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator in float32 for stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K in chunks
    for k0 in range(0, K, BLOCK_K):
        # For each kk within the chunk
        for kk in range(0, BLOCK_K):
            # Compute actual k index
            k = k0 + kk
            # Bounds mask for loads
            m_mask = (m_offsets < M) & (k < K)
            n_mask = (n_offsets < N) & (k < K)

            # Load A[m, k] as a vector over BLOCK_M, masked
            a_vec = tl.load(A_ptr + m_offsets * stride_am + k * stride_ak, mask=m_mask, other=0.0)  # [BLOCK_M]
            # Load B[k, n] as a vector over BLOCK_N, masked
            b_vec = tl.load(B_ptr + k * stride_bk + n_offsets * stride_bn, mask=n_mask, other=0.0)  # [BLOCK_N]

            # Cast to float32 for accumulation (inputs are fp16 in benchmark)
            a_vec_f32 = a_vec.to(tl.float32)  # [BLOCK_M]
            b_vec_f32 = b_vec.to(tl.float32)  # [BLOCK_N]

            # Outer product and accumulate: [BLOCK_M, 1] * [1, BLOCK_N] -> [BLOCK_M, BLOCK_N]
            # We need to broadcast a_vec_f32 across columns and b_vec_f32 across rows.
            acc += a_vec_f32[:, None] * b_vec_f32[None, :]

    # Store back to C with mask (only valid m/n indices)
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure 2D inputs
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}] and B is [{K2}, {N}]")

        # Output tensor
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)  # keep output in fp32 for correctness

        # Strides in elements
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Choose blocks to guarantee grid coverage even for tiny M,N
        BLOCK_M = 16
        BLOCK_N = 64
        BLOCK_K = 32  # iterate K in chunks of 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        matmul_AT_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=2, num_stages=2,
        )

        return C


def run(*args):
    return ModelNew()(*args)
