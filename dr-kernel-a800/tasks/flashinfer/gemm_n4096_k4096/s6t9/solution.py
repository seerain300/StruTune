import triton
import triton.language as tl


@triton.jit
def matmul_AT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Tile indices
    pid_m = tl.program_id(0)  # tile id along M
    pid_n = tl.program_id(1)  # tile id along N

    # Offsets within the tile; with BLOCK_M=1 and BLOCK_N=1, this is a single element
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # shape [1]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # shape [1]

    # Accumulator for this tile (since BLOCK_M=1 and BLOCK_N=1, it's scalar)
    acc = tl.zeros((), dtype=tl.float32)  # accumulate in float32 for stability

    # Loop over K: C[m, n] = sum_k A[m, k] * B[k, n]
    # Since BLOCK_M=1 and BLOCK_N=1, we can iterate over k directly.
    # If needed, for larger K, this loop will be fine for correctness.
    for k in range(0, K):
        # Load A[m, k], m is the single offset in the M tile
        m_idx = m_offsets[0]  # scalar
        a_ptr = A_ptr + m_idx * stride_am + k * stride_ak
        # Mask: only valid if m_idx < M. But for BLOCK_M=1 and grid_m=M, always true.
        a_val = tl.load(a_ptr)  # dtype follows tensor dtype (e.g., fp16), cast later

        # Load B[k, n], n is the single offset in the N tile
        n_idx = n_offsets[0]  # scalar
        b_ptr = B_ptr + k * stride_bk + n_idx * stride_bn
        b_val = tl.load(b_ptr)

        # Accumulate in float32
        a_f32 = a_val.to(tl.float32)
        b_f32 = b_val.to(tl.float32)
        acc += a_f32 * b_f32

    # Store the result to C[m, n]
    c_ptr = C_ptr + m_offsets[0] * stride_cm + n_offsets[0] * stride_cn
    # Since m_offsets[0] = pid_m and n_offsets[0] = pid_n, and grid=(M,N),
    # m_offsets[0] < M and n_offsets[0] < N always.
    tl.store(c_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Validate inputs
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}] and B is [{K2}, {N}]")

        # Ensure contiguous
        A = A.contiguous()
        B = B.contiguous()

        # Allocate output tensor (same dtype as A)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Grid covers all M and N dimensions (BLOCK_M=1, BLOCK_N=1)
        grid = (M, N)

        # Launch kernel; BLOCK_M and BLOCK_N are 1
        matmul_AT_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=1, BLOCK_N=1,
            num_warps=1, num_stages=1,
        )

        return C


def run(*args):
    return ModelNew()(*args)
