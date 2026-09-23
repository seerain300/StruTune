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
    # Each program handles one row tile (BLOCK_M=1) and one col tile (BLOCK_N=1)
    pid_m = tl.program_id(0)  # tile id along M
    pid_n = tl.program_id(1)  # tile id along N

    # Single row/col offsets for this program (since BLOCK_M=1 and BLOCK_N=1)
    m0 = pid_m
    n0 = pid_n

    # Accumulator for this [m0, n0] position
    # We will sum over K in chunks of BLOCK_K
    acc = tl.zeros((), dtype=tl.float32)  # scalar accumulator

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Pointer for A[m0, k] across k in the chunk
        A_ptrs = A_ptr + m0 * stride_am + k_offsets * stride_ak  # [BLOCK_K]
        # Pointer for B[k, n0] across k in the chunk
        B_ptrs = B_ptr + k_offsets * stride_bk + n0 * stride_bn  # [BLOCK_K]

        # Masks for loads
        a_mask = (m0 < M) & (k_offsets < K)
        b_mask = (k_offsets < K) & (n0 < N)

        # Load as float32 for numerical robustness
        a_vals = tl.load(A_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BLOCK_K]
        b_vals = tl.load(B_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BLOCK_K]

        # Accumulate dot for this chunk
        # Since BLOCK_M=1 and BLOCK_N=1, this reduces to a scalar sum
        acc += tl.sum(a_vals * b_vals, axis=0)  # scalar

    # Store result to C[m0, n0]
    C_ptr_out = C_ptr + m0 * stride_cm + n0 * stride_cn
    store_mask = (m0 < M) & (n0 < N)
    # acc is float32; Triton will cast on store if C is fp16/bf16. Here C has same dtype as A.
    # We store the scalar result for this element.
    tl.store(C_ptr_out, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are 2D tensors
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}] and B is [{K2}, {N}]")

        # Output tensor, same dtype as A
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides in elements
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # BLOCK sizes chosen to guarantee grid > 0 for all M,N
        BLOCK_M = 1
        BLOCK_N = 1
        # Choose a reasonable BLOCK_K; 64 works well for fp16
        BLOCK_K = 64

        # Grid over tiles. With BLOCK_M=1, BLOCK_N=1, grid_m = M, grid_n = N
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch kernel
        matmul_AT_kernel[grid](
            A, B, C,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=1, num_stages=2,
        )

        return C


def run(*args):
    return ModelNew()(*args)
