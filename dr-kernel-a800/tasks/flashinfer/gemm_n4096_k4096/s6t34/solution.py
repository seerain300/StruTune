class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure 2D inputs
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        # Ensure contiguity
        A = A.contiguous()
        B = B.contiguous()

        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K}] and B is [{K2}, {N}]")

        # Allocate output tensor; same dtype as input A (float16 in the provided example)
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Strides in elements
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bk = B.stride(0)
        stride_bn = B.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Choose small blocks to guarantee coverage even for tiny M/N
        BLOCK_M = 1
        BLOCK_N = 1
        BLOCK_K = 32  # chunk size over K

        # Grid across M and N tiles
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

        # Return C (same shape as torch.matmul(A, B.T))
        return C


def run(*args):
    return ModelNew()(*args)
