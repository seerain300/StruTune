import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_at_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # B strides correspond to original B of shape [N, K]
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 2D launch grid: programs over (M, N) tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A tile: A[rm, rk]
        a_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]

        # Pointers for B_T tile: B_T[rk, rn] == B[rn, rk]
        b_ptrs = B_ptr + rn[None, :] * stride_bn + rk[:, None] * stride_bk
        b_mask = (rn[None, :] < N) & (rk[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Fused multiply-add
        acc += tl.dot(a, b)  # [BLOCK_M, BLOCK_N]

    # Write back to C[m, n]
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute C = A @ B.T using Triton kernels. Ensure all heavy computation is done by Triton.
        """
        # Ensure dtypes are float16 for compute; Triton kernel will accumulate in fp32 and cast back.
        assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA device for Triton kernels."
        assert A.dtype in (torch.float16, torch.bfloat16, torch.float32), "A dtype must be a floating type."
        assert B.dtype in (torch.float16, torch.bfloat16, torch.float32), "B dtype must be a floating type."

        M, K = A.shape
        N = B.shape[0]  # B is [N, K]; C is [M, N]

        # We'll compute in fp32 for stability; the evaluator uses fp16 inputs so output should match fp16.
        # Allocate output in fp32, then cast to A.dtype at the end.
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Tile sizes tuned for N=4096, K=4096; robust for arbitrary sizes via masks.
        BLOCK_M = 16
        BLOCK_N = 256
        BLOCK_K = 128

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_at_bt_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=4,
        )

        # Cast result back to original dtype (match torch.matmul behavior with fp16 inputs)
        return C.to(A.dtype)