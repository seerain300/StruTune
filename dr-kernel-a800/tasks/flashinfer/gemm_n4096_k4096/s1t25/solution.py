import torch
import triton
import triton.language as tl


@triton.jit
def row_matmul_B_T_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # strides for B along its rows (N) and cols (K)
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids: each program handles one row m and a block of columns
    pid_m = tl.program_id(axis=0)
    pid_nb = tl.program_id(axis=1)

    # Compute column offsets for this block
    offs_n = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
    # Guard for columns beyond N
    col_mask = offs_n < N

    # Initialize accumulator for this row block
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # Load A[m, k0:k0+BLOCK_K]
        a_ptrs = A_ptr + pid_m * stride_am + offs_k * stride_ak
        a_vals = tl.load(a_ptrs, mask=k_mask, other=0.0)
        a_vals = a_vals.to(tl.float32)  # promote to fp32 for accumulation

        # Load B[offs_n, k0:k0+BLOCK_K] with strided access along N
        b_ptrs = B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
        b_vals = tl.load(b_ptrs, mask=col_mask[None, :] & k_mask[:, None], other=0.0)
        b_vals = b_vals.to(tl.float32)

        # Accumulate: acc[offs_n] += sum_{kk} a_vals[kk] * b_vals[:, kk]
        # b_vals shape: [BLOCK_N, BLOCK_K], a_vals shape: [BLOCK_K]
        # Compute dot per column: sum over kk of b[:, kk] * a[kk]
        # Equivalent: acc += tl.sum(b_vals * a_vals[:, None], axis=1)
        # But Triton expects elementwise ops; use tl.dot on (BLOCK_K, 1) x (1, BLOCK_N):
        a_col = a_vals[:, None]  # shape [BLOCK_K, 1]
        acc += tl.sum(b_vals * a_col, axis=0)

    # Store results for this row block
    c_ptrs = C_ptr + pid_m * stride_cm + offs_n * stride_cn
    tl.store(c_ptrs, acc, mask=col_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Compute C = A @ B.T where A: [M, K], B: [N, K], C: [M, N]
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
        M, K = A.shape
        N_b, K_b = B.shape
        assert K == K_b, f"Incompatible shapes: A is [M, {K}], B is [{N_b}, {K_b}]"
        N = N_b

        # Ensure contiguous for simple stride handling
        A = A.contiguous()
        B = B.contiguous()

        # Allocate output in fp32 for accumulation precision
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Launch Triton kernel: grid over rows (M) and column blocks
        BLOCK_N = 256  # columns processed per program
        BLOCK_K = 128  # reduction chunk size
        grid = (M, triton.cdiv(N, BLOCK_N))
        row_matmul_B_T_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Cast output to match A.dtype
        if C.dtype != A.dtype:
            C = C.to(A.dtype)
        return C


def run(*args):
    return ModelNew()(*args)
