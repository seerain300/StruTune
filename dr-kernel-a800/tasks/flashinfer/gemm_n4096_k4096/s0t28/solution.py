import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Tiny M: emphasize N parallelism
        triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 1024}, num_warps=8, num_stages=2),
    ],
    key=["N"],  # specialize by output width
)
@triton.jit
def matmul_rowwise_kernel(
    A_ptr,           # *fp16 [M, K_T]
    BT_ptr,          # *fp16 [K_T, N] (B.t().contiguous())
    C_ptr,           # *fp16 [M, N]
    M, N, K_T,       # sizes
    stride_am, stride_ak,     # strides for A
    stride_btk, stride_btn,   # strides for BT (BT is [K_T, N], contiguous after .t().contiguous())
    stride_cm, stride_cn,     # strides for C
    # meta-parameters
    BLOCK_N: tl.constexpr,
):
    # Each program computes one output row i and a BLOCK_N-wide slice of columns
    i = tl.program_id(0)
    if i >= M:
        return
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator for one row (BLOCK_N columns)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K_T in chunks of BLOCK_K inferred from BT tile; here we use implicit single iteration
    # but since we iterate only over N, we need to incorporate K as well. We'll do that by looping
    # over K in the host or using another kernel. For tiny M, it's fine to compute per row with
    # outer product accumulation. However, Triton requires explicit tiling over K. So we switch to
    # a 2D-tiled kernel below. This function will not be used when M > 8; we keep it for completeness.
    # NOTE: The above comment reflects the design change. We won't use this kernel for general cases.
    # To keep correctness, we simply mask out-of-range columns.
    # Compute C[i, offs_n] = sum_k A[i, k] * BT[k, offs_n]
    # We will implement a simple outer-product accumulation over k:
    for k in range(0, K_T):
        # Load A[i, k] scalar
        a_val = tl.load(A_ptr + i * stride_am + k * stride_ak)
        a_val = a_val.to(tl.float32)
        # Load BT[k, offs_n] vector
        bt_ptrs = BT_ptr + k * stride_btk + offs_n * stride_btn
        bt_vals = tl.load(bt_ptrs, mask=offs_n < N, other=0.0)
        bt_vals = bt_vals.to(tl.float32)
        acc += a_val * bt_vals

    # Store result
    out_ptrs = C_ptr + i * stride_cm + offs_n * stride_cn
    tl.store(out_ptrs, acc.to(tl.float16), mask=offs_n < N)


@triton.autotune(
    configs=[
        # General 2D tiled, emphasize both M and N tiles
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 512, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K_T"],
)
@triton.jit
def matmul_tiled_kernel(
    A_ptr,           # *fp16 [M, K_T]
    BT_ptr,          # *fp16 [K_T, N] (B.t().contiguous())
    C_ptr,           # *fp16 [M, N]
    M, N, K_T,       # sizes
    stride_am, stride_ak,     # strides for A
    stride_btk, stride_btn,   # strides for BT (BT is [K_T, N], contiguous)
    stride_cm, stride_cn,     # strides for C
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K_T in chunks of BLOCK_K (BLOCK_K must match autotune config)
    for k0 in range(0, K_T, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K_T)
        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Pointers for BT tile: [BLOCK_K, BLOCK_N], BT[k, n] = B_T[k, n]
        bt_ptrs = BT_ptr + (offs_k[:, None] * stride_btk + offs_n[None, :] * stride_btn)
        bt_mask = (offs_k[:, None] < K_T) & (offs_n[None, :] < N)
        bt_vals = tl.load(bt_ptrs, mask=bt_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a_vals, bt_vals)

    # Write back
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor):
        """
        Compute C = A @ B.T
        A: [M, K_T], B: [N, O_T], output C: [M, N]
        Triton kernels do the computation; host only allocates, ensures contiguity, and launches kernels.
        """
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
        assert A.dtype == torch.float16 and B.dtype == torch.float16, "Expected float16 tensors"

        # Ensure contiguous
        A = A.contiguous()
        # Construct B_T explicitly as contiguous [K_T, N] so we can index BT[k, n] safely
        BT = B.t().contiguous()  # BT[k, n] = B[n, k]

        M, K_T = A.shape
        N, O_T = B.shape
        # Output C [M, N]
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        # BT is [K_T, N], contiguous; strides (N, 1) if contiguous
        stride_btk = BT.stride(0)
        stride_btn = BT.stride(1)
        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Launch kernel
        if M <= 8:
            # Row-wise specialization for tiny M
            grid = (M, triton.cdiv(N, 128))  # BLOCK_N will be chosen by autotune; initial guess 128
            matmul_rowwise_kernel[grid](
                A, BT, C,
                M, N, K_T,
                stride_am, stride_ak,
                stride_btk, stride_btn,
                stride_cm, stride_cn,
            )
        else:
            # General 2D tiled kernel
            def grid(meta):
                return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
            matmul_tiled_kernel[grid](
                A, BT, C,
                M, N, K_T,
                stride_am, stride_ak,
                stride_btk, stride_btn,
                stride_cm, stride_cn,
            )

        return C


def run(*args):
    return ModelNew()(*args)
