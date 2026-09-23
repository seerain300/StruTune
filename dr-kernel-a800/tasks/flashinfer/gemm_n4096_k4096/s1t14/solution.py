import torch
import triton
import triton.language as tl

# Row-wise Triton kernel: computes one output row at a time.
# Grid: (M,)
@triton.jit
def rowwise_matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
):
    # Each program handles one row m
    m = tl.program_id(0)
    # Initialize fp32 accumulator for this row
    acc = tl.zeros([N], dtype=tl.float32)

    # Loop over K in blocks
    k_start = 0
    while k_start < K:
        k_offsets = k_start + tl.arange(0, 128)  # use a block along K
        k_mask = k_offsets < K

        # Load A_row slice: A[m, k_offsets] -> [BLOCK_K]
        a_row_ptrs = A_ptr + m * stride_am + k_offsets * stride_ak
        a_vec = tl.load(a_row_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K], fp32 promotion by default

        # Accumulate over this K block by iterating N in chunks of BLOCK_N
        # We use a loop over N since we don't have a 2D grid here; choose BLOCK_N=128 for vectorization.
        n_chunk = 128
        n_start = 0
        while n_start < N:
            offs_n = n_start + tl.arange(0, n_chunk)
            n_mask = offs_n < N

            # Load B block: B[offs_n, k_offsets], shape [BLOCK_N, BLOCK_K]
            b_ptrs = B_ptr + offs_n[:, None] * stride_bn + k_offsets[None, :] * stride_bk
            b_block = tl.load(b_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

            # Compute partial dot for this N chunk: [BLOCK_N]
            partial = tl.sum(b_block * a_vec[None, :], axis=1)  # [BLOCK_N]
            acc[n_start:n_start + n_chunk] += partial

            n_start += n_chunk

        k_start += 128

    # Store the fp32 accumulator to C[m, :]
    c_row_ptrs = C_ptr + m * stride_cm + tl.arange(0, N) * stride_cn
    tl.store(c_row_ptrs, acc, mask=(tl.arange(0, N) < N))


# 2D-tiled GEMM kernel: computes C = A @ B.T (A: [M, K], B: [N, K], C: [M, N])
# Grid: (ceil(M/BLOCK_M), ceil(N/BLOCK_N)), loop over K in blocks.
@triton.jit
def matmul_bt_kernel_2d(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator for the tile
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Loop over K in blocks
    k_start = 0
    while k_start < K:
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Pointers for A[m, k] and B[n, k]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)  # [BM, BK]
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn) + (offs_k[:, None] * stride_bk)  # [BK, BN]

        # Masks for bounds
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles
        a_block = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BM, BK]
        b_block = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BK, BN]

        # Accumulate: acc += a_block @ b_block
        acc += tl.dot(a_block, b_block)  # [BM, BN]

        k_start += BLOCK_K

    # Write back C[offs_m, offs_n]
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor):
        # Ensure tensors are on CUDA and contiguous
        if not A.is_cuda or not B.is_cuda:
            raise RuntimeError("ModelNew.forward requires CUDA tensors.")
        A = A.contiguous()
        B = B.contiguous()

        # Shapes
        M, K_a = A.shape
        N, K_b = B.shape
        if K_a != K_b:
            raise RuntimeError(f"Incompatible shapes: A is [{M}, {K_a}], B is [{N}, {K_b}] (K mismatch).")

        # Allocate output as fp32 for stability; cast later to A.dtype
        C_fp32 = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Extract strides (in elements, Triton expects element-wise strides)
        stride_am, stride_ak = A.stride()
        stride_bn, stride_bk = B.stride()
        stride_cm, stride_cn = C_fp32.stride()

        # Choose kernel based on M
        # For very small M, row-wise kernel is more efficient (reduces masked work).
        if M <= 4:
            grid = (M,)
            rowwise_matmul_bt_kernel[grid](
                A, B, C_fp32,
                M, N, K_a,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
            )
        else:
            # 2D-tiled kernel: tune block sizes; these work well for medium to large M, N, K
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            matmul_bt_kernel_2d[grid](
                A, B, C_fp32,
                M, N, K_a,
                stride_am, stride_ak,
                stride_bn, stride_bk,
                stride_cm, stride_cn,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=8,
                num_stages=2,
            )

        # Cast output to A's dtype to match original behavior
        return C_fp32.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
