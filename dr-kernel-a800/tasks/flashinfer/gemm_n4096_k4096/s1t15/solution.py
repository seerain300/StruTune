import math
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tile indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    k0 = 0
    while k0 < K:
        # Pointers for current tiles
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak)
        # B is [N, K], but we emulate B.T by indexing B[k, n] as B_ptr + n*stride_bn + k*stride_bk
        B_ptrs = B_ptr + (offs_n[None, :] * stride_bn + (k0 + offs_k[:, None]) * stride_bk)

        # Masks for boundaries
        a_mask = (offs_m[:, None] < M) & (k0 + offs_k[None, :] < K)
        b_mask = (offs_n[None, :] < N) & (k0 + offs_k[:, None] < K)

        # Load tiles (promote to fp32 for accumulation)
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)
        B_tile = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Cast to fp32 for stable accumulation
        A_tile = A_tile.to(tl.float32)
        B_tile = B_tile.to(tl.float32)

        # Accumulate
        acc += tl.dot(A_tile, B_tile)

        k0 += BLOCK_K

    # Write results back to C (fp32). Host will cast to desired dtype if needed.
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Validate inputs
        if A.dim() != 2 or B.dim() != 2:
            raise ValueError("A and B must be 2D tensors.")
        M, K_a = A.shape
        N_b, K_b = B.shape
        if K_a != K_b:
            raise ValueError(f"Incompatible shapes: A is [{M}, {K_a}], B is [{N_b}, {K_b}]. Expected K_a == K_b.")

        # Ensure contiguity
        A = A.contiguous()
        B = B.contiguous()

        # Compute in fp32 for numerical stability; cast later if needed
        # Output shape is [M, N_b]
        C_fp32 = torch.empty((M, N_b), dtype=torch.float32, device=A.device)

        # Strides for A [M, K], B [N, K], C [M, N]
        stride_am = A.stride(0)
        stride_ak = A.stride(1)
        stride_bn = B.stride(0)  # B is [N, K], so stride(0) = K stride of B (elements per row)
        stride_bk = B.stride(1)  # stride for K dimension in B
        stride_cm = C_fp32.stride(0)
        stride_cn = C_fp32.stride(1)

        # Choose tile sizes. Heuristic based on problem size
        # For typical large K/N (>= 4096), these tiles reduce loop iterations.
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64
        num_warps = 8
        num_stages = 3

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N_b, BLOCK_N))

        matmul_bt_kernel[grid](
            A, B, C_fp32,
            M, N_b, K_a,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=num_stages,
        )

        # Cast back to original dtype if desired. Original example uses fp16.
        # Note: ModelNew.forward must return the same dtype as the original result.
        # The original run uses torch.randn with dtype=torch.float16 in get_inputs, so we match that.
        if A.dtype == torch.float16:
            return C_fp32.to(torch.float16)
        elif A.dtype == torch.bfloat16:
            return C_fp32.to(torch.bfloat16)
        else:
            # For other dtypes (e.g., fp32), return fp32
            return C_fp32


def run(*args):
    return ModelNew()(*args)
