import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_at_bt_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTk, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M is expected to be 1 in this kernel; we still accept any M for safety
    # Each program handles a block of N columns for row m=0
    pid_n = tl.program_id(axis=0)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Initialize accumulator
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K dimension in chunks
    k_iter = 0
    while k_iter < K:
        k_range = k_iter + tl.arange(0, BLOCK_K)
        k_mask = k_range < K
        n_mask = cols < N

        # Load A[0, k] as a vector
        a_ptrs = A + 0 * stride_am + k_range * stride_ak
        a_vec = tl.load(a_ptrs, mask=k_mask, other=0.0)

        # Load BT[k, cols] tile: BT has shape [K, N], BT[k, cols] = B[cols, k]
        bt_ptrs = BT + k_range[:, None] * stride_bTk + cols[None, :] * stride_bTn
        bt_mask = k_mask[:, None] & n_mask[None, :]
        bt_tile = tl.load(bt_ptrs, mask=bt_mask, other=0.0)

        # Accumulate outer product: [BLOCK_K, 1] * [1, BLOCK_N]
        acc += tl.sum(bt_tile.to(tl.float32) * a_vec[:, None], axis=0)

        k_iter += BLOCK_K

    # Store result to C[0, cols]
    c_ptrs = C + 0 * stride_cm + cols * stride_cn
    tl.store(c_ptrs, acc, mask=cols < N)


@triton.jit
def _matmul_generic_at_bt_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bTk, stride_bTn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_iter = 0
    while k_iter < K:
        offs_k = k_iter + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # BT tile: [BLOCK_K, BLOCK_N], BT[k, n] = B[n, k]
        bt_ptrs = BT + (offs_k[:, None] * stride_bTk + offs_n[None, :] * stride_bTn)
        bt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        bt_tile = tl.load(bt_ptrs, mask=bt_mask, other=0.0)

        # Fused multiply-add
        acc += tl.dot(a_tile.to(tl.float32), bt_tile.to(tl.float32))

        k_iter += BLOCK_K

    # Write back C tile
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Validate shapes: A [M, K], B [N, K], output C [M, N]
        assert A.dim() == 2, "A must be 2D [M, K]"
        assert B.dim() == 2, "B must be 2D [N, K]"
        M, K = A.shape
        N = B.shape[0]
        assert B.shape[1] == K, f"B must have shape [N, K], got {B.shape} vs K={K}"

        # Create BT explicitly to ensure correct strides and layout
        BT = B.transpose(0, 1).contiguous()  # BT: [K, N]

        # Output buffer in fp32 for accumulation
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        if M == 1:
            # Fast path for single-row A
            BLOCK_N = 256
            BLOCK_K = 256
            grid = (triton.cdiv(N, BLOCK_N),)
            _rowwise_at_bt_kernel[grid](
                A, BT, C,
                M, N, K,
                A.stride(0), A.stride(1),
                BT.stride(0), BT.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=8, num_stages=4,
            )
        else:
            # Generic fallback
            BLOCK_M = 64
            BLOCK_N = 128
            BLOCK_K = 64
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            _matmul_generic_at_bt_kernel[grid](
                A, BT, C,
                M, N, K,
                A.stride(0), A.stride(1),
                BT.stride(0), BT.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )

        # Cast to original dtype to match torch.matmul behavior
        return C.to(A.dtype)


def run(*args):
    return ModelNew()(*args)
