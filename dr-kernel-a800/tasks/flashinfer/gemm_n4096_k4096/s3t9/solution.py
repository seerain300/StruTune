import torch
import triton
import triton.language as tl


@triton.jit
def matmul_rowwise_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,       # A is (M, K): stride_am along rows, stride_ak along cols
    stride_bk, stride_bn,       # BT is (K, N): stride_bk along rows (K), stride_bn along cols (N)
    stride_cm, stride_cn,       # C is (M, N): stride_cm along rows (M), stride_cn along cols (N)
    BLOCK_K: tl.constexpr,
):
    # One program per output row m
    m = tl.program_id(0)

    # Vector of column offsets
    offs_n = tl.arange(0, 128)  # We will only store up to N columns; actual N is passed
    # Accumulator for this row (float32 for stability)
    acc_vec = tl.zeros((128,), dtype=tl.float32)

    # Iterate over K in tiles of BLOCK_K using a static loop (K is runtime, but Triton supports this)
    for k0 in tl.static_range(0, 1 << 30, BLOCK_K):  # upper bound huge; real loop controlled by mask
        mask_k = k0 + tl.arange(0, BLOCK_K) < K
        # Load A[m, k0:k0+BLOCK_K]
        A_row_ptrs = A_ptr + m * stride_am + (k0 + tl.arange(0, BLOCK_K)) * stride_ak
        a_row = tl.load(A_row_ptrs, mask=mask_k, other=0.0)  # shape (BLOCK_K,)
        a_row = a_row.to(tl.float32)  # ensure fp32

        # Load BT[k0:k0+BLOCK_K, offs_n], this is a (BLOCK_K, 128) tile
        BT_ptrs = BT_ptr + (k0 + tl.arange(0, BLOCK_K))[:, None] * stride_bk + offs_n[None, :] * stride_bn
        mask_bt = (k0 + tl.arange(0, BLOCK_K))[:, None] < K
        bt_tile = tl.load(BT_ptrs, mask=mask_bt, other=0.0)  # shape (BLOCK_K, 128)
        bt_tile = bt_tile.to(tl.float32)  # ensure fp32

        # Accumulate: dot(a_row, bt_tile, axis=0) -> shape (128,)
        acc_vec += tl.sum(bt_tile * a_row[:, None], axis=0)

        # If k0 + BLOCK_K >= K, the next iteration's mask_k will be all False, breaking effect.

    # Store the accumulated row to C[m, :]
    C_row_ptrs = C_ptr + m * stride_cm + offs_n * stride_cn
    # We only store the first N columns; mask for n < N
    store_mask = offs_n < N
    tl.store(C_row_ptrs, acc_vec, mask=store_mask)


@triton.jit
def matmul_2d_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,       # A strides: A is (M, K)
    stride_bk, stride_bn,       # BT strides: BT is (K, N)
    stride_cm, stride_cn,       # C strides: C is (M, N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids over tiles of output C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                    # [BLOCK_K]

    # Accumulator in fp32 for numeric stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Static loop over K in chunks of BLOCK_K (Triton supports static_range with constexpr step)
    for k0 in tl.static_range(0, 1 << 30, BLOCK_K):
        a_mask = (offs_m[:, None] < M) & (k0 + offs_k[None, :] < K)
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak)
        a_tile = tl.load(A_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # (BLOCK_M, BLOCK_K)

        b_mask = (k0 + offs_k[:, None] < K) & (offs_n[None, :] < N)
        BT_ptrs = BT_ptr + ((k0 + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        bt_tile = tl.load(BT_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # (BLOCK_K, BLOCK_N)

        acc += tl.dot(a_tile, bt_tile)

    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=out_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor):
        # Ensure CUDA tensors; if not, fall back to torch for robustness
        if not A.is_cuda or not B.is_cuda:
            return torch.matmul(A, B.T)

        # Make inputs contiguous
        A = A.contiguous()
        B = B.contiguous()

        # Prepare BT = B.T (K, N) for A @ BT
        BT = B.transpose(0, 1).contiguous()  # BT shape: (K, N)

        M, K = A.shape
        K2, N = BT.shape
        assert K == K2, "B must be of shape (N, K) so that B.T is (K, N)"

        # Allocate output
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Choose kernel:
        # - If M is small, use rowwise kernel (one program per row).
        # - Otherwise, use 2D kernel.
        # Threshold can be tuned; 64 works well as a starting point.
        if M < 64:
            # Launch rowwise kernel: grid = (M,)
            # Note: we only store the first N columns using mask.
            grid = (M,)
            matmul_rowwise_kernel[grid](
                A, BT, C,
                M, N, K,
                A.stride(0), A.stride(1),
                BT.stride(0), BT.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_K=128,
                num_warps=4,
                num_stages=2,
            )
        else:
            # 2D tiling: conservative tiles to avoid resource issues
            BLOCK_M = 64
            BLOCK_N = 64
            BLOCK_K = 32
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            matmul_2d_kernel[grid](
                A, BT, C,
                M, N, K,
                A.stride(0), A.stride(1),
                BT.stride(0), BT.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4,
                num_stages=2,
            )

        return C


def run(*args):
    return ModelNew()(*args)
