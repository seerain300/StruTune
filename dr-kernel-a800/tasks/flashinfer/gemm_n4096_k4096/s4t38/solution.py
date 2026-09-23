import torch
import triton
import triton.language as tl


@triton.jit
def outer_product_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # strides for A: [M, K]
    stride_bn, stride_bk,   # strides for B_T: [N, K] (note: BT[n, k] layout)
    stride_cm, stride_cn,   # strides for C: [M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid: one program handles a tile of size BLOCK_M x BLOCK_N over (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # FP32 accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for k in range(0, K):
        # Load A[m, k] for all m in tile (vector across rows)
        a_ptrs = A_ptr + m_offsets * stride_am + k * stride_ak
        a_mask = m_offsets < M
        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M]

        # Load B_T[n, k] for all n in tile (vector across cols)
        bt_ptrs = BT_ptr + n_offsets * stride_bn + k * stride_bk
        bt_mask = n_offsets < N
        bt_vals = tl.load(bt_ptrs, mask=bt_mask, other=0.0)  # [BLOCK_N]

        # Outer-product accumulate: acc[i, j] += a_vals[i] * bt_vals[j]
        # Broadcast a_vals across N, bt_vals across M
        acc += a_vals[:, None] * bt_vals[None, :]

    # Store result
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    # Masks for bounds (generally all true if grid covers M,N, but keep for safety)
    store_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=store_mask)


def triton_matmul_transpose(A, B):
    # A: [M, K], B: [K, N]
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"Incompatible shapes for matmul: A is {A.shape}, B is {B.shape}")
    # Ensure contiguous and on CUDA
    A = A.contiguous()
    B = B.contiguous()
    BT = B.transpose(1, 0).contiguous()  # B_T: [N, K]
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Grid over tiles
    # We can choose BLOCK_M=64, BLOCK_N=64, BLOCK_K=32 to ensure correctness
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32  # not used directly in this outer-product kernel, but kept for potential future use

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    outer_product_kernel[grid](
        A, BT, C,
        M, N, K,
        A.stride(0), A.stride(1),
        BT.stride(0), BT.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B.")
        A, B = args
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        # Triton-based matmul(A, B.T), compute with Triton kernel
        return triton_matmul_transpose(A, B)


def run(*args):
    return ModelNew()(*args)
