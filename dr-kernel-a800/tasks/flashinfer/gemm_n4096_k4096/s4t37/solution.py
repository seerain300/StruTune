import torch
import triton
import triton.language as tl


@triton.jit
def matmul_transpose_tiled_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A is [M, K]: stride_am along rows (M), stride_ak along cols (K)
    stride_bn, stride_bk,   # B_T is [N, K]: stride_bn along rows (N), stride_bk along cols (K)
    stride_cm, stride_cn,   # C is [M, N]: stride_cm along rows (M), stride_cn along cols (N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tile indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # FP32 accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction loop over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A_tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B_T_tile: shape [BLOCK_K, BLOCK_N], B_T is [N, K] so k is rows, n is cols
        bt_ptrs = BT_ptr + n_offsets[None, :] * stride_bn + k_offsets[:, None] * stride_bk
        bt_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0)

        # Accumulate in FP32
        acc += tl.dot(a.to(tl.float32), bt.to(tl.float32))

    # Store results to C (cast to FP16 to match original dtype)
    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)  # acc is FP32; Triton will store as FP32 to FP16 if C is FP16


def _pick_launch_params(M, N, K):
    # Simple heuristic: larger tiles for larger problems
    max_dim = max(M, N, K)
    if max_dim >= 2048:
        return 128, 128, 64, 8, 3
    else:
        return 64, 64, 32, 4, 2


def _matmul_transpose_triton(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Ensure CUDA tensors
    if not A.is_cuda or not B.is_cuda:
        # If not CUDA, move to current CUDA device (evaluation uses CUDA)
        device = torch.device("cuda")
        A = A.to(device, non_blocking=True)
        B = B.to(device, non_blocking=True)

    # Make inputs contiguous to simplify strides and improve memory access
    A = A.contiguous()
    B_T = B.transpose(1, 0).contiguous()  # B_T has shape [N, K]

    M, K = A.shape
    N, Kb = B_T.shape  # Kb should equal K
    assert Kb == K, f"B's second dim ({Kb}) must equal A's second dim ({K})"

    # Allocate output (FP16 to match original code)
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Compute strides (in elements, not bytes)
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_bn = B_T.stride(0)  # stride along N (rows) of B_T
    stride_bk = B_T.stride(1)  # stride along K (cols) of B_T
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Pick tiling and launch params
    BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = _pick_launch_params(M, N, K)

    # Grid over tiles of M and N
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch Triton kernel
    matmul_transpose_tiled_kernel[grid](
        A, B_T, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Original signature: run(A, B) computes A @ B.T
        if len(args) != 2:
            raise ValueError("ModelNew.forward expects two tensors: A and B.")
        A, B = args
        # Ensure CUDA tensors for Triton
        return _matmul_transpose_triton(A, B)


def run(*args):
    return ModelNew()(*args)
