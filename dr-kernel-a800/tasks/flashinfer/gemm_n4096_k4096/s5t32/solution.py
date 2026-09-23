import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_a_bt_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_btk, stride_btn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over output C[M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K], A[m, k]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load BT tile: BT has shape [K, N], we index BT[k, n]
        BT_ptrs = BT_ptr + k_offsets[:, None] * stride_btk + n_offsets[None, :] * stride_btn
        bt_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        bt = tl.load(BT_ptrs, mask=bt_mask, other=0.0)

        # Accumulate in fp32
        acc += tl.dot(a, bt)

    # Store result C[m, n] as fp16
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    out_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float16), mask=out_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Triton requires CUDA tensors
        if not (A.is_cuda and B.is_cuda):
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")

        # Shapes: A [M, K], B [K, N]
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("Inputs must be 2D: A [M, K], B [K, N].")
        M, K = A.shape
        K_b, N = B.shape
        if K != K_b:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (K={K_b}, N={N}). B's K must equal A's K.")

        # Ensure contiguity for simple, consistent strides
        A_c = A.contiguous()
        B_c = B.contiguous()

        # Materialize BT = B.T as contiguous [K, N] to avoid stride ambiguity
        BT = B_c.t().contiguous()  # [K, N]

        # Output tensor [M, N], float16, contiguous
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Tiling parameters: balanced for fp16 GEMM
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_a_bt_kernel[grid](
            A_c, BT, C,
            M, N, K,
            A_c.stride(0), A_c.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )
        return C


def run(*args):
    return ModelNew()(*args)
