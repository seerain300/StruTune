import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bt, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program IDs for 2D tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute row/col offsets this program handles
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator in float16 to match input dtype
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

    # Iterate over K dimension in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Pointers for A: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        # Pointers for BT (which is B.T): BT has shape [K, N], strides are (stride_bt, stride_bn)
        # We index BT as BT[offs_k, offs_n] -> BT_ptr + offs_k[:, None]*stride_bt + offs_n[None, :]*stride_bn
        b_ptrs = BT_ptr + (offs_k[:, None] * stride_bt + offs_n[None, :] * stride_bn)

        # 1D masks for validity
        row_mask = offs_m < M
        col_mask = offs_n < N
        k_mask = offs_k < K

        # Broadcast masks to 2D explicitly to avoid Triton's 2D slicing sensitivity
        a_mask = row_mask[:, None] & (k_mask[None, :])  # [BLOCK_M, BLOCK_K]
        b_mask = k_mask[:, None] & (col_mask[None, :])  # [BLOCK_K, BLOCK_N]

        # Loads with masks; masked elements become zero
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K], float16
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N], float16

        # Accumulate: acc += a @ b  -> a: [BM, BK], b: [BK, BN]
        acc += tl.dot(a, b)  # Triton will handle lowering; keep dtype float16

    # Store results to C with appropriate mask
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    row_mask_2d = row_mask[:, None]  # [BLOCK_M, 1]
    col_mask_2d = col_mask[None, :]  # [1, BLOCK_N]
    store_mask = row_mask_2d & col_mask_2d  # [BLOCK_M, BLOCK_N]
    tl.store(c_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Compute C = A @ B.T, where A is (M, K), B is (N, K), C is (M, N)
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
        M, K = A.shape
        N, K2 = B.shape
        assert K == K2, f"Inner dimension mismatch: A.shape[1]={K}, B.shape[1]={K2}"
        # Ensure BT (B.T) is contiguous and float16
        BT = B.transpose(0, 1).contiguous()  # BT shape: (K, N), float16
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Launch Triton kernel
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_at_bT_kernel[grid](
            A, BT, C,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )
        return C


def run(*args):
    return ModelNew()(*args)
