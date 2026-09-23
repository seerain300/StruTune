import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_btk, stride_btn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the offsets this program will handle
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in float16 to match input dtype
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A and BT tiles
        # A is [M, K], BT is [K, N] (since B.T)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        bt_ptrs = BT_ptr + (offs_k[:, None] * stride_btk + offs_n[None, :] * stride_btn)

        # 1D masks for rows and columns
        row_mask = offs_m < M
        col_mask = offs_n < N
        # For A tile: valid rows (offs_m) and valid K (offs_k)
        a_mask = row_mask[:, None] & (offs_k[None, :] < K)
        # For BT tile: valid K (offs_k) and valid cols (offs_n)
        bt_mask = (offs_k[:, None] < K) & col_mask[None, :]

        # Load tiles; cast to float16 to match A/BT dtype
        # Triton will load as pointer dtype (float16 here), we avoid upcasting
        a = tl.load(a_ptrs, mask=a_mask, other=tl.zeros((), dtype=tl.float16))
        bt = tl.load(bt_ptrs, mask=bt_mask, other=tl.zeros((), dtype=tl.float16))

        # Accumulate
        acc += tl.dot(a, bt)

    # Store results with mask for valid M,N
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure tensors are on CUDA for Triton
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton execution."
        # We must do all computation in Triton; no torch.matmul in forward (CUDA path).
        # Prepare BT = B.T as a contiguous [K, N] tensor
        BT = B.transpose(0, 1).contiguous()

        M, K = A.shape
        N, Kb = BT.shape
        assert Kb == K, "B must have shape (N, K) so that BT has shape (K, N)."

        # Allocate output
        out = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Tile sizes (conservative to avoid shared memory issues)
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        # Grid over tiles of M and N
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel
        matmul_at_bT_kernel[grid](
            A, BT, out,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        return out


def run(*args):
    return ModelNew()(*args)
