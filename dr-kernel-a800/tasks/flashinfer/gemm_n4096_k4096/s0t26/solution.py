import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Tiny M (e.g., M <= 8): reduce BLOCK_M, increase parallelism along N
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 512, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 512, "BLOCK_K": 32}, num_warps=4, num_stages=2),

        # Medium M
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),

        # Larger M
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 512, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K_T"],  # K_T is the "K" dimension of A; also used in grid key
)
@triton.jit
def matmul_kernel(
    A_ptr,           # *fp16 [M, K_T]
    BT_ptr,          # *fp16 [O_T, N] where BT[k, n] = B[n, k] (B.t().contiguous())
    C_ptr,           # *fp16 [M, N]
    M, N, K_T, O_T,  # sizes
    stride_am, stride_ak,     # strides for A
    stride_bto, stride_btn,   # strides for BT (BT is [O_T, N], contiguous after .t().contiguous())
    stride_cm, stride_cn,     # strides for C
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute the offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K_T (A's second dimension) in chunks of BLOCK_K
    for k0 in range(0, K_T, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        # Pointers for BT tile: [BLOCK_K, BLOCK_N], BT[k, n] = B_T[k, n]
        # BT is [O_T, N] contiguous; strides are (N, 1) for contiguous [rows=O_T, cols=N]
        bt_ptrs = BT_ptr + (offs_k[:, None] * stride_bto + offs_n[None, :] * stride_btn)

        # Masks for boundary conditions
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K_T)
        bt_mask = (offs_k[:, None] < K_T) & (offs_n[None, :] < N)

        # Load tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0)

        # Cast to fp32 for accumulation
        a = a.to(tl.float32)
        bt = bt.to(tl.float32)

        # Accumulate: acc += a @ bt
        # a: [BM, BK], bt: [BK, BN] -> acc: [BM, BN]
        acc += tl.dot(a, bt)

    # Store result
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c = acc.to(tl.float16)
    tl.store(c_ptrs, c, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        """
        A: [M, K_T], B: [N, O_T]
        Compute C = A @ B.T, where B.T is [O_T, N].
        """
        # Ensure inputs are on CUDA and contiguous
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton kernel."
        A = A.contiguous()
        # Explicitly compute B.T as a contiguous tensor to guarantee correct layout
        BT = B.t().contiguous()

        M, K_T = A.shape
        N, O_T = B.shape

        # Output C has shape [M, N]
        C = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Strides (in elements)
        stride_am = A.stride(0)
        stride_ak = A.stride(1)

        # BT is [O_T, N] contiguous after .t().contiguous(), so strides are (N, 1) for a standard contiguous tensor.
        # However, we pass actual strides from BT to be robust.
        stride_bto, stride_btn = BT.stride()

        stride_cm = C.stride(0)
        stride_cn = C.stride(1)

        # Define grid: 2D over tiles of M and N
        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

        # Launch kernel
        matmul_kernel[grid](
            A, BT, C,
            M, N, K_T, O_T,
            stride_am, stride_ak,
            stride_bto, stride_btn,
            stride_cm, stride_cn,
        )
        return C


def run(*args):
    return ModelNew()(*args)
