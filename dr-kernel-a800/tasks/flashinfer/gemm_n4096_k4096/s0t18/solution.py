import torch
import triton
import triton.language as tl


@triton.jit
def make_B_transpose_contiguous_kernel(B, BT, M, N, K):
    # BT is output of shape (K, N), BT[k, n] = B[n, k]
    pid_n = tl.program_id(0)  # tile along N (columns of BT)
    pid_k = tl.program_id(1)  # tile along K (rows of BT)
    BLOCK_N = 256
    BLOCK_K = 64

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # corresponds to original B's first dim
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)  # corresponds to original B's second dim

    mask_n = offs_n < N
    mask_k = offs_k < K

    # Load a tile from B: shape [BLOCK_K, BLOCK_N], where element is B[offs_n, offs_k]
    b_ptrs = B + offs_n[None, :] * N + offs_k[:, None]  # since row-major, index = n*N + k
    b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

    # Store to BT: BT[offs_k, offs_n] = b
    bt_ptrs = BT + offs_k[:, None] * N + offs_n[None, :]
    tl.store(bt_ptrs, b, mask=mask_k[:, None] & mask_n[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(A, BT, C, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # 2D tile over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K], A[m, k]
        a_ptrs = A + offs_m[:, None] * M + offs_k[None, :]
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # BT tile: [BLOCK_K, BLOCK_N], BT[k, n] = B_T[k, n]
        bt_ptrs = BT + offs_k[:, None] * N + offs_n[None, :]
        bt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0)

        acc += tl.dot(a, bt)

    # Store to C
    c_ptrs = C + offs_m[:, None] * M + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # A: [M, K], B: [N, K] (as per original run), output C: [M, N] = A @ B.T
        M, K = A.shape
        N = B.shape[0]

        # Allocate output
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        # Create BT = B.T contiguous inside Triton kernel
        BT = torch.empty((K, N), device=A.device, dtype=A.dtype)

        # Launch kernel to create BT
        grid_bt = (triton.cdiv(N, 256), triton.cdiv(K, 64))
        make_B_transpose_contiguous_kernel[grid_bt](B, BT, M, N, K)

        # Launch matmul kernel
        def grid_matmul(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        matmul_kernel[grid_matmul](A, BT, C, M, N, K)

        return C


def run(*args):
    return ModelNew()(*args)
