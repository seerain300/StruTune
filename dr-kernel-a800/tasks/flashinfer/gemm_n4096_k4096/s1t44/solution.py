import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Configs focused on moderate tiles to avoid shared memory overflow while covering common regimes
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},   num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},   num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},   num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128},  num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bT_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for tiling
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this program
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to A and B tiles
    # A: [M, K], B: [N, K] but we need B.T -> we index B as B[n, k] for n in [offs_n], k in [offs_k]
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_mask = (offs_k + k) < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        # advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Write back to C
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure inputs are contiguous
        A = A.contiguous()
        B = B.contiguous()

        # Output tensor (fp32 for accumulation, then cast to A.dtype)
        M, K = A.shape
        N, Kb = B.shape
        assert Kb == K, "B's last dim must equal A's last dim (K)"
        # Allocate fp32 output for safe accumulation
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Launch Triton kernel
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )
        matmul_bT_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
        )

        # Cast back to input dtype to match expected output
        if C.dtype != A.dtype:
            C = C.to(A.dtype)
        return C


def run(*args):
    return ModelNew()(*args)
