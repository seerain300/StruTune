import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel(
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

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        # Load A tile: shape (BLOCK_M, BLOCK_K)
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak)
        a_mask = (offs_m[:, None] < M) & (k0 + offs_k[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load BT tile: shape (BLOCK_K, BLOCK_N), BT is (K, N)
        BT_ptrs = BT_ptr + ((k0 + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = (k0 + offs_k[:, None] < K) & (offs_n[None, :] < N)
        bt = tl.load(BT_ptrs, mask=b_mask, other=0.0)

        # Accumulate in fp32
        acc += tl.dot(a, bt)

    # Store result C[m, n] = acc (cast to output dtype on store)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Compute C = A @ B.T, A is (M, K), B is (N, K), output is (M, N)
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        M, K_a = A.shape
        N_b, K_b = B.shape
        assert K_a == K_b, f"Incompatible shapes: A is (M, {K_a}) and B is ({N_b}, {K_b})"
        # Make inputs contiguous for better performance
        A_c = A.contiguous()
        B_c = B.contiguous()

        # BT is B.T contiguous: (K, N)
        BT = B_c.transpose(0, 1).contiguous()
        N = B_c.shape[0]  # output N dimension
        K = K_a

        # Allocate output as float16 (matching get_inputs dtype)
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Heuristic selection of tile sizes and launch parameters
        if (M >= 4096) or (N >= 4096):
            BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
            num_warps, num_stages = 8, 4
        elif (M * N) >= 4_000_000:
            BLOCK_M, BLOCK_N, BLOCK_K = 128, 64, 64
            num_warps, num_stages = 8, 3
        elif (M >= 1024) or (N >= 1024):
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 64
            num_warps, num_stages = 8, 3
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
            num_warps, num_stages = 4, 2

        # Launch Triton kernel
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_at_bT_kernel[grid](
            A_c, BT, C,
            M, N, K,
            A_c.stride(0), A_c.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=num_stages,
        )

        return C


def run(*args):
    return ModelNew()(*args)
