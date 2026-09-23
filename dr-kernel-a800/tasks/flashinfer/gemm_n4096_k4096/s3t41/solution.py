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
    # 2D program ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        A_tile_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_tile = tl.load(A_tile_ptrs, mask=a_mask, other=0.0)

        # Load BT tile: [BLOCK_K, BLOCK_N]
        # BT has shape (K, N) after transposing B
        BT_tile_ptrs = BT_ptr + (offs_k[:, None] * stride_btk + offs_n[None, :] * stride_btn)
        bt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        BT_tile = tl.load(BT_tile_ptrs, mask=bt_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, BT_tile)

    # Store results to C with mask
    C_tile_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_tile_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # If tensors are on CUDA, compute via Triton; else fallback to torch
        if A.is_cuda and B.is_cuda:
            M, K = A.shape
            N, Kb = B.shape
            assert Kb == K, "B must have shape [N, K], matching A's second dim."

            # Prepare BT = B.T as contiguous [K, N]
            BT = B.transpose(0, 1).contiguous()  # shape: [N, K]
            BT = BT.transpose(0, 1).contiguous()  # shape: [K, N]

            # Output tensor (match original dtype: float16)
            C = torch.empty((M, N), device=A.device, dtype=torch.float16)

            # Tiling parameters
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
                num_warps=8, num_stages=3,
            )
            return C
        else:
            # CPU fallback (not used in evaluation)
            return torch.matmul(A, B.transpose(0, 1))


def run(*args):
    return ModelNew()(*args)
