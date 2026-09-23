import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel(
    A_ptr,  # *ptr to A (M x K), dtype: f16
    BT_ptr,  # *ptr to B.T (K x N), dtype: f16
    C_ptr,  # *ptr to C (M x N), dtype: f16
    M, N, K,
    stride_am, stride_ak,  # strides for A
    stride_btk, stride_btn,  # strides for BT (K, N)
    stride_cm, stride_cn,  # strides for C
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D program ids for tiles along M and N
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in f16 to match input/output dtype; shape matches tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

    # Loop over K dimension in chunks; use Python range for runtime K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Compute pointers for the current A and BT tiles
        A_tile_ptr = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        BT_tile_ptr = BT_ptr + (offs_k[:, None] * stride_btk + offs_n[None, :] * stride_btn)

        # Masks for boundary handling: valid rows/cols in A and BT, and valid cols in C
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        # Store mask is only about C boundaries; offs_m < M and offs_n < N cover valid stores
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

        # Load tiles as f16
        a = tl.load(A_tile_ptr, mask=a_mask, other=0.0)  # f16
        b = tl.load(BT_tile_ptr, mask=b_mask, other=0.0)  # f16

        # Accumulate in f16 (no upcasting to match PyTorch's f16 behavior)
        acc += a @ b

    # Store results with mask (C boundaries)
    C_tile_ptr = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(C_tile_ptr, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Triton requires CUDA tensors
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton kernels."

        # Shapes: A is (M, K), B is (N, K) in provided inputs. Compute C = A @ B.T
        M, K = A.shape
        N = B.shape[0]  # B has shape (N, K)
        BT = B.T.contiguous()  # BT has shape (K, N), dtype f16

        # Output tensor (M, N), dtype f16
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Tile sizes: conservative to avoid shared memory issues and ensure good occupancy
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        # Grid over tiles
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch kernel
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


def run(*args):
    return ModelNew()(*args)
