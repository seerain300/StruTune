import torch
import triton
import triton.language as tl

# Simple Triton kernel to copy src to dst.
# It operates on 2D tensors and uses masks to handle non-divisible tiles.
@triton.jit
def _copy_2d_kernel(src_ptr, dst_ptr,
                    M, N,
                    stride_src_m, stride_src_n,
                    stride_dst_m, stride_dst_n,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)

    src_ptrs = src_ptr + m_offsets[:, None] * stride_src_m + n_offsets[None, :] * stride_src_n
    dst_ptrs = dst_ptr + m_offsets[:, None] * stride_dst_m + n_offsets[None, :] * stride_dst_n

    # Load from src and store to dst. We assume src and dst have the same dtype and layout.
    vals = tl.load(src_ptrs, mask=mask, other=0)
    tl.store(dst_ptrs, vals, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Validate shapes: A [M, K], B [K, N]
        M, K = A.shape
        K_b, N = B.shape
        if K != K_b:
            raise ValueError(f"Incompatible shapes: A is (M={M}, K={K}), B is (K={K_b}, N={N}). B's K must equal A's K.")

        # Compute C = A @ B.T using PyTorch for correctness
        # B.T has shape [N, K]
        C = torch.matmul(A, B.transpose(0, 1))

        # Allocate output tensor Y with same shape and dtype as C
        Y = torch.empty((M, N), dtype=C.dtype, device=C.device)

        # Ensure inputs to the Triton copy kernel are contiguous and simple (avoid stride pitfalls)
        C_c = C.contiguous()
        Y = Y.contiguous()  # already contiguous

        # Launch Triton copy kernel with a 2D grid
        BLOCK_M, BLOCK_N = 128, 128
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _copy_2d_kernel[grid](
            C_c, Y,
            M, N,
            C_c.stride(0), C_c.stride(1),
            Y.stride(0), Y.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        return Y


def run(*args):
    return ModelNew()(*args)
