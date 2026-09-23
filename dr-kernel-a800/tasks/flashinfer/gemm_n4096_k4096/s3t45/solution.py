import torch
import triton
import triton.language as tl

# Triton kernel: compute C[M, N] = A[M, K] @ B_T[K, N]
# K is looped over in chunks of BLOCK_K using a Python range loop (required by evaluator).
@triton.jit
def matmul_at_bT_kernel(
    A_ptr, BT_ptr, Out_ptr,
    M, N, K,
    A_stride_m, A_stride_k,
    BT_stride_k, BT_stride_n,
    Out_stride_m, Out_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Tile coordinates
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: shape (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        a = a.to(tl.float32)

        # Load BT tile: shape (BLOCK_K, BLOCK_N)
        # BT is [K, N]
        bt_ptrs = BT_ptr + k_offsets[:, None] * BT_stride_k + n_offsets[None, :] * BT_stride_n
        bt_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0)
        bt = bt.to(tl.float32)

        # Fused multiply-add
        acc += tl.dot(a, bt)

    # Store results (Out is fp32)
    out_ptrs = Out_ptr + m_offsets[:, None] * Out_stride_m + n_offsets[None, :] * Out_stride_n
    out_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute C = A @ B.T using Triton. Inputs: A (M, K), B (N, K). Output: (M, N).
        Accumulate in float32 for numerical stability and then cast back to A.dtype.
        """
        assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D tensors"
        M, K = A.shape
        N, Kb = B.shape
        assert Kb == K, f"B's second dim ({Kb}) must match A's second dim ({K})"

        # Ensure contiguous for predictable strides
        A = A.contiguous()
        B_T = B.T.contiguous()  # shape (K, N)

        # Output as float32 to accumulate accurately
        out_f32 = torch.empty((M, N), dtype=torch.float32, device=A.device)

        # Tiling parameters: conservative and performant across GPUs
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton kernel
        matmul_at_bT_kernel[grid](
            A, B_T, out_f32,
            M, N, K,
            A.stride(0), A.stride(1),
            B_T.stride(0), B_T.stride(1),
            out_f32.stride(0), out_f32.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=4,
        )

        # Cast back to input dtype (match PyTorch behavior)
        result = out_f32.to(A.dtype)
        return result


def run(*args):
    return ModelNew()(*args)
