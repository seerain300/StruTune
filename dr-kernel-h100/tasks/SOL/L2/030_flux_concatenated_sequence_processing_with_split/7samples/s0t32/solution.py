import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    C_ptr, A_ptr, W_ptr,
    B, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs: (batch, tiles along M, tiles along N)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for boundary
    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in tiles
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A[b, m, k] tile: shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        A_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load W[k, n] tile: shape [BLOCK_K, BLOCK_N]
        W_ptrs = W_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wn
        W_mask = mask_k[:, None] & mask_n[None, :]
        w = tl.load(W_ptrs, mask=W_mask, other=0.0)

        # Accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(a, w)

    # Store result into C[b, m, n]
    C_ptrs = C_ptr + b * stride_cb + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    C_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenates along sequence dim (dim=1) using torch.cat (host-side).
        - Computes C = A @ process_weight.T using a Triton GEMM kernel.
        - Splits C back into two streams and returns (processed_encoder, processed_hidden).
        """
        # 1) Concatenate along sequence dimension
        # A: [B, T + I, K]
        A = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # Ensure A is contiguous for Triton
        A = A.contiguous()

        # 2) Prepare weight: process_weight is [K, K], input @ weight.T -> use W = process_weight.T
        # Cast to float32 for Triton compute (accumulation in fp32)
        W = process_weight.t().contiguous().to(torch.float32)

        # Shapes
        B = A.shape[0]
        M = A.shape[1]  # T + I
        K = A.shape[2]

        # 3) Allocate output C: [B, M, K]
        C = torch.empty((B, M, K), device=A.device, dtype=torch.float32)

        # 4) Launch Triton GEMM kernel
        # Tiling parameters: choose moderate sizes for robustness
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        batched_matmul_kernel[grid](
            C, A, W,
            B, M, K, K,
            A.stride(0), A.stride(1), A.stride(2),
            W.stride(0), W.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 5) Split outputs back into encoder and hidden streams
        processed_encoder = C[:, :encoder_hidden_states.shape[1], :]
        processed_hidden = C[:, encoder_hidden_states.shape[1]:, :]

        # Cast outputs back to original input dtypes (match original behavior)
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
