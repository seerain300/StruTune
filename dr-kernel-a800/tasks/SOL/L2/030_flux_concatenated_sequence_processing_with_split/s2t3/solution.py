import torch
import triton
import triton.language as tl


@triton.jit
def _batched_matmul_kernel(
    A_ptr,  # [B, L, K]
    B_ptr,  # [K, N]  (process_weight.T)
    C_ptr,  # [B, L, N]
    B_shape, L, K, N,
    A_stride_b, A_stride_m, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_b, C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid over (batch, sequence blocks, output column blocks)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    m_valid = m_offsets < L
    n_valid = n_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offsets < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + b * A_stride_b + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        A_mask = m_valid[:, None] & k_mask[None, :]
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load B tile: [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + k_offsets[:, None] * B_stride_k + n_offsets[None, :] * B_stride_n
        B_mask = k_mask[:, None] & n_valid[None, :]
        B_tile = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Accumulate: acc += A_tile @ B_tile
        # For each output column nn in [BLOCK_N], acc[:, nn] += sum_k A_tile[:, k] * B_tile[k, nn]
        # Using einsum-like broadcast:
        acc += tl.sum(A_tile[:, :, None] * B_tile[None, :, :], axis=1)

    # Store results
    C_ptrs = C_ptr + b * C_stride_b + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    store_mask = m_valid[:, None] & n_valid[None, :]
    tl.store(C_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version that:
          - Concatenates [encoder_hidden_states, hidden_states] along sequence dim (host-side torch.cat for data movement)
          - Applies linear projection via Triton matmul
          - Splits into encoder and hidden outputs
        """
        # Ensure CUDA tensors; compute in float32
        device = hidden_states.device
        if device.type != "cuda":
            raise RuntimeError("ModelNew requires CUDA tensors. Move inputs to CUDA.")

        # Cast to float32 and ensure contiguous
        encoder = encoder_hidden_states.contiguous().float()
        hidden = hidden_states.contiguous().float()
        # process_weight: [H, H], we need B = process_weight.T [H, H]
        B = process_weight.t().contiguous().float()

        # Shapes
        B_shape = encoder.shape[0]
        T = encoder.shape[1]
        K = encoder.shape[2]
        I = hidden.shape[1]
        L = T + I
        N = B.shape[1]  # should equal K

        # Concatenate sequences along sequence dim
        A_cat = torch.cat([encoder, hidden], dim=1)  # [B, L, K]

        # Allocate output [B, L, N]
        out = torch.empty((B_shape, L, N), device=device, dtype=torch.float32)

        # Strides
        A_stride_b, A_stride_m, A_stride_k = A_cat.stride(0), A_cat.stride(1), A_cat.stride(2)
        B_stride_k, B_stride_n = B.stride(0), B.stride(1)
        C_stride_b, C_stride_m, C_stride_n = out.stride(0), out.stride(1), out.stride(2)

        # Tile sizes (heuristics)
        BLOCK_N = 128 if N >= 128 else 64
        BLOCK_K = 64 if K >= 64 else 32
        BLOCK_M = 128

        # Grid over batch, sequence blocks, output column blocks
        grid = (B_shape, triton.cdiv(L, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Launch Triton matmul kernel
        _batched_matmul_kernel[grid](
            A_cat, B, out,
            B_shape, L, K, N,
            A_stride_b, A_stride_m, A_stride_k,
            B_stride_k, B_stride_n,
            C_stride_b, C_stride_m, C_stride_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Split results
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
