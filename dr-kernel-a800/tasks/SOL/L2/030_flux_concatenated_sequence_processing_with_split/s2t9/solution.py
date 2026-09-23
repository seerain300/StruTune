import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_seq_kernel(
    encoder_ptr,  # [B, T, H]
    hidden_ptr,   # [B, I, H]
    out_ptr,      # [B, L, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr, L: tl.constexpr,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    hidden_stride_b, hidden_stride_i, hidden_stride_h,
    out_stride_b, out_stride_l, out_stride_h,
):
    # Grid: (B, L)
    b = tl.program_id(0)
    m = tl.program_id(1)  # sequence index 0..L-1
    # Choose source based on m < T
    is_encoder = m < T
    k = 0
    # Loop over hidden dimension H
    while k < H:
        # Compute source pointers and load
        if is_encoder:
            val = tl.load(encoder_ptr + b * encoder_stride_b + m * encoder_stride_t + k * encoder_stride_h)
        else:
            mi = m - T
            val = tl.load(hidden_ptr + b * hidden_stride_b + mi * hidden_stride_i + k * hidden_stride_h)
        # Store into out[b, m, k]
        tl.store(out_ptr + b * out_stride_b + m * out_stride_l + k * out_stride_h, val)
        k += 1


@triton.jit
def _batched_matmul_kernel(
    A_ptr,   # [B, L, K], here we will pass the concatenated tensor
    B_ptr,   # [K, N], process_weight.T, shape [H, H]
    C_ptr,   # [B, L, N], output
    B: tl.constexpr, L: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    A_stride_b, A_stride_l, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_b, C_stride_l, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, ceil_div(L, BLOCK_M))
    b = tl.program_id(0)
    m_block = tl.program_id(1)

    # Offsets for sequence positions in this block
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)

    # Masks for sequence and output columns
    m_mask = m_offsets < L
    n_mask = n_offsets < N

    # Initialize accumulator [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_range < K

        # Load A rows: shape [BLOCK_M, BLOCK_K]
        # A[b, m, k] layout: b, l, k
        A_ptrs = A_ptr + b * A_stride_b + m_offsets[:, None] * A_stride_l + k_range[None, :] * A_stride_k
        A_tile = tl.load(A_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        # Load B chunk: shape [BLOCK_K, BLOCK_N], B[k, n] layout: k, n
        B_ptrs = B_ptr + k_range[:, None] * B_stride_k + n_offsets[None, :] * B_stride_n
        B_tile = tl.load(B_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

        # Accumulate: acc += A_tile @ B_tile
        acc += tl.dot(A_tile, B_tile)

    # Store results into C[b, m, n]
    C_ptrs = C_ptr + b * C_stride_b + m_offsets[:, None] * C_stride_l + n_offsets[None, :] * C_stride_n
    # Broadcast masks: handle case where L or N not multiples of blocks
    tl.store(C_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
          - Concatenates encoder_hidden_states and hidden_states along sequence dimension via Triton kernel
          - Computes linear projection via Triton batched GEMM on the concatenated tensor
          - Splits results back into encoder and hidden streams
        """
        # Ensure CUDA tensors
        device = hidden_states.device
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"

        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        L = T + I

        # Ensure contiguity
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        W_T = process_weight.t().contiguous()  # [H, H]

        # Output for concatenated tensor [B, L, H]
        A_cat = torch.empty((B, L, H), device=device, dtype=torch.float32)

        # Launch concatenation kernel: grid over (B, L)
        grid_concat = (B, L)
        _concatenate_seq_kernel[grid_concat](
            encoder, hidden, A_cat,
            B, T, I, H, L,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            A_cat.stride(0), A_cat.stride(1), A_cat.stride(2),
            num_warps=1, num_stages=1,
        )

        # Output for processed [B, L, H]
        processed = torch.empty((B, L, H), device=device, dtype=torch.float32)

        # Choose tile sizes (conservative for robustness)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        # Grid over batch and sequence tiles
        grid_mm = (B, triton.cdiv(L, BLOCK_M))

        _batched_matmul_kernel[grid_mm](
            A_cat, W_T, processed,
            B, L, H, H,  # K=H, N=H
            A_cat.stride(0), A_cat.stride(1), A_cat.stride(2),
            W_T.stride(0), W_T.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Split back into encoder and hidden streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
