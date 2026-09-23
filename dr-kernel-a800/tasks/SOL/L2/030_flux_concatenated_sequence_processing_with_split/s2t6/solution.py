import torch
import triton
import triton.language as tl


@triton.jit
def cat_seq_kernel(
    encoder_ptr,  # *float32 [B, T, H]
    hidden_ptr,   # *float32 [B, I, H]
    out_ptr,      # *float32 [B, L, H]
    B, T, I, H, L,
    encoder_stride_b, encoder_stride_m, encoder_stride_k,
    hidden_stride_b, hidden_stride_m, hidden_stride_k,
    out_stride_b, out_stride_m, out_stride_k,
):
    # One program per batch
    b = tl.program_id(0)

    # Loop over sequence positions m in 0..L-1
    for m in range(0, L):
        # Determine source: encoder for m < T, hidden for m >= T
        is_encoder = m < T
        if is_encoder:
            src = encoder_ptr + b * encoder_stride_b + m * encoder_stride_m
        else:
            m_src = m - T
            src = hidden_ptr + b * hidden_stride_b + m_src * hidden_stride_m

        # Copy H elements to out[b, m, :]
        dst = out_ptr + b * out_stride_b + m * out_stride_m
        for k in range(0, H):
            val = tl.load(src + k * encoder_stride_k)
            tl.store(dst + k * out_stride_k, val)


@triton.jit
def batched_matmul_kernel(
    A_ptr,  # *float32 [B, L, K]
    B_ptr,  # *float32 [K, N]
    C_ptr,  # *float32 [B, L, N]
    B, L, K, N,
    A_stride_b, A_stride_m, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_b, C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    # Compute tile offsets
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for bounds
    m_mask = m_offsets < L
    n_mask = n_offsets < N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A tile: [BLOCK_M, BLOCK_K] for (b, m_offsets, k_offsets)
        A_ptrs = A_ptr + b * A_stride_b + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        A_mask = m_mask[:, None] & k_mask[None, :]
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load B tile: [BLOCK_K, BLOCK_N] for (k_offsets, n_offsets)
        B_ptrs = B_ptr + k_offsets[:, None] * B_stride_k + n_offsets[None, :] * B_stride_n
        B_mask = k_mask[:, None] & n_mask[None, :]
        B_tile = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, B_tile)

    # Store results into C[b, m_offsets, n_offsets]
    C_ptrs = C_ptr + b * C_stride_b + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(C_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension in a Triton kernel.
        - Performs batched matmul (A_cat @ process_weight.T) in a Triton kernel.
        - Returns processed_encoder and processed_hidden slices.
        """
        device = hidden_states.device

        # Shapes
        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = encoder_hidden_states.shape[2]
        L = T + I

        # Cast to float32 and ensure contiguous for Triton
        encoder = encoder_hidden_states.contiguous().to(torch.float32)
        hidden = hidden_states.contiguous().to(torch.float32)
        W = process_weight.contiguous().to(torch.float32)  # [H, H]
        W_T = W.t().contiguous()  # [H, H]

        # Output for concatenation: [B, L, H]
        A_cat = torch.empty((B, L, H), device=device, dtype=torch.float32)

        # Launch concatenation kernel: one program per batch
        grid_cat = (B,)
        cat_seq_kernel[grid_cat](
            encoder, hidden, A_cat,
            B, T, I, H, L,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            A_cat.stride(0), A_cat.stride(1), A_cat.stride(2),
            num_warps=1, num_stages=2,
        )

        # Allocate output for processed: [B, L, H]
        processed = torch.empty((B, L, H), device=device, dtype=torch.float32)

        # Tile sizes (robust defaults)
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64

        # Grid over batch, sequence tiles, and output tiles
        grid = (B, triton.cdiv(L, BLOCK_M), triton.cdiv(H, BLOCK_N))

        # Launch batched matmul kernel
        batched_matmul_kernel[grid](
            A_cat, W_T, processed,
            B, L, H, H,
            A_cat.stride(0), A_cat.stride(1), A_cat.stride(2),
            W_T.stride(0), W_T.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Split back into separate streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
