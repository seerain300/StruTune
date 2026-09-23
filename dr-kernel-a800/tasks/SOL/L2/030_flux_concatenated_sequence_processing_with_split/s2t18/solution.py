import torch
import triton
import triton.language as tl


@triton.jit
def cat_seq_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, I, L, H,
    encoder_s0, encoder_s1, encoder_s2,
    hidden_s0, hidden_s1, hidden_s2,
    out_s0, out_s1, out_s2,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (B, ceil(L / BLOCK_M))
    b = tl.program_id(0)
    seq_block = tl.program_id(1)

    m_offsets = seq_block * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offsets = tl.arange(0, BLOCK_K)

    # Loop over hidden dimension tiles
    for k_start in range(0, H, BLOCK_K):
        k = k_start + k_offsets
        # Masks for bounds
        mask_m = m_offsets < L
        mask_k = k < H

        # Select source based on sequence index
        is_text = m_offsets < T
        text_m = tl.where(is_text, m_offsets, 0)  # valid only for True; used for mask later
        image_m = m_offsets - T  # valid when not text

        # Compute pointers for encoder and hidden
        enc_ptrs = encoder_ptr + b * encoder_s0 + text_m[:, None] * encoder_s1 + k[None, :] * encoder_s2
        hid_ptrs = hidden_ptr + b * hidden_s0 + image_m[:, None] * hidden_s1 + k[None, :] * hidden_s2

        # Load with masks
        enc_vals = tl.load(enc_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        hid_vals = tl.load(hid_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Select based on is_text; for non-text, use hidden values; else use encoder
        vals = tl.where(is_text[:, None], enc_vals, hid_vals)

        # Store to output
        out_ptrs = out_ptr + b * out_s0 + m_offsets[:, None] * out_s1 + k[None, :] * out_s2
        tl.store(out_ptrs, vals, mask=mask_m[:, None] & mask_k[None, :])


@triton.jit
def batched_matmul_kernel_2d(
    A_ptr, Wt_ptr, C_ptr,
    B, L, H,
    A_s0, A_s1, A_s2,
    Wt_s0, Wt_s1,
    C_s0, C_s1, C_s2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (B, ceil(L / BLOCK_M))
    b = tl.program_id(0)
    seq_block = tl.program_id(1)

    m_offsets = seq_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)

    # Iterate over output tiles
    for n_start in range(0, H, BLOCK_N):
        n = n_start + n_offsets
        mask_m = m_offsets < L
        mask_n = n < H

        # Accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # accumulate in fp32 for numerical stability

        # Reduction over K
        for k_start in range(0, H, BLOCK_K):
            k = k_start + tl.arange(0, BLOCK_K)
            mask_k = k < H

            # Load A_cat[b, m, k] tile
            A_ptrs = A_ptr + b * A_s0 + m_offsets[:, None] * A_s1 + k[None, :] * A_s2
            A_tile = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

            # Load Wt[k, n] tile
            W_ptrs = Wt_ptr + k[:, None] * Wt_s0 + n[None, :] * Wt_s1
            W_tile = tl.load(W_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

            # Accumulate
            acc += tl.dot(A_tile.to(tl.float32), W_tile.to(tl.float32))

        # Store back to C[b, m, n]
        C_ptrs = C_ptr + b * C_s0 + m_offsets[:, None] * C_s1 + n[None, :] * C_s2
        tl.store(C_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized implementation:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension in Triton.
        - Apply linear projection (concatenated @ process_weight.T) in Triton.
        - Split back into separate encoder and image streams.

        Returns:
          processed_encoder: [B, T, H]
          processed_hidden: [B, I, H]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All tensors must be on CUDA for Triton kernels."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "Hidden dims must match."
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]."

        # Ensure contiguity
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        W = process_weight.contiguous()

        # Concatenate along sequence dimension using Triton: out_cat [B, L, H]
        L = T + I
        A_cat = torch.empty((B, L, H), device=hidden.device, dtype=hidden.dtype)

        # Launch concatenation kernel
        BLOCK_M_cat = 128  # sequence block
        BLOCK_K_cat = 64   # hidden block
        grid_cat = (B, triton.cdiv(L, BLOCK_M_cat))
        cat_seq_kernel[grid_cat](
            encoder, hidden, A_cat,
            B, T, I, L, H,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            A_cat.stride(0), A_cat.stride(1), A_cat.stride(2),
            BLOCK_M=BLOCK_M_cat, BLOCK_K=BLOCK_K_cat,
            num_warps=4, num_stages=2,
        )

        # Transpose process_weight to [H, H] for Triton: Wt[k, n] = process_weight[n, k]
        Wt = W.t().contiguous()  # [H, H], same dtype as process_weight

        # Allocate output for processed: [B, L, H]
        processed = torch.empty((B, L, H), device=hidden.device, dtype=hidden.dtype)

        # Launch batched matmul kernel: C[b, m, n] = sum_k A_cat[b, m, k] * Wt[k, n]
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64
        grid_mm = (B, triton.cdiv(L, BLOCK_M))
        batched_matmul_kernel_2d[grid_mm](
            A_cat, Wt, processed,
            B, L, H,
            A_cat.stride(0), A_cat.stride(1), A_cat.stride(2),
            Wt.stride(0), Wt.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Split back into separate streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
