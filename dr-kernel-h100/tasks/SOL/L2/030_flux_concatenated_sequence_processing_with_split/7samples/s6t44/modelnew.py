import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_split_seqtile_kernel(
    encoder_hidden_states,  # [B, T, H]
    hidden_states,          # [B, I, H]
    process_weight,         # [H, H]
    processed_concat,       # [B, T+I, H]
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    H: tl.constexpr,
    stride_e_n, stride_e_s, stride_e_h,
    stride_h_n, stride_h_s, stride_h_h,
    stride_w_h, stride_w_k,
    stride_out_n, stride_out_s, stride_out_h,
    total_seq,
    tiles_h,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs: batch, sequence-tile, hidden-tile
    n = tl.program_id(0)
    S_block = tl.program_id(1)
    H_tile = tl.program_id(2)

    # Compute sequence indices for this block
    s_idx = S_block * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_idx < total_seq

    # Hidden feature indices for this tile
    h_idx = H_tile * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h_idx < H

    # Output pointers for this (n, s, h) block
    out_ptrs = processed_concat + n * stride_out_n + s_idx[:, None] * stride_out_s + h_idx[None, :] * stride_out_h

    # Accumulator [BLOCK_S, BLOCK_H]
    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # Accumulate over K dimension in tiles
    # For each k tile, load input row and weight row, then accumulate
    for k0 in range(0, H, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_idx < H

        # Determine source tensor for each s_idx element
        # s_idx < T -> encoder; else hidden (offset by T)
        is_encoder = s_idx < T

        # Build input pointer arrays [BLOCK_S, BLOCK_K]
        # For encoder: e[n, s, k]
        # For hidden: h[n, s - T, k]
        input_ptrs_e = encoder_hidden_states + n * stride_e_n + s_idx[:, None] * stride_e_s + k_idx[None, :] * stride_e_h
        input_ptrs_h = hidden_states + n * stride_h_n + (s_idx[:, None] - T) * stride_h_s + k_idx[None, :] * stride_h_h

        # Mask combining s validity and k validity
        mask_input = (mask_s[:, None]) & (mask_k[None, :])
        # Choose input pointers based on is_encoder
        input_ptrs = tl.where(is_encoder[:, None], input_ptrs_e, input_ptrs_h)

        # Load input rows as [BLOCK_S, BLOCK_K]
        # For masked elements, 'other' won't be used because we won't multiply them
        input_rows = tl.load(input_ptrs, mask=mask_input, other=0.0).to(tl.float32)  # [BLOCK_S, BLOCK_K]

        # Load weight rows as [BLOCK_K, BLOCK_H]
        weight_ptrs = process_weight + k_idx[:, None] * stride_w_k + h_idx[None, :] * stride_w_h
        weight_rows = tl.load(weight_ptrs, mask=(mask_k[:, None] & mask_h[None, :]), other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_H]

        # Accumulate: acc += input_rows @ weight_rows
        acc += tl.dot(input_rows, weight_rows)

    # Store results to output for valid s and h
    tl.store(out_ptrs, acc, mask=(mask_s[:, None] & mask_h[None, :]))


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure dtype and contiguity
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton."
        B, T, H = encoder_hidden_states.shape
        B2, I, H2 = hidden_states.shape
        assert B == B2 and H == H2, "encoder_hidden_states and hidden_states must have same batch and hidden_dim."

        # Make sure everything is float32 and contiguous
        encoder_hidden_states = encoder_hidden_states.contiguous().to(torch.float32)
        hidden_states = hidden_states.contiguous().to(torch.float32)
        process_weight = process_weight.contiguous().to(torch.float32)  # [H, H]

        total_seq = T + I
        processed_concat = torch.empty((B, total_seq, H), device=encoder_hidden_states.device, dtype=torch.float32)

        # Tile sizes (tuned for common hidden sizes; masks handle tails safely)
        BLOCK_H = 128
        BLOCK_K = 64
        # Sequence tiling: choose 64 to increase parallelism; mask handles tails
        BLOCK_S = 64

        tiles_h = (H + BLOCK_H - 1) // BLOCK_H
        # Grid: (batch, tiles over sequence, tiles over hidden)
        grid = (B, triton.cdiv(total_seq, BLOCK_S), tiles_h)

        concat_linear_split_seqtile_kernel[grid](
            encoder_hidden_states, hidden_states, process_weight, processed_concat,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            processed_concat.stride(0), processed_concat.stride(1), processed_concat.stride(2),
            total_seq,
            tiles_h,
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
            num_warps=8,
            num_stages=2,
        )

        processed_encoder = processed_concat[:, :T, :]
        processed_hidden = processed_concat[:, T:, :]
        return processed_encoder, processed_hidden