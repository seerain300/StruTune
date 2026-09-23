import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_split_kernel(
    encoder_hidden_states,  # [B, T, H]
    hidden_states,          # [B, I, H]
    process_weight,         # [H, H] (note: original uses process_weight.T)
    processed_concat,       # [B, T+I, H] output
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
    stride_e_n, stride_e_s, stride_e_h,
    stride_h_n, stride_h_s, stride_h_h,
    stride_w_h, stride_w_k,
    stride_out_n, stride_out_s, stride_out_h,
    total_seq,
    tiles_h,
    BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # 3D grid: (batch, sequence position, H tiles)
    n = tl.program_id(0)
    s = tl.program_id(1)  # 0 <= s < total_seq = T + I
    tile_h = tl.program_id(2)

    # Offsets over hidden features for this tile
    h_offsets = tile_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    # Determine source: encoder if s < T, else hidden at s - T
    is_encoder = s < T
    base_row = tl.where(is_encoder,
                        encoder_hidden_states + n * stride_e_n + s * stride_e_s,
                        hidden_states   + n * stride_h_n + (s - T) * stride_h_s)

    # Accumulator for out[n, s, h_offsets]
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Loop over K (input features) in tiles
    for k_start in range(0, H, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < H

        # Load input vector row for this s across k_offsets
        # Note: base_row has shape [*, H], we index by k_offsets to get a vector of size BLOCK_K
        x_k = tl.load(base_row + k_offsets * stride_e_h, mask=k_mask, other=0.0)  # same stride_h_h applies to both
        # Load weight row across h_offsets for this k tile: weight[k_offsets, h_offsets]
        # weight is [H, H], so pointer arithmetic uses k_offsets (rows) and h_offsets (cols)
        w_kh = tl.load(process_weight + k_offsets[:, None] * stride_w_k + h_offsets[None, :] * stride_w_h,
                       mask=k_mask[:, None] & h_mask[None, :], other=0.0)
        # Accumulate: sum over K tile
        acc += tl.sum(w_kh * x_k[:, None], axis=0)

    # Store the computed out[n, s, h_offsets]
    # Note: s is guaranteed valid in the grid (0 <= s < total_seq)
    tl.store(processed_concat + n * stride_out_n + s * stride_out_s + h_offsets * stride_out_h,
             acc, mask=h_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        1) Concatenate [encoder_hidden_states, hidden_states] along sequence dimension
        2) Apply linear projection with process_weight.T (no bias)
        3) Split back into encoder and hidden outputs
        """
        # Ensure dtype float32 and contiguous for predictable numerics and performance
        # Original code implies float32; enforce to avoid dtype mismatches.
        dtype = torch.float32
        if hidden_states.dtype != dtype:
            hidden_states = hidden_states.to(dtype)
        if encoder_hidden_states.dtype != dtype:
            encoder_hidden_states = encoder_hidden_states.to(dtype)
        if process_weight.dtype != dtype:
            process_weight = process_weight.to(dtype)

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        total_seq = T + I

        # Allocate output: [B, total_seq, H]
        processed_concat = torch.empty((B, total_seq, H), dtype=dtype, device=hidden_states.device)

        # Strides
        stride_e_n, stride_e_s, stride_e_h = encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2)
        stride_h_n, stride_h_s, stride_h_h = hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2)
        stride_w_h, stride_w_k = process_weight.stride(0), process_weight.stride(1)
        stride_out_n, stride_out_s, stride_out_h = processed_concat.stride(0), processed_concat.stride(1), processed_concat.stride(2)

        # Tiling parameters
        BLOCK_H = 128
        BLOCK_K = 64
        tiles_h = (H + BLOCK_H - 1) // BLOCK_H

        # Grid over (batch, sequence positions, H tiles)
        grid = (B, total_seq, tiles_h)

        concat_linear_split_kernel[grid](
            encoder_hidden_states, hidden_states, process_weight, processed_concat,
            B, T, I, H,
            stride_e_n, stride_e_s, stride_e_h,
            stride_h_n, stride_h_s, stride_h_h,
            stride_w_h, stride_w_k,
            stride_out_n, stride_out_s, stride_out_h,
            total_seq,
            tiles_h,
            BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )

        # Split outputs
        processed_encoder = processed_concat[:, :T, :]
        processed_hidden = processed_concat[:, T:, :]
        return processed_encoder, processed_hidden