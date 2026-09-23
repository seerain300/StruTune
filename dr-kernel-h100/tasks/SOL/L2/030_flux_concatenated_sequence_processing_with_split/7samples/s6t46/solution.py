import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_split_kernel(
    encoder_hidden_states,  # [B, T, H]
    hidden_states,          # [B, I, H]
    process_weight,         # [H, H]  (original code uses process_weight.T)
    processed_concat,       # [B, T+I, H] output
    B, T, I, H, total_seq,  # runtime ints
    stride_e_n, stride_e_s, stride_e_h,
    stride_h_n, stride_h_s, stride_h_h,
    stride_w_h, stride_w_k,
    stride_out_n, stride_out_s, stride_out_h,
    tiles_h,                # number of H tiles: ceil_div(H, BLOCK_H)
    BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    # Program IDs
    n = tl.program_id(0)  # batch index
    s = tl.program_id(1)  # sequence position in concatenated stream (0..total_seq-1)
    tile_h = tl.program_id(2)  # hidden tile index

    # Compute offsets for this H tile
    h_offsets = tile_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    # Determine input source: encoder if s < T, else hidden (offset s - T)
    use_encoder = s < T
    s_src = s if use_encoder else (s - T)

    # Load input row vector (float32, contiguous along H)
    if use_encoder:
        inp_row = tl.load(
            encoder_hidden_states + n * stride_e_n + s_src * stride_e_s + tl.arange(0, H) * stride_e_h,
            mask=tl.arange(0, H) < H,
            other=0.0,
        )
    else:
        inp_row = tl.load(
            hidden_states + n * stride_h_n + s_src * stride_h_s + tl.arange(0, H) * stride_h_h,
            mask=tl.arange(0, H) < H,
            other=0.0,
        )

    # Accumulator for output H-vector
    out_vec = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Accumulate over K (input features) in tiles of BLOCK_K
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < H

        # Load weight tile [BLOCK_K, BLOCK_H]: W^T where W is [H, H]
        # weight has shape [H, H], we load rows k_offsets (input features) and columns h_offsets (output features)
        w_tile = tl.load(
            process_weight + k_offsets[:, None] * stride_w_k + h_offsets[None, :] * stride_w_h,
            mask=k_mask[:, None] & h_mask[None, :],
            other=0.0,
        )

        # Compute partial dot: sum over K of inp_row[k] * w_tile[k, :]
        # inp_row is [H]; w_tile is [BLOCK_K, BLOCK_H]
        # We need to extract inp_row[k] for k in k_offsets and multiply per column in h_offsets
        partial = tl.zeros([BLOCK_H], dtype=tl.float32)
        # Manual accumulation across BLOCK_K to avoid potential issues
        for kk in range(BLOCK_K):
            k_idx = k0 + kk
            k_valid = k_idx < H
            # inp_row[k_idx] is scalar; multiply with w_tile[kk, :]
            # w_tile[kk, :] is [BLOCK_H]; masked by h_mask
            w_col = w_tile[kk, :]
            inp_val = tl.where(k_valid, inp_row[k_idx], 0.0)
            partial += w_col * inp_val

        out_vec += partial

    # Store result to processed_concat at position (n, s, h_offsets)
    tl.store(
        processed_concat + n * stride_out_n + s * stride_out_s + h_offsets * stride_out_h,
        out_vec,
        mask=h_mask,
    )


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton implementation of the original run function:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension.
        - Apply linear projection with process_weight.T.
        - Split back into encoder and hidden outputs.
        """
        # Ensure dtype float32 and contiguous
        B, T, H = encoder_hidden_states.shape
        _, I, H2 = hidden_states.shape
        assert H == H2, "hidden_dim must match between encoder_hidden_states and hidden_states"
        total_seq = T + I

        # Make inputs contiguous and float32
        encoder = encoder_hidden_states.contiguous().to(torch.float32)
        hidden = hidden_states.contiguous().to(torch.float32)
        weight = process_weight.contiguous().to(torch.float32)  # [H, H]

        # Allocate output [B, T+I, H]
        processed_concat = torch.empty((B, total_seq, H), device=encoder.device, dtype=torch.float32)

        # Strides
        stride_e_n, stride_e_s, stride_e_h = encoder.stride(0), encoder.stride(1), encoder.stride(2)
        stride_h_n, stride_h_s, stride_h_h = hidden.stride(0), hidden.stride(1), hidden.stride(2)
        stride_w_h, stride_w_k = weight.stride(0), weight.stride(1)  # [H, H]
        stride_out_n, stride_out_s, stride_out_h = processed_concat.stride(0), processed_concat.stride(1), processed_concat.stride(2)

        # Tile over hidden dimension
        BLOCK_H = 128
        tiles_h = (H + BLOCK_H - 1) // BLOCK_H

        # Grid: (batch, total_seq, tiles_h)
        grid = (B, total_seq, tiles_h)

        # Launch Triton kernel
        concat_linear_split_kernel[grid](
            encoder, hidden, weight, processed_concat,
            B, T, I, H, total_seq,
            stride_e_n, stride_e_s, stride_e_h,
            stride_h_n, stride_h_s, stride_h_h,
            stride_w_h, stride_w_k,
            stride_out_n, stride_out_s, stride_out_h,
            tiles_h,
            BLOCK_K=64, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        # Split outputs
        processed_encoder = processed_concat[:, :T, :]
        processed_hidden = processed_concat[:, T:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
