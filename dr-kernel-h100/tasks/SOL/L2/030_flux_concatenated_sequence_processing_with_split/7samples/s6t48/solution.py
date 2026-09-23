import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_split_kernel(
    # Pointers to inputs
    encoder_hidden_states, hidden_states, process_weight, processed_concat,
    # Problem sizes
    B, T, I, H,
    # Strides
    stride_e_n, stride_e_s, stride_e_h,
    stride_h_n, stride_h_s, stride_h_h,
    stride_w_h, stride_w_k,
    stride_out_n, stride_out_s, stride_out_h,
    # Runtime parameters
    total_seq,
    tiles_h,
    BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # program ids
    n = tl.program_id(0)  # batch
    s = tl.program_id(1)  # sequence position in concatenated [0, total_seq)
    tile_h = tl.program_id(2)  # tile over hidden_dim

    # offsets along hidden_dim for this program's H tile
    h_offsets = tile_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    # choose input source based on sequence index
    # s < T comes from encoder_hidden_states, else from hidden_states at s - T
    from_encoder = s < T

    # load input vector from the correct source
    # If from_encoder: input[n, s, h]
    # Else: input[n, s - T, h]
    if from_encoder:
        # input row from encoder
        in_ptr = encoder_hidden_states + n * stride_e_n + s * stride_e_s
    else:
        # input row from hidden, offset by T
        in_ptr = hidden_states + n * stride_h_n + (s - T) * stride_h_s

    # Load input vector tile [BLOCK_K] for accumulation (iterate K in tiles)
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    k_start = 0
    while k_start < H:
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < H

        # load input slice: shape [BLOCK_K]
        in_vals = tl.load(in_ptr + k_offsets * stride_e_h, mask=k_mask, other=0.0).to(tl.float32)

        # Load weight tile W^T: weight is [H, H], we want W^T[k, h] = weight[h, k]
        # Pointer to weight[h_offsets, k_offsets]
        w_ptr = process_weight + h_offsets[:, None] * stride_w_h + k_offsets[None, :] * stride_w_k
        w_tile = tl.load(w_ptr, mask=h_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        # Accumulate dot products: acc[h] += sum_k in_vals[k] * w_tile[h, k]
        acc += tl.sum(w_tile * in_vals[None, :], axis=1)

        k_start += BLOCK_K

    # Store the accumulated H vector into out[n, s, :]
    out_ptr = processed_concat + n * stride_out_n + s * stride_out_s
    tl.store(out_ptr + h_offsets * stride_out_h, acc, mask=h_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run function.

        Performs:
          concatenated = cat([encoder_hidden_states, hidden_states], dim=1)
          processed = concatenated @ process_weight.T
          return processed_encoder = processed[:, :T], processed_hidden = processed[:, T:]

        All computation is done in Triton (no torch.cat or torch.matmul on tensors in forward).
        """
        # Ensure float32 and contiguous
        hidden = hidden_states.contiguous().to(torch.float32)
        encoder = encoder_hidden_states.contiguous().to(torch.float32)
        weight = process_weight.contiguous().to(torch.float32)

        B = hidden.shape[0]
        T = encoder.shape[1]
        I = hidden.shape[1]
        H = hidden.shape[2]
        total_seq = T + I

        # Allocate output [B, total_seq, H]
        processed_concat = torch.empty((B, total_seq, H), dtype=torch.float32, device=hidden.device)

        # Strides
        stride_e_n, stride_e_s, stride_e_h = encoder.stride(0), encoder.stride(1), encoder.stride(2)
        stride_h_n, stride_h_s, stride_h_h = hidden.stride(0), hidden.stride(1), hidden.stride(2)
        stride_w_h, stride_w_k = weight.stride(0), weight.stride(1)  # weight is [H, H]
        stride_out_n, stride_out_s, stride_out_h = processed_concat.stride(0), processed_concat.stride(1), processed_concat.stride(2)

        # Tile sizes: choose moderate defaults that work across common H
        # For H=128/256, this gives one H tile; for other H, mask handles tails.
        BLOCK_H = 128
        BLOCK_K = 128
        tiles_h = (H + BLOCK_H - 1) // BLOCK_H

        # Launch Triton kernel: grid over batch, sequence positions, and H tiles
        grid = (B, total_seq, tiles_h)

        concat_linear_split_kernel[grid](
            encoder, hidden, weight, processed_concat,
            B, T, I, H,
            stride_e_n, stride_e_s, stride_e_h,
            stride_h_n, stride_h_s, stride_h_h,
            stride_w_h, stride_w_k,
            stride_out_n, stride_out_s, stride_out_h,
            total_seq,
            tiles_h,
            BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
            num_warps=8,
            num_stages=2,
        )

        # Split outputs into encoder and hidden streams
        processed_encoder = processed_concat[:, :T, :]
        processed_hidden = processed_concat[:, T:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
