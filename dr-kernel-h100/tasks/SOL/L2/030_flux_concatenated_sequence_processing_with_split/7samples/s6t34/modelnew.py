import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_split_kernel(
    encoder_hidden_states_ptr,  # [B, T, H]
    hidden_states_ptr,          # [B, I, H]
    process_weight_ptr,         # [H, H]
    processed_concat_ptr,       # [B, T+I, H]
    B: tl.constexpr,            # batch size
    T: tl.constexpr,            # text_seq_len
    I: tl.constexpr,            # img_seq_len
    H: tl.constexpr,            # hidden_dim
    stride_e_n, stride_e_s, stride_e_h,  # encoder strides
    stride_h_n, stride_h_s, stride_h_h,  # hidden strides
    stride_w_h, stride_w_k,               # process_weight strides (shape [H, H])
    stride_out_n, stride_out_s, stride_out_h,  # output strides
    total_seq: tl.constexpr,  # T + I
    tiles_h: tl.constexpr,    # number of H tiles
    BLOCK_K: tl.constexpr,    # tile over input features (K = H)
    BLOCK_H: tl.constexpr,    # tile over output features (H)
):
    # program ids: (n over batch, s over total sequence, h_tile over H tiles)
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_ht = tl.program_id(2)

    # Compute output H offsets for this tile
    h_offsets = pid_ht * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    # Determine input row source
    if pid_s < total_seq:
        if pid_s < T:
            # input from encoder: [n, s, :]
            e_ptr = encoder_hidden_states_ptr + pid_n * stride_e_n + pid_s * stride_e_s
            input_row = tl.load(e_ptr + tl.arange(0, H) * stride_e_h, mask=tl.full((H,), True, tl.int1), other=0.0)
        else:
            # input from hidden: [n, s - T, :]
            h_ptr = hidden_states_ptr + pid_n * stride_h_n + (pid_s - T) * stride_h_s
            input_row = tl.load(h_ptr + tl.arange(0, H) * stride_h_h, mask=tl.full((H,), True, tl.int1), other=0.0)

        # Load weight tiles: W^T is [H, H] (weight is [H, H]). We need W^T[h, k].
        # Accumulator for output vector
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)

        # Tiled accumulation over K (input features)
        for k_start in range(0, H, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < H
            # weight tile: [BLOCK_H, BLOCK_K] = W^T[h, k] for h in this tile, k in this tile
            # address: weight_ptr + h_offsets[:, None]*stride_w_h + k_offsets[None, :]*stride_w_k
            w_tile = tl.load(
                process_weight_ptr + h_offsets[:, None] * stride_w_h + k_offsets[None, :] * stride_w_k,
                mask=h_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            # input_row_k: [BLOCK_K]
            in_k = tl.load(
                input_row_ptr + k_offsets * stride_in_k,
                mask=k_mask,
                other=0.0,
            )
            # Broadcast and accumulate: acc[h] += sum_k w_tile[h, k] * in_k[k]
            acc += tl.sum(w_tile * in_k[None, :], axis=1)

        # Store the computed output vector into processed_concat[n, s, :]
        out_ptr = processed_concat_ptr + pid_n * stride_out_n + pid_s * stride_out_s
        tl.store(out_ptr + h_offsets * stride_out_h, acc, mask=h_mask)


# The rest of the forward logic in ModelNew
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-based implementation of concatenated sequence processing.
    """
    B, T, H = encoder_hidden_states.shape
    B2, I, H2 = hidden_states.shape
    assert B == B2 and H == H2, "Batch or hidden_dim mismatch"

    # Ensure inputs are contiguous and float32
    e = encoder_hidden_states.contiguous().to(torch.float32)
    h = hidden_states.contiguous().to(torch.float32)
    w = process_weight.contiguous().to(torch.float32)

    total_seq = T + I
    processed_concat = torch.empty((B, total_seq, H), device=e.device, dtype=torch.float32)

    # Tile sizes: choose 128 to cover typical hidden dims (128/256) robustly
    BLOCK_H = 128
    BLOCK_K = 128
    tiles_h = (H + BLOCK_H - 1) // BLOCK_H

    # Launch Triton kernel: grid over (batch, total_seq, H-tiles)
    grid = (B, total_seq, tiles_h)
    concat_linear_split_kernel[grid](
        e, h, w, processed_concat,
        B, T, I, H,
        e.stride(0), e.stride(1), e.stride(2),
        h.stride(0), h.stride(1), h.stride(2),
        w.stride(0), w.stride(1),
        processed_concat.stride(0), processed_concat.stride(1), processed_concat.stride(2),
        total_seq,
        tiles_h,
        BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
        num_warps=4,
        num_stages=2,
    )

    # Split outputs into encoder and hidden streams
    processed_encoder = processed_concat[:, :T, :]
    processed_hidden = processed_concat[:, T:, :]
    return processed_encoder, processed_hidden

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        return run(encoder_hidden_states, hidden_states, process_weight)