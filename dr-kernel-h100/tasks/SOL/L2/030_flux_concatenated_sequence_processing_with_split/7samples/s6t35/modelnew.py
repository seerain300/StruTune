import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_split_kernel(
    encoder_hidden_states_ptr,  # [B, T, H]
    hidden_states_ptr,          # [B, I, H]
    process_weight_ptr,         # [H, H]
    processed_concat_ptr,       # [B, T+I, H]
    B,                          # int: batch size
    T,                          # int: text_seq_len
    I,                          # int: img_seq_len
    H,                          # int: hidden_dim
    stride_e_n, stride_e_s, stride_e_h,  # encoder strides
    stride_h_n, stride_h_s, stride_h_h,  # hidden strides
    stride_w_h, stride_w_k,               # process_weight strides (shape [H, H])
    stride_out_n, stride_out_s, stride_out_h,  # output strides
    total_seq,                  # int: T + I
    tiles_h,                    # int: number of H tiles
    BLOCK_K: tl.constexpr,      # tile size for K (input features)
    BLOCK_H: tl.constexpr,      # tile size for output features (H)
):
    # program ids: (n over batch, s over total sequence, h_tile over H tiles)
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_ht = tl.program_id(2)

    # H offsets for this tile
    h_offsets = pid_ht * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    # Select the input row based on position s
    if pid_s < total_seq:
        if pid_s < T:
            # input from encoder: [n, s, :]
            # compute base pointer then load row
            e_row_ptr = encoder_hidden_states_ptr + pid_n * stride_e_n + pid_s * stride_e_s
            input_row = tl.load(e_row_ptr + h_offsets * stride_e_h, mask=h_mask, other=0.0)
        else:
            # input from hidden: [n, s - T, :]
            h_row_ptr = hidden_states_ptr + pid_n * stride_h_n + (pid_s - T) * stride_h_s
            input_row = tl.load(h_row_ptr + h_offsets * stride_h_h, mask=h_mask, other=0.0)

        # Accumulator for this tile
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)

        # Iterate over K (input features) in tiles
        for k_start in range(0, H, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < H

            # Load weight tile [BLOCK_K, BLOCK_H]: process_weight[k, h] with k=k_offsets, h=h_offsets
            # Weight is [H, H], we want W^T, i.e., [K, N] with K=H, N=H.
            w_tile = tl.load(
                process_weight_ptr + k_offsets[:, None] * stride_w_k + h_offsets[None, :] * stride_w_h,
                mask=k_mask[:, None] & h_mask[None, :],
                other=0.0,
            )

            # Load corresponding input_row[k_offsets]
            x_k = tl.load(
                input_row_ptr + k_offsets * stride_e_h,  # input_row_ptr already computed above
                mask=k_mask,
                other=0.0,
            )

            # acc += sum_k x_k * w_tile[k, :]
            # Broadcast multiply and reduce along K axis
            acc += tl.sum(w_tile * x_k[None, :], axis=0)

        # Store the result for this (n, s) row across H offsets
        out_row_ptr = processed_concat_ptr + pid_n * stride_out_n + pid_s * stride_out_s
        tl.store(out_row_ptr + h_offsets * stride_out_h, acc, mask=h_mask)


def run(encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Ensure dtype and contiguity
    assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton."
    B = encoder_hidden_states.shape[0]
    T = encoder_hidden_states.shape[1]
    I = hidden_states.shape[1]
    H = encoder_hidden_states.shape[2]
    assert hidden_states.shape[2] == H, "hidden_states and encoder_hidden_states must have the same hidden_dim."
    assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [hidden_dim, hidden_dim]."

    e = encoder_hidden_states.contiguous().to(torch.float32)
    h = hidden_states.contiguous().to(torch.float32)
    w = process_weight.contiguous().to(torch.float32)

    total_seq = T + I
    # Output tensor [B, total_seq, H]
    processed_concat = torch.empty((B, total_seq, H), dtype=torch.float32, device=e.device)

    # Tile sizes: use 128 for robustness (common hidden_dim); masks handle tails
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