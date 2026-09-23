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
    BLOCK_K: tl.constexpr,      # tile over input features (K = H)
    BLOCK_H: tl.constexpr,      # tile over output features (H)
):
    # program ids: (n over batch, s over total sequence, h_tile over H tiles)
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_ht = tl.program_id(2)

    # Compute H tile offsets and mask
    h_offsets = pid_ht * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    # Accumulator for this (n, s) row across H tile
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Select input source based on sequence index
    if pid_s < total_seq:
        if pid_s < T:
            # input from encoder: [n, s, :]
            e_ptr = encoder_hidden_states_ptr + pid_n * stride_e_n + pid_s * stride_e_s
            input_vec = tl.load(e_ptr + tl.arange(0, H) * stride_e_h, mask=tl.arange(0, H) < H, other=0.0)
        else:
            # input from hidden: [n, s - T, :]
            h_ptr = hidden_states_ptr + pid_n * stride_h_n + (pid_s - T) * stride_h_s
            input_vec = tl.load(h_ptr + tl.arange(0, H) * stride_h_h, mask=tl.arange(0, H) < H, other=0.0)

    # Accumulate over K tiles (input features)
    for k_start in range(0, H, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < H
        # Load process_weight tile: [BLOCK_K, BLOCK_H]
        w_ptrs = process_weight_ptr + k_offsets[:, None] * stride_w_h + h_offsets[None, :] * stride_w_k
        w_tile = tl.load(w_ptrs, mask=k_mask[:, None] & h_mask[None, :], other=0.0)

        # Compute dot-product for this K tile across BLOCK_H
        # acc[h] += sum_{kk in K tile} input_vec[kk] * w_tile[kk, h]
        # Note: input_vec is length-H; we only keep valid kk entries via masked load above.
        for kk in range(BLOCK_K):
            kk_valid = (k_start + kk) < H
            if kk_valid:
                # multiply scalar input_vec[k_start + kk] with w_tile[kk, :]
                # acc += input_vec[k_start + kk] * w_tile[kk, :]
                acc += input_vec[k_start + kk] * w_tile[kk, :]

    # Store results for this (n, s, h_offsets) tile
    out_ptrs = processed_concat_ptr + pid_n * stride_out_n + pid_s * stride_out_s + h_offsets * stride_out_h
    tl.store(out_ptrs, acc, mask=h_mask)


def run(encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor):
    # Ensure contiguous and float32 for predictable numerics
    e = encoder_hidden_states.contiguous().to(torch.float32)
    h = hidden_states.contiguous().to(torch.float32)
    w = process_weight.contiguous().to(torch.float32)  # shape [H, H]

    B, T, H_e = e.shape
    B2, I, H_h = h.shape
    assert B == B2, "Batch size must match"
    assert H_e == H_h, "Hidden dim must match"
    H = H_e
    total_seq = T + I

    # Allocate output tensor [B, T+I, H]
    processed_concat = torch.empty((B, total_seq, H), device=e.device, dtype=torch.float32)

    # Tile sizes: choose 128 for robustness across common hidden sizes (128/256)
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
        # Call the Triton-optimized run
        processed_encoder, processed_hidden = run(encoder_hidden_states, hidden_states, process_weight)
        return processed_encoder, processed_hidden