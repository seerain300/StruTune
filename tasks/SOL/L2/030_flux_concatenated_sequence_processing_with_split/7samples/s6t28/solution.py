import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_split_kernel(
    e_ptr,        # [B, T, H], float32
    h_ptr,        # [B, I, H], float32
    w_ptr,        # [H, H], float32
    out_ptr,      # [B, T+I, H], float32 (we'll write only to s < T+I)
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    stride_e_n: tl.int32, stride_e_s: tl.int32, stride_e_h: tl.int32,
    stride_h_n: tl.int32, stride_h_s: tl.int32, stride_h_h: tl.int32,
    stride_w_h: tl.int32, stride_w_k: tl.int32,
    stride_out_n: tl.int32, stride_out_s: tl.int32, stride_out_h: tl.int32,
    total_seq: tl.int32,
    tiles_h: tl.int32,
    BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # program ids
    pid_n = tl.program_id(0)  # batch index
    pid_s = tl.program_id(1)  # sequence index (0..total_seq-1)
    pid_th = tl.program_id(2) # tile index over H

    # Compute h offsets for this tile
    h_offsets = pid_th * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    # Determine input source: if pid_s < T, use encoder; else use hidden at s - T
    use_encoder = pid_s < T

    # Compute input row pointer: either e[n, s, :] or h[n, s - T, :]
    if use_encoder:
        # s is valid since use_encoder implies pid_s < T
        in_row_ptr = e_ptr + pid_n * stride_e_n + pid_s * stride_e_s
    else:
        s_hidden = pid_s - T
        in_row_ptr = h_ptr + pid_n * stride_h_n + s_hidden * stride_h_s

    # Accumulator for output vector
    out_vec = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Loop over K (input features) in tiles
    for k_start in range(0, H, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # Load input row slice: shape [BLOCK_K]
        in_vals = tl.load(in_row_ptr + k_offsets * stride_e_h, mask=mask_k, other=0.0)

        # Load weight slices: w[k, h] -> [BLOCK_K, BLOCK_H]
        w_k = k_offsets[:, None]            # [BLOCK_K, 1]
        w_h = h_offsets[None, :]            # [1, BLOCK_H]
        w_tile = tl.load(
            w_ptr + w_k * stride_w_h + w_h * stride_w_k,
            mask=mask_k[:, None] & mask_h[None, :],
            other=0.0,
        )

        # Accumulate: out_vec[h] += sum_k in_vals[k] * w_tile[k, h]
        out_vec += tl.sum(w_tile * in_vals[:, None], axis=0)

    # Store result into out[n, s, :]
    # For pid_s >= total_seq, we do not write; for pid_s < total_seq, we write.
    store_mask = (pid_s < total_seq) & mask_h
    out_ptr_row = out_ptr + pid_n * stride_out_n + pid_s * stride_out_s
    tl.store(out_ptr_row + h_offsets * stride_out_h, out_vec, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized implementation:
        - Concatenates encoder_hidden_states and hidden_states along sequence dim.
        - Applies process_weight.T linear projection in Triton.
        - Splits back into encoder and image outputs.

        Returns:
          (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        # Ensure dtype and contiguity for Triton
        B, T, H_e = encoder_hidden_states.shape
        B_h, I, H_h = hidden_states.shape
        assert B == B_h, "Batch size mismatch between encoder_hidden_states and hidden_states"
        assert H_e == H_h, "Hidden dim mismatch"
        H = H_e

        # Enforce float32 and contiguity
        encoder = encoder_hidden_states.contiguous().to(torch.float32)
        hidden = hidden_states.contiguous().to(torch.float32)
        weight = process_weight.contiguous().to(torch.float32)  # [H, H]

        total_seq = T + I
        # Allocate output concatenated tensor; we'll only write s < total_seq
        processed_concat = torch.empty((B, total_seq, H), dtype=torch.float32, device=encoder.device)

        # Compute grid: (B, total_seq, ceil_div(H, BLOCK_H))
        BLOCK_H = 128
        BLOCK_K = 64
        tiles_h = (H + BLOCK_H - 1) // BLOCK_H
        grid = (B, total_seq, tiles_h)

        # Launch Triton kernel
        concat_linear_split_kernel[grid](
            encoder, hidden, weight, processed_concat,
            B, T, I, H,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            weight.stride(0), weight.stride(1),
            processed_concat.stride(0), processed_concat.stride(1), processed_concat.stride(2),
            total_seq,
            tiles_h,
            BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        # Split outputs
        processed_encoder = processed_concat[:, :T, :]
        processed_hidden = processed_concat[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
