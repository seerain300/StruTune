import torch
import triton
import triton.language as tl

@triton.jit
def concat_linear_per_token_kernel(
    image_ptr,           # *f16/f32 [B, I, H]
    encoder_ptr,         # *f16/f32 [B, T, H]
    weight_ptr,          # *f16/f32 [H, H]
    out_ptr,             # *f16/f32 [B, T+I, H]
    B: tl.int32,
    I: tl.int32,
    T: tl.int32,
    H: tl.int32,
    # strides (elements)
    image_stride_b, image_stride_i, image_stride_h,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    weight_stride_w, weight_stride_k,   # weight is [H, H]
    out_stride_b, out_stride_t, out_stride_h,
    # meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid is (B, T+I): one program per (batch b, output position t)
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    # Determine whether this output position corresponds to encoder (t < T) or image (t >= T)
    is_encoder = pid_t < T

    # Loop over hidden dimension H in tiles
    for h_off in range(0, H, BLOCK_H):
        h_offsets = h_off + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        # Accumulator for this H tile (float32)
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # Loop over K (input feature dimension) in tiles
        for k_off in range(0, H, BLOCK_K):
            k_offsets = k_off + tl.arange(0, BLOCK_K)
            mask_k = k_offsets < H

            # Select source tensor based on is_encoder
            # Compute base offsets for the selected input vector
            if is_encoder:
                base_in = encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t
            else:
                base_in = image_ptr + pid_b * image_stride_b + (pid_t - T) * image_stride_i

            # Load input vector chunk as 1D
            # Pointer for x_chunk: base_in + k_offsets * input_stride_h
            # Since we don't have a separate input stride, we rely on the fact that
            # both image and encoder are laid out as [B, dim, H] with stride_h == 1 for contiguous H,
            # but to be robust, use the existing strides assuming contiguous H along the last dim:
            # We pass image_stride_h and encoder_stride_h; for contiguous tensors these are 1,
            # but we can still use them generically by treating the last dim as stride_h.
            x_chunk = tl.load(base_in + k_offsets * 1, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_K]

            # Load weight block [BLOCK_H, BLOCK_K]
            # Pointer: weight_ptr + h_offsets[:, None]*weight_stride_w + k_offsets[None, :]*weight_stride_k
            w_block = tl.load(
                weight_ptr + h_offsets[:, None] * weight_stride_w + k_offsets[None, :] * weight_stride_k,
                mask=(mask_h[:, None] & mask_k[None, :]),
                other=0.0
            ).to(tl.float32)  # [BLOCK_H, BLOCK_K]

            # Accumulate: acc += sum(w_block * x_chunk[None, :], axis=1)
            # Ensure x_chunk is 2D to broadcast correctly
            acc += tl.sum(w_block * x_chunk[None, :], axis=1)

        # Store the accumulated output for this H tile to out[pid_b, pid_t, :]
        out_ptr_t = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t
        tl.store(out_ptr_t + h_offsets * out_stride_h, acc, mask=mask_h)


def triton_concat_linear(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    Compute processed = torch.cat([encoder_hidden_states, hidden_states], dim=1) @ process_weight.T
    without materializing the concatenation, using Triton. Returns tensor of shape [B, T+I, H].
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton."
    assert hidden_states.shape[0] == encoder_hidden_states.shape[0] == process_weight.shape[1], "Shape mismatch."
    assert hidden_states.shape[2] == encoder_hidden_states.shape[2] == process_weight.shape[0], "Hidden dim mismatch."

    B = hidden_states.shape[0]
    I = hidden_states.shape[1]
    T = encoder_hidden_states.shape[1]
    H = hidden_states.shape[2]

    # Output tensor [B, T+I, H]
    total = torch.empty((B, T + I, H), device=hidden_states.device, dtype=torch.float32)

    # Ensure inputs are contiguous in memory (strides are used anyway, but contiguous improves performance)
    image = hidden_states.contiguous()
    encoder = encoder_hidden_states.contiguous()
    weight = process_weight.contiguous()

    # Strides (in elements)
    image_stride_b, image_stride_i, image_stride_h = image.stride()
    encoder_stride_b, encoder_stride_t, encoder_stride_h = encoder.stride()
    weight_stride_w, weight_stride_k = weight.stride()
    out_stride_b, out_stride_t, out_stride_h = total.stride()

    # Tile sizes; moderate defaults to avoid register pressure. Tune if needed.
    BLOCK_H = 64
    BLOCK_K = 64

    grid = (B, T + I)
    concat_linear_per_token_kernel[grid](
        image, encoder, weight, total,
        B, I, T, H,
        image_stride_b, image_stride_i, image_stride_h,
        encoder_stride_b, encoder_stride_t, encoder_stride_h,
        weight_stride_w, weight_stride_k,
        out_stride_b, out_stride_t, out_stride_h,
        BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    return total


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_hidden_states).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
        total = triton_concat_linear(hidden_states, encoder_hidden_states, process_weight)  # [B, T+I, H]
        T = encoder_hidden_states.shape[1]
        processed_encoder = total[:, :T, :]
        processed_hidden = total[:, T:, :]
        return processed_encoder, processed_hidden