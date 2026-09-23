import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_per_token_1d_kernel(
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
    # 1D grid: total programs = B * (T + I)
    pid = tl.program_id(0)
    total_out = T + I
    b = pid // total_out
    t = pid % total_out
    is_encoder = t < T

    # Loop over hidden dimension in tiles
    for h_off in range(0, H, BLOCK_H):
        h_offsets = h_off + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        # Accumulator for this H tile (float32)
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # Loop over input feature dimension K in tiles
        for k_off in range(0, H, BLOCK_K):
            k_offsets = k_off + tl.arange(0, BLOCK_K)
            mask_k = k_offsets < H

            # Select input vector source
            if is_encoder:
                base = encoder_ptr + b * encoder_stride_b + t * encoder_stride_t
            else:
                base = image_ptr + b * image_stride_b + (t - T) * image_stride_i

            # Load input chunk x_chunk (masked)
            x_chunk = tl.load(base + k_offsets * image_stride_h, mask=mask_k, other=0.0)
            x_chunk = x_chunk.to(tl.float32)

            # Load weight block w_block of shape [BLOCK_H, BLOCK_K]
            w_base = weight_ptr + h_offsets[:, None] * weight_stride_w + k_offsets[None, :] * weight_stride_k
            w_block = tl.load(w_base, mask=mask_h[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

            # Accumulate: acc += sum(w_block * x_chunk, axis=1)
            # w_block: [BLOCK_H, BLOCK_K], x_chunk: [BLOCK_K]
            partial = tl.sum(w_block * x_chunk[None, :], axis=1)
            acc += partial

        # Store acc to output
        out_base = out_ptr + b * out_stride_b + t * out_stride_t
        tl.store(out_base + h_offsets * out_stride_h, acc, mask=mask_h)


def triton_concat_linear(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of the linear projection over the concatenated sequence without materializing it.
    Returns processed tensor of shape [B, T+I, H].
    """
    B = hidden_states.shape[0]
    I = hidden_states.shape[1]
    T = encoder_hidden_states.shape[1]
    H = hidden_states.shape[2]

    # Ensure contiguity for simple stride handling
    image = hidden_states.contiguous()
    encoder = encoder_hidden_states.contiguous()
    weight = process_weight.contiguous()

    # Output tensor in float32 for numerical stability
    total = torch.empty((B, T + I, H), device=hidden_states.device, dtype=torch.float32)

    # Strides (in elements)
    image_stride_b, image_stride_i, image_stride_h = image.stride()
    encoder_stride_b, encoder_stride_t, encoder_stride_h = encoder.stride()
    weight_stride_w, weight_stride_k = weight.stride()
    out_stride_b, out_stride_t, out_stride_h = total.stride()

    # Tile sizes. Start conservatively to avoid register pressure.
    BLOCK_H = 64
    BLOCK_K = 64

    grid = (B * (T + I),)
    concat_linear_per_token_1d_kernel[grid](
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


def run(*args):
    return ModelNew()(*args)
