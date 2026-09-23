import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_per_token_kernel_1d(
    image_ptr,           # *f16/f32 [B, I, H]
    encoder_ptr,         # *f16/f32 [B, T, H]
    weight_ptr,          # *f16/f32 [H, H]
    out_ptr,             # *f16/f32 [B, T+I, H]
    B: tl.int32,
    I: tl.int32,
    T: tl.int32,
    H: tl.int32,
    # strides (in elements)
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
    # Map program id to (b, t)
    b = pid // (T + I)
    t = pid % (T + I)

    # Determine whether this output position corresponds to encoder (t < T) or image (t >= T)
    is_encoder = t < T

    # Base pointer for the selected input tensor
    in_ptr = encoder_ptr + b * encoder_stride_b
    if not is_encoder:
        in_ptr = image_ptr + b * image_stride_b

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

            # Load input vector chunk (1D), elementwise
            x_chunk = tl.load(in_ptr + t * (1 if is_encoder else -T) * (encoder_stride_t if is_encoder else image_stride_i) + k_offsets * (encoder_stride_h if is_encoder else image_stride_h),
                              mask=mask_k, other=0.0)

            # Load weight block: [BLOCK_H, BLOCK_K]
            w_block = tl.load(
                weight_ptr + h_offsets[:, None] * weight_stride_w + k_offsets[None, :] * weight_stride_k,
                mask=mask_h[:, None] & mask_k[None, :],
                other=0.0,
            )

            # Accumulate: acc += sum(w_block * x_chunk, axis=1)
            # Triton supports elementwise multiply and reduction via tl.sum
            acc += tl.sum(w_block * x_chunk[None, :], axis=1)

        # Store the accumulated vector into out[b, t, :]
        out_row_ptr = out_ptr + b * out_stride_b + t * out_stride_t
        tl.store(out_row_ptr + h_offsets * out_stride_h, acc, mask=mask_h)


def triton_concat_linear(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of:
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, H]
        processed = torch.matmul(concatenated, process_weight.T)               # [B, T+I, H]
    Returns processed [B, T+I, H], computed without materializing the concatenation.
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
    B = hidden_states.shape[0]
    I = hidden_states.shape[1]
    T = encoder_hidden_states.shape[1]
    H = hidden_states.shape[2]
    assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == H
    assert process_weight.shape[0] == H and process_weight.shape[1] == H

    # Allocate output tensor [B, T+I, H]
    total = torch.empty((B, T + I, H), device=hidden_states.device, dtype=process_weight.dtype)

    # Ensure inputs are contiguous for predictable strides
    image = hidden_states.contiguous()
    encoder = encoder_hidden_states.contiguous()
    weight = process_weight.contiguous()

    # Strides (in elements)
    image_stride_b, image_stride_i, image_stride_h = image.stride()
    encoder_stride_b, encoder_stride_t, encoder_stride_h = encoder.stride()
    weight_stride_w, weight_stride_k = weight.stride()
    out_stride_b, out_stride_t, out_stride_h = total.stride()

    # Tile sizes; moderate defaults to avoid register pressure. Tune as needed.
    BLOCK_H = 64
    BLOCK_K = 64

    # 1D grid: B * (T + I)
    grid = (B * (T + I),)
    concat_linear_per_token_kernel_1d[grid](
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
