import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_kernel_tiled(
    image_ptr,           # *f16/f32 [B, I, H]
    encoder_ptr,         # *f16/f32 [B, T, H]
    weight_ptr,          # *f16/f32 [H, H]
    out_ptr,             # *f16/f32 [B, T+I, H]  (we will store float32 for stability)
    B: tl.int32,
    I: tl.int32,
    T: tl.int32,
    H: tl.int32,
    # strides (in elements)
    image_stride_b, image_stride_i, image_stride_h,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    weight_stride_w, weight_stride_k,   # weight is [H, W=H]
    out_stride_b, out_stride_t, out_stride_h,
    # meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)  # batch
    pid_t = tl.program_id(1)  # output position in [0, T+I)
    pid_th = tl.program_id(2)  # tile along hidden dimension

    # Compute hidden offsets for this tile
    h_off = pid_th * BLOCK_H
    offs_h = h_off + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H

    # Decide source: if t < T, use encoder, else use image
    use_encoder = pid_t < T

    # Initialize accumulator for this (b, t, tile_h)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Loop over hidden dimension in chunks of BLOCK_K to load input vector and weight block
    for k_off in range(0, H, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H

        if use_encoder:
            # Load input vector from encoder[b, t, k_off:k_off+BLOCK_K]
            enc_ptrs = encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t + offs_k * encoder_stride_h
            x_chunk = tl.load(enc_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]
        else:
            # Load input vector from image[b, t - T, k_off:k_off+BLOCK_K]
            img_i = pid_t - T
            # Bounds are guaranteed by grid (pid_t in [T, T+I)), so no extra mask needed
            img_ptrs = image_ptr + pid_b * image_stride_b + img_i * image_stride_i + offs_k * image_stride_h
            x_chunk = tl.load(img_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load weight block W[n:n+BLOCK_H, k:k+BLOCK_K] -> shape [BLOCK_H, BLOCK_K]
        w_ptrs = weight_ptr + offs_h[:, None] * weight_stride_w + offs_k[None, :] * weight_stride_k
        w_block = tl.load(w_ptrs, mask=mask_h[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_H, BLOCK_K]
        w_block = w_block.to(tl.float32)
        x_chunk = x_chunk.to(tl.float32)

        # Accumulate: acc[n] += sum_k w_block[n, k] * x_chunk[k]
        # Implement as matrix-vector product: acc += w_block @ x_chunk
        # Sum across K axis
        acc += tl.sum(w_block * x_chunk[None, :], axis=1)

    # Store the accumulated output slice to out[b, t, h_off:h_off+BLOCK_H]
    out_ptrs = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + offs_h * out_stride_h
    tl.store(out_ptrs, acc, mask=mask_h)


def triton_concat_linear(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    Triton kernel that computes:
      total[b, t, :] = (if t < T: encoder_hidden_states[b, t, :] else hidden_states[b, t - T, :]) @ process_weight.T
    Returns total of shape [B, T+I, H].
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Triton requires CUDA tensors"
    B = hidden_states.shape[0]
    I = hidden_states.shape[1]
    T = encoder_hidden_states.shape[1]
    H = hidden_states.shape[2]
    assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"

    # Make sure inputs are contiguous (Triton expects consistent strides)
    image = hidden_states.contiguous()
    encoder = encoder_hidden_states.contiguous()
    weight = process_weight.contiguous()

    # Output as float32 for numeric stability (can cast later if needed)
    total = torch.empty((B, T + I, H), device=image.device, dtype=torch.float32)

    # Compute strides (in elements)
    image_stride_b, image_stride_i, image_stride_h = image.stride()
    encoder_stride_b, encoder_stride_t, encoder_stride_h = encoder.stride()
    weight_stride_w, weight_stride_k = weight.stride()  # weight is [H, H]
    out_stride_b, out_stride_t, out_stride_h = total.stride()

    # Choose tiling parameters (tune as needed)
    BLOCK_H = 64
    BLOCK_K = 64

    # Grid: (B, T+I, tiles along H)
    tiles_h = (H + BLOCK_H - 1) // BLOCK_H
    grid = (B, T + I, tiles_h)

    # Launch Triton kernel
    # num_warps can be tuned; 4 or 8 are typical
    concat_linear_kernel_tiled[grid](
        image, encoder, weight, total,
        B, I, T, H,
        image_stride_b, image_stride_i, image_stride_h,
        encoder_stride_b, encoder_stride_t, encoder_stride_h,
        weight_stride_w, weight_stride_k,
        out_stride_b, out_stride_t, out_stride_h,
        BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    # If original inputs were not float32, consider casting output back. Here we keep float32.
    # To match original behavior exactly, you can cast to original dtype if needed.
    return total


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_image_hidden_states).
        """
        # Ensure tensors are on CUDA for Triton
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
        total = triton_concat_linear(hidden_states, encoder_hidden_states, process_weight)  # [B, T+I, H]
        T = encoder_hidden_states.shape[1]
        processed_encoder = total[:, :T, :]
        processed_hidden = total[:, T:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
