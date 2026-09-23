import torch
import triton
import triton.language as tl

# Triton kernel: compute processed = cat([encoder, image], dim=1) @ process_weight.T
# without materializing the concatenation. Each program handles one (batch b, output position t),
# and iterates over hidden dimension H in tiles.
@triton.jit
def concat_linear_simple_kernel(
    image_ptr,           # *f32 [B, I, H]
    encoder_ptr,         # *f32 [B, T, H]
    weight_ptr,          # *f32 [H, H]
    out_ptr,             # *f32 [B, T+I, H]
    B: tl.int32,
    I: tl.int32,
    T: tl.int32,
    H: tl.int32,
    # strides (elements)
    image_stride_b, image_stride_i, image_stride_h,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    weight_stride_w, weight_stride_k,
    out_stride_b, out_stride_t, out_stride_h,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch index
    pid_t = tl.program_id(1)  # output sequence position [0, T+I)

    # Initialize output vector for this position (float32 accumulation)
    output_vec = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Loop over hidden dimension in tiles
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # Accumulator for current H tile
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # Loop over input hidden dimension in chunks (K dimension)
        # We iterate k from 0 to H (weight is HxH) in tiles, loading input vectors and weight blocks.
        for k_off in range(0, H, BLOCK_H):  # k_off is the chunk offset along input hidden dim
            x_chunk = tl.zeros((BLOCK_H,), dtype=tl.float32)

            # Decide source tensor: if t < T, read from encoder, else from image (shift by T)
            if pid_t < T:
                src_b = pid_b
                src_t = pid_t
                # Load x_chunk = encoder[src_b, src_t, k_off : k_off + BLOCK_H]
                src_k = k_off + tl.arange(0, BLOCK_H)
                mask_k = src_k < H
                enc_ptrs = encoder_ptr + src_b * encoder_stride_b + src_t * encoder_stride_t + src_k * encoder_stride_h
                x_chunk = tl.load(enc_ptrs, mask=mask_k, other=0.0)
            else:
                src_b = pid_b
                src_t_local = pid_t - T  # since t >= T, src_t in [0, I)
                # Load x_chunk = image[src_b, src_t_local, k_off : k_off + BLOCK_H]
                src_k = k_off + tl.arange(0, BLOCK_H)
                mask_k = src_k < H
                img_ptrs = image_ptr + src_b * image_stride_b + src_t_local * image_stride_i + src_k * image_stride_h
                x_chunk = tl.load(img_ptrs, mask=mask_k, other=0.0)

            # Load weight block [BLOCK_H, BLOCK_H] along k-chunk
            w_ptrs = weight_ptr + (offs_h[:, None] * weight_stride_w + (k_off + tl.arange(0, BLOCK_H))[None, :] * weight_stride_k)
            mask_w = (offs_h[:, None] < H) & ((k_off + tl.arange(0, BLOCK_H))[None, :] < H)
            w_block = tl.load(w_ptrs, mask=mask_w, other=0.0)

            # Accumulate: acc += sum over K of w_block * x_chunk
            # Broadcast x_chunk to [1, BLOCK_H] and multiply with w_block [BLOCK_H, BLOCK_H]
            # Then reduce over K (last axis) into acc
            # Note: x_chunk is [BLOCK_H], w_block is [BLOCK_H, BLOCK_H]
            # We'll compute acc += tl.sum(w_block * x_chunk[None, :], axis=1)
            # But x_chunk must be broadcast across rows. Triton supports broadcasting via unsqueezing:
            acc += tl.sum(w_block * x_chunk[None, :], axis=1)

        # Add this tile's contribution to the output vector
        output_vec += acc

    # Store output_vec to out[pid_b, pid_t, :]
    out_ptrs = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + offs_h * out_stride_h
    store_mask = offs_h < H
    tl.store(out_ptrs, output_vec, mask=store_mask)


def triton_concat_linear(image: torch.Tensor, encoder: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of processing concatenated sequences:
    total = cat([encoder, image], dim=1) @ weight.T
    Returns total of shape [B, T+I, H].
    Falls back to PyTorch if Triton cannot run safely.
    """
    # Ensure we are on CUDA and dtypes are supported
    use_triton = (
        image.is_cuda
        and encoder.is_cuda
        and weight.is_cuda
        and image.dtype == torch.float32
        and encoder.dtype == torch.float32
        and weight.dtype == torch.float32
    )

    # If not supported, fall back to PyTorch
    if not use_triton:
        # Fallback: materialize concatenation and use matmul
        concat = torch.cat([encoder, image], dim=1)  # [B, T+I, H]
        return torch.matmul(concat, weight.t())      # [B, T+I, H]

    B, I, H = image.shape
    T = encoder.shape[1]
    total = torch.empty((B, T + I, H), device=image.device, dtype=torch.float32)

    # Compute grid: one program per (batch, output position)
    grid = (B, T + I)

    # Choose a safe tile size for hidden dim. Keep moderate to avoid excessive register use.
    BLOCK_H = 128

    # Launch kernel
    concat_linear_simple_kernel[grid](
        image,
        encoder,
        weight,
        total,
        B, I, T, H,
        image.stride(0), image.stride(1), image.stride(2),
        encoder.stride(0), encoder.stride(1), encoder.stride(2),
        weight.stride(0), weight.stride(1),
        total.stride(0), total.stride(1), total.stride(2),
        BLOCK_H=BLOCK_H,
        num_warps=4,
        num_stages=2,
    )
    return total


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_hidden_states).
        Falls back to PyTorch if Triton execution is not safe.
        """
        # If inputs are not CUDA or dtypes unsupported, use PyTorch path
        if not (hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda):
            concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            processed = torch.matmul(concatenated, process_weight.t())
            return processed[:, :encoder_hidden_states.shape[1], :], processed[:, encoder_hidden_states.shape[1]:, :]
        total = triton_concat_linear(hidden_states, encoder_hidden_states, process_weight)  # [B, T+I, H]
        T = encoder_hidden_states.shape[1]
        processed_encoder = total[:, :T, :]
        processed_hidden = total[:, T:, :]
        return processed_encoder, processed_hidden