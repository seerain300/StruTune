import torch
import triton
import triton.language as tl

# Triton kernel: compute processed = (cat([encoder, image], dim=1)) @ process_weight.T
# Strategy: one program per (b, t) where t in [0, T+I). For each output position,
# decide whether to read from encoder[b, t, :] or image[b, t - T, :].
@triton.jit
def concat_linear_token_kernel(
    image_ptr,          # *f16/f32 [B, I, H]
    encoder_ptr,        # *f16/f32 [B, T, H]
    weight_ptr,         # *f16/f32 [H, H]
    out_ptr,            # *f32 [B, T+I, H] (we'll cast to input dtype after)
    B: tl.int32,
    I: tl.int32,
    T: tl.int32,
    H: tl.int32,
    # strides (elements)
    image_stride_b, image_stride_i, image_stride_h,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    weight_stride_h, weight_stride_k,  # weight is [H, K] with K=H
    out_stride_b, out_stride_t, out_stride_h,
    # meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch id
    pid_t = tl.program_id(1)  # output sequence position

    # Determine source tensor: if pid_t < T, use encoder; else use image shifted by T
    use_encoder = pid_t < T

    # Accumulator for output vector (float32)
    output_vec = tl.zeros((H,), dtype=tl.float32)

    # Loop over hidden dimension in tiles of BLOCK_H
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # Initialize acc for this H-tile
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # Loop over K (hidden dim) in tiles of BLOCK_K to load weight blocks
        for k_off in range(0, H, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H

            # Load input vector x_chunk (either from encoder or image)
            if use_encoder:
                # Input is encoder[b, pid_t, :]
                x_ptr = encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t + offs_k * encoder_stride_h
                x_chunk = tl.load(x_ptr, mask=mask_k, other=0.0)  # [BLOCK_K]
            else:
                # Input is image[b, pid_t - T, :]
                t_img = pid_t - T
                x_ptr = image_ptr + pid_b * image_stride_b + t_img * image_stride_i + offs_k * image_stride_h
                x_chunk = tl.load(x_ptr, mask=mask_k, other=0.0)  # [BLOCK_K]

            # Load weight block W_chunk of shape [BLOCK_H, BLOCK_K]
            w_ptr = weight_ptr + offs_h[:, None] * weight_stride_h + offs_k[None, :] * weight_stride_k
            w_chunk = tl.load(w_ptr, mask=mask_h[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_H, BLOCK_K]
            # Accumulate: acc += sum_k w_chunk[:, k] * x_chunk[k]
            # Broadcast x_chunk to [1, BLOCK_K], multiply with w_chunk, reduce along axis=1
            acc += tl.sum(w_chunk * x_chunk[None, :], axis=1)

        # Add this tile's contribution to the output vector
        output_vec = output_vec + acc

    # Store result for (b, t) at all H positions
    out_ptr_vec = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + offs_h * out_stride_h
    # Note: out tensor is float32; we'll cast to original dtype in host if needed.
    tl.store(out_ptr_vec, output_vec, mask=mask_h)


def triton_concat_linear(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    Triton-optimized concat + linear. Returns processed tensor of shape [B, T+I, H].
    Assumes all tensors are on CUDA and contiguous.
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
    assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3 and process_weight.dim() == 2, "Invalid tensor dims."
    B = hidden_states.shape[0]
    I = hidden_states.shape[1]  # image seq len
    T = encoder_hidden_states.shape[1]  # text seq len
    H = hidden_states.shape[2]  # hidden dim (must equal process_weight.shape[1])
    assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]."

    # Ensure contiguous tensors
    image = hidden_states.contiguous()
    encoder = encoder_hidden_states.contiguous()
    weight = process_weight.contiguous()

    # Output tensor in float32 for numerical stability; we'll cast to image.dtype after kernel.
    out = torch.empty((B, T + I, H), device=image.device, dtype=torch.float32)

    # Strides (elements)
    image_stride_b, image_stride_i, image_stride_h = image.stride()
    encoder_stride_b, encoder_stride_t, encoder_stride_h = encoder.stride()
    weight_stride_h, weight_stride_k = weight.stride()  # K=H for linear
    out_stride_b, out_stride_t, out_stride_h = out.stride()

    # Launch grid: one program per (batch, output token)
    grid = (B, T + I)

    # Choose reasonable tile sizes; can be tuned
    BLOCK_H = 64
    BLOCK_K = 64
    num_warps = 4
    num_stages = 2

    concat_linear_token_kernel[grid](
        image, encoder, weight, out,
        B, I, T, H,
        image_stride_b, image_stride_i, image_stride_h,
        encoder_stride_b, encoder_stride_t, encoder_stride_h,
        weight_stride_h, weight_stride_k,
        out_stride_b, out_stride_t, out_stride_h,
        BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )

    # Cast to original dtype to match expected output behavior
    # Original PyTorch code returns same dtype as input tensors; typically float32.
    # If you need exact dtype match, adjust cast based on inputs.
    if out.dtype != hidden_states.dtype:
        out = out.to(hidden_states.dtype)

    return out

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