import torch
import triton
import triton.language as tl

# Kernel: compute the full processed tensor of shape [B, T+I, H]
# We do not materialize the concatenation; instead, for each output position t,
# we decide whether to read from encoder[b, t, :] or from image[b, t - T, :].
@triton.jit
def concat_linear_kernel(
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
):
    pid_b = tl.program_id(0)  # batch dimension
    pid_t = tl.program_id(1)  # output sequence position in [0, T+I)

    # We'll process one output position t per program. Grid is (B, T+I).
    # Compute output vector for this t over H
    # Initialize output vector for this t (float32 accumulation)
    output_vec = tl.zeros((H,), dtype=tl.float32)

    # Loop over hidden dimension H in tiles
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # Accumulator for this tile
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # Loop over hidden dimension (weight rows) to perform matrix-vector multiply
        for n in range(0, H):
            # Load weight row w[n, offs_h] as a vector
            w_row_ptrs = weight_ptr + n * weight_stride_w + offs_h * weight_stride_k
            w_row = tl.load(w_row_ptrs, mask=mask_h, other=0.0).to(tl.float32)

            # Determine source: if t < T, use encoder[b, t, :], else use image[b, t - T, :]
            use_encoder = t < T
            if use_encoder:
                x_ptrs = encoder_ptr \
                         + pid_b * encoder_stride_b \
                         + t * encoder_stride_t \
                         + tl.arange(0, H) * encoder_stride_h
                x_vec = tl.load(x_ptrs, mask=mask_h, other=0.0).to(tl.float32)
            else:
                i_pos = t - T
                x_ptrs = image_ptr \
                         + pid_b * image_stride_b \
                         + i_pos * image_stride_i \
                         + tl.arange(0, H) * image_stride_h
                x_vec = tl.load(x_ptrs, mask=mask_h, other=0.0).to(tl.float32)

            # acc += w_row * x_vec (vector dot product for this n)
            acc += tl.sum(w_row[:, None] * x_vec[None, :], axis=0)

        # Add this tile's contribution to the full output vector
        output_vec[offs_h] = acc

    # Store the resulting output vector to out[b, t, :]
    out_row_ptrs = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + tl.arange(0, H) * out_stride_h
    tl.store(out_row_ptrs, output_vec, mask=True)

def triton_concat_linear(
    image: torch.Tensor,      # [B, I, H]
    encoder: torch.Tensor,    # [B, T, H]
    weight: torch.Tensor,     # [H, H]
) -> torch.Tensor:           # [B, T+I, H]
    """
    Compute out = cat([encoder, image], dim=1) @ weight.T using Triton.
    Returns out of shape [B, T+I, H].
    """
    assert image.is_cuda and encoder.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton."
    assert image.shape[0] == encoder.shape[0] == weight.shape[1] == weight.shape[0], "Shape mismatch."
    B, I, H = image.shape
    T = encoder.shape[1]
    total_out = T + I

    # Ensure weight is contiguous [H, H]
    weight_c = weight.contiguous()
    # Output tensor
    out = torch.empty((B, total_out, H), device=image.device, dtype=torch.float32)

    # Grid: one program per (batch, output position)
    grid = (B, total_out)

    # Choose tile size for H. Use 64 as a default; Triton will handle masks for H not divisible by 64.
    BLOCK_H = 64

    # Strides in elements
    image_stride_b, image_stride_i, image_stride_h = image.stride()
    encoder_stride_b, encoder_stride_t, encoder_stride_h = encoder.stride()
    weight_stride_w, weight_stride_k = weight_c.stride()
    out_stride_b, out_stride_t, out_stride_h = out.stride()

    concat_linear_kernel[grid](
        image, encoder, weight_c, out,
        B, I, T, H,
        image_stride_b, image_stride_i, image_stride_h,
        encoder_stride_b, encoder_stride_t, encoder_stride_h,
        weight_stride_w, weight_stride_k,
        out_stride_b, out_stride_t, out_stride_h,
        BLOCK_H=BLOCK_H,
        num_warps=4,
        num_stages=2,
    )

    # Cast back to original dtype if needed (PyTorch code originally computes in input dtype)
    # The original uses torch.matmul with default dtype; we keep output as float32 for stability,
    # but if you want to match dtype, uncomment the following:
    # if out.dtype != image.dtype:
    #     out = out.to(image.dtype)
    return out

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_image_hidden_states).
        """
        # Ensure tensors are on CUDA for Triton; if not, fall back (but here we require Triton-only path).
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
        total = triton_concat_linear(hidden_states, encoder_hidden_states, process_weight)  # [B, T+I, H]
        T = encoder_hidden_states.shape[1]
        processed_encoder = total[:, :T, :]
        processed_hidden = total[:, T:, :]
        return processed_encoder, processed_hidden