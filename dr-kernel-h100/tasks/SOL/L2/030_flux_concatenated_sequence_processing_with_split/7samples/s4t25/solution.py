import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_kernel(
    image_ptr,           # *f16/f32 [B, I, H]
    encoder_ptr,         # *f16/f32 [B, T, H]
    weight_ptr,          # *f16/f32 [H, H]
    out_ptr,             # *f16/f32 [B, T+I, H] (float32 accumulation and write)
    B: tl.int32,
    I: tl.int32,
    T: tl.int32,
    H: tl.int32,
    # strides (in elements)
    image_stride_b, image_stride_i, image_stride_h,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    weight_stride_w, weight_stride_k,
    out_stride_b, out_stride_t, out_stride_h,
    BLOCK_H: tl.constexpr,  # tile size along hidden dimension
):
    # Grid is (B, T+I). Each program computes one output vector at position t for batch b.
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    # Output vector accumulator in float32
    output_vec = tl.zeros((H,), dtype=tl.float32)

    # Tile over hidden dimension H
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # For each tile of H, accumulate over K using weight blocks and input vectors
        for k_off in range(0, H, 64):  # We use K-blocks of size 64; H is usually <= 4096
            offs_k = k_off + tl.arange(0, 64)
            mask_k = offs_k < H

            # Decide source: if t < T, use encoder[b, t, :]; else use image[b, t - T, :]
            if pid_t < T:
                # Load input vector chunk from encoder
                # Pointer: encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t + offs_k * encoder_stride_h
                src_ptr = encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t + offs_k * encoder_stride_h
                x_chunk = tl.load(src_ptr, mask=mask_k, other=0.0)
                # Convert to float32 for accumulation
                x_chunk = x_chunk.to(tl.float32)
            else:
                # Load input vector chunk from image
                src_i = pid_t - T
                src_ptr = image_ptr + pid_b * image_stride_b + src_i * image_stride_i + offs_k * image_stride_h
                x_chunk = tl.load(src_ptr, mask=mask_k, other=0.0)
                x_chunk = x_chunk.to(tl.float32)

            # Load weight block [BLOCK_H, 64] for current H tile and K chunk
            # Weights are [H, H] with strides (weight_stride_w, weight_stride_k)
            w_ptrs = weight_ptr + offs_h[:, None] * weight_stride_w + offs_k[None, :] * weight_stride_k
            mask_w = (mask_h[:, None] & mask_k[None, :])
            w_block = tl.load(w_ptrs, mask=mask_w, other=0.0)

            # Accumulate: output_vec[offs_h] += sum_k w_block[:, k] * x_chunk[k]
            # Convert w_block to float32 for accumulation
            w_block = w_block.to(tl.float32)
            partial = tl.sum(w_block * x_chunk[None, :], axis=1)  # shape [BLOCK_H]
            output_vec[offs_h] += partial

    # Store the computed output vector to out[b, t, :]
    out_ptrs = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + offs_h * out_stride_h
    tl.store(out_ptrs, output_vec, mask=mask_h)


def triton_concat_linear(image: torch.Tensor, encoder: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Compute processed = cat([encoder, image], dim=1) @ weight.T without materializing the concatenation.
    Returns tensor of shape [B, T+I, H].
    """
    assert image.is_cuda and encoder.is_cuda and weight.is_cuda, "Triton requires CUDA tensors."
    B, I, H = image.shape
    T = encoder.shape[1]
    total = torch.empty((B, T + I, H), device=image.device, dtype=torch.float32)

    # Ensure tensors are contiguous
    image_c = image.contiguous()
    encoder_c = encoder.contiguous()
    weight_c = weight.contiguous()
    total_c = total  # float32 output

    # Strides (in elements)
    image_stride_b, image_stride_i, image_stride_h = image_c.stride()
    encoder_stride_b, encoder_stride_t, encoder_stride_h = encoder_c.stride()
    weight_stride_w, weight_stride_k = weight_c.stride()  # weight is [H, H]
    out_stride_b, out_stride_t, out_stride_h = total_c.stride()

    # Launch grid: one program per (batch, token)
    grid = (B, T + I)

    # Choose meta-parameters conservatively
    BLOCK_H = 64

    concat_linear_kernel[grid](
        image_c, encoder_c, weight_c, total_c,
        B, I, T, H,
        image_stride_b, image_stride_i, image_stride_h,
        encoder_stride_b, encoder_stride_t, encoder_stride_h,
        weight_stride_w, weight_stride_k,
        out_stride_b, out_stride_t, out_stride_h,
        BLOCK_H=BLOCK_H,
        num_warps=4,
        num_stages=2,
    )
    return total_c


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_hidden).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
        total = triton_concat_linear(hidden_states, encoder_hidden_states, process_weight)  # [B, T+I, H]
        T = encoder_hidden_states.shape[1]
        processed_encoder = total[:, :T, :]
        processed_hidden = total[:, T:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
