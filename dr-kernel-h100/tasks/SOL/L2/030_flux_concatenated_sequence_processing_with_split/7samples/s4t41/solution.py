import torch
import triton
import triton.language as tl


@triton.jit
def matvec_concat_kernel(
    image_ptr,            # *f32 [B, I, H]
    encoder_ptr,          # *f32 [B, T, H]
    weight_ptr,           # *f32 [H, H]
    out_ptr,              # *f32 [B, T+I, H]
    B: tl.int32, I: tl.int32, T: tl.int32, H: tl.int32,
    # strides (in elements)
    image_stride_b, image_stride_i, image_stride_h,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    weight_stride_w, weight_stride_k,
    out_stride_b, out_stride_t, out_stride_h,
    # meta-parameters
    BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch index
    pid_t = tl.program_id(1)  # output token index in [0, T+I)

    # We compute the output vector for this (b, t).
    # Decide source based on t < T:
    # If t < T: use encoder[b, t, :]
    # Else: use image[b, t - T, :]
    is_encoder = pid_t < T

    # Base pointers for input vector
    # For encoder: input_ptr = encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t
    # For image: input_ptr = image_ptr + pid_b * image_stride_b + (pid_t - T) * image_stride_i
    if is_encoder:
        input_ptr = encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t
    else:
        input_t = pid_t - T
        input_ptr = image_ptr + pid_b * image_stride_b + input_t * image_stride_i

    # Accumulator for output vector in float32
    out_vec = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Loop over hidden dimension H in tiles
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # We'll accumulate acc = sum over k of weight[offs_h, k] * x[input, k]
        # Loop over K in chunks for input vector loads
        # Note: x_k loads are 1D vectors of size BLOCK_K
        for k_off in range(0, H, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H

            # Load input vector chunk x_k: shape [BLOCK_K]
            x_k = tl.load(input_ptr + offs_k * encoder_stride_h, mask=mask_k, other=0.0).to(tl.float32)

            # Load weight block: weight[offs_h, offs_k] -> shape [BLOCK_H, BLOCK_K]
            w_ptrs = weight_ptr + offs_h[:, None] * weight_stride_w + offs_k[None, :] * weight_stride_k
            w_block = tl.load(w_ptrs, mask=mask_h[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

            # Accumulate: out_vec += sum(w_block * x_k[None, :], axis=1)
            out_vec += tl.sum(w_block * x_k[None, :], axis=1)

        # After processing all K tiles, store partial output
        # We must ensure we only store valid h positions
        # out[b, t, h] at linear index: pid_b * out_stride_b + pid_t * out_stride_t + h * out_stride_h
        out_ptrs = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + offs_h * out_stride_h
        tl.store(out_ptrs, out_vec, mask=mask_h)


def triton_concat_linear(image: torch.Tensor, encoder: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Compute processed = cat([encoder, image], dim=1) @ weight.T without materializing the concatenation.
    Returns tensor of shape [B, T+I, H], all computations done in Triton.
    """
    assert image.is_cuda and encoder.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton execution."
    assert image.dtype == torch.float32 and encoder.dtype == torch.float32 and weight.dtype == torch.float32, "Use float32 tensors for this Triton kernel."
    B, I, H = image.shape
    T = encoder.shape[1]
    total = torch.empty((B, T + I, H), device=image.device, dtype=image.dtype)

    # Ensure tensors are contiguous for predictable strides
    image = image.contiguous()
    encoder = encoder.contiguous()
    weight = weight.contiguous()
    total = total.contiguous()

    # Strides in elements
    image_stride_b, image_stride_i, image_stride_h = image.stride()
    encoder_stride_b, encoder_stride_t, encoder_stride_h = encoder.stride()
    weight_stride_w, weight_stride_k = weight.stride()  # weight is [H, H]
    out_stride_b, out_stride_t, out_stride_h = total.stride()

    # Launch Triton kernel: grid over (batch, output tokens)
    grid = (B, T + I)
    # Choose tile sizes; conservative defaults for robustness
    BLOCK_H = 128
    BLOCK_K = 128

    matvec_concat_kernel[grid](
        image, encoder, weight, total,
        B, I, T, H,
        image_stride_b, image_stride_i, image_stride_h,
        encoder_stride_b, encoder_stride_t, encoder_stride_h,
        weight_stride_w, weight_stride_k,
        out_stride_b, out_stride_t, out_stride_h,
        BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    return total


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_image_hidden_states).
        """
        # Triton path: all heavy computation in Triton kernels
        total = triton_concat_linear(hidden_states, encoder_hidden_states, process_weight)  # [B, T+I, H]
        T = encoder_hidden_states.shape[1]
        processed_encoder = total[:, :T, :]
        processed_hidden = total[:, T:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
