import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_single_token_kernel(
    image_ptr,           # *f32 [B, I, H]
    encoder_ptr,         # *f32 [B, T, H]
    weight_ptr,          # *f32 [H, H]
    out_ptr,             # *f32 [B, T+I, H]
    # shapes
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
    BLOCK_K: tl.constexpr,  # typically set to H
    BLOCK_H: tl.constexpr,  # typically set to H
):
    # Each program handles one (b, t)
    pid_b = tl.program_id(0)  # batch id
    pid_t = tl.program_id(1)  # output token id in [0, T+I)

    # Compute base pointers for input vector depending on whether t < T
    # If t < T: input is from encoder[b, t, :]
    # Else: input is from image[b, t - T, :]
    is_encoder = pid_t < T

    # Prepare output vector
    offs_h = tl.arange(0, BLOCK_H)  # since BLOCK_H == H, this covers full hidden dim
    out_vec = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Iterate over hidden dimension in chunks of BLOCK_K (set to H, so single iteration)
    for k_off in range(0, H, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H

        # Load input vector chunk (1D) from the chosen source
        if is_encoder:
            # ptr to encoder[b, pid_t, offs_k]
            enc_ptr = encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t + offs_k * encoder_stride_h
            x_chunk = tl.load(enc_ptr, mask=mask_k, other=0.0)  # [BLOCK_K]
        else:
            # ptr to image[b, pid_t - T, offs_k]
            img_idx = pid_t - T
            img_ptr = image_ptr + pid_b * image_stride_b + img_idx * image_stride_i + offs_k * image_stride_h
            x_chunk = tl.load(img_ptr, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load weight tile [BLOCK_K, BLOCK_H]
        w_ptr = weight_ptr + offs_k[:, None] * weight_stride_w + offs_h[None, :] * weight_stride_k  # [BLOCK_K, BLOCK_H]
        w_tile = tl.load(w_ptr, mask=mask_k[:, None], other=0.0)  # [BLOCK_K, BLOCK_H]

        # Accumulate: out_vec += sum over k of w_tile[k, :] * x_chunk[k]
        # Broadcast multiply and reduce over axis=0
        prod = w_tile * x_chunk[:, None]  # [BLOCK_K, BLOCK_H]
        out_vec += tl.sum(prod, axis=0)

    # Store output vector to out[b, pid_t, :]
    out_ptr_vec = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + offs_h * out_stride_h
    # Since BLOCK_H == H, we can store with mask for safety
    tl.store(out_ptr_vec, out_vec, mask=offs_h < H)


def triton_concat_linear(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    Compute the full processed tensor of shape [B, T+I, H] using Triton:
    processed = cat([encoder_hidden_states, hidden_states], dim=1) @ process_weight.T
    without materializing the concatenation. Returns float32 for robustness.
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
    # Ensure contiguous tensors
    image = hidden_states.contiguous()
    encoder = encoder_hidden_states.contiguous()
    weight = process_weight.contiguous()

    B, I, H = image.shape
    T = encoder.shape[1]

    # Output as float32 for numerical robustness
    out = torch.empty((B, T + I, H), device=image.device, dtype=torch.float32)

    # Strides in elements
    image_stride_b, image_stride_i, image_stride_h = image.stride()
    encoder_stride_b, encoder_stride_t, encoder_stride_h = encoder.stride()
    weight_stride_w, weight_stride_k = weight.stride()
    out_stride_b, out_stride_t, out_stride_h = out.stride()

    # Launch one program per (batch, token)
    grid = (B, T + I)

    # Set BLOCK sizes; for robustness we set BLOCK_H == H and BLOCK_K == H
    # Choose meta-parameters conservatively; H in provided workloads is modest (<=4096).
    # If you need to support very large H, consider tiling H with a smaller BLOCK_H and loop.
    BLOCK_H = H
    # For BLOCK_K, set to H as well to avoid multiple iterations; Triton will specialize per H
    BLOCK_K = H

    # num_warps and num_stages can be tuned; start with moderate values
    concat_linear_single_token_kernel[grid](
        image, encoder, weight, out,
        B, I, T, H,
        image_stride_b, image_stride_i, image_stride_h,
        encoder_stride_b, encoder_stride_t, encoder_stride_h,
        weight_stride_w, weight_stride_k,
        out_stride_b, out_stride_t, out_stride_h,
        BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
        num_warps=4, num_stages=2,
    )

    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_image_hidden_states).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
        total = triton_concat_linear(hidden_states, encoder_hidden_states, process_weight)  # [B, T+I, H]
        T = encoder_hidden_states.shape[1]
        processed_encoder = total[:, :T, :]
        processed_hidden = total[:, T:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
