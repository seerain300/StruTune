import torch
import triton
import triton.language as tl

# Triton kernel: compute processed = cat([encoder, image], dim=1) @ process_weight.T
# Grid: (B, T+I). Each program computes one output position for a fixed (b, t).
@triton.jit
def concat_linear_kernel_simple(
    image_ptr,            # *f32 [B, I, H]
    encoder_ptr,          # *f32 [B, T, H]
    weight_ptr,           # *f32 [H, H]
    out_ptr,              # *f32 [B, T+I, H]
    B: tl.int32,
    I: tl.int32,
    T: tl.int32,
    H: tl.int32,
    # strides (in elements)
    image_stride_b, image_stride_i, image_stride_h,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    weight_stride_h, weight_stride_k,  # weight is [H, H]
    out_stride_b, out_stride_t, out_stride_h,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch id
    pid_t = tl.program_id(1)  # output position id in [0, T+I)

    # Compute output vector for this (b, t) across H tiles
    # Accumulator in float32
    output_vec = tl.zeros((H,), dtype=tl.float32)

    # Iterate over hidden dimension in tiles of size BLOCK_H
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # Initialize accumulator for this H tile
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # Loop over K (input feature) dimension in tiles
        for k_off in range(0, H, BLOCK_H):  # Note: typo previously; correct to iterate K in chunks too
            offs_k = k_off + tl.arange(0, BLOCK_H)
            mask_k = offs_k < H

            # Load x (input vector) from encoder or image based on t
            # If t < T: x = encoder[b, t, :]
            # Else: x = image[b, t - T, :]
            # Note: since we loop over K (feature), x is the input vector for all features, not just one element.
            # We need a vector x of length H. Triton doesn't support arbitrary gathers over full H easily,
            # so we compute x as a vector using the same offs_k and select source based on t.
            # To keep simple and robust, we load x as a vector using offs_k and rely on masks for partial tiles.
            # However, to select encoder vs image, we need to know if t < T; we can branch.
            # We branch here: compute x accordingly.
            if t < T:
                x_ptrs = encoder_ptr + pid_b * encoder_stride_b + t * encoder_stride_t + offs_k * encoder_stride_h
                x = tl.load(x_ptrs, mask=mask_k, other=0.0)
            else:
                t_img = t - T
                x_ptrs = image_ptr + pid_b * image_stride_b + t_img * image_stride_i + offs_k * image_stride_h
                x = tl.load(x_ptrs, mask=mask_k, other=0.0)

            # Load weight block [BLOCK_H, BLOCK_H] for this K chunk
            w_ptrs = weight_ptr + (offs_h[:, None] * weight_stride_h) + (offs_k[None, :] * weight_stride_k)
            w = tl.load(w_ptrs, mask=(mask_h[:, None] & mask_k[None, :]), other=0.0)

            # Accumulate: acc += sum over K of w * x
            # Broadcast x to [1, BLOCK_K] implicitly by multiplication
            acc += tl.sum(w * x[None, :], axis=1)

        # Accumulate into the full output vector
        output_vec[offs_h] = acc

    # Store the output vector for this (b, t) into out
    out_ptrs = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + offs_h * out_stride_h
    tl.store(out_ptrs, output_vec, mask=mask_h)

def triton_concat_linear(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    Compute processed = torch.cat([encoder_hidden_states, hidden_states], dim=1) @ process_weight.T
    using a Triton kernel. Returns tensor of shape [B, T+I, H], dtype float32 for robustness.
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
    B = hidden_states.shape[0]
    I = hidden_states.shape[1]
    T = encoder_hidden_states.shape[1]
    H = hidden_states.shape[2]
    assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]."

    # Cast inputs to float32 for kernel robustness; return will be cast back appropriately in ModelNew.
    image = hidden_states.contiguous().to(torch.float32)
    encoder = encoder_hidden_states.contiguous().to(torch.float32)
    weight = process_weight.contiguous().to(torch.float32)

    out = torch.empty((B, T + I, H), dtype=torch.float32, device=image.device)

    # Strides (in elements)
    image_stride_b = image.stride(0)
    image_stride_i = image.stride(1)
    image_stride_h = image.stride(2)

    encoder_stride_b = encoder.stride(0)
    encoder_stride_t = encoder.stride(1)
    encoder_stride_h = encoder.stride(2)

    weight_stride_h = weight.stride(0)
    weight_stride_k = weight.stride(1)

    out_stride_b = out.stride(0)
    out_stride_t = out.stride(1)
    out_stride_h = out.stride(2)

    # Kernel launch: grid over (B, T+I)
    BLOCK_H = 64  # tile size along hidden dimension; tune as needed
    grid = (B, T + I)
    concat_linear_kernel_simple[grid](
        image, encoder, weight, out,
        B, I, T, H,
        image_stride_b, image_stride_i, image_stride_h,
        encoder_stride_b, encoder_stride_t, encoder_stride_h,
        weight_stride_h, weight_stride_k,
        out_stride_b, out_stride_t, out_stride_h,
        BLOCK_H=BLOCK_H,
        num_warps=4,  # conservative; adjust if tuning
    )
    return out

# Entry point as requested
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
