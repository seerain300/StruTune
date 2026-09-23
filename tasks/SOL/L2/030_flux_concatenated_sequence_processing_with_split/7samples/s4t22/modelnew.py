import torch
import triton
import triton.language as tl

@triton.jit
def matvec_concat_kernel(
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
    pid_b = tl.program_id(0)  # batch index
    pid_t = tl.program_id(1)  # output token index in [0, T+I)

    # Decide source: if t < T, use encoder; else use image (shift by -T)
    use_encoder = pid_t < T

    # Accumulator for output vector for this (b, t)
    output_vec = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Loop over H in tiles
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # Initialize accumulator for this H tile
        # We'll accumulate over K tiles
        acc_tile = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # Loop over K (input hidden dim) in tiles
        for k_off in range(0, H, BLOCK_K):  # H is the input hidden dimension; process_weight is [H, H]
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H

            # Load input vector chunk x_chunk (either from encoder or image)
            # We'll load as float32 for stability, cast later if needed
            if use_encoder:
                x_ptr = encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t
            else:
                idx_i = pid_t - T
                x_ptr = image_ptr + pid_b * image_stride_b + idx_i * image_stride_i

            x_chunk = tl.load(x_ptr + offs_k * encoder_stride_h, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_K]

            # Load weight tile [BLOCK_H, BLOCK_K]
            # Note: weight_ptr has stride_w along rows (H) and stride_k along cols (H).
            w_ptrs = weight_ptr + (offs_h[:, None] * weight_stride_w) + (offs_k[None, :] * weight_stride_k)
            w_mask = (mask_h[:, None]) & (mask_k[None, :])
            w_block = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)  # [BLOCK_H, BLOCK_K]

            # Accumulate: acc_tile += sum over K (axis=1) of w_block * x_chunk[:, None]
            # Broadcasting: w_block [BLOCK_H, BLOCK_K], x_chunk [BLOCK_K] -> [BLOCK_H, BLOCK_K]
            prod = w_block * x_chunk[None, :]
            acc_tile += tl.sum(prod, axis=1)

        # Store the accumulated tile to output
        # Cast to output dtype if needed: Triton stores as provided; we keep float32 for stability
        out_ptrs = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t
        out_ptrs += (offs_h) * out_stride_h
        tl.store(out_ptrs, acc_tile, mask=mask_h)

# Host-side function that launches the Triton kernel
def triton_concat_linear(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    Compute processed = cat([encoder_hidden_states, hidden_states], dim=1) @ process_weight.T
    without materializing the concatenation, using Triton kernels.
    Returns [B, T+I, H].
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
    # Shapes
    B = hidden_states.shape[0]
    I = hidden_states.shape[1]
    T = encoder_hidden_states.shape[1]
    H = hidden_states.shape[2]  # hidden_dim
    assert encoder_hidden_states.shape[2] == H, "Hidden dimensions must match."

    # Ensure contiguous memory for predictable strides
    image = hidden_states.contiguous()          # [B, I, H]
    encoder = encoder_hidden_states.contiguous()  # [B, T, H]
    weight = process_weight.contiguous()        # [H, H]

    # Allocate output
    out = torch.empty((B, T + I, H), device=hidden_states.device, dtype=torch.float32)  # compute in float32

    # Strides (elements)
    image_stride_b, image_stride_i, image_stride_h = image.stride()
    encoder_stride_b, encoder_stride_t, encoder_stride_h = encoder.stride()
    weight_stride_w, weight_stride_k = weight.stride()  # weight is [H, H]
    out_stride_b, out_stride_t, out_stride_h = out.stride()

    # Meta-parameters: tune as needed
    BLOCK_H = 64
    BLOCK_K = 64
    grid = (B, T + I)

    matvec_concat_kernel[grid](
        image, encoder, weight, out,
        B, I, T, H,
        image_stride_b, image_stride_i, image_stride_h,
        encoder_stride_b, encoder_stride_t, encoder_stride_h,
        weight_stride_w, weight_stride_k,
        out_stride_b, out_stride_t, out_stride_h,
        BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
        num_warps=4,  # conservative; can increase to 8 for larger tiles
        num_stages=2
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