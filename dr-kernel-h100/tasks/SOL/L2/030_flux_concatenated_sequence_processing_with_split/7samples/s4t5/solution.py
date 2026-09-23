import torch
import triton
import triton.language as tl

@triton.jit
def concat_linear_per_token_kernel(
    image_ptr,           # *fp16/fp32 [B, I, H]
    encoder_ptr,         # *fp16/fp32 [B, T, H]
    weight_ptr,          # *fp16/fp32 [H, H]  (original code uses [hidden_dim, hidden_dim])
    out_ptr,             # *fp32          [B, T+I, H]  (we accumulate in fp32)
    B: tl.int32,
    I: tl.int32,
    T: tl.int32,
    H: tl.int32,
    # strides (in elements)
    image_stride_b, image_stride_i, image_stride_h,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    weight_stride_w, weight_stride_k,
    out_stride_b, out_stride_t, out_stride_h,
    # meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # program ids
    pid_b = tl.program_id(0)  # batch index
    pid_t = tl.program_id(1)  # output token index in [0, T+I)

    # accumulate output vector for this (b, t) across H
    output_vec = tl.zeros((H,), dtype=tl.float32)

    # loop over H dimension in tiles
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        # initialize partial output for this H-tile
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # loop over K (input hidden dimension) in tiles
        for k_off in range(0, H, BLOCK_K):  # H is the input hidden dim, equals weight's K dimension
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H

            # decide source: if t < T, use encoder; else use image
            # base pointers for input vector
            if pid_t < T:
                base_ptr = encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t
            else:
                rel_t = pid_t - T
                base_ptr = image_ptr + pid_b * image_stride_b + rel_t * image_stride_i

            # load input vector chunk as float32
            x = tl.load(base_ptr + offs_k * image_stride_h, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_K]
            x = tl.where(mask_k[None, :], x, 0.0)  # ensure masked elements are zero

            # load weight block: [BLOCK_H, BLOCK_K]
            weight_block = tl.load(
                weight_ptr + offs_h[:, None] * weight_stride_w + offs_k[None, :] * weight_stride_k,
                mask=mask_h[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)  # [BLOCK_H, BLOCK_K]

            # accumulate: acc += sum over K of weight_block * x
            # broadcast x to [1, BLOCK_K] to match [BLOCK_H, BLOCK_K]
            acc += tl.sum(weight_block * x[None, :], axis=1)

        # add to full output vector
        output_vec = output_vec + acc

    # store result for (b, t, :)
    out_row_ptr = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t
    tl.store(out_row_ptr + offs_h * out_stride_h, output_vec, mask=mask_h)


def triton_concat_linear(image: torch.Tensor, encoder: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Compute processed = cat([encoder, image], dim=1) @ weight.T
    without materializing the concatenation, using Triton.
    Returns a torch.Tensor of shape [B, T+I, H], dtype float32 for accumulation.
    """
    assert image.is_cuda and encoder.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors for Triton."
    assert image.dim() == 3 and encoder.dim() == 3 and weight.dim() == 2, "Invalid shapes."
    B, I, H = image.shape
    T = encoder.shape[1]
    K = weight.shape[0]  # hidden dim
    assert weight.shape[1] == K, "weight must be [H, H] where H is hidden_dim."

    # Allocate output in fp32 for stable accumulation
    total = torch.empty((B, T + I, H), device=image.device, dtype=torch.float32)

    # Grid: one program per (batch, output position)
    grid = (B, T + I)

    # Choose reasonable tile sizes; tune if needed
    BLOCK_H = 128
    BLOCK_K = 128
    # num_warps can be tuned; 4 is a good default for these tile sizes
    concat_linear_per_token_kernel[grid](
        image, encoder, weight, total,
        B, I, T, H,
        image.stride(0), image.stride(1), image.stride(2),
        encoder.stride(0), encoder.stride(1), encoder.stride(2),
        weight.stride(0), weight.stride(1),
        total.stride(0), total.stride(1), total.stride(2),
        BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    return total


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_hidden_states).
        """
        # Ensure inputs are CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
        total = triton_concat_linear(hidden_states, encoder_hidden_states, process_weight)  # [B, T+I, H], float32
        T = encoder_hidden_states.shape[1]
        processed_encoder = total[:, :T, :]
        processed_hidden = total[:, T:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
