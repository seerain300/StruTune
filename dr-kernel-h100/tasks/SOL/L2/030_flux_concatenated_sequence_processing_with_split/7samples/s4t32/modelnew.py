import torch
import triton
import triton.language as tl

@triton.jit
def concat_linear_row_kernel(
    image_ptr,           # *fp16/fp32 [B, I, H]
    encoder_ptr,         # *fp16/fp32 [B, T, H]
    weight_ptr,          # *fp16/fp32 [H, H]
    out_ptr,             # *fp32 [B, T+I, H]  (we accumulate/store in fp32 for stability)
    B: tl.int32,
    I: tl.int32,
    T: tl.int32,
    H: tl.int32,
    # strides (elements)
    image_stride_b, image_stride_i, image_stride_h,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    weight_stride_w, weight_stride_k,
    out_stride_b, out_stride_t, out_stride_h,
    # meta
    BLOCK_H: tl.constexpr,  # tile size along hidden dim
    BLOCK_K: tl.constexpr,  # tile size along reduction dim
):
    pid_b = tl.program_id(0)  # batch
    pid_t = tl.program_id(1)  # output token position in [0, T+I)

    # initialize output vector accumulator for this (b, t)
    output_vec = tl.zeros((H,), dtype=tl.float32)

    # iterate over hidden dimension in tiles
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # accumulate for this H tile
        acc_tile = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # iterate over reduction (input) dimension in tiles
        for k_off in range(0, H, BLOCK_K):  # H is the input feature dim (same as hidden dim)
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H

            # decide source tensor: if t < T, use encoder[b, t, :], else use image[b, t - T, :]
            use_encoder = pid_t < T
            # compute input pointers for this chunk
            if use_encoder:
                x_ptr = encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t + offs_k * encoder_stride_h
            else:
                idx_i = pid_t - T
                x_ptr = image_ptr + pid_b * image_stride_b + idx_i * image_stride_i + offs_k * image_stride_h

            # load input vector chunk (fp32 for math)
            x_chunk = tl.load(x_ptr, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_K]

            # load weight block: weight[offs_h, offs_k] -> shape [BLOCK_H, BLOCK_K]
            w_ptr = weight_ptr + offs_h[:, None] * weight_stride_w + offs_k[None, :] * weight_stride_k
            mask_w = mask_h[:, None] & mask_k[None, :]
            w_block = tl.load(w_ptr, mask=mask_w, other=0.0).to(tl.float32)  # [BLOCK_H, BLOCK_K]

            # accumulate: acc_tile += sum(w_block * x_chunk[None, :], axis=1)
            acc_tile += tl.sum(w_block * x_chunk[None, :], axis=1)

        # add this H-tile accumulation to output
        output_vec[offs_h] = acc_tile

    # store output for (b, t, :)
    out_row_ptr = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t  # row start
    out_cols_ptr = out_row_ptr + offs_h * out_stride_h  # vector of column pointers
    tl.store(out_cols_ptr, output_vec, mask=mask_h)


def triton_concat_linear(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    Compute processed = cat([encoder_hidden_states, hidden_states], dim=1) @ process_weight.T
    using a Triton kernel. Returns [B, T+I, H] in float32 for numerical stability.
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
    B = hidden_states.shape[0]
    T = encoder_hidden_states.shape[1]
    I = hidden_states.shape[1]
    H = hidden_states.shape[2]

    # Ensure contiguity
    image = hidden_states.contiguous()          # [B, I, H]
    encoder = encoder_hidden_states.contiguous()  # [B, T, H]
    weight = process_weight.contiguous()       # [H, H]

    # Output buffer: we compute in float32 for stability
    out = torch.empty((B, T + I, H), device=hidden_states.device, dtype=torch.float32)

    # Strides (in elements)
    image_stride_b, image_stride_i, image_stride_h = image.stride()
    encoder_stride_b, encoder_stride_t, encoder_stride_h = encoder.stride()
    weight_stride_w, weight_stride_k = weight.stride()
    out_stride_b, out_stride_t, out_stride_h = out.stride()

    # Choose tile sizes conservatively
    BLOCK_H = 64
    BLOCK_K = 64
    grid = (B, T + I)

    concat_linear_row_kernel[grid](
        image, encoder, weight, out,
        B, I, T, H,
        image_stride_b, image_stride_i, image_stride_h,
        encoder_stride_b, encoder_stride_t, encoder_stride_h,
        weight_stride_w, weight_stride_k,
        out_stride_b, out_stride_t, out_stride_h,
        BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
        num_warps=4,  # tune as needed
    )

    # If you need to match input dtype, cast here:
    # out = out.to(hidden_states.dtype)
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