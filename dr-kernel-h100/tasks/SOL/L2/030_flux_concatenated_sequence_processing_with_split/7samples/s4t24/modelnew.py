import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_kernel(
    image_ptr,           # *fp16/fp32 [B, I, H]
    encoder_ptr,         # *fp16/fp32 [B, T, H]
    weight_ptr,          # *fp16/fp32 [H, H]
    out_ptr,             # *fp32 [B, T+I, H]  (we accumulate in fp32)
    B: tl.int32,
    I: tl.int32,
    T: tl.int32,
    H: tl.int32,
    # strides in elements
    image_stride_b, image_stride_i, image_stride_h,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    weight_stride_w, weight_stride_k,     # weight is [H, H]
    out_stride_b, out_stride_t, out_stride_h,
    # meta-parameters
    BLOCK_H: tl.constexpr,        # tile size along hidden dimension
    BLOCK_K: tl.constexpr,        # tile size along input dimension (we iterate K-chunks)
    NUM_WARPS: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch
    pid_t = tl.program_id(1)  # output sequence position [0, T+I)

    # Decide which input to read from: if t < T -> encoder[b, t, :], else -> image[b, t - T, :]
    is_img = pid_t >= T
    src_b = pid_b
    src_t = pid_t if not is_img else (pid_t - T)

    # Initialize output vector accumulator (fp32)
    output_vec = tl.zeros((H,), dtype=tl.float32)

    # Loop over hidden dimension in tiles
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # Accumulator for this hidden tile
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # Iterate over input dimension in chunks
        for k_off in range(0, H, BLOCK_K):  # note: using H as input dimension (concatenated is [B, T+I, H])
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H

            # Load input vector chunk from selected source
            # For fp16 input, Triton will produce fp16; we'll cast to fp32 for accumulation
            if is_img:
                # x_vec[k] = image[src_b, src_t, k]
                x_ptrs = image_ptr + src_b * image_stride_b + src_t * image_stride_i + offs_k * image_stride_h
                x_vec = tl.load(x_ptrs, mask=mask_k, other=0.0)
            else:
                # x_vec[k] = encoder[src_b, src_t, k]
                x_ptrs = encoder_ptr + src_b * encoder_stride_b + src_t * encoder_stride_t + offs_k * encoder_stride_h
                x_vec = tl.load(x_ptrs, mask=mask_k, other=0.0)

            # Load weight block [BLOCK_H, BLOCK_K]: weight[offs_h, offs_k]
            w_ptrs = weight_ptr + (offs_h[:, None] * weight_stride_w) + (offs_k[None, :] * weight_stride_k)
            w_block = tl.load(w_ptrs, mask=(mask_h[:, None] & mask_k[None, :]), other=0.0)

            # Accumulate: acc += sum_k (w_block[:, k] * x_vec[k]) over K-chunk
            # Do it in fp32 for stability
            x_vec32 = x_vec.to(tl.float32)  # [BLOCK_K]
            prod = w_block.to(tl.float32) * x_vec32[None, :]  # [BLOCK_H, BLOCK_K]
            acc += tl.sum(prod, axis=1)  # [BLOCK_H]

        # Combine partial acc into output vector
        output_vec = output_vec + acc

    # Store the result for (b, t)
    out_ptrs = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + tl.arange(0, H) * out_stride_h
    # Cast back to desired output dtype (out is fp32 in this implementation)
    tl.store(out_ptrs, output_vec, mask=tl.arange(0, H) < H)


def triton_concat_linear(image: torch.Tensor, encoder: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Compute processed = cat([encoder, image], dim=1) @ weight.T without materializing the concat.
    Returns tensor of shape [B, T+I, H].
    """
    assert image.is_cuda and encoder.is_cuda and weight.is_cuda, "All inputs must be CUDA tensors for Triton."
    B = image.shape[0]
    I = image.shape[1]
    T = encoder.shape[1]
    H = image.shape[2]
    assert weight.shape[1] == H, "Weight second dim must match hidden_dim."
    assert weight.shape[0] == H, "Weight first dim must match hidden_dim."

    # Ensure contiguous tensors (we will use strides, but contiguous often helps performance)
    image = image.contiguous()
    encoder = encoder.contiguous()
    weight = weight.contiguous()

    # Output: we'll accumulate in float32 for numeric stability; then return float32
    # If you need to preserve original dtype, you can cast at the end.
    out = torch.empty((B, T + I, H), device=image.device, dtype=torch.float32)

    # Choose reasonable meta-parameters. You can tune these for your GPU and shapes.
    # Smaller tiles reduce risk of register pressure; larger tiles improve throughput.
    BLOCK_H = 64
    BLOCK_K = 64
    grid = (B, T + I)

    concat_linear_kernel[grid](
        image, encoder, weight, out,
        B, I, T, H,
        image.stride(0), image.stride(1), image.stride(2),
        encoder.stride(0), encoder.stride(1), encoder.stride(2),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K, NUM_WARPS=4,
        num_warps=4, num_stages=2,
    )

    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_image_hidden_states).
        """
        # Ensure inputs are CUDA for Triton execution
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
        total = triton_concat_linear(hidden_states, encoder_hidden_states, process_weight)  # [B, T+I, H]
        T = encoder_hidden_states.shape[1]
        processed_encoder = total[:, :T, :]
        processed_hidden = total[:, T:, :]
        return processed_encoder, processed_hidden