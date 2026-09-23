import torch
import triton
import triton.language as tl

@triton.jit
def matvec_concat_kernel(
    image_ptr,          # *f32 [B, I, H]
    encoder_ptr,        # *f32 [B, T, H]
    weight_ptr,         # *f32 [H, H]
    out_ptr,            # *f32 [B, T+I, H]
    B: tl.int32,
    I: tl.int32,
    T: tl.int32,
    H: tl.int32,
    # strides (elements)
    image_stride_b, image_stride_i, image_stride_h,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    weight_stride_w, weight_stride_k,
    out_stride_b, out_stride_t, out_stride_h,
    # meta-params
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, T+I)
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    # Initialize output vector accumulator (float32)
    out_vec = tl.zeros((H,), dtype=tl.float32)

    # Loop over hidden dimension in tiles
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # Accumulate across K in chunks
        for k_off in range(0, H, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H

            # Load input vector x for this token
            # Decide source: if pid_t < T, use encoder; else use image offset by T
            if pid_t < T:
                # encoder[b, pid_t, :]
                x_ptrs = encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t + offs_k * encoder_stride_h
                x_chunk = tl.load(x_ptrs, mask=mask_k, other=0.0)
            else:
                # image[b, pid_t - T, :]
                src_t = pid_t - T
                x_ptrs = image_ptr + pid_b * image_stride_b + src_t * image_stride_i + offs_k * image_stride_h
                x_chunk = tl.load(x_ptrs, mask=mask_k, other=0.0)

            # Load weight block [BLOCK_H, BLOCK_K]
            w_ptrs = weight_ptr + offs_h[:, None] * weight_stride_w + offs_k[None, :] * weight_stride_k
            w_block = tl.load(w_ptrs, mask=mask_h[:, None] & mask_k[None, :], other=0.0)

            # Accumulate: out_vec[h] += sum_k w_block[h, k] * x_chunk[k]
            # Broadcast multiply and reduce over K
            prod = w_block * x_chunk[None, :]  # shape [BLOCK_H, BLOCK_K]
            acc_tile = tl.sum(prod, axis=1)    # shape [BLOCK_H]
            out_vec = out_vec + acc_tile

        # Store the current tile of output vector
        out_ptrs = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + offs_h * out_stride_h
        tl.store(out_ptrs, out_vec, mask=mask_h)

    # Done: each program writes its entire output vector


def triton_concat_linear(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    Compute processed = (cat([encoder_hidden_states, hidden_states], dim=1)) @ process_weight.T
    without materializing the concatenation, using Triton kernels.
    Returns tensor of shape [B, T+I, H].
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton."
    # Ensure float32 for robust accumulation (weights and inputs should be float32)
    if hidden_states.dtype != torch.float32 or encoder_hidden_states.dtype != torch.float32 or process_weight.dtype != torch.float32:
        raise RuntimeError("This Triton implementation currently expects float32 tensors for inputs and weights.")

    B = hidden_states.shape[0]
    I = hidden_states.shape[1]
    T = encoder_hidden_states.shape[1]
    H = hidden_states.shape[2]

    # Allocate output
    total = torch.empty((B, T + I, H), device=hidden_states.device, dtype=torch.float32)

    # Compute strides (in elements)
    image_stride_b, image_stride_i, image_stride_h = hidden_states.stride()
    encoder_stride_b, encoder_stride_t, encoder_stride_h = encoder_hidden_states.stride()
    weight_stride_w, weight_stride_k = process_weight.stride()  # process_weight is [H, H]
    out_stride_b, out_stride_t, out_stride_h = total.stride()

    # Launch kernel: one program per (batch, output token)
    grid = (B, T + I)
    # Tunable meta-parameters
    BLOCK_H = 128
    BLOCK_K = 128
    matvec_concat_kernel[grid](
        hidden_states, encoder_hidden_states, process_weight, total,
        B, I, T, H,
        image_stride_b, image_stride_i, image_stride_h,
        encoder_stride_b, encoder_stride_t, encoder_stride_h,
        weight_stride_w, weight_stride_k,
        out_stride_b, out_stride_t, out_stride_h,
        BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return total


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_image_hidden_states).
        """
        # Triton path (requires CUDA tensors and float32)
        total = triton_concat_linear(hidden_states, encoder_hidden_states, process_weight)  # [B, T+I, H]
        T = encoder_hidden_states.shape[1]
        processed_encoder = total[:, :T, :]
        processed_hidden = total[:, T:, :]
        return processed_encoder, processed_hidden