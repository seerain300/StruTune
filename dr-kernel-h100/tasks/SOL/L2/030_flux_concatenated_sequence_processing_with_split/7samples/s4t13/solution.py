import torch
import triton
import triton.language as tl


@triton.jit
def matvec_triton_kernel(
    src_ptr,        # *f32 [B, T+I, H] concatenated tensor
    weight_ptr,     # *f32 [H, H]
    out_ptr,        # *f32 [B, T+I, H]
    B: tl.int32,
    T: tl.int32,
    I: tl.int32,      # not used directly but can be passed for clarity
    H: tl.int32,
    # strides for src
    src_stride_b, src_stride_t, src_stride_h,
    # strides for weight (W is [H, H])
    weight_stride_w, weight_stride_k,
    # strides for out
    out_stride_b, out_stride_t, out_stride_h,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)       # batch index
    t = tl.program_id(1)       # output token index in [0, T+I)

    # Output vector accumulator in float32
    offs_h = tl.arange(0, BLOCK_H)
    # We will loop over H in tiles of size BLOCK_H
    for h_start in range(0, H, BLOCK_H):
        h_idx = h_start + offs_h
        mask_h = h_idx < H

        # Initialize accumulator for this tile
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # Loop over K (hidden dimension) in chunks
        for k_start in range(0, H, BLOCK_K):
            k_idx = k_start + tl.arange(0, BLOCK_K)
            mask_k = k_idx < H

            # Load input vector slice src[b, t, k_idx] as [BLOCK_K]
            src_vec_ptrs = src_ptr + b * src_stride_b + t * src_stride_t + k_idx * src_stride_h
            x_chunk = tl.load(src_vec_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K], f32

            # Load weight block [BLOCK_H, BLOCK_K]
            w_ptrs = weight_ptr + h_idx[:, None] * weight_stride_w + k_idx[None, :] * weight_stride_k
            w_block = tl.load(w_ptrs, mask=mask_h[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_H, BLOCK_K], f32

            # Accumulate: acc += sum_k (w_block[:, k] * x_chunk[k])
            # Broadcast multiply and reduce over K axis
            # Ensure x_chunk is [1, BLOCK_K] for broadcasting
            acc += tl.sum(w_block * x_chunk[None, :], axis=1)

        # Store the accumulated tile to out[b, t, :]
        out_ptrs = out_ptr + b * out_stride_b + t * out_stride_t + h_idx * out_stride_h
        tl.store(out_ptrs, acc, mask=mask_h)


@triton.jit
def split_seq_triton_kernel(
    src_ptr,        # *f32 [B, T+I, H]
    out_ptr,        # *f32 [B, T, H]
    B: tl.int32,
    T: tl.int32,
    I: tl.int32,
    H: tl.int32,
    # src strides
    src_stride_b, src_stride_t, src_stride_h,
    # out strides (out is [B, T, H])
    out_stride_b, out_stride_t, out_stride_h,
    BLOCK_H: tl.constexpr,
):
    # Grid is (B, T) for encoder part
    b = tl.program_id(0)
    t = tl.program_id(1)
    # Loop over H in tiles and copy src[b, t, :] to out[b, t, :]
    for h_start in range(0, H, BLOCK_H):
        h_idx = h_start + tl.arange(0, BLOCK_H)
        mask_h = h_idx < H
        src_ptrs = src_ptr + b * src_stride_b + t * src_stride_t + h_idx * src_stride_h
        x = tl.load(src_ptrs, mask=mask_h, other=0.0)
        out_ptrs = out_ptr + b * out_stride_b + t * out_stride_t + h_idx * out_stride_h
        tl.store(out_ptrs, x, mask=mask_h)


@triton.jit
def split_seq_triton_kernel_hidden(
    src_ptr,        # *f32 [B, T+I, H]
    out_ptr,        # *f32 [B, I, H]
    B: tl.int32,
    T: tl.int32,
    I: tl.int32,
    H: tl.int32,
    # src strides
    src_stride_b, src_stride_t, src_stride_h,
    # out strides (out is [B, I, H])
    out_stride_b, out_stride_i, out_stride_h,
    BLOCK_H: tl.constexpr,
):
    # Grid is (B, I) for image part
    b = tl.program_id(0)
    i = tl.program_id(1)
    t = T + i
    for h_start in range(0, H, BLOCK_H):
        h_idx = h_start + tl.arange(0, BLOCK_H)
        mask_h = h_idx < H
        src_ptrs = src_ptr + b * src_stride_b + t * src_stride_t + h_idx * src_stride_h
        x = tl.load(src_ptrs, mask=mask_h, other=0.0)
        out_ptrs = out_ptr + b * out_stride_b + i * out_stride_i + h_idx * out_stride_h
        tl.store(out_ptrs, x, mask=mask_h)


def triton_matvec_concat(encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    Compute processed = cat([encoder_hidden_states, hidden_states], dim=1) @ process_weight.T
    entirely in Triton. Returns [B, T+I, H].
    """
    assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."

    B = encoder_hidden_states.shape[0]
    T = encoder_hidden_states.shape[1]
    I = hidden_states.shape[1]
    H = encoder_hidden_states.shape[2]
    assert hidden_states.shape[2] == H
    assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]."

    # Concatenate in PyTorch (data movement, not heavy compute)
    concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, H]

    # Prepare output
    total = torch.empty((B, T + I, H), device=concatenated.device, dtype=torch.float32)

    # Launch Triton matvec kernel: one program per (b, t)
    grid = (B, T + I)
    # Choose tile sizes; 64x64 is a robust default for many GPUs
    BLOCK_H = 64
    BLOCK_K = 64
    matvec_triton_kernel[grid](
        concatenated,
        process_weight,  # [H, H]
        total,
        B, T, I, H,
        # src strides
        concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
        # weight strides (W is [H, H])
        process_weight.stride(0), process_weight.stride(1),
        # out strides
        total.stride(0), total.stride(1), total.stride(2),
        BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return total


def triton_split_seq(total: torch.Tensor, T: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split total [B, T+I, H] into:
    - processed_encoder: [B, T, H] for first T tokens
    - processed_hidden: [B, I, H] for remaining tokens
    Both computed by Triton kernels.
    """
    assert total.is_cuda, "Total must be on CUDA for Triton execution."
    B, T_total, H = total.shape
    assert T_total == T + hidden_size if hidden_size > 0 else True  # guard in caller
    I = T_total - T

    processed_encoder = torch.empty((B, T, H), device=total.device, dtype=torch.float32)
    processed_hidden = torch.empty((B, I, H), device=total.device, dtype=torch.float32)

    BLOCK_H = 64
    grid_encoder = (B, T)
    split_seq_triton_kernel[grid_encoder](
        total,
        processed_encoder,
        B, T, I, H,
        total.stride(0), total.stride(1), total.stride(2),
        processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
        BLOCK_H=BLOCK_H,
        num_warps=2, num_stages=2,
    )

    grid_hidden = (B, I)
    split_seq_triton_kernel_hidden[grid_hidden](
        total,
        processed_hidden,
        B, T, I, H,
        total.stride(0), total.stride(1), total.stride(2),
        processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
        BLOCK_H=BLOCK_H,
        num_warps=2, num_stages=2,
    )

    return processed_encoder, processed_hidden


# Note: hidden_size should be derived from hidden_states.shape[2], but since forward has access to hidden_states,
# we will define ModelNew and call the helpers appropriately.
class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_image_hidden_states).
        """
        # Ensure inputs are on CUDA for Triton execution
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."

        # Concatenate and compute matvec in Triton
        total = triton_matvec_concat(encoder_hidden_states, hidden_states, process_weight)  # [B, T+I, H]

        # Split into encoder and hidden parts in Triton
        T = encoder_hidden_states.shape[1]
        processed_encoder, processed_hidden = triton_split_seq(total, T)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
