import torch
import triton
import triton.language as tl

@triton.jit
def concat_and_compute_kernel(
    encoder_ptr,   # *f32 [B, T, H]
    image_ptr,     # *f32 [B, I, H]
    weight_ptr,    # *f32 [H, H]
    out_ptr,       # *f32 [B, T+I, H]
    B: tl.int32,
    T: tl.int32,
    I: tl.int32,
    H: tl.int32,
    # strides (elements)
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    image_stride_b, image_stride_i, image_stride_h,
    weight_stride_h, weight_stride_k,   # weight is [H, H], k-index corresponds to weight[h, k]
    out_stride_b, out_stride_t, out_stride_h,
    # meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program computes one output position t for one batch b
    pid_b = tl.program_id(0)  # batch id
    pid_t = tl.program_id(1)  # output position in [0, T+I)

    # Decide source input: encoder vs image
    is_encoder = pid_t < T
    # Base pointer for input vector
    # For encoder: index t = pid_t
    # For image: index t = pid_t - T
    if is_encoder:
        src_ptr = encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t
    else:
        src_ptr = image_ptr + pid_b * image_stride_b + (pid_t - T) * image_stride_i

    # Output pointer for this (b, t_out)
    out_ptr_t = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t

    # Accumulate output vector for this position across hidden dimension H in tiles
    # We'll store as float32
    offs_h = tl.arange(0, BLOCK_H)
    for h_off in range(0, H, BLOCK_H):
        h_idx = h_off + offs_h
        mask_h = h_idx < H
        # Initialize accumulator for this H tile
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        # Loop over K in tiles
        for k_off in range(0, H, BLOCK_K):  # H (output hidden dim) is used as K dimension
            k_idx = k_off + tl.arange(0, BLOCK_K)
            mask_k = k_idx < H
            # Load input vector chunk x[k]
            # src_ptr is a base pointer; we add k_idx * stride_h
            x_chunk = tl.load(src_ptr + k_idx * image_stride_h, mask=mask_k, other=0.0)
            # Load weight block W[h, k] for h in this tile and k in this tile
            w_block = tl.load(
                weight_ptr + h_idx[:, None] * weight_stride_h + k_idx[None, :] * weight_stride_k,
                mask=mask_h[:, None] & mask_k[None, :],
                other=0.0,
            )
            # Accumulate: acc[h] += sum_k w_block[h, k] * x_chunk[k]
            # Broadcast x_chunk to [1, BLOCK_K], multiply with [BLOCK_H, BLOCK_K], then reduce along axis=1
            # Cast x_chunk to float32 for accumulation
            x_chunk_f32 = x_chunk.to(tl.float32)
            acc += tl.sum(w_block * x_chunk_f32[None, :], axis=1)
        # Store the accumulated output vector for this H tile
        # We store to out_ptr_t + h_idx * out_stride_h with mask
        tl.store(out_ptr_t + h_idx * out_stride_h, acc, mask=mask_h)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_hidden_states).
        """
        # Ensure inputs are CUDA tensors for Triton execution
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]  # hidden_dim assumed same across tensors

        # Allocate output tensor [B, T+I, H]
        total = torch.empty((B, T + I, H), device=hidden_states.device, dtype=torch.float32)

        # Get strides (in elements)
        # Note: PyTorch strides are in elements already
        encoder_stride_b, encoder_stride_t, encoder_stride_h = encoder_hidden_states.stride()
        image_stride_b, image_stride_i, image_stride_h = hidden_states.stride()
        # weight is [H, H], so stride_h and stride_k (second dim) are its strides
        weight_stride_h, weight_stride_k = process_weight.stride()
        out_stride_b, out_stride_t, out_stride_h = total.stride()

        # Launch Triton kernel: one program per (b, t_out)
        grid = (B, T + I)
        # Choose conservative tile sizes; adjust if needed for performance
        BLOCK_H = 64
        BLOCK_K = 64
        concat_and_compute_kernel[grid](
            encoder_hidden_states, hidden_states, process_weight, total,
            B, T, I, H,
            encoder_stride_b, encoder_stride_t, encoder_stride_h,
            image_stride_b, image_stride_i, image_stride_h,
            weight_stride_h, weight_stride_k,
            out_stride_b, out_stride_t, out_stride_h,
            BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Split back into separate streams
        processed_encoder = total[:, :T, :]
        processed_hidden = total[:, T:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
