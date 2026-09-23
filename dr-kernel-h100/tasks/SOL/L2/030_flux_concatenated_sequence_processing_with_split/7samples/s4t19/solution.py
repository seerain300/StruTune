import torch
import triton
import triton.language as tl

# Triton kernel: computes processed_image[b, i, :] = hidden_states[b, i, :] @ weight.T for all b, i
@triton.jit
def process_image_kernel(
    hidden_ptr,        # *f32 [B, I, H]
    weight_ptr,        # *f32 [H, H]
    out_ptr,           # *f32 [B, I, H]
    B: tl.int32, I: tl.int32, H: tl.int32,
    hidden_stride_b, hidden_stride_i, hidden_stride_h,
    weight_stride_h, weight_stride_k,
    out_stride_b, out_stride_i, out_stride_h,
    BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)

    # Initialize output vector (float32 for stability)
    out_vec = tl.zeros((H,), dtype=tl.float32)

    # Loop over hidden dimension in tiles
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # Loop over K (input hidden dim) in tiles to accumulate
        for k_off in range(0, H, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H

            # Load input vector chunk x[k] = hidden[b, i, k]
            x = tl.load(hidden_ptr + pid_b * hidden_stride_b + pid_i * hidden_stride_i + offs_k * hidden_stride_h, mask=mask_k, other=0.0)  # [BLOCK_K]
            # Load weight tile w[h, k]
            w = tl.load(weight_ptr + offs_h[:, None] * weight_stride_h + offs_k[None, :] * weight_stride_k, mask=mask_h[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_H, BLOCK_K]
            # Accumulate: out_vec[h] += sum_k w[h, k] * x[k]
            # Multiply [BLOCK_H, 1] by [1, BLOCK_K] -> [BLOCK_H, BLOCK_K], then reduce along K
            prod = w * x[None, :]  # broadcast x across H tile
            out_vec = out_vec + tl.sum(prod, axis=1)

        # Store the accumulated output vector for this (b, i)
        tl.store(out_ptr + pid_b * out_stride_b + pid_i * out_stride_i + tl.arange(0, H) * out_stride_h, out_vec, mask=tl.arange(0, H) < H)

# Triton kernel: computes processed_encoder[b, t, :] = encoder_hidden_states[b, t, :] @ weight.T for all b, t
@triton.jit
def process_encoder_kernel(
    encoder_ptr,       # *f32 [B, T, H]
    weight_ptr,        # *f32 [H, H]
    out_ptr,           # *f32 [B, T, H]
    B: tl.int32, T: tl.int32, H: tl.int32,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    weight_stride_h, weight_stride_k,
    out_stride_b, out_stride_t, out_stride_h,
    BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    out_vec = tl.zeros((H,), dtype=tl.float32)

    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        for k_off in range(0, H, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H

            # Load x = encoder[b, t, k]
            x = tl.load(encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t + offs_k * encoder_stride_h, mask=mask_k, other=0.0)
            # Load weight tile
            w = tl.load(weight_ptr + offs_h[:, None] * weight_stride_h + offs_k[None, :] * weight_stride_k, mask=mask_h[:, None] & mask_k[None, :], other=0.0)
            # Accumulate
            prod = w * x[None, :]
            out_vec = out_vec + tl.sum(prod, axis=1)

        # Store out_vec
        tl.store(out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + tl.arange(0, H) * out_stride_h, out_vec, mask=tl.arange(0, H) < H)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_hidden_states).
        """
        # Ensure tensors are on CUDA for Triton
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."

        # Make sure inputs are contiguous for simple stride math
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]

        # Output buffers for each stream
        processed_encoder = torch.empty((B, T, H), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, H), device=hidden_states.device, dtype=torch.float32)

        # Choose tile sizes; conservative defaults
        BLOCK_H = 64
        BLOCK_K = 64

        # Launch kernels: one program per (b, i) for image, and one per (b, t) for encoder
        grid_image = (B, I)
        grid_encoder = (B, T)

        # For process_weight, weight_stride_k is the stride along its second dimension (which is H if contiguous)
        # If weight is [H, H] contiguous, weight.stride() returns (H, 1)
        process_weight_stride = process_weight.stride()

        process_image_kernel[grid_image](
            hidden_states, process_weight, processed_hidden,
            B, I, H,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        process_encoder_kernel[grid_encoder](
            encoder_hidden_states, process_weight, processed_encoder,
            B, T, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Return split streams matching original function's output
        # Note: original returns two tensors; we compute them separately and return.
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
