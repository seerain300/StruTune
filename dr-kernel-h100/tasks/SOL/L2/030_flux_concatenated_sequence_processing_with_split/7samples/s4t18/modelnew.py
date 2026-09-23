import torch
import triton
import triton.language as tl

# Kernel: compute processed_image = hidden_states @ weight.T, output shape [B, I, H]
@triton.jit
def process_image_kernel(
    image_ptr,           # *f32 [B, I, H]
    weight_ptr,          # *f32 [H, H]
    out_ptr,             # *f32 [B, I, H]
    B: tl.int32, I: tl.int32, H: tl.int32,
    # strides in elements
    image_stride_b, image_stride_i, image_stride_h,
    weight_stride_h, weight_stride_k,   # weight is [H, H]
    out_stride_b, out_stride_i, out_stride_h,
    # meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch
    pid_i = tl.program_id(1)  # image sequence position

    # Output vector accumulator in float32
    output_vec = tl.zeros((H,), dtype=tl.float32)

    # Loop over H in tiles
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # Initialize accumulator for this H tile
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # Loop over K (hidden dimension) in tiles
        for k_off in range(0, H, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H

            # Load input chunk vector x (shape: [BLOCK_K])
            # Pointer: image_ptr + pid_b*image_stride_b + pid_i*image_stride_i + offs_k*image_stride_h
            x_ptrs = image_ptr + pid_b * image_stride_b + pid_i * image_stride_i + offs_k * image_stride_h
            x_chunk = tl.load(x_ptrs, mask=mask_k, other=0.0)

            # Load weight block w (shape: [BLOCK_H, BLOCK_K])
            w_ptrs = weight_ptr + offs_h[:, None] * weight_stride_h + offs_k[None, :] * weight_stride_k
            w_block = tl.load(w_ptrs, mask=mask_h[:, None] & mask_k[None, :], other=0.0)

            # Accumulate: acc[h] += sum_k w_block[h, k] * x_chunk[k]
            acc += tl.sum(w_block * x_chunk[None, :], axis=1)

        # Add this tile's accumulation into output vector
        output_vec[h_off:h_off + BLOCK_H] = acc

    # Store the output vector to out
    out_ptrs = out_ptr + pid_b * out_stride_b + pid_i * out_stride_i + tl.arange(0, H) * out_stride_h
    tl.store(out_ptrs, output_vec, mask=tl.arange(0, H) < H)

# Kernel: compute processed_encoder = encoder_hidden_states @ weight.T, output shape [B, T, H]
@triton.jit
def process_encoder_kernel(
    encoder_ptr,         # *f32 [B, T, H]
    weight_ptr,          # *f32 [H, H]
    out_ptr,             # *f32 [B, T, H]
    B: tl.int32, T: tl.int32, H: tl.int32,
    # strides in elements
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    weight_stride_h, weight_stride_k,
    out_stride_b, out_stride_t, out_stride_h,
    # meta-parameters
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch
    pid_t = tl.program_id(1)  # encoder sequence position

    # Output vector accumulator in float32
    output_vec = tl.zeros((H,), dtype=tl.float32)

    # Loop over H in tiles
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # Loop over K (hidden dimension) in tiles
        for k_off in range(0, H, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H

            # Load input chunk vector x (shape: [BLOCK_K])
            x_ptrs = encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t + offs_k * encoder_stride_h
            x_chunk = tl.load(x_ptrs, mask=mask_k, other=0.0)

            # Load weight block w (shape: [BLOCK_H, BLOCK_K])
            w_ptrs = weight_ptr + offs_h[:, None] * weight_stride_h + offs_k[None, :] * weight_stride_k
            w_block = tl.load(w_ptrs, mask=mask_h[:, None] & mask_k[None, :], other=0.0)

            # Accumulate: acc[h] += sum_k w_block[h, k] * x_chunk[k]
            acc += tl.sum(w_block * x_chunk[None, :], axis=1)

        # Add this tile's accumulation into output vector
        output_vec[h_off:h_off + BLOCK_H] = acc

    # Store the output vector to out
    out_ptrs = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + tl.arange(0, H) * out_stride_h
    tl.store(out_ptrs, output_vec, mask=tl.arange(0, H) < H)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_hidden_states).
        """
        # Ensure inputs are on CUDA for Triton
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."

        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "encoder_hidden_states hidden_dim must match hidden_states"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"

        # Make sure tensors are contiguous for simple stride handling
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        # Prepare outputs (float32 accumulation, output dtype can be kept as float32)
        processed_hidden = torch.empty((B, I, H), device=hidden_states.device, dtype=torch.float32)
        processed_encoder = torch.empty((B, T, H), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton kernels: grid over (batch, length)
        # Tile sizes and warps — tune as needed
        BLOCK_H = 128
        BLOCK_K = 128

        # Kernel for image stream: [B, I] grid
        grid_image = (B, I)
        process_image_kernel[grid_image](
            hidden_states, process_weight, processed_hidden,
            B, I, H,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=2,
        )

        # Kernel for encoder stream: [B, T] grid
        grid_encoder = (B, T)
        process_encoder_kernel[grid_encoder](
            encoder_hidden_states, process_weight, processed_encoder,
            B, T, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=2,
        )

        return processed_encoder, processed_hidden