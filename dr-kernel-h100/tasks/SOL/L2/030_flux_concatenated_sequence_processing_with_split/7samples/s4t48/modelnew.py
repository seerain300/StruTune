import torch
import triton
import triton.language as tl

@triton.jit
def matvec_per_token_kernel(
    image_ptr,         # *f32 [B, I, H]
    encoder_ptr,       # *f32 [B, T, H]
    weight_ptr,        # *f32 [H, H]
    out_ptr,           # *f32 [B, T+I, H]
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
    BLOCK_K: tl.constexpr,  # typically set to H
):
    # Each program handles one (batch, output token) pair
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    # Compute the output vector for this (b, t) and accumulate in float32
    # output_vec: [H] as float32
    # We set BLOCK_K == H to avoid tiling over hidden dimension.
    output_vec = tl.zeros((H,), dtype=tl.float32)

    # Determine source tensor based on t < T
    # If t < T: read from encoder[b, t, :]
    # Else: read from image[b, t - T, :]
    # Loop over K in chunks; since BLOCK_K == H, this is a single iteration
    for k_off in range(0, H, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H

        # Load weight tile [BLOCK_K, H] (since H == BLOCK_K and BLOCK_K == H)
        # Note: here we load weight for rows offs_k (input indices) and columns 0..H-1 (output indices).
        w_ptrs = weight_ptr + (offs_k[:, None] * weight_stride_k + tl.arange(0, H)[None, :] * weight_stride_w)
        weight_tile = tl.load(w_ptrs, mask=mask_k[:, None], other=0.0)

        # Determine which source to load from
        if pid_t < T:
            x_ptrs = encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t + tl.arange(0, H) * encoder_stride_h
        else:
            x_idx = pid_t - T
            x_ptrs = image_ptr + pid_b * image_stride_b + x_idx * image_stride_i + tl.arange(0, H) * image_stride_h

        x_vec = tl.load(x_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K], float32

        # Accumulate: output_vec += weight_tile @ x_vec
        # weight_tile: [BLOCK_K, H], x_vec: [BLOCK_K] -> result [H]
        acc = tl.dot(weight_tile, x_vec)
        output_vec += acc

    # Store the output vector to out[b, t, :]
    out_ptrs = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + tl.arange(0, H) * out_stride_h
    tl.store(out_ptrs, output_vec)

def triton_concat_linear(image: torch.Tensor, encoder: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Compute processed = cat([encoder, image], dim=1) @ weight.T
    Returns [B, T+I, H] with dtype float32 (accumulation in fp32).
    """
    assert image.is_cuda and encoder.is_cuda and weight.is_cuda, "Triton kernels require CUDA tensors"
    # Ensure contiguous tensors
    image = image.contiguous()
    encoder = encoder.contiguous()
    weight = weight.contiguous()

    B = image.shape[0]
    I = image.shape[1]
    T = encoder.shape[1]
    H = image.shape[2]
    total_len = T + I

    # We will compute in float32 for stability. If you need to match original dtype,
    # you can cast the result at the end. Here we return float32 tensor.
    out = torch.empty((B, total_len, H), device=image.device, dtype=torch.float32)

    # Launch kernel: one program per (b, t)
    grid = (B, total_len)
    # Choose BLOCK_K = H for simplicity; ensure weight is [H, H]. We'll pass H as BLOCK_K meta.
    triton.run(
        matvec_per_token_kernel,
        grid=grid,
        num_warps=4,
        num_stages=2,
        image_ptr=image,
        encoder_ptr=encoder,
        weight_ptr=weight,
        out_ptr=out,
        B=B, I=I, T=T, H=H,
        # strides
        image_stride_b=image.stride(0), image_stride_i=image.stride(1), image_stride_h=image.stride(2),
        encoder_stride_b=encoder.stride(0), encoder_stride_t=encoder.stride(1), encoder_stride_h=encoder.stride(2),
        weight_stride_w=weight.stride(0), weight_stride_k=weight.stride(1),
        out_stride_b=out.stride(0), out_stride_t=out.stride(1), out_stride_h=out.stride(2),
        BLOCK_K=H,  # set BLOCK_K to H to avoid tiling issues
    )
    return out

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_image_hidden_states).
        """
        # Ensure inputs are on CUDA for Triton
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
        # Compute the full processed tensor [B, T+I, H]
        total = triton_concat_linear(hidden_states, encoder_hidden_states, process_weight)
        T = encoder_hidden_states.shape[1]
        processed_encoder = total[:, :T, :]
        processed_hidden = total[:, T:, :]
        return processed_encoder, processed_hidden