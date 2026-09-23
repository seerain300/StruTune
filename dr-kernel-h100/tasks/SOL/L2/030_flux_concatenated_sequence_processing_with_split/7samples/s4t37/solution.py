import torch
import triton
import triton.language as tl

@triton.jit
def concat_linear_simple_kernel(
    image_ptr,           # *f32 [B, I, H]
    encoder_ptr,         # *f32 [B, T, H]
    weight_ptr,          # *f32 [H, H]
    out_ptr,             # *f32 [B, T+I, H]
    B: tl.int32,
    I: tl.int32,
    T: tl.int32,
    H: tl.int32,
    # strides (elements)
    image_stride_b, image_stride_i, image_stride_h,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    weight_stride_w, weight_stride_k,   # weight is [H, H]
    out_stride_b, out_stride_t, out_stride_h,
    # meta
    BLOCK_H: tl.constexpr,
):
    # program ids
    pid_b = tl.program_id(0)   # batch
    pid_t = tl.program_id(1)   # output sequence position in [0, T+I)

    # We will compute the entire output vector for this (b, t) in tiles of H.
    # Float32 accumulation for stability.
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # Loop over hidden dimension (weight rows) in tiles to form dot-products
        for k_off in range(0, H, BLOCK_H):  # here use BLOCK_H as both, you can tune
            offs_k = k_off + tl.arange(0, BLOCK_H)
            mask_k = offs_k < H

            # Load input vector x (either from encoder or image depending on pid_t < T)
            # Compute input index: if pid_t < T -> encoder[b, pid_t, :] else image[b, pid_t - T, :]
            is_image = pid_t >= T
            base_b = pid_b
            t_src = pid_t if is_image else pid_t  # not used if is_image, but safe
            # pointer to input row
            x_ptr = encoder_ptr + base_b * encoder_stride_b + (t_src if not is_image else (pid_t - T)) * encoder_stride_t
            x_vec = tl.load(x_ptr + offs_k * encoder_stride_h, mask=mask_k, other=0.0)  # [BLOCK_H]
            x_vec = x_vec.to(tl.float32)

            # Load weight block W_sub = weight[offs_h, offs_k], shape [BLOCK_H, BLOCK_H]
            w_ptr = weight_ptr + offs_h[:, None] * weight_stride_w + offs_k[None, :] * weight_stride_k
            w_mask = mask_h[:, None] & mask_k[None, :]
            w_sub = tl.load(w_ptr, mask=w_mask, other=0.0)  # [BLOCK_H, BLOCK_H], float32

            # Accumulate acc += sum over k (axis=1) of w_sub * x_vec[:, None]
            # Ensure x_vec is broadcast correctly to [BLOCK_H, BLOCK_H]
            x_b = x_vec[:, None]  # [BLOCK_H, 1]
            acc += tl.sum(w_sub * x_b, axis=1)

        # Store the result for this output position (b, pid_t)
        out_ptr_t = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t
        out_ptrs = out_ptr_t + offs_h * out_stride_h
        tl.store(out_ptrs, acc, mask=mask_h)


@triton.jit
def slice_two_streams_kernel(
    total_ptr,        # *f32 [B, T+I, H]
    out_encoder_ptr,  # *f32 [B, T, H]
    out_image_ptr,    # *f32 [B, I, H]
    B: tl.int32,
    T: tl.int32,
    I: tl.int32,
    H: tl.int32,
    total_stride_b, total_stride_t, total_stride_h,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    image_stride_b, image_stride_t, image_stride_h,
    # grid: (B, T) for encoder, (B, I) for image
):
    # First slice: encoder part
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)  # token in [0, T)

    base_total = total_ptr + pid_b * total_stride_b + pid_t * total_stride_t
    # Load entire H for this token
    for h_off in range(0, H):
        val = tl.load(base_total + h_off * total_stride_h)
        tl.store(out_encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t + h_off * encoder_stride_h, val)

    # Second slice: image part
    pid_b_img = tl.program_id(2)
    pid_i = tl.program_id(3)  # token in [0, I)
    base_total_img = total_ptr + pid_b_img * total_stride_b + (pid_t + T) * total_stride_t
    for h_off in range(0, H):
        val = tl.load(base_total_img + h_off * total_stride_h)
        tl.store(out_image_ptr + pid_b_img * image_stride_b + pid_i * image_stride_t + h_off * image_stride_h, val)


def triton_concat_linear(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    Triton kernel to compute total = cat([encoder_hidden_states, hidden_states], dim=1) @ process_weight.T
    Returns total of shape [B, T+I, H].
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
    # Ensure contiguous memory
    hidden_states = hidden_states.contiguous()
    encoder_hidden_states = encoder_hidden_states.contiguous()
    process_weight = process_weight.contiguous()

    B = hidden_states.shape[0]
    I = hidden_states.shape[1]
    T = encoder_hidden_states.shape[1]
    H = hidden_states.shape[2]  # hidden_dim, consistent across tensors

    # Output [B, T+I, H]
    out = torch.empty((B, T + I, H), dtype=torch.float32, device=hidden_states.device)

    # Launch kernel: grid = (B, T+I)
    grid = (B, T + I)
    # Choose a conservative BLOCK_H; 64 works well for many cases, adjust if needed
    concat_linear_simple_kernel[grid](
        encoder_hidden_states, hidden_states, process_weight, out,
        B, I, T, H,
        encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        process_weight.stride(0), process_weight.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_H=64,  # tile over hidden dimension
        num_warps=4,  # tune for performance
        num_stages=2
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_hidden_states).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."

        # Compute full processed tensor
        total = triton_concat_linear(hidden_states, encoder_hidden_states, process_weight)  # [B, T+I, H]

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]

        # Allocate outputs
        processed_encoder = torch.empty((B, T, hidden_states.shape[2]), dtype=torch.float32, device=hidden_states.device)
        processed_hidden = torch.empty((B, hidden_states.shape[1], hidden_states.shape[2]), dtype=torch.float32, device=hidden_states.device)

        # Launch slice kernels: grid = (B, T) and (B, I)
        grid_encoder = (B, T)
        grid_image = (B, hidden_states.shape[1])

        slice_two_streams_kernel[grid_encoder](
            total, processed_encoder, torch.empty(1, dtype=torch.float32, device=hidden_states.device),
            B, T, hidden_states.shape[1], hidden_states.shape[2],
            total.stride(0), total.stride(1), total.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            num_warps=1, num_stages=1
        )

        slice_two_streams_kernel[grid_image](
            total, torch.empty(1, dtype=torch.float32, device=hidden_states.device), processed_hidden,
            B, T, hidden_states.shape[1], hidden_states.shape[2],
            total.stride(0), total.stride(1), total.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            num_warps=1, num_stages=1
        )

        # The second call above writes processed_hidden. The first call writes processed_encoder into its first argument.
        # We can avoid the second dummy tensor by computing both slices with separate launches. Here we provide second output as dummy to satisfy signature and ignore it.

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
