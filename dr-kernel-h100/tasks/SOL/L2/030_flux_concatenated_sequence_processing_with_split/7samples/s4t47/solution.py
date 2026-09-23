import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_gemm_kernel(
    image_ptr,           # *T [B, I, H] input dtype
    encoder_ptr,         # *T [B, T, H] input dtype
    weight_ptr,          # *T [H, H] (no bias), input dtype
    out_ptr,             # *T [B, T+I, H] output, float32
    B: tl.int32,         # batch size
    I: tl.int32,         # image seq len
    T: tl.int32,         # text seq len
    H: tl.int32,         # hidden dim
    # strides (in elements)
    image_stride_b, image_stride_i, image_stride_h,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    weight_stride_w, weight_stride_k,   # weight is [H, H], row-major: stride_w=H, stride_k=1
    out_stride_b, out_stride_t, out_stride_h,
    BLOCK_K: tl.constexpr,              # tile along K (hidden input dimension)
):
    pid_b = tl.program_id(0)   # batch index
    pid_t = tl.program_id(1)   # output token index in [0, T+I)

    # Decide source tensor: encoder for t < T, else image at t - T
    use_encoder = pid_t < T
    base_idx = pid_t if not use_encoder else (pid_t - T)

    # Prepare output vector accumulator (float32 for stability)
    out_vec = tl.zeros((H,), dtype=tl.float32)

    # Loop over hidden input dimension K in tiles
    for k_off in range(0, H, BLOCK_K):
        k_range = k_off + tl.arange(0, BLOCK_K)
        # Load input vector chunk (1D) from chosen source
        if use_encoder:
            ptr_x = encoder_ptr + pid_b * encoder_stride_b + base_idx * encoder_stride_t + k_range * encoder_stride_h
        else:
            ptr_x = image_ptr + pid_b * image_stride_b + base_idx * image_stride_i + k_range * image_stride_h
        x_chunk = tl.load(ptr_x, mask=k_range < H, other=0.0)  # [BLOCK_K]
        # Load weight tile [BLOCK_K, BLOCK_H] where BLOCK_H=H
        ptr_w = weight_ptr + k_range[:, None] * weight_stride_k + tl.arange(0, H) * weight_stride_w
        # Mask for K: only first (H - k_off) rows may be valid if H not divisible by BLOCK_K; but we set H as BLOCK_H, so no partial rows here.
        w_tile = tl.load(ptr_w, mask=(k_range[:, None] < H) & (tl.arange(0, H)[None, :] < H), other=0.0)  # [BLOCK_K, H]
        # Accumulate: out_vec += w_tile @ x_chunk.T  -> shape: [H]
        # Convert to float32 for dot
        w_tile_f32 = w_tile.to(tl.float32)
        x_chunk_f32 = x_chunk.to(tl.float32)
        # Perform the dot product across K (axis=0), resulting in [H]
        partial = tl.sum(w_tile_f32 * x_chunk_f32[None, :], axis=0)  # [H]
        out_vec += partial

    # Store result to out[b, t, :]
    out_ptr_vec = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + tl.arange(0, H) * out_stride_h
    tl.store(out_ptr_vec, out_vec, mask=tl.arange(0, H) < H)


def triton_concat_linear(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    Triton kernel that computes the full processed tensor:
    processed = cat([encoder_hidden_states, hidden_states], dim=1) @ process_weight.T
    Returns tensor of shape [B, T + I, H] in float32 for numerical stability.
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
    B = encoder_hidden_states.shape[0]
    T = encoder_hidden_states.shape[1]
    I = hidden_states.shape[1]
    H = encoder_hidden_states.shape[2]
    assert hidden_states.shape[2] == H, "hidden_dim mismatch"
    assert process_weight.shape == (H, H), "process_weight must be [H, H]"

    # Ensure contiguous for predictable strides
    image = hidden_states.contiguous()
    encoder = encoder_hidden_states.contiguous()
    weight = process_weight.contiguous()

    # Allocate output in float32
    out = torch.empty((B, T + I, H), dtype=torch.float32, device=hidden_states.device)

    # Strides (elements)
    image_stride_b, image_stride_i, image_stride_h = image.stride()
    encoder_stride_b, encoder_stride_t, encoder_stride_h = encoder.stride()
    weight_stride_w, weight_stride_k = weight.stride()  # row-major expected: (H, 1)
    out_stride_b, out_stride_t, out_stride_h = out.stride()

    # Choose BLOCK_K. Since H is typically moderate, using H directly is fine and avoids H tiling.
    # If H is very large, you can set BLOCK_K to 128 or 256 and loop, but here we set BLOCK_K=H for simplicity.
    BLOCK_K = H

    # Launch one program per (batch, output token)
    grid = (B, T + I)
    # Heuristics: num_warps=4 is a good default for these sizes; num_stages=2 for pipelining.
    concat_linear_gemm_kernel[grid](
        image, encoder, weight, out,
        B, I, T, H,
        image_stride_b, image_stride_i, image_stride_h,
        encoder_stride_b, encoder_stride_t, encoder_stride_h,
        weight_stride_w, weight_stride_k,
        out_stride_b, out_stride_t, out_stride_h,
        BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
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


def run(*args):
    return ModelNew()(*args)
