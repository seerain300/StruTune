import torch
import triton
import triton.language as tl

# Triton kernel: For each (b, t), compute out[b, t, :] = process_weight @ input_vec,
# where input_vec is either encoder_hidden_states[b, t, :] (t < T) or
# hidden_states[b, t - T, :] (t >= T). We do not materialize concatenation.
@triton.jit
def process_per_token_kernel(
    encoder_ptr,         # *f32 [B, T, H]
    image_ptr,           # *f32 [B, I, H]
    weight_ptr,          # *f32 [H, H]
    out_ptr,             # *f32 [B, T+I, H]
    B: tl.int32,
    T: tl.int32,
    I: tl.int32,
    H: tl.int32,
    # strides in elements
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    image_stride_b, image_stride_i, image_stride_h,
    weight_stride_w, weight_stride_k,  # weight is [H, H]
    out_stride_b, out_stride_t, out_stride_h,
    # meta-parameters
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)  # t in [0, T+I)

    # Decide which input row to use: if t < T -> encoder, else -> image shifted by T
    src = 0  # 0 => encoder, 1 => image
    if t < T:
        src = 0
        k_idx = t
    else:
        src = 1
        k_idx = t - T

    # Initialize output vector in float32
    output_vec = tl.zeros((H,), dtype=tl.float32)

    # Iterate over hidden dimension H in chunks of BLOCK_K
    for h_off in range(0, H, BLOCK_K):
        offs_h = h_off + tl.arange(0, BLOCK_K)
        mask_h = offs_h < H

        # Load input vector chunk depending on src
        if src == 0:
            # encoder[b, t, h_off:h_off+BLOCK_K]
            input_chunk = tl.load(
                encoder_ptr + b * encoder_stride_b + k_idx * encoder_stride_t + offs_h * encoder_stride_h,
                mask=mask_h,
                other=0.0,
            )
        else:
            # image[b, k_idx, h_off:h_off+BLOCK_K]
            input_chunk = tl.load(
                image_ptr + b * image_stride_b + k_idx * image_stride_i + offs_h * image_stride_h,
                mask=mask_h,
                other=0.0,
            )
        input_chunk = input_chunk.to(tl.float32)

        # Load weight block [BLOCK_K, BLOCK_K] along K (input_dim) and accumulate
        # For each k in this chunk, multiply weight rows [H] by input_chunk[k]
        for k_off in range(0, BLOCK_K):
            kk = h_off + k_off
            # Scalar kk might be out-of-bounds in last iteration; but input_chunk[kk] would be 0 if kk >= H
            # So we just guard input_chunk[kk] by kk < H; load weight row for kk.
            w_row = tl.load(
                weight_ptr + kk * weight_stride_w + offs_h * weight_stride_k,
                mask=mask_h,
                other=0.0,
            )
            output_vec += w_row * input_chunk[kk]

    # Store the output vector for (b, t, :)
    tl.store(out_ptr + b * out_stride_b + t * out_stride_t + offs_h * out_stride_h, output_vec, mask=mask_h)


def triton_run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Triton-optimized computation that avoids torch.cat and torch.matmul in host code.
    Returns processed tensor of shape [B, T+I, H] computed as:
      concatenated[b, t, :] = encoder_hidden_states[b, t, :] if t < T else hidden_states[b, t - T, :]
      processed[b, t, :] = process_weight @ concatenated[b, t, :]
    All heavy computation is done by Triton kernels.
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."

    B, I, H = hidden_states.shape
    T = encoder_hidden_states.shape[1]

    # Allocate output as float32 (accumulation dtype). We'll cast to original dtype if needed after.
    out = torch.empty((B, T + I, H), device=hidden_states.device, dtype=torch.float32)

    # Ensure contiguous for simple stride handling
    encoder = encoder_hidden_states.contiguous()
    image = hidden_states.contiguous()
    weight = process_weight.contiguous()
    out = out.contiguous()

    # Launch Triton kernel: one program per (b, t)
    grid = (B, T + I)
    # Choose a reasonable BLOCK_K. If H is small, 64 is fine; for larger H, 128 may help.
    process_per_token_kernel[grid](
        encoder, image, weight, out,
        B, T, I, H,
        encoder.stride(0), encoder.stride(1), encoder.stride(2),
        image.stride(0), image.stride(1), image.stride(2),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_K=64,  # tuneable; 64 or 128 depending on H and GPU
        num_warps=4, num_stages=2
    )

    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_hidden_hidden_states).
        """
        # Ensure inputs are on CUDA for Triton
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."

        # Compute processed tensor entirely in Triton (no torch.cat / torch.matmul on host)
        total = triton_run(hidden_states, encoder_hidden_states, process_weight)  # [B, T+I, H], float32

        # Split into encoder and image parts
        T = encoder_hidden_states.shape[1]
        processed_encoder = total[:, :T, :]
        processed_hidden = total[:, T:, :]

        # If you want to return in the same dtype as inputs, cast here (original code returns float tensors by default)
        # Check input dtypes and cast accordingly:
        # We'll assume the original inputs are float32 (common in transformer setups). If not, cast back to hidden_states.dtype.
        # If hidden_states.dtype is not float32, cast outputs to hidden_states.dtype.
        if processed_encoder.dtype != hidden_states.dtype:
            processed_encoder = processed_encoder.to(hidden_states.dtype)
        if processed_hidden.dtype != hidden_states.dtype:
            processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
