import torch
import triton
import triton.language as tl

# Triton kernel: compute out[b, t, :] = process_weight @ concatenated[b, t, :]
# We avoid materializing concatenation by selecting source per t:
#   if t < T: use encoder_hidden_states[b, t, :]
#   else:      use hidden_states[b, t - T, :]
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
    # strides (in elements)
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    image_stride_b, image_stride_i, image_stride_h,
    weight_stride_h, weight_stride_h2,   # weight is [H, H], second dim is same as first
    out_stride_b, out_stride_t, out_stride_h,
):
    pid_b = tl.program_id(0)  # batch index
    pid_t = tl.program_id(1)  # output token index in [0, T+I)

    # Output vector for this (b, t)
    # We will fill out_ptr[pid_b, pid_t, :]
    # Decide source index k in input: k = t if t < T else t - T
    src_index = tl.where(pid_t < T, pid_t, pid_t - T)

    # Initialize output vector as float32
    out_vec = tl.zeros((H,), dtype=tl.float32)

    # Loop over hidden dimension and compute dot with weight row
    # For each h, input_scalar is either encoder[b, t, h] or image[b, t - T, h]
    for h in range(0, H):
        # Load input scalar
        # encoder path if pid_t < T, else image path
        # Use masks as constants since pid_t is scalar
        # Pointer arithmetic:
        # encoder_ptr + b*encoder_stride_b + src_index*encoder_stride_t + h*encoder_stride_h
        # image_ptr   + b*image_stride_b  + (src_index - T)*image_stride_i + h*image_stride_h
        # Build pointers with tl.where for selection
        ptr_encoder = encoder_ptr + pid_b * encoder_stride_b + src_index * encoder_stride_t + h * encoder_stride_h
        ptr_image = image_ptr + pid_b * image_stride_b + (src_index - T) * image_stride_i + h * image_stride_h
        use_encoder = pid_t < T
        input_scalar = tl.load(tl.where(use_encoder, ptr_encoder, ptr_image))  # scalar load
        # Load weight[h, h]
        ptr_weight = weight_ptr + h * weight_stride_h + h * weight_stride_h2
        weight_val = tl.load(ptr_weight)  # scalar load
        # Accumulate
        out_vec[h] = out_vec[h] + input_scalar * weight_val

    # Store out_vec to out[b, t, :]
    # out_ptr + b*out_stride_b + t*out_stride_t + h*out_stride_h
    for h in range(0, H):
        ptr_out = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + h * out_stride_h
        tl.store(ptr_out, out_vec[h])


def triton_run(encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    Triton-only computation:
    Compute processed = (cat([encoder_hidden_states, hidden_states], dim=1)) @ process_weight.T
    without using torch.cat or torch.matmul in host code.
    Returns out of shape [B, T+I, H] in float32.
    """
    assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
    B = encoder_hidden_states.shape[0]
    T = encoder_hidden_states.shape[1]
    I = hidden_states.shape[1]
    H = encoder_hidden_states.shape[2]
    assert hidden_states.shape[0] == B and hidden_states.shape[2] == H, "Shape mismatch"
    assert process_weight.shape[0] == H and process_weight.shape[1] == H, "Weight must be [H, H]"

    # Allocate output tensor in float32 (accumulation dtype)
    out = torch.empty((B, T + I, H), dtype=torch.float32, device=encoder_hidden_states.device)

    # Launch Triton kernel: one program per (b, t)
    grid = (B, T + I)
    process_per_token_kernel[grid](
        encoder_hidden_states, hidden_states, process_weight, out,
        B, T, I, H,
        # strides (elements)
        encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        process_weight.stride(0), process_weight.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        num_warps=1, num_stages=1,
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
        # Compute the full processed tensor with Triton
        total = triton_run(encoder_hidden_states, hidden_states, process_weight)  # [B, T+I, H]
        T = encoder_hidden_states.shape[1]
        processed_encoder = total[:, :T, :]
        processed_hidden = total[:, T:, :]
        return processed_encoder, processed_hidden