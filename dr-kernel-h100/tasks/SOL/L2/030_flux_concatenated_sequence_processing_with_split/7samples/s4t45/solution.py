import torch
import triton
import triton.language as tl

@triton.jit
def process_per_token_kernel(
    encoder_ptr,      # *f32 [B, T, H]
    hidden_ptr,       # *f32 [B, I, H]
    weight_ptr,       # *f32 [H, H]
    out_ptr,          # *f32 [B, T+I, H]
    B: tl.int32,
    T: tl.int32,
    I: tl.int32,
    H: tl.int32,
    # strides for encoder [B, T, H]
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    # strides for hidden [B, I, H]
    hidden_stride_b, hidden_stride_i, hidden_stride_h,
    # strides for weight [H, H]
    weight_stride_n, weight_stride_k,
    # strides for out [B, T+I, H]
    out_stride_b, out_stride_t, out_stride_h,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    assert b < B and t < (T + I), "Kernel grid out of bounds"

    # Decide source: if t < T, use encoder row; else use hidden row at position t - T
    use_encoder = t < T
    # Build input pointer:
    # If use_encoder:
    #   input_row = encoder[b, t, :]
    # else:
    #   input_row = hidden[b, t - T, :]
    # We'll compute the base pointer for the chosen tensor.
    if use_encoder:
        # base pointer for encoder[b, t, :]
        base_x = encoder_ptr + b * encoder_stride_b + t * encoder_stride_t
    else:
        # base pointer for hidden[b, t - T, :]
        t_img = t - T
        base_x = hidden_ptr + b * hidden_stride_b + t_img * hidden_stride_i

    # Output pointer for this (b, t)
    out_ptr_t = out_ptr + b * out_stride_b + t * out_stride_t

    # Accumulate output vector in float32
    out_vec = tl.zeros((H,), dtype=tl.float32)

    # Loop over hidden dimension and compute out_vec = W @ x_t
    for h in range(0, H):
        # Load input scalar x_t[h]
        x_val = tl.load(base_x + h * hidden_stride_h)  # hidden_stride_h is 1 for contiguous H; we pass it for generality
        # Load weight row slice for this h
        w_row = tl.load(weight_ptr + h * weight_stride_n + tl.arange(0, H) * weight_stride_k)
        # Accumulate: out_vec += x_val * w_row
        out_vec += x_val * w_row

    # Store the result vector
    tl.store(out_ptr_t + tl.arange(0, H) * out_stride_h, out_vec)


def triton_run(encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation that computes:
        total = cat([encoder_hidden_states, hidden_states], dim=1) @ process_weight.T
        without using torch.cat or torch.matmul in the host code.

    Returns total of shape [B, T + I, H] as float32 (accumulation dtype).
    """
    assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton."
    B, T, H = encoder_hidden_states.shape
    B2, I, H2 = hidden_states.shape
    assert B == B2 and H == H2, "Encoder and hidden tensors must have the same batch and hidden_dim."
    assert process_weight.shape == (H, H), "process_weight must be [H, H]."

    # Allocate output tensor in float32
    total = torch.empty((B, T + I, H), device=encoder_hidden_states.device, dtype=torch.float32)

    # Launch one program per (batch, output token)
    grid = (B, T + I)
    process_per_token_kernel[grid](
        encoder_hidden_states, hidden_states, process_weight, total,
        B, T, I, H,
        encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        process_weight.stride(0), process_weight.stride(1),
        total.stride(0), total.stride(1), total.stride(2),
        num_warps=1, num_stages=1,
    )
    return total


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_image_hidden_states).
        """
        # Ensure inputs are CUDA for Triton execution
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
        total = triton_run(encoder_hidden_states, hidden_states, process_weight)  # [B, T+I, H], float32
        T = encoder_hidden_states.shape[1]
        processed_encoder = total[:, :T, :]
        processed_hidden = total[:, T:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
