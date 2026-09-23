import torch
import triton
import triton.language as tl


@triton.jit
def matvec_row_kernel(
    in_ptr,           # *f32, concatenated [B, T, H_in]
    weightT_ptr,      # *f32, process_weight.T [H_in, H_out] (original [H_out, H_in] transposed)
    out_ptr,          # *f32, [B, T, H_out]
    B, T, H_in, H_out,
    stride_i_b, stride_i_t, stride_i_h,
    stride_w_i, stride_w_o,  # weight_T strides: i (dim-0) and o (dim-1)
    stride_o_b, stride_o_t, stride_o_h,
):
    # Grid: (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    # Base pointers for this (b, t) input row and output row
    in_base = in_ptr + b * stride_i_b + t * stride_i_t
    out_base = out_ptr + b * stride_o_b + t * stride_o_t

    # Accumulate dot product for each output channel
    for n in range(0, H_out):
        acc = tl.zeros([1], dtype=tl.float32)
        # Loop over input channels k = 0..H_in-1
        for k in range(0, H_in):
            in_val = tl.load(in_base + k * stride_i_h)
            w_val = tl.load(weightT_ptr + k * stride_w_i + n * stride_w_o)
            acc += in_val * w_val
        tl.store(out_base + n * stride_o_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run() function.
        - Concatenates encoder_hidden_states and hidden_states along the sequence dimension (PyTorch).
        - Applies linear projection using process_weight.T via Triton kernel.
        - Splits back into processed_encoder_hidden_states and processed_hidden.
        """
        # Ensure CUDA and dtype
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA device."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."

        # Shapes
        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H_in = hidden_states.shape[2]
        T = Stext + Simg
        H_out = process_weight.shape[0]  # process_weight is [H_out, H_in]

        # Make inputs contiguous
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        # Concatenate along sequence dimension
        concatenated = torch.cat([encoder, hidden], dim=1)  # [B, T, H_in]
        # Transpose process_weight to [H_in, H_out] for row-wise matvec
        weight_T = process_weight.t().contiguous()  # [H_out, H_in]

        # Output buffer
        out = torch.empty((B, T, H_out), device=hidden.device, dtype=hidden.dtype)

        # Launch Triton kernel: grid (B, T)
        grid = (B, T)
        matvec_row_kernel[grid](
            concatenated, weight_T, out,
            B, T, H_in, H_out,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            weight_T.stride(0), weight_T.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
        )

        # Split back along sequence dimension
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
