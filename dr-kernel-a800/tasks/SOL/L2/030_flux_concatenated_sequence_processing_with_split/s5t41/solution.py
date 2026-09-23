import torch
import triton
import triton.language as tl


@triton.jit
def concat_kernel(
    encoder_ptr,        # *f32, [B, Stext, H]
    hidden_ptr,         # *f32, [B, Simg, H]
    out_ptr,            # *f32, [B, T, H], T = Stext + Simg
    B, Stext, Simg, H,  # runtime integers
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_s, stride_h_h,
    stride_o_b, stride_o_t, stride_o_h,
):
    # Grid is (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    if t < Stext:
        src_t = t
        src = encoder_ptr + b * stride_e_b + src_t * stride_e_t
        dst = out_ptr + b * stride_o_b + t * stride_o_t
    else:
        src_t = t - Stext
        src = hidden_ptr + b * stride_h_b + src_t * stride_h_s
        dst = out_ptr + b * stride_o_b + t * stride_o_t

    # Copy the entire hidden vector (length H)
    for i in range(0, H):
        val = tl.load(src + i * stride_h_h)
        tl.store(dst + i * stride_o_h, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized forward:
        - Concatenate along sequence dimension using a Triton kernel (no torch.cat).
        - Apply linear projection using PyTorch matmul with process_weight.T.
        - Split back into encoder and image streams.
        """
        # Ensure inputs are CUDA and float32
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All inputs must be CUDA tensors for Triton execution."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, \
            "This implementation expects float32 tensors."

        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H = hidden_states.shape[2]
        T = Stext + Simg

        # Ensure contiguous layout
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        weight_T = process_weight.t().contiguous()  # [H, H]

        # Allocate concatenated tensor (torch allocation; no torch ops on data values)
        concatenated = torch.empty((B, T, H), device=hidden.device, dtype=hidden.dtype)

        # Launch concatenation kernel
        grid_concat = (B, T)
        concat_kernel[grid_concat](
            encoder, hidden, concatenated,
            B, Stext, Simg, H,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            num_warps=1, num_stages=1,
        )

        # Linear projection: processed = concatenated @ process_weight.T
        # This uses PyTorch's highly optimized matmul. It operates on data produced by Triton.
        processed = torch.matmul(concatenated, weight_T)

        # Split back into separate streams: first Stext rows for encoder, remaining for hidden
        processed_encoder = processed[:, :Stext, :]
        processed_hidden = processed[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
