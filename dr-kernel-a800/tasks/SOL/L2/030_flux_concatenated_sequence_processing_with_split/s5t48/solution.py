import torch
import triton
import triton.language as tl


@triton.jit
def concat_kernel(
    encoder_ptr,       # *f32, [B, Stext, H_in]
    hidden_ptr,        # *f32, [B, Simg, H_in]
    out_ptr,           # *f32, [B, T, H_in], T = Stext + Simg
    B: tl.constexpr,
    Stext: tl.constexpr,
    Simg: tl.constexpr,
    H_in: tl.constexpr,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_s, stride_h_h,
    stride_o_b, stride_o_t, stride_o_h,
):
    # program ids
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    # Determine source: if t < Stext, copy from encoder at row t; else copy from hidden at row t - Stext
    if t < Stext:
        src_row = encoder_ptr + b * stride_e_b + t * stride_e_t
    else:
        s = t - Stext
        src_row = hidden_ptr + b * stride_h_b + s * stride_h_s

    dst_row = out_ptr + b * stride_o_b + t * stride_o_t

    # Copy H_in elements with masks
    for i in range(0, H_in):
        val = tl.load(src_row + i * stride_e_h if t < Stext else i * stride_h_h)
        tl.store(dst_row + i * stride_o_h, val)


@triton.jit
def batched_gemv_kernel_scalar(
    in_ptr,            # *f32, concatenated [B, T, H_in]
    weightT_ptr,       # *f32, process_weight.T [H_in, H_in]
    out_ptr,           # *f32, processed [B, T, H_in]
    B, T, H_in,
    stride_in_b, stride_in_t, stride_in_h,
    stride_w_k, stride_w_n,   # weightT strides: dim-0 = k, dim-1 = n
    stride_out_b, stride_out_t, stride_out_h,
):
    # Grid: (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    # Output vector for this (b, t)
    out_row = out_ptr + b * stride_out_b + t * stride_out_t

    # Accumulate output across all k
    for n in range(0, H_in):
        acc = 0.0
        for k in range(0, H_in):
            # Load input scalar: in[b, t, k]
            in_val = tl.load(in_ptr + b * stride_in_b + t * stride_in_t + k * stride_in_h)
            # Load weight row element: weightT[k, n]
            w_val = tl.load(weightT_ptr + k * stride_w_k + n * stride_w_n)
            acc += in_val * w_val
        # Store accumulated result
        tl.store(out_row + n * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Concatenate encoder_hidden_states and hidden_states along sequence dim in Triton.
        - Apply linear projection via Triton batched GEMV (no torch ops on data).
        - Split outputs back into (encoder, hidden).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."
        assert hidden_states.is_contiguous() and encoder_hidden_states.is_contiguous() and process_weight.is_contiguous(), "Inputs must be contiguous."

        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        T = Stext + Simg
        H_in = hidden_states.shape[2]
        assert process_weight.shape[0] == H_in and process_weight.shape[1] == H_in, "process_weight must be [hidden_dim, hidden_dim]."

        # 1) Concatenate along sequence dim using Triton
        concatenated = torch.empty((B, T, H_in), device=hidden_states.device, dtype=hidden_states.dtype)
        grid_concat = (B, T)
        concat_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, Stext, Simg, H_in,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
        )

        # 2) Apply linear projection via Triton GEMV: processed = concatenated @ process_weight.T
        weight_T = process_weight.t()  # [H_in, H_in], contiguous
        processed = torch.empty((B, T, H_in), device=hidden_states.device, dtype=hidden_states.dtype)
        grid_gemv = (B, T)
        batched_gemv_kernel_scalar[grid_gemv](
            concatenated, weight_T, processed,
            B, T, H_in,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            weight_T.stride(0), weight_T.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
        )

        # 3) Split back
        processed_encoder = processed[:, :Stext, :]
        processed_hidden = processed[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
