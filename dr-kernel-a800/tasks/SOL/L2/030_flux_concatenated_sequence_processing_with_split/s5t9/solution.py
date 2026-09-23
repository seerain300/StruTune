import torch
import triton
import triton.language as tl


@triton.jit
def concat_by_row_kernel(
    encoder_ptr,    # *f32, [B, Stext, H]
    hidden_ptr,     # *f32, [B, Simg, H]
    out_ptr,        # *f32, [B, T, H], T = Stext + Simg
    B, Stext, Simg, H,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_t, stride_h_h,
    stride_out_b, stride_out_t, stride_out_h,
):
    # 2D grid: (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)

    # Compute the source row index in hidden based on t
    is_text = t < Stext
    src_t = t  # encoder row index
    if is_text:
        src_ptr = encoder_ptr + b * stride_e_b + src_t * stride_e_t
    else:
        src_t = t - Stext
        src_ptr = hidden_ptr + b * stride_h_b + src_t * stride_h_t

    # Write out_cat[b, t, :] = src_row
    # We loop over H with a vectorized offset to handle any H
    h_offsets = tl.arange(0, H)
    vals = tl.load(src_ptr + h_offsets * stride_e_h, mask=h_offsets < H, other=0.0)
    out_row_ptr = out_ptr + b * stride_out_b + t * stride_out_t
    tl.store(out_row_ptr + h_offsets * stride_out_h, vals, mask=h_offsets < H)


@triton.jit
def matvec_row_kernel(
    in_ptr,     # *f32, [B, T, H] (out_cat)
    wT_ptr,     # *f32, [H, H] (process_weight transposed)
    out_ptr,    # *f32, [B, T, H] result
    B, T, H,
    stride_in_b, stride_in_t, stride_in_h,
    stride_w_b, stride_w_t, stride_w_h,  # wT is [H, H] => (h_in, h_out)
    stride_out_b, stride_out_t, stride_out_h,
):
    # 2D grid: (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)

    # Compute out[b, t, :] = wT @ in[b, t, :]
    # We'll write each h_out by accumulating over k
    # Loop over k (hidden input dims)
    for k in range(0, H):
        # Load in[b, t, k]
        in_ptr_k = in_ptr + b * stride_in_b + t * stride_in_t + k * stride_in_h
        a_k = tl.load(in_ptr_k, mask=k < H, other=0.0)

        # Accumulate out[b, t, h_out] += a_k * wT[k, h_out]
        for h_out in range(0, H):
            w_ptr = wT_ptr + k * stride_w_b + h_out * stride_w_h
            w_val = tl.load(w_ptr, mask=h_out < H, other=0.0)
            out_ptr_h = out_ptr + b * stride_out_b + t * stride_out_t + h_out * stride_out_h
            curr = tl.load(out_ptr_h, mask=h_out < H, other=0.0)
            new_val = curr + a_k * w_val
            tl.store(out_ptr_h, new_val, mask=h_out < H)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,         # [B, Simg, H]
        encoder_hidden_states: torch.Tensor, # [B, Stext, H]
        process_weight: torch.Tensor,        # [H, H], no bias
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension (Triton).
        - Applies linear projection process_weight.T to the concatenated sequence (Triton).
        - Splits the result back into separate encoder and image streams.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."

        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H = hidden_states.shape[2]
        T = Stext + Simg

        # Ensure contiguous
        e = encoder_hidden_states.contiguous()
        h = hidden_states.contiguous()
        w = process_weight.contiguous()  # [H, H]
        w_T = w.transpose(0, 1).contiguous()  # [H, H] (transposed for Triton)

        # Allocate output for concatenation
        out_cat = torch.empty((B, T, H), device=e.device, dtype=e.dtype)

        # Launch concat kernel: grid (B, T)
        grid_concat = (B, T)
        concat_by_row_kernel[grid_concat](
            e, h, out_cat,
            B, Stext, Simg, H,
            e.stride(0), e.stride(1), e.stride(2),
            h.stride(0), h.stride(1), h.stride(2),
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            num_warps=1, num_stages=1,
        )

        # Allocate output for GEMV
        out = torch.empty((B, T, H), device=e.device, dtype=e.dtype)

        # Launch GEMV kernel: grid (B, T)
        grid_gemm = (B, T)
        matvec_row_kernel[grid_gemm](
            out_cat, w_T, out,
            B, T, H,
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            w_T.stride(0), w_T.stride(1), w_T.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            num_warps=1, num_stages=1,
        )

        # Split along sequence dimension
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
