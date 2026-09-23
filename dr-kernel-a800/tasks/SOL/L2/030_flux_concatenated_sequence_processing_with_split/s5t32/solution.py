import torch
import triton
import triton.language as tl


@triton.jit
def concat_kernel(
    encoder_ptr,      # *f32, [B, Stext, H]
    hidden_ptr,       # *f32, [B, Simg, H]
    out_ptr,          # *f32, [B, T, H], T = Stext + Simg
    B, Stext, Simg, H,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_s, stride_h_h,
    stride_o_b, stride_o_t, stride_o_h,
):
    # 2D grid over (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    if t < Stext:
        # take from encoder at position t
        src = encoder_ptr + b * stride_e_b + t * stride_e_t
        dst = out_ptr + b * stride_o_b + t * stride_o_t
        for i in range(0, H):
            val = tl.load(src + i * stride_e_h)
            tl.store(dst + i * stride_o_h, val)
    else:
        # take from hidden at position s = t - Stext
        s = t - Stext
        src = hidden_ptr + b * stride_h_b + s * stride_h_s
        dst = out_ptr + b * stride_o_b + t * stride_o_t
        for i in range(0, H):
            val = tl.load(src + i * stride_h_h)
            tl.store(dst + i * stride_o_h, val)


@triton.jit
def matvec_row_kernel(
    in_ptr,           # *f32, concatenated [B, T, H], T = Stext + Simg
    weightT_ptr,      # *f32, process_weight.T [H, H]
    out_ptr,          # *f32, processed [B, T, H]
    B, T, H_in, H_out,
    stride_in_b, stride_in_t, stride_in_h,
    stride_w_k, stride_w_n,   # weight_T: dim-0=k, dim-1=n
    stride_out_b, stride_out_t, stride_out_h,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    # output vector for this (b, t)
    out_row = out_ptr + b * stride_out_b + t * stride_out_t

    # Accumulate dot products over input dimension
    acc = tl.zeros([H_out], dtype=tl.float32)

    k = 0
    while k < H_in:
        # load input scalar at this position
        in_val = tl.load(in_ptr + b * stride_in_b + t * stride_in_t + k * stride_in_h)
        # load weight_T[k, :] vector of length H_out
        w_ptrs = weightT_ptr + k * stride_w_k + tl.arange(0, H_out) * stride_w_n
        w_vals = tl.load(w_ptrs)  # length H_out

        # accumulate dot product for this k
        acc += in_val * w_vals
        k += 1

    # store the accumulated output row
    out_ptrs = out_row + tl.arange(0, H_out) * stride_out_h
    tl.store(out_ptrs, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dim in Triton.
        2) Apply linear projection using weight.T @ concatenated in Triton (per (b, t) row).
        3) Split back into encoder and hidden outputs via slicing.
        """
        # Ensure CUDA and float32
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA device for Triton."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."

        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H = hidden_states.shape[2]
        T = Stext + Simg

        # Ensure contiguous inputs
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        weight_T = process_weight.t().contiguous()  # [H, H]

        # Allocate output concatenated tensor
        concatenated = torch.empty((B, T, H), dtype=torch.float32, device=hidden.device)

        # Launch concatenation Triton kernel
        grid_concat = (B, T)
        concat_kernel[grid_concat](
            encoder, hidden, concatenated,
            B, Stext, Simg, H,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            num_warps=1,
        )

        # Allocate processed output tensor [B, T, H]
        processed = torch.empty((B, T, H), dtype=torch.float32, device=hidden.device)

        # Launch matvec row-wise Triton kernel
        grid_matvec = (B, T)
        matvec_row_kernel[grid_matvec](
            concatenated, weight_T, processed,
            B, T, H, H,  # H_in == H_out == H
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            weight_T.stride(0), weight_T.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            num_warps=1,
        )

        # Split back along sequence dimension
        processed_encoder = processed[:, :Stext, :]
        processed_hidden = processed[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
