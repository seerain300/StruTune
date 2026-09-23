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

    h = tl.arange(0, H)

    if t < Stext:
        # take from encoder at position t
        src = encoder_ptr + b * stride_e_b + t * stride_e_t
        dst = out_ptr + b * stride_o_b + t * stride_o_t
    else:
        # take from hidden at source s = t - Stext
        s = t - Stext
        src = hidden_ptr + b * stride_h_b + s * stride_h_s
        dst = out_ptr + b * stride_o_b + t * stride_o_t

    # Load and store the row; masked by h < H for safety
    vals = tl.load(src + h * stride_e_h, mask=h < H, other=0.0)
    tl.store(dst + h * stride_o_h, vals, mask=h < H)


@triton.jit
def matvec_row_kernel(
    in_ptr,           # *f32, [B, T, H]
    weightT_ptr,      # *f32, [H, H] (transpose of process_weight)
    out_ptr,          # *f32, [B, T, H]
    B, T, H,
    stride_in_b, stride_in_t, stride_in_h,
    stride_w_k, stride_w_n,  # weight_T strides: dim-0 = k, dim-1 = n
    stride_out_b, stride_out_t, stride_out_h,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    h = tl.arange(0, H)

    # Accumulator for this (b, t) across all H outputs
    acc = tl.zeros([H], dtype=tl.float32)

    # Loop over input dimension k
    for k in range(0, H):
        # Load input vector slice: in[b, t, k]
        in_ptr_k = in_ptr + b * stride_in_b + t * stride_in_t + k * stride_in_h
        in_val = tl.load(in_ptr_k, mask=True, other=0.0)

        # Load weight_T[k, :] vector
        w_ptr_k = weightT_ptr + k * stride_w_k + h * stride_w_n
        w_vec = tl.load(w_ptr_k, mask=h < H, other=0.0)

        # Accumulate dot
        acc += in_val * w_vec

    # Store the result
    out_ptr_row = out_ptr + b * stride_out_b + t * stride_out_t
    tl.store(out_ptr_row + h * stride_out_h, acc, mask=h < H)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenate along sequence dimension in Triton.
        - Apply linear projection in Triton.
        - Split back into encoder and image streams.
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "inputs must be 3D [B, seq_len, H]"
        assert process_weight.dim() == 2, "process_weight must be 2D [H, H]"
        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H = hidden_states.shape[2]

        device = hidden_states.device
        dtype = hidden_states.dtype
        assert dtype == torch.float32, "use float32 tensors for this Triton implementation"

        # Ensure contiguous tensors
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        weight = process_weight.contiguous()

        # Allocate concatenated output [B, T, H]
        T = Stext + Simg
        concatenated = torch.empty((B, T, H), device=device, dtype=dtype)

        # Launch concat kernel
        grid_concat = (B, T)
        concat_kernel[grid_concat](
            encoder, hidden, concatenated,
            B, Stext, Simg, H,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            num_warps=1, num_stages=1,
        )

        # Prepare weight_T = process_weight.T [H, H]
        weight_T = weight.transpose(0, 1).contiguous()

        # Allocate processed output [B, T, H]
        processed = torch.empty((B, T, H), device=device, dtype=dtype)

        # Launch matvec per row kernel
        grid_matvec = (B, T)
        matvec_row_kernel[grid_matvec](
            concatenated, weight_T, processed,
            B, T, H,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            weight_T.stride(0), weight_T.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            num_warps=1, num_stages=1,
        )

        # Split back into encoder and hidden streams
        processed_encoder = processed[:, :Stext, :]
        processed_hidden = processed[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
