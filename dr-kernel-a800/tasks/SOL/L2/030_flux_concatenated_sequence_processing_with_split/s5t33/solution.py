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
    BLOCK_H: tl.constexpr,
):
    # 2D grid over (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    # Vector of hidden indices
    h = tl.arange(0, BLOCK_H)
    mask = h < H

    # Select source based on t
    if t < Stext:
        src = encoder_ptr + b * stride_e_b + t * stride_e_t + h * stride_e_h
    else:
        s = t - Stext
        src = hidden_ptr + b * stride_h_b + s * stride_h_s + h * stride_h_h

    dst = out_ptr + b * stride_o_b + t * stride_o_t + h * stride_o_h
    vals = tl.load(src, mask=mask, other=0.0)
    tl.store(dst, vals, mask=mask)


@triton.jit
def matvec_row_kernel(
    in_ptr,           # *f32, concatenated [B, T, H]
    weightT_ptr,      # *f32, [H, H] (transpose of process_weight)
    out_ptr,          # *f32, [B, T, H]
    B, T, H,
    stride_i_b, stride_i_t, stride_i_h,
    stride_w_k, stride_w_n,   # weight_T strides: dim-0 is k, dim-1 is n
    stride_o_b, stride_o_t, stride_o_h,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    # Accumulator for output vector
    out_row = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Loop over input dimension k
    for k in range(0, H):
        # Load x[b, t, k] as scalar
        x = tl.load(in_ptr + b * stride_i_b + t * stride_i_t + k * stride_i_h)

        # Load weight_T[k, :] vector
        w = tl.load(weightT_ptr + k * stride_w_k + h * stride_w_n, mask=h < H, other=0.0)

        # Accumulate
        out_row += x * w

    # Store the resulting vector to out[b, t, :]
    dst = out_ptr + b * stride_o_b + t * stride_o_t + h * stride_o_h
    tl.store(dst, out_row, mask=h < H)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure on CUDA and float32
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA device."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."

        # Shapes
        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "hidden_dim must match between inputs and weight."
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [hidden_dim, hidden_dim]."

        # Make tensors contiguous
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        weight = process_weight.contiguous()

        # Allocate concatenated tensor [B, T, H], T = Stext + Simg
        T = Stext + Simg
        concatenated = torch.empty((B, T, H), device=hidden.device, dtype=hidden.dtype)

        # Launch Triton concat kernel: 2D grid (B, T)
        grid_concat = (B, T)
        concat_kernel[grid_concat](
            encoder, hidden, concatenated,
            B, Stext, Simg, H,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_H=H,
            num_warps=1,
            num_stages=1,
        )

        # Prepare weight.T [H, H]
        weight_T = weight.transpose(0, 1).contiguous()

        # Allocate output tensor for processed [B, T, H]
        processed = torch.empty((B, T, H), device=hidden.device, dtype=hidden.dtype)

        # Launch Triton matvec kernel: 2D grid (B, T)
        grid_matvec = (B, T)
        matvec_row_kernel[grid_matvec](
            concatenated, weight_T, processed,
            B, T, H,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            weight_T.stride(0), weight_T.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_H=H,
            num_warps=1,
            num_stages=1,
        )

        # Split back into encoder and image streams
        processed_encoder = processed[:, :Stext, :]
        processed_hidden = processed[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
