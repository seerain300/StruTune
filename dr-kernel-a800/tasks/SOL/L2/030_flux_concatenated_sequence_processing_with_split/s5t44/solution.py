import torch
import triton
import triton.language as tl


@triton.jit
def concat_rows_kernel(
    encoder_ptr,      # *f32, [B, Stext, H]
    hidden_ptr,       # *f32, [B, Simg, H]
    out_ptr,          # *f32, [B, T, H], T = Stext + Simg
    B, Stext, Simg, H,  # runtime sizes (no tl.constexpr here to avoid static_range issues)
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_t, stride_h_h,
    stride_o_b, stride_o_t, stride_o_h,
):
    # Grid over (b, t)
    b = tl.program_id(0)
    t = tl.program_id(1)
    # We rely on grid being (B, T); no need for extra bounds checks.

    # Determine source based on t
    if t < Stext:
        src_ptr = encoder_ptr + b * stride_e_b + t * stride_e_t
    else:
        src_t = t - Stext
        src_ptr = hidden_ptr + b * stride_h_b + src_t * stride_h_t

    dst_ptr = out_ptr + b * stride_o_b + t * stride_o_t

    i = 0
    while i < H:
        if t < Stext:
            val = tl.load(src_ptr + i * stride_e_h)
        else:
            val = tl.load(src_ptr + i * stride_h_h)
        tl.store(dst_ptr + i * stride_o_h, val)
        i += 1


@triton.jit
def batched_linear_rows_kernel(
    in_ptr,           # *f32, concatenated [B, T, H]
    weight_ptr,       # *f32, process_weight [H, H] (no bias)
    out_ptr,          # *f32, output [B, T, H]
    B, T, H,          # runtime sizes
    stride_i_b, stride_i_t, stride_i_h,
    stride_w_k, stride_w_n,   # weight strides: dim-0 = k, dim-1 = n
    stride_o_b, stride_o_t, stride_o_h,
):
    # Grid over (b, t)
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    in_row_ptr = in_ptr + b * stride_i_b + t * stride_i_t
    out_row_ptr = out_ptr + b * stride_o_b + t * stride_o_t

    # Compute out_row = in_row @ weight
    # Iterate over k = 0..H-1
    k = 0
    while k < H:
        w_row_ptr = weight_ptr + k * stride_w_k  # row k of weight
        dot_val = 0.0
        j = 0
        while j < H:
            a = tl.load(in_row_ptr + j * stride_i_h)
            bval = tl.load(w_row_ptr + j * stride_w_n)
            dot_val += a * bval
            j += 1
        tl.store(out_row_ptr + k * stride_o_h, dot_val)
        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure tensors are on CUDA and float32
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA device for Triton kernels."
        if hidden_states.dtype != torch.float32 or encoder_hidden_states.dtype != torch.float32 or process_weight.dtype != torch.float32:
            hidden_states = hidden_states.float()
            encoder_hidden_states = encoder_hidden_states.float()
            process_weight = process_weight.float()

        # Shapes
        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H = hidden_states.shape[2]
        T = Stext + Simg

        # Allocate concatenated tensor [B, T, H]
        concatenated = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)

        # Launch concat kernel
        grid = (B, T)
        concat_rows_kernel[grid](
            encoder_hidden_states, hidden_states, concatenated,
            B, Stext, Simg, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            num_warps=1
        )

        # Allocate output processed [B, T, H]
        processed = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)

        # Launch linear kernel: out[b, t, :] = concatenated[b, t, :] @ process_weight
        batched_linear_rows_kernel[grid](
            concatenated, process_weight, processed,
            B, T, H,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            num_warps=1
        )

        # Split back along sequence dimension
        processed_encoder = processed[:, :Stext, :]
        processed_hidden = processed[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
