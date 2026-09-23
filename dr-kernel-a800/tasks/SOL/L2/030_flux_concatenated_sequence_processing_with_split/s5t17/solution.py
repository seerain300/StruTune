import torch
import triton
import triton.language as tl


@triton.jit
def concat_kernel(
    encoder_ptr,      # *f32, encoder_hidden_states [B, Stext, H]
    hidden_ptr,       # *f32, hidden_states [B, Simg, H]
    out_ptr,          # *f32, concatenated [B, T, H], T = Stext + Simg
    B: tl.constexpr,
    Stext: tl.constexpr,
    Simg: tl.constexpr,
    H: tl.constexpr,
    stride_e_b: tl.constexpr,
    stride_e_t: tl.constexpr,
    stride_e_h: tl.constexpr,
    stride_h_b: tl.constexpr,
    stride_h_s: tl.constexpr,
    stride_h_h: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
):
    # Grid is (B, T). One program handles one (b, t).
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= (Stext + Simg)):
        return

    # If t < Stext: source = encoder[b, t, :]
    # Else: source = hidden[b, t - Stext, :]
    if t < Stext:
        src = encoder_ptr + b * stride_e_b + t * stride_e_t
    else:
        src = hidden_ptr + b * stride_h_b + (t - Stext) * stride_h_s

    # Write the entire hidden dimension to out[b, t, :]
    for h in range(0, H):
        val = 0.0
        if t < Stext:
            val = tl.load(encoder_ptr + b * stride_e_b + t * stride_e_t + h * stride_e_h)
        else:
            val = tl.load(hidden_ptr + b * stride_h_b + (t - Stext) * stride_h_s + h * stride_h_h)
        tl.store(out_ptr + b * stride_o_b + t * stride_o_t + h * stride_o_h, val)


@triton.jit
def matvec_row_kernel(
    input_ptr,         # *f32, in [B, T, H], where T = Stext + Simg
    weightT_ptr,       # *f32, process_weight.T [H, H]
    output_ptr,        # *f32, out [B, T, H]
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    stride_in_b: tl.constexpr,
    stride_in_t: tl.constexpr,
    stride_in_h: tl.constexpr,
    stride_w_k: tl.constexpr,
    stride_w_h: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_out_h: tl.constexpr,
):
    # One program computes the whole output vector for a single (b, t)
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    # Base pointers for input row and output row
    base_in = input_ptr + b * stride_in_b + t * stride_in_t
    base_out = output_ptr + b * stride_out_b + t * stride_out_t

    # Compute out[b, t, :] = sum_k in[b, t, k] * weightT[k, :]
    for h in range(0, H):
        acc = 0.0
        for k in range(0, H):
            a = tl.load(base_in + k * stride_in_h)
            w = tl.load(weightT_ptr + k * stride_w_k + h * stride_w_h)
            acc += a * w
        tl.store(base_out + h * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [batch, img_seq_len, hidden_dim]
        encoder_hidden_states: [batch, text_seq_len, hidden_dim]
        process_weight: [hidden_dim, hidden_dim]
        Returns:
            (processed_encoder: [batch, text_seq_len, hidden_dim],
             processed_hidden: [batch, img_seq_len, hidden_dim])
        """
        # Ensure on CUDA and contiguous; use float32 as in original
        device = hidden_states.device
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H = hidden_states.shape[2]
        T = Stext + Simg

        # 1) Concatenate along sequence dimension using Triton
        out_cat = torch.empty((B, T, H), device=device, dtype=hidden_states.dtype)
        grid = (B, T)
        concat_kernel[grid](
            encoder_hidden_states, hidden_states, out_cat,
            B, Stext, Simg, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Apply linear projection: out[b, t, :] = process_weight.T @ out_cat[b, t, :]
        weight_T = process_weight.t().contiguous()  # [H, H]
        out = torch.empty((B, T, H), device=device, dtype=hidden_states.dtype)

        matvec_row_kernel[grid](
            out_cat, weight_T, out,
            B, T, H,
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            weight_T.stride(0), weight_T.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Split back along sequence dimension (metadata ops, no torch data compute)
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
