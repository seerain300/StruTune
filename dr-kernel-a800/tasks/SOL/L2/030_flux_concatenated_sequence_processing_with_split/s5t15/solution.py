import torch
import triton
import triton.language as tl


@triton.jit
def concat_by_row_kernel(
    e_ptr,            # *f32, encoder_hidden_states [B, Stext, H]
    h_ptr,            # *f32, hidden_states [B, Simg, H]
    out_ptr,          # *f32, output concatenated [B, T, H]
    B: tl.constexpr,
    Stext: tl.constexpr,
    Simg: tl.constexpr,
    H: tl.constexpr,
    stride_e_b: tl.constexpr,
    stride_e_t: tl.constexpr,
    stride_e_h: tl.constexpr,
    stride_h_b: tl.constexpr,
    stride_h_s: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_out_h: tl.constexpr,
):
    # Each program handles one (b, t) pair and copies the row to out
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= (Stext + Simg)):
        return

    # If t < Stext: source is encoder
    if t < Stext:
        # out[b, t, k] = e[b, t, k]
        for k in range(0, H):
            e_offset = b * stride_e_b + t * stride_e_t + k * stride_e_h
            out_offset = b * stride_out_b + t * stride_out_t + k * stride_out_h
            val = tl.load(e_ptr + e_offset)
            tl.store(out_ptr + out_offset, val)
    else:
        # else: source is hidden, shifted by t - Stext
        h_idx = t - Stext
        for k in range(0, H):
            h_offset = b * stride_h_b + h_idx * stride_h_s + k * stride_h_h
            out_offset = b * stride_out_b + t * stride_out_t + k * stride_out_h
            # Note: h_ptr has hidden dim stride_h_h (normally 1), but we pass stride_h_s for simplicity and consistency
            # Here we assume hidden has last dim stride 1; we pass stride_h_h as 1. For safety, we rely on .contiguous().
            # To be robust, we compute with h_ptr's actual stride (we pass stride_h_h=1). To avoid confusion, we set it to 1.
            val = tl.load(h_ptr + b * stride_h_b + h_idx * stride_h_s + k)  # assuming H dimension stride is 1
            tl.store(out_ptr + out_offset, val)


@triton.jit
def matvec_row_kernel(
    in_ptr,           # *f32, input [B, T, H]
    wT_ptr,           # *f32, process_weight.T [H, H]
    out_ptr,          # *f32, output [B, T, H]
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    stride_in_b: tl.constexpr,
    stride_in_t: tl.constexpr,
    stride_in_h: tl.constexpr,
    stride_w_h: tl.constexpr,   # weightT dim 0
    stride_w_k: tl.constexpr,   # weightT dim 1
    stride_out_b: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_out_h: tl.constexpr,
):
    # Each program handles one (b, t) and computes the entire vector h for that token
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    for h in range(0, H):
        acc = 0.0
        for k in range(0, H):
            in_offset = b * stride_in_b + t * stride_in_t + k * stride_in_h
            w_offset = k * stride_w_h + h * stride_w_k
            a_val = tl.load(in_ptr + in_offset)
            b_val = tl.load(wT_ptr + w_offset)
            acc += a_val * b_val
        out_offset = b * stride_out_b + t * stride_out_t + h * stride_out_h
        tl.store(out_ptr + out_offset, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton version of the original run:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension using Triton.
        - Apply linear projection using Triton (matrix-vector per (b, t)).
        - Split back into encoder and hidden parts using PyTorch slicing.
        """
        # Ensure CUDA and contiguity
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors"
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        H = hidden_states.shape[2]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        T = Stext + Simg

        # 1) Concatenate using Triton kernel: one program per (b, t)
        concatenated = torch.empty((B, T, H), device=hidden_states.device, dtype=hidden_states.dtype)
        grid_concat = (B, T)
        concat_by_row_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, Stext, Simg, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Linear projection using Triton: out[b, t, h] = sum_k concatenated[b, t, k] * weight_T[k, h]
        processed = torch.empty((B, T, H), device=hidden_states.device, dtype=hidden_states.dtype)
        grid_gemm = (B, T)
        matvec_row_kernel[grid_gemm](
            concatenated, process_weight.t(), processed,
            B, T, H,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            process_weight.t().stride(0), process_weight.t().stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Split back into encoder and hidden streams
        processed_encoder = processed[:, :Stext, :]
        processed_hidden = processed[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
