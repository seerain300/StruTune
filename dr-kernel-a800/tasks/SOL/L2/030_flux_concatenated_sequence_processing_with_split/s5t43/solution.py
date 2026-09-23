import torch
import triton
import triton.language as tl


@triton.jit
def concat_rows_kernel(
    encoder_ptr,      # *f32, [B, Stext, H]
    hidden_ptr,       # *f32, [B, Simg, H]
    out_ptr,          # *f32, [B, T, H], T = Stext + Simg
    B,                # int
    Stext,            # int
    Simg,             # int
    H,                # int
    stride_e_b,       # int
    stride_e_t,       # int
    stride_e_h,       # int
    stride_h_b,       # int
    stride_h_s,       # int
    stride_h_h,       # int
    stride_o_b,       # int
    stride_o_t,       # int
    stride_o_h,       # int
):
    # 2D grid over (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    h = 0
    if t < Stext:
        # Copy from encoder at position t
        src_ptr = encoder_ptr + b * stride_e_b + t * stride_e_t
        dst_ptr = out_ptr + b * stride_o_b + t * stride_o_t
        while h < H:
            val = tl.load(src_ptr + h * stride_e_h)
            tl.store(dst_ptr + h * stride_o_h, val)
            h += 1
    else:
        # Copy from hidden at position s = t - Stext
        s = t - Stext
        src_ptr = hidden_ptr + b * stride_h_b + s * stride_h_s
        dst_ptr = out_ptr + b * stride_o_b + t * stride_o_t
        while h < H:
            val = tl.load(src_ptr + h * stride_h_h)
            tl.store(dst_ptr + h * stride_o_h, val)
            h += 1


@triton.jit
def batched_linear_kernel(
    concat_ptr,       # *f32, [B, T, H]
    weight_ptr,       # *f32, [H, H]
    out_ptr,          # *f32, [B, T, H]
    B,                # int
    T,                # int
    H,                # int
    stride_c_b,       # int
    stride_c_t,       # int
    stride_c_h,       # int
    stride_w_h,       # int for dim-0 (rows)
    stride_w_p,       # int for dim-1 (cols)
    stride_o_b,       # int
    stride_o_t,       # int
    stride_o_h,       # int
):
    # Grid over (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    # Compute out_row = concat_row @ weight
    # Use a vector accumulator of length H
    acc = tl.zeros([H], dtype=tl.float32)

    # Loop over input hidden dimension K = H in chunks
    k0 = 0
    while k0 < H:
        k_offsets = k0 + tl.arange(0, H)  # vectorized offsets within [k0, k0+H)
        mask_k = k_offsets < H  # scalar mask; Triton handles per-element
        # Load in_chunk: concat[b, t, k_offsets]
        in_ptrs = concat_ptr + b * stride_c_b + t * stride_c_t + k_offsets * stride_c_h
        in_chunk = tl.load(in_ptrs, mask=mask_k, other=0.0)  # shape [H]
        # Load weight block W[k_offsets, :], shape [H]
        w_ptrs = weight_ptr + k_offsets * stride_w_h + tl.arange(0, H) * stride_w_p
        w_block = tl.load(w_ptrs, mask=mask_k, other=0.0)  # shape [H], each entry is [k]*[H] row
        # Accumulate: acc += in_chunk * w_block
        acc += in_chunk * w_block
        k0 += H

    # Store result acc to out[b, t, :]
    out_row_ptrs = out_ptr + b * stride_o_b + t * stride_o_t + tl.arange(0, H) * stride_o_h
    store_mask = tl.arange(0, H) < H  # always true, but keep for safety
    tl.store(out_row_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
          - Concatenates along sequence dimension in Triton
          - Applies linear projection in Triton (concat_row @ process_weight)
          - Splits back into encoder and hidden streams
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton execution"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors"
        assert hidden_states.is_contiguous() and encoder_hidden_states.is_contiguous() and process_weight.is_contiguous(), "Tensors must be contiguous"

        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H = hidden_states.shape[2]
        T = Stext + Simg

        # Allocate concatenated tensor
        concatenated = torch.empty((B, T, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch Triton kernel to concatenate along sequence dimension
        grid = (B, T)
        concat_rows_kernel[grid](
            encoder_hidden_states, hidden_states, concatenated,
            B, Stext, Simg, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            num_warps=1, num_stages=1,
        )

        # Prepare weight_T: original code uses process_weight (shape [H, H]), no transpose
        weight = process_weight  # [H, H]
        out = torch.empty((B, T, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch Triton GEMV-like kernel: out[b, t, :] = concatenated[b, t, :] @ weight
        batched_linear_kernel[grid](
            concatenated, weight, out,
            B, T, H,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            weight.stride(0), weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            num_warps=1, num_stages=1,
        )

        # Split back into streams
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
