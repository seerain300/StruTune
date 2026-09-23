import torch
import triton
import triton.language as tl


@triton.jit
def concat_by_row_kernel(
    e_ptr,        # *f32 [B, Stext, H]
    h_ptr,        # *f32 [B, Simg, H]
    out_ptr,      # *f32 [B, T, H], T = Stext + Simg
    B: tl.constexpr,
    Stext: tl.constexpr,
    Simg: tl.constexpr,
    H: tl.constexpr,
    stride_e_b: tl.constexpr, stride_e_t: tl.constexpr, stride_e_h: tl.constexpr,
    stride_h_b: tl.constexpr, stride_h_s: tl.constexpr, stride_h_h: tl.constexpr,
    stride_out_b: tl.constexpr, stride_out_t: tl.constexpr, stride_out_h: tl.constexpr,
):
    # One program per (b, t)
    b = tl.program_id(0)
    t = tl.program_id(1)

    if (b >= B) or (t >= (Stext + Simg)):
        return

    # Determine source based on t
    is_encoder = t < Stext
    src_s = t if is_encoder else (t - Stext)

    # Loop over H and copy
    h_offsets = tl.arange(0, H)
    if is_encoder:
        e_row_ptrs = e_ptr + b * stride_e_b + src_s * stride_e_t + h_offsets * stride_e_h
        e_vals = tl.load(e_row_ptrs)
        out_row_ptrs = out_ptr + b * stride_out_b + t * stride_out_t + h_offsets * stride_out_h
        tl.store(out_row_ptrs, e_vals)
    else:
        h_row_ptrs = h_ptr + b * stride_h_b + src_s * stride_h_s + h_offsets * stride_h_h
        h_vals = tl.load(h_row_ptrs)
        out_row_ptrs = out_ptr + b * stride_out_b + t * stride_out_t + h_offsets * stride_out_h
        tl.store(out_row_ptrs, h_vals)


@triton.jit
def matvec_row_kernel(
    in_ptr,       # *f32 [B, T, H]
    wT_ptr,       # *f32 [H, H] (transposed process_weight)
    out_ptr,      # *f32 [B, T, H]
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    stride_in_b: tl.constexpr, stride_in_t: tl.constexpr, stride_in_h: tl.constexpr,
    stride_w_b: tl.constexpr, stride_w_k: tl.constexpr, stride_w_h: tl.constexpr,  # wT is [H, H], we index as [k, h]
    stride_out_b: tl.constexpr, stride_out_t: tl.constexpr, stride_out_h: tl.constexpr,
):
    # One program per (b, t)
    b = tl.program_id(0)
    t = tl.program_id(1)

    if (b >= B) or (t >= T):
        return

    # Compute out[b, t, :] = wT @ in[b, t, :]
    # Loop over k and accumulate over H
    h_offsets = tl.arange(0, H)
    mask_h = h_offsets < H

    out_row_ptrs = out_ptr + b * stride_out_b + t * stride_out_t + h_offsets * stride_out_h
    out_vals = tl.zeros((H,), dtype=tl.float32)

    for k in range(0, H):
        # Load in[b, t, k] scalar
        in_val = tl.load(in_ptr + b * stride_in_b + t * stride_in_t + k * stride_in_h)
        # Load wT[k, :] as vector
        w_row_ptrs = wT_ptr + k * stride_w_k + h_offsets * stride_w_h
        w_vals = tl.load(w_row_ptrs, mask=mask_h, other=0.0)
        # FMA
        out_vals += in_val * w_vals

    tl.store(out_row_ptrs, out_vals, mask=mask_h)


@triton.jit
def split_rows_kernel(
    in_ptr,       # *f32 [B, T, H]
    eout_ptr,     # *f32 [B, Stext, H]
    hout_ptr,     # *f32 [B, Simg, H]
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    Stext: tl.constexpr,
    stride_in_b: tl.constexpr, stride_in_t: tl.constexpr, stride_in_h: tl.constexpr,
    stride_e_b: tl.constexpr, stride_e_s: tl.constexpr, stride_e_h: tl.constexpr,
    stride_h_b: tl.constexpr, stride_h_s: tl.constexpr, stride_h_h: tl.constexpr,
):
    b = tl.program_id(0)
    if b >= B:
        return

    h_offsets = tl.arange(0, H)
    mask_h = h_offsets < H

    # Copy encoder part: in[b, :Stext, :]
    for s in range(0, Stext):
        in_row_ptrs = in_ptr + b * stride_in_b + s * stride_in_t + h_offsets * stride_in_h
        in_vals = tl.load(in_row_ptrs, mask=mask_h, other=0.0)
        e_row_ptrs = eout_ptr + b * stride_e_b + s * stride_e_s + h_offsets * stride_e_h
        tl.store(e_row_ptrs, in_vals, mask=mask_h)

    # Copy hidden part: in[b, Stext:, :]
    for s in range(Stext, T):
        in_row_ptrs = in_ptr + b * stride_in_b + s * stride_in_t + h_offsets * stride_in_h
        in_vals = tl.load(in_row_ptrs, mask=mask_h, other=0.0)
        h_row_ptrs = hout_ptr + b * stride_h_b + (s - Stext) * stride_h_s + h_offsets * stride_h_h
        tl.store(h_row_ptrs, in_vals, mask=mask_h)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, Simg, H]
        encoder_hidden_states: torch.Tensor,  # [B, Stext, H]
        process_weight: torch.Tensor,         # [H, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension in Triton.
        - Apply linear projection with process_weight.T in Triton (batched matvec per (b, t)).
        - Split the result back into (encoder part, hidden part) using Triton row-wise copies.
        """
        # Ensure CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA device"
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        Simg = hidden_states.shape[1]
        Stext = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H and process_weight.shape[0] == H and process_weight.shape[1] == H, "Hidden dim mismatch"

        T = Stext + Simg

        # 1) Concatenate along sequence dimension: out_cat [B, T, H]
        out_cat = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)
        grid_cat = (B, T)
        concat_by_row_kernel[grid_cat](
            encoder_hidden_states, hidden_states, out_cat,
            B=B, Stext=Stext, Simg=Simg, H=H,
            stride_e_b=encoder_hidden_states.stride(0), stride_e_t=encoder_hidden_states.stride(1), stride_e_h=encoder_hidden_states.stride(2),
            stride_h_b=hidden_states.stride(0), stride_h_s=hidden_states.stride(1), stride_h_h=hidden_states.stride(2),
            stride_out_b=out_cat.stride(0), stride_out_t=out_cat.stride(1), stride_out_h=out_cat.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Apply linear projection: out [B, T, H] = out_cat @ process_weight.T
        # Transpose weight to [H, H] (contiguous)
        W_T = process_weight.transpose(0, 1).contiguous()  # [H, H]

        out = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)
        grid_gemm = (B, T)
        matvec_row_kernel[grid_gemm](
            out_cat, W_T, out,
            B=B, T=T, H=H,
            stride_in_b=out_cat.stride(0), stride_in_t=out_cat.stride(1), stride_in_h=out_cat.stride(2),
            stride_w_b=W_T.stride(0), stride_w_k=W_T.stride(1), stride_w_h=W_T.stride(2),
            stride_out_b=out.stride(0), stride_out_t=out.stride(1), stride_out_h=out.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Split into encoder and hidden parts: [B, Stext, H] and [B, Simg, H]
        processed_encoder = torch.empty((B, Stext, H), dtype=torch.float32, device=hidden_states.device)
        processed_hidden = torch.empty((B, Simg, H), dtype=torch.float32, device=hidden_states.device)

        grid_split = (B,)
        split_rows_kernel[grid_split](
            out, processed_encoder, processed_hidden,
            B=B, T=T, H=H, Stext=Stext,
            stride_in_b=out.stride(0), stride_in_t=out.stride(1), stride_in_h=out.stride(2),
            stride_e_b=processed_encoder.stride(0), stride_e_s=processed_encoder.stride(1), stride_e_h=processed_encoder.stride(2),
            stride_h_b=processed_hidden.stride(0), stride_h_s=processed_hidden.stride(1), stride_h_h=processed_hidden.stride(2),
            num_warps=1, num_stages=1,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
