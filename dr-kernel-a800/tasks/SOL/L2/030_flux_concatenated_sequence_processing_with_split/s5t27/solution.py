import torch
import triton
import triton.language as tl


@triton.jit
def concat_kernel(
    encoder_ptr,      # *f32, [B, Stext, H]
    hidden_ptr,       # *f32, [B, Simg, H]
    out_ptr,          # *f32, [B, T, H], T = Stext + Simg
    B: tl.constexpr,
    Stext: tl.constexpr,
    Simg: tl.constexpr,
    H: tl.constexpr,
    stride_e_b: tl.constexpr,
    stride_e_t: tl.constexpr,
    stride_e_h: tl.constexpr,
    stride_h_b: tl.constexpr,
    stride_h_s: tl.constexpr,  # s = t - Stext
    stride_h_h: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    # Vectorize over hidden dimension with mask to avoid OOB
    n = tl.arange(0, H)
    mask_h = n < H

    if t < Stext:
        src_ptr = encoder_ptr + b * stride_e_b + t * stride_e_t + n * stride_e_h
        dst_ptr = out_ptr + b * stride_o_b + t * stride_o_t + n * stride_o_h
    else:
        s = t - Stext
        src_ptr = hidden_ptr + b * stride_h_b + s * stride_h_s + n * stride_h_h
        dst_ptr = out_ptr + b * stride_o_b + t * stride_o_t + n * stride_o_h

    vals = tl.load(src_ptr, mask=mask_h, other=0.0)
    tl.store(dst_ptr, vals, mask=mask_h)


@triton.jit
def gemv_row_kernel(
    concat_ptr,       # *f32, [B, T, H_in]
    weightT_ptr,      # *f32, [H_in, H_out] (here H_in == H_out == H)
    out_ptr,          # *f32, [B, T, H_out]
    B: tl.constexpr,
    T: tl.constexpr,
    H_in: tl.constexpr,   # input hidden dimension
    H_out: tl.constexpr,  # output hidden dimension (same as H_in here)
    stride_c_b: tl.constexpr,
    stride_c_t: tl.constexpr,
    stride_c_h: tl.constexpr,
    stride_w_k: tl.constexpr,  # weight_T dim-0 (input H_in)
    stride_w_n: tl.constexpr,  # weight_T dim-1 (output H_out)
    stride_o_b: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    # Load entire input vector for this (b, t): concat[b, t, :]
    n = tl.arange(0, H_in)
    in_ptrs = concat_ptr + b * stride_c_b + t * stride_c_t + n * stride_c_h
    in_chunk = tl.load(in_ptrs, mask=n < H_in, other=0.0)  # vector [H_in]

    # Compute processed vector: weight_T @ in_chunk
    out_row_ptrs = out_ptr + b * stride_o_b + t * stride_o_t + tl.arange(0, H_out) * stride_o_h
    for n_idx in range(0, H_out):
        # Load weightT row chunk for n_idx: weightT[:, n_idx]
        k = tl.arange(0, H_in)
        w_ptrs = weightT_ptr + k * stride_w_k + n_idx * stride_w_n
        w_chunk = tl.load(w_ptrs, mask=k < H_in, other=0.0)  # vector [H_in]
        val = tl.sum(in_chunk * w_chunk, axis=0)  # scalar
        # Store scalar at position n_idx
        tl.store(out_row_ptrs + n_idx, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Concatenate along sequence dimension via Triton
        - Linear projection via Triton per (batch, token)
        - Split back via torch slicing (no compute on data)
        Returns (processed_encoder, processed_hidden)
        """
        # Triton requires CUDA tensors; assert and make contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32"

        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H = hidden_states.shape[2]
        T = Stext + Simg

        device = hidden_states.device

        # Ensure contiguity
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        W_T = process_weight.t().contiguous()  # [H, H]

        # Allocate concatenated output
        concatenated = torch.empty((B, T, H), dtype=torch.float32, device=device)

        # Launch concatenation kernel: grid (B, T)
        grid_concat = (B, T)
        concat_kernel[grid_concat](
            enc, hid, concatenated,
            B, Stext, Simg, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            hid.stride(0), Simg, hid.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            num_warps=4,
            num_stages=2,
        )

        # Allocate processed output
        processed = torch.empty((B, T, H), dtype=torch.float32, device=device)

        # Launch GEMV per (b, t): grid (B, T)
        grid_gemv = (B, T)
        gemv_row_kernel[grid_gemv](
            concatenated, W_T, processed,
            B, T, H, H,  # H_in == H_out == H
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            W_T.stride(0), W_T.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            num_warps=4,
            num_stages=2,
        )

        # Split back along sequence
        processed_encoder = processed[:, :Stext, :]
        processed_hidden = processed[:, Stext:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
