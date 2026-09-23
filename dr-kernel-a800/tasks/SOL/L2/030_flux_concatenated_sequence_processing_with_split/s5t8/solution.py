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

    # Bounds check
    if b >= B or t >= (Stext + Simg):
        return

    # Choose source based on t
    is_text = t < Stext

    # Vector over H
    h_idx = tl.arange(0, H)

    if is_text:
        # Load encoder row
        e_ptrs = encoder_ptr + b * stride_e_b + t * stride_e_t + h_idx * stride_e_h
        e_vec = tl.load(e_ptrs)
        out_ptrs = out_ptr + b * stride_out_b + t * stride_out_t + h_idx * stride_out_h
        tl.store(out_ptrs, e_vec)
    else:
        # Load hidden row (shift by Stext)
        h_ptrs = hidden_ptr + b * stride_h_b + (t - Stext) * stride_h_t + h_idx * stride_h_h
        h_vec = tl.load(h_ptrs)
        out_ptrs = out_ptr + b * stride_out_b + t * stride_out_t + h_idx * stride_out_h
        tl.store(out_ptrs, h_vec)


@triton.jit
def matvec_row_kernel(
    in_mat_ptr,     # *f32, [B, T, H]
    weightT_ptr,    # *f32, [H, H], process_weight.T
    out_ptr,        # *f32, [B, T, H]
    B, T, H,
    stride_in_b, stride_in_t, stride_in_h,
    stride_w_k, stride_w_h,
    stride_out_b, stride_out_t, stride_out_h,
):
    # 2D grid: (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)
    if b >= B or t >= T:
        return

    # Output vector over H
    h_idx = tl.arange(0, H)

    # Compute out[b, t, h_idx] = sum_k in_mat[b, t, k] * weightT[k, h_idx]
    # We'll compute each h separately to keep it simple and robust.
    # out pointer for this (b, t)
    out_ptrs = out_ptr + b * stride_out_b + t * stride_out_t + h_idx * stride_out_h
    acc = tl.zeros((H,), dtype=tl.float32)

    # Loop over k in 0..H-1
    # Note: This loop is over a compile-time H; Triton supports such scalar loops.
    for k in range(0, H):
        in_ptrs = in_mat_ptr + b * stride_in_b + t * stride_in_t + k * stride_in_h
        in_val = tl.load(in_ptrs)  # scalar
        # Load weightT[k, h_idx]
        w_ptrs = weightT_ptr + k * stride_w_k + h_idx * stride_w_h
        w_vec = tl.load(w_ptrs)     # vector over H
        acc += in_val * w_vec

    tl.store(out_ptrs, acc)


@triton.jit
def split_rows_kernel(
    in_ptr,         # *f32, [B, T, H]
    out_e_ptr,      # *f32, [B, Stext, H]
    out_i_ptr,      # *f32, [B, Simg, H]
    B, T, Stext, H,
    stride_in_b, stride_in_t, stride_in_h,
    stride_e_b, stride_e_t, stride_e_h,
    stride_i_b, stride_i_t, stride_i_h,
):
    # 1D grid over batch
    b = tl.program_id(0)
    if b >= B:
        return

    h_idx = tl.arange(0, H)

    # Copy encoder part: t in [0, Stext)
    for t in range(0, Stext):
        in_ptrs = in_ptr + b * stride_in_b + t * stride_in_t + h_idx * stride_in_h
        vals = tl.load(in_ptrs)
        e_ptrs = out_e_ptr + b * stride_e_b + t * stride_e_t + h_idx * stride_e_h
        tl.store(e_ptrs, vals)

    # Copy hidden part: t in [Stext, T)
    for t in range(Stext, T):
        in_ptrs = in_ptr + b * stride_in_b + t * stride_in_t + h_idx * stride_in_h
        vals = tl.load(in_ptrs)
        i_ptrs = out_i_ptr + b * stride_i_b + (t - Stext) * stride_i_t + h_idx * stride_i_h
        tl.store(i_ptrs, vals)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dim in Triton.
        2) Apply linear projection (process_weight.T) to each row of the concatenated tensor in Triton.
        3) Split the result back into encoder and hidden parts (Triton kernel).
        """
        # Ensure CUDA and contiguity
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA device."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        H = hidden_states.shape[2]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        T = Stext + Simg

        # 1) Concatenation
        out_cat = torch.empty((B, T, H), device=hidden_states.device, dtype=torch.float32)

        grid_concat = (B, T)
        concat_by_row_kernel[grid_concat](
            encoder_hidden_states, hidden_states, out_cat,
            B, Stext, Simg, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
        )

        # 2) Linear projection: out[b, t, :] = process_weight.T @ out_cat[b, t, :]
        # process_weight.T is [H, H]; ensure contiguous
        W_T = process_weight.transpose(0, 1).contiguous()

        out = torch.empty((B, T, H), device=hidden_states.device, dtype=torch.float32)

        grid_gemm = (B, T)
        matvec_row_kernel[grid_gemm](
            out_cat, W_T, out,
            B, T, H,
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            W_T.stride(0), W_T.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
        )

        # 3) Split into encoder and hidden parts
        processed_encoder = torch.empty((B, Stext, H), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, Simg, H), device=hidden_states.device, dtype=torch.float32)

        grid_split = (B,)
        split_rows_kernel[grid_split](
            out, processed_encoder, processed_hidden,
            B, T, Stext, H,
            out.stride(0), out.stride(1), out.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
