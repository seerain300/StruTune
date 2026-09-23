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
    stride_h_s: tl.constexpr,   # s = t - Stext
    stride_h_h: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
):
    # Grid over (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    # Vector of H indices for this row
    h = tl.arange(0, H)

    # Determine source: if t < Stext, from encoder at row t, else from hidden at row t - Stext
    if t < Stext:
        src_ptr = encoder_ptr + b * stride_e_b + t * stride_e_t + h * stride_e_h
    else:
        s = t - Stext
        src_ptr = hidden_ptr + b * stride_h_b + s * stride_h_s + h * stride_h_h

    dst_ptr = out_ptr + b * stride_o_b + t * stride_o_t + h * stride_o_h

    vals = tl.load(src_ptr)
    tl.store(dst_ptr, vals)


@triton.jit
def matvec_per_bt_kernel(
    in_ptr,           # *f32, concatenated [B, T, H]
    weightT_ptr,      # *f32, process_weight.T [H, H]
    out_ptr,          # *f32, processed [B, T, H]
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,             # input/output hidden size
    BLOCK_K: tl.constexpr,       # chunk over H
    stride_i_b: tl.constexpr,
    stride_i_t: tl.constexpr,
    stride_i_h: tl.constexpr,
    stride_w_k: tl.constexpr,    # weight_T dim-0 stride (k)
    stride_w_h: tl.constexpr,    # weight_T dim-1 stride (n)
    stride_o_b: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
):
    # One program per (b, t)
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    # Accumulator for output vector
    acc = tl.zeros([H], dtype=tl.float32)

    # Loop over input dimension H in chunks of BLOCK_K
    k0 = 0
    while k0 < H:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # Load input chunk: in[b, t, k_offsets] -> shape [BLOCK_K]
        in_ptrs = in_ptr + b * stride_i_b + t * stride_i_t + k_offsets * stride_i_h
        in_chunk = tl.load(in_ptrs, mask=mask_k, other=0.0)

        # Load weight tile: weightT[k_offsets, :] -> shape [BLOCK_K]
        # For each k in chunk, accumulate dot with in_chunk[k]
        # We'll iterate over BLOCK_K and multiply by the corresponding scalar w element.
        for kk in range(0, BLOCK_K):
            kk_valid = k0 + kk < H
            # Load scalar w for this k
            w_ptr = weightT_ptr + (k0 + kk) * stride_w_k + 0 * stride_w_h
            w_val = tl.load(w_ptr) if kk_valid else 0.0
            # Multiply and accumulate
            acc += w_val * in_chunk[kk] if kk_valid else 0.0

        k0 += BLOCK_K

    # Store the accumulated vector out[b, t, :]
    o_ptr = out_ptr + b * stride_o_b + t * stride_o_t + tl.arange(0, H) * stride_o_h
    tl.store(o_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure inputs are CUDA, contiguous, and float32
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors"

        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H = hidden_states.shape[2]
        T = Stext + Simg

        # 1) Concatenate along sequence dimension in Triton
        concatenated = torch.empty((B, T, H), device=hidden_states.device, dtype=hidden_states.dtype)

        e = encoder_hidden_states.contiguous()
        h = hidden_states.contiguous()
        o = concatenated

        e_b, e_t, e_h = e.stride(0), e.stride(1), e.stride(2)
        h_b, h_s, h_h = h.stride(0), h.stride(1), h.stride(2)
        o_b, o_t, o_h = o.stride(0), o.stride(1), o.stride(2)

        grid_concat = (B, T)
        concat_kernel[grid_concat](
            e, h, o,
            B, Stext, Simg, H,
            e_b, e_t, e_h,
            h_b, h_s, h_h,
            o_b, o_t, o_h,
            num_warps=4, num_stages=2,
        )

        # 2) Apply linear projection using Triton per-(b, t) kernel
        weightT = process_weight.t().contiguous()  # [H, H]
        out = torch.empty((B, T, H), device=hidden_states.device, dtype=hidden_states.dtype)

        i = concatenated.contiguous()
        w = weightT
        o_b, o_t, o_h = out.stride(0), out.stride(1), out.stride(2)

        grid_matvec = (B, T)
        # Use BLOCK_K=128 for typical hidden sizes; loop handles any H
        matvec_per_bt_kernel[grid_matvec](
            i, w, out,
            B, T, H,
            BLOCK_K=128,
            stride_i_b=i.stride(0), stride_i_t=i.stride(1), stride_i_h=i.stride(2),
            stride_w_k=w.stride(0), stride_w_h=w.stride(1),
            stride_o_b=o_b, stride_o_t=o_t, stride_o_h=o_h,
            num_warps=4, num_stages=2,
        )

        # 3) Split back into encoder and hidden streams (metadata ops, no computation)
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
