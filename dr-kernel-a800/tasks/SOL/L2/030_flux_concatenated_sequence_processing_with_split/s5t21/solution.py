import torch
import triton
import triton.language as tl


# Kernel 1: Concatenate along the sequence dimension using Triton.
# Input:
#   encoder_ptr: *f32, encoder_hidden_states [B, Stext, H]
#   hidden_ptr: *f32, hidden_states [B, Simg, H]
# Output:
#   out_ptr: *f32, concatenated [B, T, H], where T = Stext + Simg
@triton.jit
def concat_kernel(
    encoder_ptr,      # *f32
    hidden_ptr,       # *f32
    out_ptr,          # *f32
    B: tl.constexpr,
    Stext: tl.constexpr,
    Simg: tl.constexpr,
    H: tl.constexpr,
    stride_e_b: tl.constexpr,
    stride_e_t: tl.constexpr,
    stride_e_h: tl.constexpr,
    stride_h_b: tl.constexpr,
    stride_h_t: tl.constexpr,
    stride_h_h: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= (Stext + Simg)):
        return
    if t < Stext:
        src_t = t
        e = encoder_ptr + b * stride_e_b + src_t * stride_e_t
        o = out_ptr + b * stride_o_b + t * stride_o_t
        for i in range(0, H):
            val = tl.load(e + i * stride_e_h)
            tl.store(o + i * stride_o_h, val)
    else:
        src_t = t - Stext
        h = hidden_ptr + b * stride_h_b + src_t * stride_h_t
        o = out_ptr + b * stride_o_b + t * stride_o_t
        for i in range(0, H):
            val = tl.load(h + i * stride_h_h)
            tl.store(o + i * stride_o_h, val)


# Kernel 2: Batched GEMV with 2D tiling over (b, t) and output tiles using tl.dot.
# out[b, t, n] = sum_k concat[b, t, k] * weight_T[k, n]
# Grid: (B, T, ceil_div(H_out, BLOCK_N)); BLOCK_N chosen by autotune.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 64,  'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 128, 'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 64,  'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 256, 'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 128, 'BLOCK_N': 256}, num_warps=8, num_stages=2),
    ],
    key=['H_in', 'H_out'],
)
@triton.jit
def batched_gemv_kernel_tiled_dot(
    concat_ptr,         # *f32, [B, T, H_in]
    weightT_ptr,        # *f32, [H_in, H_out] (process_weight.T)
    out_ptr,            # *f32, [B, T, H_out]
    B, T, H_in, H_out,
    stride_c_b, stride_c_t, stride_c_h,
    stride_w_k, stride_w_n,  # weight strides: dim-0 = k, dim-1 = n
    stride_o_b, stride_o_t, stride_o_h,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    n_block = tl.program_id(2)

    if (b >= B) or (t >= T):
        return

    n_start = n_block * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < H_out

    # Accumulator for this (b, t) tile over outputs
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Iterate over input dimension in chunks of BLOCK_K
    for k0 in range(0, H_in, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H_in

        # Load input chunk for (b, t): shape [BLOCK_K]
        in_ptrs = concat_ptr + b * stride_c_b + t * stride_c_t + k_offsets * stride_c_h
        in_chunk = tl.load(in_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load weight chunk for outputs: shape [BLOCK_K, BLOCK_N]
        w_ptrs = weightT_ptr + k_offsets[:, None] * stride_w_k + n_offsets[None, :] * stride_w_n
        w_chunk = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate: acc[n] += sum_k in_chunk[k] * w_chunk[k, n] = tl.dot(in_chunk, w_chunk) over axis 0
        # in_chunk: [BLOCK_K, 1] by broadcasting; tl.dot([BLOCK_K, 1], [BLOCK_K, BLOCK_N]) -> [1, BLOCK_N]
        # We need [BLOCK_N], so multiply by 1.
        acc += tl.dot(in_chunk[:, None], w_chunk)[0, :]

    # Store results
    out_ptrs = out_ptr + b * stride_o_b + t * stride_o_t + n_offsets * stride_o_h
    tl.store(out_ptrs, acc, mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure device and dtype compatibility
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All tensors must be on CUDA for Triton execution."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and \
            process_weight.dtype == torch.float32, "Use float32 tensors."
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        H = hidden_states.shape[2]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        T = Stext + Simg

        # 1) Triton concatenate along sequence dimension: out_cat [B, T, H]
        out_cat = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)
        grid_concat = (B, T)
        concat_kernel[grid_concat](
            encoder_hidden_states, hidden_states, out_cat,
            B, Stext, Simg, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
        )

        # 2) Triton batched GEMV: out_cat [B, T, H] @ process_weight.T [H, H] -> out [B, T, H]
        H_in = H  # concat's hidden dim
        H_out = H  # weight's output dim (same as hidden dim)
        out = torch.empty((B, T, H_out), dtype=torch.float32, device=hidden_states.device)

        # Launch with 3D grid: (B, T, N_blocks). N_blocks depends on selected BLOCK_N.
        # Triton passes meta params from the selected autotune config; we compute grid accordingly.
        # We'll use a lambda grid function to pass meta BLOCK_N and compute N_blocks.
        def grid(meta):
            blocks_n = triton.cdiv(H_out, meta['BLOCK_N'])
            return (B, T, blocks_n)

        batched_gemv_kernel_tiled_dot[grid](
            out_cat, process_weight.transpose(0, 1), out,  # weight_T is process_weight.T
            B, T, H_in, H_out,
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            process_weight.transpose(0, 1).stride(0), process_weight.transpose(0, 1).stride(1),
            out.stride(0), out.stride(1), out.stride(2),
        )

        # 3) Split back into encoder and hidden streams using torch slicing (no compute)
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
