import torch
import triton
import triton.language as tl


@triton.jit
def concat_kernel(
    encoder_ptr,       # *f32, encoder_hidden_states [B, Stext, H_in]
    hidden_ptr,        # *f32, hidden_states [B, Simg, H_in]
    out_ptr,           # *f32, concatenated [B, T, H_in], T = Stext + Simg
    B: tl.constexpr,
    Stext: tl.constexpr,
    Simg: tl.constexpr,
    H_in: tl.constexpr,
    stride_e_b: tl.constexpr,
    stride_e_t: tl.constexpr,
    stride_e_h: tl.constexpr,
    stride_h_b: tl.constexpr,
    stride_h_s: tl.constexpr,   # hidden sequence index s = t - Stext
    stride_h_h: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
):
    # Grid: (B, T). Each program copies one row from encoder or hidden to concatenated.
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    if t < Stext:
        src_row = encoder_ptr + b * stride_e_b + t * stride_e_t
    else:
        s = t - Stext
        src_row = hidden_ptr + b * stride_h_b + s * stride_h_s

    dst_row = out_ptr + b * stride_o_b + t * stride_o_t

    # Copy H_in elements; H_in is constexpr for Triton unrolling.
    for i in range(0, H_in):
        val = tl.load(src_row + i * stride_e_h if t < Stext else i * stride_h_h)
        tl.store(dst_row + i * stride_o_h, val)


@triton.jit
def batched_gemv_kernel_tiled(
    in_ptr,            # *f32, concatenated [B, T, H_in]
    weightT_ptr,       # *f32, process_weight.T [H_in, H_out]
    out_ptr,           # *f32, processed [B, T, H_out]
    B, T, H_in, H_out,
    stride_in_b, stride_in_t, stride_in_h,
    stride_w_k, stride_w_n,   # weight_T strides: dim-0=k, dim-1=n
    stride_out_b, stride_out_t, stride_out_h,
    BLOCK_K: tl.constexpr,    # tile along input H_in
    BLOCK_N: tl.constexpr,    # tile along output H_out
):
    # 3D grid over (B, T, N_tiles)
    b = tl.program_id(0)
    t = tl.program_id(1)
    n_block = tl.program_id(2)
    if (b >= B) or (t >= T):
        return

    # Output offsets for this tile
    n_start = n_block * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < H_out

    # Accumulator for this (b, t) over BLOCK_N outputs
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over input dimension in chunks of BLOCK_K
    k0 = 0
    while k0 < H_in:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H_in

        # Load input chunk: in[b, t, k_offsets] -> shape [BLOCK_K]
        in_ptrs = in_ptr + b * stride_in_b + t * stride_in_t + k_offsets * stride_in_h
        in_chunk = tl.load(in_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load weight tile: weight_T[k_offsets, n_offsets] -> shape [BLOCK_K, BLOCK_N]
        w_ptrs = weightT_ptr + k_offsets[:, None] * stride_w_k + n_offsets[None, :] * stride_w_n
        w_chunk = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate: [BLOCK_K] dot [BLOCK_K, BLOCK_N] -> [BLOCK_N]
        acc += tl.dot(in_chunk, w_chunk)

        k0 += BLOCK_K

    # Store accumulated outputs
    out_ptrs = out_ptr + b * stride_out_b + t * stride_out_t + n_offsets * stride_out_h
    tl.store(out_ptrs, acc, mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run function:
        - Concatenates encoder_hidden_states and hidden_states along sequence dim in Triton.
        - Applies linear projection (no bias) using a tiled Triton GEMV.
        - Splits results back into encoder and image streams.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All inputs must be CUDA tensors."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, \
            "Use float32 tensors."

        B = hidden_states.shape[0]
        H_in = hidden_states.shape[2]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        T = Stext + Simg
        H_out = process_weight.shape[1]  # process_weight is [H_in, H_out]

        # Ensure contiguous
        e = encoder_hidden_states.contiguous()     # [B, Stext, H_in]
        h = hidden_states.contiguous()             # [B, Simg, H_in]
        w = process_weight.contiguous()            # [H_in, H_out]

        # Allocate concatenated [B, T, H_in]
        concatenated = torch.empty((B, T, H_in), device=e.device, dtype=torch.float32)

        # Launch concat kernel: grid (B, T)
        grid_concat = (B, T)
        concat_kernel[grid_concat](
            e, h, concatenated,
            B=B, Stext=Stext, Simg=Simg, H_in=H_in,
            stride_e_b=e.stride(0), stride_e_t=e.stride(1), stride_e_h=e.stride(2),
            stride_h_b=h.stride(0), stride_h_s=h.stride(1), stride_h_h=h.stride(2),
            stride_o_b=concatenated.stride(0), stride_o_t=concatenated.stride(1), stride_o_h=concatenated.stride(2),
            num_warps=1, num_stages=1,
        )

        # Allocate output processed [B, T, H_out]
        processed = torch.empty((B, T, H_out), device=concatenated.device, dtype=torch.float32)

        # Tiling parameters
        BLOCK_N = 128
        BLOCK_K = 128
        n_tiles = (H_out + BLOCK_N - 1) // BLOCK_N  # number of tiles along output dim

        # Launch tiled GEMV kernel: grid (B, T, n_tiles)
        grid_gemv = (B, T, n_tiles)
        batched_gemv_kernel_tiled[grid_gemv](
            concatenated, w, processed,
            B, T, H_in, H_out,
            stride_in_b=concatenated.stride(0), stride_in_t=concatenated.stride(1), stride_in_h=concatenated.stride(2),
            stride_w_k=w.stride(0), stride_w_n=w.stride(1),
            stride_out_b=processed.stride(0), stride_out_t=processed.stride(1), stride_out_h=processed.stride(2),
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # Split back into streams
        processed_encoder = processed[:, :Stext, :]
        processed_hidden = processed[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
