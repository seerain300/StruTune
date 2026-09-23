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
    stride_w_k, stride_w_n,   # weight_T strides: k (dim-0), n (dim-1)
    stride_out_b, stride_out_t, stride_out_h,
    BLOCK_K: tl.constexpr,    # tile size for input dimension
    BLOCK_N: tl.constexpr,    # tile size for output dimension
):
    # 3D grid: (B, T, ceil_div(H_out, BLOCK_N))
    b = tl.program_id(0)
    t = tl.program_id(1)
    n_block = tl.program_id(2)

    if (b >= B) or (t >= T):
        return

    n_start = n_block * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < H_out

    # Accumulator for this (b, t) tile
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Iterate over input K in chunks
    k0 = 0
    while k0 < H_in:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H_in

        # Load input vector chunk: in[b, t, k_offsets] -> shape [BLOCK_K]
        in_ptrs = in_ptr + b * stride_in_b + t * stride_in_t + k_offsets * stride_in_h
        in_chunk = tl.load(in_ptrs, mask=mask_k, other=0.0)

        # Load weight tile: weightT[k_offsets, n_offsets] -> shape [BLOCK_K, BLOCK_N]
        w_ptrs = weightT_ptr + k_offsets[:, None] * stride_w_k + n_offsets[None, :] * stride_w_n
        w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate: acc += sum_{kk in BLOCK_K} in_chunk[kk] * w_tile[kk, :]
        # Unroll the small loop for performance (BLOCK_K is constexpr).
        for kk in range(0, BLOCK_K):
            val = in_chunk[kk]
            acc += val * w_tile[kk, :]

        k0 += BLOCK_K

    # Store results
    out_ptrs = out_ptr + b * stride_out_b + t * stride_out_t + n_offsets * stride_out_h
    tl.store(out_ptrs, acc, mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension using Triton.
        - Applies linear projection using Triton batched GEMV with tiling.
        - Splits outputs back into encoder and hidden streams via slicing (no torch computation).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Triton requires CUDA tensors"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors"

        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H_in = hidden_states.shape[2]
        H_out = process_weight.shape[0]  # process_weight is [H_in, H_out] in this usage
        T = Stext + Simg

        # Ensure contiguity
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        weight_T = process_weight.transpose(0, 1).contiguous()  # [H_in, H_out]

        # 1) Concatenate in Triton
        concatenated = torch.empty((B, T, H_in), device=hidden.device, dtype=hidden.dtype)

        stride_e_b, stride_e_t, stride_e_h = encoder.stride()
        stride_h_b, stride_h_s, stride_h_h = hidden.stride()  # s is position in hidden seq (t - Stext)
        stride_o_b, stride_o_t, stride_o_h = concatenated.stride()

        grid_concat = (B, T)
        concat_kernel[grid_concat](
            encoder, hidden, concatenated,
            B, Stext, Simg, H_in,
            stride_e_b, stride_e_t, stride_e_h,
            stride_h_b, stride_h_s, stride_h_h,
            stride_o_b, stride_o_t, stride_o_h,
            num_warps=1, num_stages=1,
        )

        # 2) Batched GEMV using tiled Triton kernel
        processed = torch.empty((B, T, H_out), device=hidden.device, dtype=hidden.dtype)

        stride_in_b, stride_in_t, stride_in_h = concatenated.stride()
        stride_w_k, stride_w_n = weight_T.stride()
        stride_out_b, stride_out_t, stride_out_h = processed.stride()

        # Tile sizes: 128 works well for typical hidden sizes up to 4096.
        BLOCK_K = 128
        BLOCK_N = 128
        grid_gemv = (B, T, triton.cdiv(H_out, BLOCK_N))
        batched_gemv_kernel_tiled[grid_gemv](
            concatenated, weight_T, processed,
            B, T, H_in, H_out,
            stride_in_b, stride_in_t, stride_in_h,
            stride_w_k, stride_w_n,
            stride_out_b, stride_out_t, stride_out_h,
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=8, num_stages=2,
        )

        # 3) Split into encoder and hidden streams (slicing only, no computation)
        processed_encoder = processed[:, :Stext, :]
        processed_hidden = processed[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
