import torch
import triton
import triton.language as tl


# Kernel 1: Concatenate along the sequence dimension using Triton.
# Input:
#   encoder_ptr: [B, Stext, H]
#   hidden_ptr: [B, Simg, H]
# Output:
#   out_ptr: [B, T, H], where T = Stext + Simg
@triton.jit
def concat_seq_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, Stext, Simg, H,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_t, stride_h_h,
    stride_o_b, stride_o_t, stride_o_h,
    T,  # T = Stext + Simg, runtime
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
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


# Kernel 2: Batched GEMV using 2D tiling for better performance:
# out[b, t, n] = sum_k concat[b, t, k] * weight_T[k, n]
# We compute a tile of size [BLOCK_K, BLOCK_N] per program for a single (b, t).
@triton.jit
def batched_gemv_kernel_tiled(
    concat_ptr, weightT_ptr, out_ptr,
    B, T, H_in, H_out,
    stride_c_b, stride_c_t, stride_c_h,
    stride_w_k, stride_w_n,  # weight_T strides: dim-0 is k, dim-1 is n
    stride_o_b, stride_o_t, stride_o_h,
    BLOCK_K: tl.constexpr,  # tile along input H_in
    BLOCK_N: tl.constexpr,  # tile along output H_out
):
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

        # Load input vector chunk: concat[b, t, k_offsets] -> shape [BLOCK_K]
        in_ptrs = concat_ptr + b * stride_c_b + t * stride_c_t + k_offsets * stride_c_h
        in_chunk = tl.load(in_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load weight block: weightT[k_offsets, n_offsets] -> shape [BLOCK_K, BLOCK_N]
        w_ptrs = weightT_ptr + k_offsets[:, None] * stride_w_k + n_offsets[None, :] * stride_w_n
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate: sum over k of in_chunk[k] * w_block[k, :]
        acc += tl.sum(w_block * in_chunk[:, None], axis=0)

        k0 += BLOCK_K

    # Store results to out[b, t, n_offsets]
    out_ptrs = out_ptr + b * stride_o_b + t * stride_o_t + n_offsets * stride_o_h
    tl.store(out_ptrs, acc, mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - Concatenates encoder_hidden_states and hidden_states along the sequence dimension in Triton.
        - Applies linear projection with process_weight.T using a 2D-tiled Triton GEMV kernel.
        - Splits the result back into encoder and hidden streams.

        Assumes all inputs are CUDA tensors of dtype float32 and contiguous.
        """
        # Ensure device, dtype, contiguity
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors"
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.size(0)
        Stext = encoder_hidden_states.size(1)
        Simg = hidden_states.size(1)
        H = hidden_states.size(2)
        H_in = H
        T = Stext + Simg

        # 1) Concatenate in Triton: [B, T, H]
        concatenated = torch.empty((B, T, H), device=hidden_states.device, dtype=hidden_states.dtype)
        grid_concat = (B, T)
        concat_seq_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, Stext, Simg, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            T,
            num_warps=1,  # simple copy op
        )

        # 2) Prepare output and weight_T [H_in, H_in]
        out = torch.empty((B, T, H_in), device=hidden_states.device, dtype=hidden_states.dtype)
        weight_T = process_weight.t().contiguous()  # [H, H]

        # 3) Launch tiled GEMV kernel
        # Use larger tiles and more warps to improve throughput.
        BLOCK_K = 256
        BLOCK_N = 256
        grid_gemv = (B, T, triton.cdiv(H_in, BLOCK_N))
        batched_gemv_kernel_tiled[grid_gemv](
            concatenated, weight_T, out,
            B, T, H_in, H_in,  # H_out == H_in
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            weight_T.stride(0), weight_T.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=8,
            num_stages=2,
        )

        # 4) Split back into encoder and hidden streams
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
