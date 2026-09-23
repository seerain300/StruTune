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
    stride_h_s: tl.constexpr,   # index into Simg corresponds to t - Stext
    stride_h_h: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
):
    # 2D grid over (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    h = tl.arange(0, H)

    if t < Stext:
        src_ptr = encoder_ptr + b * stride_e_b + t * stride_e_t + h * stride_e_h
    else:
        src_t = t - Stext
        src_ptr = hidden_ptr + b * stride_h_b + src_t * stride_h_s + h * stride_h_h

    dst_ptr = out_ptr + b * stride_o_b + t * stride_o_t + h * stride_o_h

    vals = tl.load(src_ptr)
    tl.store(dst_ptr, vals)


@triton.jit
def batched_linear_kernel_tiled(
    concatenated_ptr,    # *f32, [B, T, H_in], where T = Stext + Simg
    weightT_ptr,         # *f32, [H_in, H_out] (process_weight transposed)
    out_ptr,             # *f32, [B, T, H_out]
    B, T, H_in, H_out,
    stride_c_b, stride_c_t, stride_c_h,
    stride_w_k, stride_w_n,  # weightT strides: dim-0=k (H_in), dim-1=n (H_out)
    stride_o_b, stride_o_t, stride_o_h,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program IDs for batch, tile along T, tile along H_out
    b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Compute tile offsets
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    n_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_t = t_offsets < T
    mask_n = n_offsets < H_out

    # Initialize accumulator for this tile: [BLOCK_T, BLOCK_H]
    acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)

    # Loop over input hidden dimension in chunks
    k0 = 0
    while k0 < H_in:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H_in

        # Load input chunk for all t in this tile: shape [BLOCK_T, BLOCK_K]
        # concatenated[b, t, k] => offset = b*stride_c_b + t*stride_c_t + k*stride_c_h
        c_ptrs = concatenated_ptr + b * stride_c_b + t_offsets[:, None] * stride_c_t + k_offsets[None, :] * stride_c_h
        load_mask = mask_t[:, None] & mask_k[None, :]
        in_chunk = tl.load(c_ptrs, mask=load_mask, other=0.0)  # [BLOCK_T, BLOCK_K]

        # Load weight tile: weightT[k, n] -> shape [BLOCK_K, BLOCK_H]
        w_ptrs = weightT_ptr + k_offsets[:, None] * stride_w_k + n_offsets[None, :] * stride_w_n
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_chunk = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_H]

        # Accumulate: acc += in_chunk @ w_chunk  (each is [BLOCK_T, BLOCK_K] and [BLOCK_K, BLOCK_H] -> [BLOCK_T, BLOCK_H])
        acc += tl.dot(in_chunk, w_chunk)

        k0 += BLOCK_K

    # Store results: out[b, t, n] for all t in tile and n in tile
    out_ptrs = out_ptr + b * stride_o_b + t_offsets[:, None] * stride_o_t + n_offsets[None, :] * stride_o_h
    store_mask = mask_t[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.

        1) Concatenate encoder_hidden_states and hidden_states along sequence dim (Triton).
        2) Apply linear projection using a Triton-tiled kernel: processed = concat @ process_weight.T
        3) Split processed back into encoder and hidden parts.
        """
        # Validate and ensure CUDA
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors"

        # Ensure contiguous inputs
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        H_in = hidden_states.shape[2]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        T = Stext + Simg
        H_out = H_in  # linear has same hidden dim in and out (no bias)

        # Step 1: Concatenate along sequence dimension using Triton
        concatenated = torch.empty((B, T, H_in), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch Triton concat kernel
        grid_concat = (B, T)
        concat_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, Stext, Simg, H_in,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            num_warps=2, num_stages=2,
        )

        # Step 2: Prepare weight_T = process_weight.T (transpose in Triton kernel usage; we pass as is)
        weight_T = process_weight.transpose(0, 1).contiguous()  # [H_in, H_out]

        # Output tensor for processed [B, T, H_out]
        out = torch.empty((B, T, H_out), device=concatenated.device, dtype=concatenated.dtype)

        # Launch Triton tiled linear kernel
        BLOCK_T = 64
        BLOCK_H = 64
        BLOCK_K = 64
        grid = (B, triton.cdiv(T, BLOCK_T), triton.cdiv(H_out, BLOCK_H))
        batched_linear_kernel_tiled[grid](
            concatenated, weight_T, out,
            B, T, H_in, H_out,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            weight_T.stride(0), weight_T.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_T=BLOCK_T, BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Step 3: Split back into encoder and hidden streams
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
