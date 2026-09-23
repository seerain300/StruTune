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
    stride_h_s: tl.constexpr,   # Simg index corresponding to t - Stext
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

    # Determine source: if t < Stext, take from encoder; else take from hidden at s = t - Stext
    h = tl.arange(0, H)  # vector of column indices
    if t < Stext:
        src_ptr = encoder_ptr + b * stride_e_b + t * stride_e_t + h * stride_e_h
    else:
        s = t - Stext
        src_ptr = hidden_ptr + b * stride_h_b + s * stride_h_s + h * stride_h_h

    dst_ptr = out_ptr + b * stride_o_b + t * stride_o_t + h * stride_o_h

    vals = tl.load(src_ptr)  # load row
    tl.store(dst_ptr, vals)  # store to output


@triton.jit
def gemv_row_tiled_kernel(
    concat_ptr,       # *f32, [B, T, H_in]
    weightT_ptr,      # *f32, [H_in, H_out] (transposed process_weight)
    out_ptr,          # *f32, [B, T, H_out]
    B: tl.constexpr,
    T: tl.constexpr,
    H_in: tl.constexpr,
    H_out: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_t: tl.constexpr,
    stride_c_h: tl.constexpr,
    stride_w_k: tl.constexpr,  # dim-0 of weight_T is k (input hidden)
    stride_w_n: tl.constexpr,  # dim-1 of weight_T is n (output hidden)
    stride_o_b: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 3D grid: (B, T, H_out_tiles)
    b = tl.program_id(0)
    t = tl.program_id(1)
    h_block = tl.program_id(2)
    if (b >= B) or (t >= T) or (h_block >= (H_out + BLOCK_H - 1) // BLOCK_H):
        return

    n_start = h_block * BLOCK_H
    n_offsets = n_start + tl.arange(0, BLOCK_H)
    mask_n = n_offsets < H_out

    # Accumulator for this (b, t) over BLOCK_H outputs
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Loop over input dimension in chunks of BLOCK_K
    for k0 in range(0, H_in, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H_in

        # Load input chunk: concat[b, t, k_offsets] -> shape [BLOCK_K]
        in_ptrs = concat_ptr + b * stride_c_b + t * stride_c_t + k_offsets * stride_c_h
        in_chunk = tl.load(in_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load weight tile: weight_T[k_offsets, n_offsets] -> shape [BLOCK_K, BLOCK_H]
        w_ptrs = weightT_ptr + k_offsets[:, None] * stride_w_k + n_offsets[None, :] * stride_w_n
        w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)  # [BLOCK_K, BLOCK_H]

        # Accumulate: acc[n] += sum_k in_chunk[k] * w_tile[k, n]
        # Use tl.dot to vectorize the accumulation across the K-chunk
        acc += tl.dot(in_chunk, w_tile)

    # Store the accumulated results to out[b, t, n_offsets]
    out_ptrs = out_ptr + b * stride_o_b + t * stride_o_t + n_offsets * stride_o_h
    tl.store(out_ptrs, acc, mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dimension (Triton).
        2) Apply linear projection via Triton GEMV: out = concat @ process_weight.T
        3) Split back into processed_encoder_hidden_states and processed_hidden_states (PyTorch slicing).
        """
        device = hidden_states.device
        assert hidden_states.device == encoder_hidden_states.device == process_weight.device, "All tensors must be on the same device"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors"
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H_in = hidden_states.shape[2]
        T = Stext + Simg
        H_out = process_weight.shape[1]  # process_weight is [H_in, H_out] after .t()

        # Allocate concatenated tensor [B, T, H_in]
        concatenated = torch.empty((B, T, H_in), device=device, dtype=torch.float32)

        # Launch concat kernel: (B, T)
        concat_kernel[(B, T)](
            encoder_hidden_states, hidden_states, concatenated,
            B, Stext, Simg, H_in,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            num_warps=2, num_stages=2,
        )

        # Transpose weight to [H_in, H_out] for easy indexing in kernel
        weight_T = process_weight  # [H_in, H_out]

        # Allocate output tensor [B, T, H_out]
        processed = torch.empty((B, T, H_out), device=device, dtype=torch.float32)

        # Launch GEMV tiled kernel: (B, T, H_out_tiles)
        BLOCK_H = 64
        BLOCK_K = 64
        H_out_tiles = (H_out + BLOCK_H - 1) // BLOCK_H
        gemv_row_tiled_kernel[(B, T, H_out_tiles)](
            concatenated, weight_T, processed,
            B, T, H_in, H_out,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            weight_T.stride(0), weight_T.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Split back along sequence dimension
        processed_encoder = processed[:, :Stext, :]
        processed_hidden = processed[:, Stext:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
