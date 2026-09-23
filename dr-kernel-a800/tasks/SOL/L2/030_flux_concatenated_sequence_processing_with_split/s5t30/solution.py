import torch
import triton
import triton.language as tl


# Triton kernel: concatenate encoder_hidden_states and hidden_states along the sequence dimension.
# Input:
#   encoder_ptr: [B, Stext, H]
#   hidden_ptr: [B, Simg, H]
# Output:
#   out_ptr: [B, T, H], T = Stext + Simg
@triton.jit
def concat_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, Stext, Simg, H,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_s, stride_h_h,
    stride_o_b, stride_o_t, stride_o_h,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    if t < Stext:
        src_t = t
        src_ptr = encoder_ptr + b * stride_e_b + src_t * stride_e_t
    else:
        src_t = t - Stext
        src_ptr = hidden_ptr + b * stride_h_b + src_t * stride_h_s

    dst_ptr = out_ptr + b * stride_o_b + t * stride_o_t

    # Vectorized copy across hidden dim in chunks of BLOCK_H
    for i in range(0, H, BLOCK_H):
        h_offsets = i + tl.arange(0, BLOCK_H)
        mask = h_offsets < H
        vals = tl.load(src_ptr + h_offsets * stride_e_h, mask=mask, other=0.0)
        tl.store(dst_ptr + h_offsets * stride_o_h, vals, mask=mask)


# Triton kernel: batched GEMM over output tiles. Computes:
# out[b, t, n] = sum_k C[b, t, k] * W_T[k, n]
# Grid: (B, T_tiles, N_tiles), where T_tiles = ceil_div(T, BLOCK_T), N_tiles = ceil_div(H_out, BLOCK_N)
@triton.jit
def batched_gemm_tiled_kernel(
    concat_ptr, weightT_ptr, out_ptr,
    B, T, H_in, H_out,
    stride_c_b, stride_c_t, stride_c_h,
    stride_w_k, stride_w_n,  # weight_T strides: k (rows) and n (cols)
    stride_o_b, stride_o_t, stride_o_h,
    BLOCK_T: tl.constexpr,   # tile along sequence T
    BLOCK_N: tl.constexpr,   # tile along output H_out
    BLOCK_K: tl.constexpr,   # reduction tile along input H_in
):
    # program ids
    b = tl.program_id(0)
    t_block = tl.program_id(1)
    n_block = tl.program_id(2)
    if (b >= B) or (t_block >= (T + BLOCK_T - 1) // BLOCK_T) or (n_block >= (H_out + BLOCK_N - 1) // BLOCK_N):
        return

    # offsets for this tile
    t_offsets = t_block * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_t = t_offsets < T
    mask_n = n_offsets < H_out

    # accumulator [BLOCK_T, BLOCK_N]
    acc = tl.zeros([BLOCK_T, BLOCK_N], dtype=tl.float32)

    # loop over input reduction dimension in chunks
    k0 = 0
    while k0 < H_in:
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < H_in

        # load C tile: shape [BLOCK_T, BLOCK_K]
        c_ptrs = concat_ptr + b * stride_c_b + t_offsets[:, None] * stride_c_t + k_offsets[None, :] * stride_c_h
        C_tile = tl.load(c_ptrs, mask=mask_t[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_T, BLOCK_K]

        # load W tile: shape [BLOCK_T, BLOCK_N]
        # weight_T is [H_out, H_in]; we want rows at k_offsets and cols at n_offsets
        w_ptrs = weightT_ptr + k_offsets[:, None] * stride_w_k + n_offsets[None, :] * stride_w_n  # [BLOCK_K, BLOCK_N]
        W_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)  # [BLOCK_K, BLOCK_N]

        # accumulate: acc += C_tile @ W_tile
        # Note: C_tile shape [BLOCK_T, BLOCK_K], W_tile shape [BLOCK_K, BLOCK_N]
        # We want [BLOCK_T, BLOCK_N], done by looping k in chunks or using tl.dot with a loop.
        # Since Triton's tl.dot expects 2D operands, we manually compute the outer product sum over K:
        # acc += sum over k of C_tile[:, k][:, None] * W_tile[k, :][None, :]
        # Implement via for-loop over k-chunks:
        for kk in range(0, BLOCK_K):
            k_valid = kk + k0 < H_in
            # If kk >= BLOCK_K, the mask above ensures we don't read, but k_valid guards the computation
            c_vec = C_tile[:, kk]  # [BLOCK_T]
            w_row = W_tile[kk, :]  # [BLOCK_N]
            # outer product: [BLOCK_T, BLOCK_N]
            acc += c_vec[:, None] * w_row[None, :]

        k0 += BLOCK_K

    # store results: out[b, t, n]
    out_ptrs = out_ptr + b * stride_o_b + t_offsets[:, None] * stride_o_t + n_offsets[None, :] * stride_o_h
    mask = mask_t[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenate encoder_hidden_states and hidden_states along sequence dim (Triton).
        - Compute out = process_weight.T @ concatenated using a tiled GEMM Triton kernel.
        - Split results back into encoder and image streams.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."

        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H_in = hidden_states.shape[2]
        H_out = process_weight.shape[0]  # process_weight is [H_in, H_out]
        assert process_weight.shape[1] == H_in, "process_weight second dim must match hidden_dim."

        T = Stext + Simg

        # Ensure contiguous
        e = encoder_hidden_states.contiguous()
        h = hidden_states.contiguous()
        wT = process_weight.transpose(0, 1).contiguous()  # [H_out, H_in]

        # Allocate concatenated tensor and output
        concatenated = torch.empty((B, T, H_in), device=e.device, dtype=torch.float32)
        processed = torch.empty((B, T, H_out), device=e.device, dtype=torch.float32)

        # Launch concatenation kernel: grid (B, T)
        BLOCK_H = 128
        grid_concat = (B, T)
        concat_kernel[grid_concat](
            e, h, concatenated,
            B, Stext, Simg, H_in,
            e.stride(0), e.stride(1), e.stride(2),
            h.stride(0), h.stride(1), h.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=2, num_stages=2,
        )

        # Launch tiled GEMM kernel: grid over (B, T_tiles, N_tiles)
        # Choose tile sizes for good occupancy and reduced loops
        BLOCK_T = 64
        BLOCK_N = 128
        BLOCK_K = 64

        T_tiles = (T + BLOCK_T - 1) // BLOCK_T
        N_tiles = (H_out + BLOCK_N - 1) // BLOCK_N

        grid_gemm = (B, T_tiles, N_tiles)
        batched_gemm_tiled_kernel[grid_gemm](
            concatenated, wT, processed,
            B, T, H_in, H_out,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            wT.stride(0), wT.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_T=BLOCK_T, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Split back into encoder and hidden streams
        processed_encoder = processed[:, :Stext, :]
        processed_hidden = processed[:, Stext:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
