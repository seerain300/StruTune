import torch
import triton
import triton.language as tl


@triton.jit
def batched_gemv_kernel_rowtile(
    concat_ptr,      # *f32, concatenated [B, T, H_in]
    weightT_ptr,     # *f32, process_weight.T [H_in, H_out]
    out_ptr,         # *f32, processed [B, T, H_out]
    B, T, H_in, H_out,
    stride_c_b, stride_c_t, stride_c_h,   # strides for concatenated
    stride_w_k, stride_w_n,               # strides for weight_T (k, n)
    stride_o_b, stride_o_t, stride_o_h,   # strides for output
    BLOCK_H: tl.constexpr,                # tile size along H_out
    BLOCK_K: tl.constexpr,                # chunk size along H_in
):
    # Grid is (B, T, ceil_div(H_out, BLOCK_H))
    b = tl.program_id(0)
    t = tl.program_id(1)
    h_block = tl.program_id(2)
    if (b >= B) or (t >= T):
        return

    # Output tile offsets
    h_start = h_block * BLOCK_H
    n_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_n = n_offsets < H_out

    # Accumulator for this (b, t) tile
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Loop over input dimension in chunks
    k0 = 0
    while k0 < H_in:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H_in

        # Load input chunk for (b, t): concatenated[b, t, k_offsets] -> [BLOCK_K]
        in_ptrs = concat_ptr + b * stride_c_b + t * stride_c_t + k_offsets * stride_c_h
        in_chunk = tl.load(in_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load weight tile: weight_T[k_offsets, n_offsets] -> [BLOCK_K, BLOCK_H]
        w_ptrs = weightT_ptr + k_offsets[:, None] * stride_w_k + n_offsets[None, :] * stride_w_n
        w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)  # [BLOCK_K, BLOCK_H]

        # Accumulate: acc += sum_k(in_chunk[k] * w_tile[k, :])
        # Equivalent to dot: acc += tl.dot(in_chunk, w_tile, axis=0)
        acc += tl.dot(in_chunk, w_tile, axis=0)  # in_chunk: [BLOCK_K], w_tile: [BLOCK_K, BLOCK_H]

        k0 += BLOCK_K

    # Store results for this (b, t, n_offsets) tile
    out_ptrs = out_ptr + b * stride_o_b + t * stride_o_t + n_offsets * stride_o_h
    tl.store(out_ptrs, acc, mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dimension.
        2) Apply linear projection via Triton kernel: out = concatenated @ process_weight.T
        3) Split back into processed_encoder and processed_hidden.
        """
        # Ensure tensors are on CUDA and float32
        assert hidden_states.device.type == 'cuda' and encoder_hidden_states.device.type == 'cuda' and process_weight.device.type == 'cuda', \
            "All tensors must be on CUDA for Triton execution."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, \
            "This implementation supports float32 tensors."

        # Make tensors contiguous
        e = encoder_hidden_states.contiguous()  # [B, Stext, H]
        h = hidden_states.contiguous()          # [B, Simg, H]
        w = process_weight.contiguous()         # [H, H_out]

        B = e.shape[0]
        Stext = e.shape[1]
        Simg = h.shape[1]
        H_in = e.shape[2]  # hidden_dim for inputs
        H_out = w.shape[1] # output hidden_dim

        # 1) Concatenate along the sequence dimension using torch (fast and robust)
        T = Stext + Simg
        concatenated = torch.cat([e, h], dim=1)  # [B, T, H_in]

        # 2) Allocate output
        processed = torch.empty((B, T, H_out), device=concatenated.device, dtype=concatenated.dtype)

        # 3) Launch Triton kernel to compute processed = concatenated @ w.T
        BLOCK_H = 64
        BLOCK_K = 64
        grid = (B, T, triton.cdiv(H_out, BLOCK_H))

        batched_gemv_kernel_rowtile[grid](
            concatenated, w.t(), processed,
            B, T, H_in, H_out,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            w.t().stride(0), w.t().stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 4) Split back along the sequence dimension
        processed_encoder = processed[:, :Stext, :]
        processed_hidden = processed[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
