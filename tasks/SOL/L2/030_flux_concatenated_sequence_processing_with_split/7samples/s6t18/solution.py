import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_cat_with_weightT_kernel(
    encoder_ptr,  # [B, T, H], float32
    hidden_ptr,   # [B, I, H], float32
    weight_ptr,   # [H, H], float32
    out_ptr,      # [B, T+I, H], float32
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    stride_e_n: tl.int32, stride_e_s: tl.int32, stride_e_h: tl.int32,
    stride_h_n: tl.int32, stride_h_s: tl.int32, stride_h_h: tl.int32,
    stride_w_h: tl.int32, stride_w_k: tl.int32,
    stride_out_n: tl.int32, stride_out_s: tl.int32, stride_out_h: tl.int32,
    BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # Grid: (B, T+I). Each program computes one output row (n, s).
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    total_seq = T + I

    # Prepare output vector for this (n, s)
    out_row_ptr = out_ptr + pid_n * stride_out_n + pid_s * stride_out_s

    # Determine if this s belongs to encoder or hidden stream
    is_encoder = pid_s < T

    # Accumulator for output vector
    # We will write H elements; use a vector of size BLOCK_H, but load/store with masks.
    # To avoid overwriting or OOB, we iterate over H tiles explicitly.
    # However, we still need a base accumulator vector for the current H tile.
    # Since we don't know the tile offset, we compute per-tile accumulation and store immediately.

    # Loop over output H in tiles of size BLOCK_H, starting from h0 = 0
    for h0 in range(0, H, BLOCK_H):
        # acc holds the computed output for this tile
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # Loop over input K (which equals H) in tiles of size BLOCK_K
        for k0 in range(0, H, BLOCK_K):
            # Load input vector slice for this (n, s). Mask for tail if H not multiple of BLOCK_K.
            if is_encoder:
                # encoder_hidden_states[n, s, k] for k in [k0, k0+BLOCK_K)
                # Pointer: base + s*stride_e_s + k*stride_e_h
                k_idx = k0 + tl.arange(0, BLOCK_K)
                k_mask = k_idx < H
                input_vec = tl.load(encoder_ptr + pid_n * stride_e_n + pid_s * stride_e_s + k_idx * stride_e_h, mask=k_mask, other=0.0)
            else:
                # hidden_states[n, s - T, k]
                s_i = pid_s - T
                k_idx = k0 + tl.arange(0, BLOCK_K)
                k_mask = k_idx < H
                input_vec = tl.load(hidden_ptr + pid_n * stride_h_n + s_i * stride_h_s + k_idx * stride_h_h, mask=k_mask, other=0.0)

            # Accumulate input_vec[:, None] * weight[k_idx, h0:h0+BLOCK_H][None, :]
            # weight is [H, H], we need weight[k, h] for k in [k0..k0+BLOCK_K), h in [h0..h0+BLOCK_H)
            h_idx = h0 + tl.arange(0, BLOCK_H)
            h_mask = h_idx < H

            # Build 2D tile pointers for weight: shape [BLOCK_K, BLOCK_H]
            # weight[k, h] => weight_ptr + k*stride_w_h + h*stride_w_k
            k_tile = k_idx[None, :]          # [1, BLOCK_K]
            h_tile = h_idx[:, None]          # [BLOCK_H, 1]
            weight_tile_ptrs = weight_ptr + k_tile * stride_w_h + h_tile * stride_w_k  # broadcast to [1, BLOCK_K, BLOCK_H]
            # Triton expects [K, H] with K as rows, H as cols. We have [1, BLOCK_K, BLOCK_H]; take first axis as K (we'll reshape).
            # Safer: build [BLOCK_K, BLOCK_H] by expanding k/h dims manually:
            # weight_tile_ptrs = weight_ptr + (k_idx[:, None] * stride_w_h) + (h_idx[None, :] * stride_w_k)
            weight_tile_ptrs = weight_ptr + (k_idx[:, None] * stride_w_h) + (h_idx[None, :] * stride_w_k)

            # Mask for weight tile: k_mask[:, None] & h_mask[None, :]
            weight_mask = k_mask[:, None] & h_mask[None, :]
            weight_tile = tl.load(weight_tile_ptrs, mask=weight_mask, other=0.0)

            # Now accumulate: for each kk in BLOCK_K, sum over BLOCK_H
            # acc[h] += input_vec[kk] * weight_tile[kk, h]
            # Implement via a small loop over kk to avoid shape ambiguity in Triton
            for kk in range(BLOCK_K):
                k_valid = k0 + kk < H
                # If k_valid, add input_vec[kk] * weight_tile[kk, :]
                # weight_tile[kk, :] is the kk-th row of the loaded [BLOCK_K, BLOCK_H] tile
                # Ensure we only add when k_valid
                # acc += input_vec[kk] * weight_tile[kk, :]
                # Triton supports elementwise ops on tensors; we can broadcast scalar:
                # However, to be precise, multiply by a mask: if k_valid, add; else 0
                # Extract kk-th row: weight_row = weight_tile[kk, :]
                weight_row = weight_tile[kk, :]
                row_mask = h_mask  # all h in this tile valid
                acc += tl.where(k_valid, input_vec[kk] * weight_row, 0.0)

        # After accumulating all K tiles, store acc to out for h in [h0, h0+BLOCK_H)
        # out[n, s, h] = acc[h - h0] for h < H, else 0
        h_idx = h0 + tl.arange(0, BLOCK_H)
        h_mask = h_idx < H
        # Store acc into out_ptr with mask h_mask
        # out_row_ptr is base for this (n, s) row; we store H elements
        tl.store(out_row_ptr + h_idx * stride_out_h, acc, mask=h_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - Concatenates encoder_hidden_states and hidden_states along sequence dim.
        - Applies linear projection via Triton GEMM: concatenated @ process_weight.T
        - Splits back into processed_encoder and processed_hidden.
        """

        # Ensure dtype and contiguity
        dtype = hidden_states.dtype
        assert process_weight.dtype == dtype, "process_weight dtype must match hidden/encoder dtype"
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton"
        # Make contiguous
        e = encoder_hidden_states.contiguous()
        h = hidden_states.contiguous()
        w = process_weight.contiguous()

        B, T, H = e.shape
        I = h.shape[1]
        total_seq = T + I

        # Allocate output [B, T+I, H]
        out = torch.empty((B, total_seq, H), dtype=dtype, device=e.device)

        # Compute strides (contiguous assumptions simplify to H stride = 1 if we ensure contiguity)
        # But we’ll use actual strides from tensors
        stride_e_n, stride_e_s, stride_e_h = e.stride(0), e.stride(1), e.stride(2)
        stride_h_n, stride_h_s, stride_h_h = h.stride(0), h.stride(1), h.stride(2)
        stride_w_h, stride_w_k = w.stride(0), w.stride(1)  # w is [H, H]
        stride_out_n, stride_out_s, stride_out_h = out.stride(0), out.stride(1), out.stride(2)

        # Launch Triton kernel: grid over (B, T+I)
        # Choose conservative block sizes to handle various H up to 4096
        BLOCK_K = 64
        BLOCK_H = 128
        grid = (B, total_seq)

        batched_matmul_cat_with_weightT_kernel[grid](
            e, h, w, out,
            B, T, I, H,
            stride_e_n, stride_e_s, stride_e_h,
            stride_h_n, stride_h_s, stride_h_h,
            stride_w_h, stride_w_k,
            stride_out_n, stride_out_s, stride_out_h,
            BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        # Split into encoder and hidden outputs
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
