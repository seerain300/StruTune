import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_split_kernel(
    encoder_hidden_states,  # [B, T, H]
    hidden_states,          # [B, I, H]
    process_weight,         # [H, H]  (original uses process_weight.T)
    processed_concat,       # [B, T+I, H] output
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    H: tl.constexpr,
    total_seq: tl.constexpr,  # T + I
    stride_e_n, stride_e_s, stride_e_h,
    stride_h_n, stride_h_s, stride_h_h,
    stride_w_h, stride_w_k,
    stride_out_n, stride_out_s, stride_out_h,
    tiles_h: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # program ids
    n = tl.program_id(0)  # batch
    s = tl.program_id(1)  # sequence position in [0, total_seq)

    # decide source: if s < T, take from encoder; else, take from hidden at s - T
    # load input row (length H) from appropriate source
    # Build input vector 'x' of length H for this s
    # Use masked loads for k-dimension when tail
    # We'll load in tiles of BLOCK_K and accumulate across K (which equals H in this problem)
    # Note: K == H, so we iterate over k = 0..H-1 with BLOCK_K tiles.

    # Initialize output vector
    out_vec = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Loop over K tiles
    # K == H, but we keep a generic loop for robustness
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < H

        # Load weight tile W[:, k_offsets] -> shape [BLOCK_K, BLOCK_H]
        w_ptrs = process_weight + (k_offsets[:, None] * stride_w_k + tl.arange(0, BLOCK_H)[None, :] * stride_w_h)
        # Since process_weight is [H, H], we access weight[k, h] = process_weight[k, h]
        # For masked loads, h must be in [0, H); k_offsets are valid by k_mask
        w_tile = tl.load(w_ptrs, mask=k_mask[:, None], other=0.0)  # [BLOCK_K, BLOCK_H]

        # Determine source tensor and load corresponding x_vec tile
        # For encoder: ptrs_e = encoder[n, s, k_offsets]
        # For hidden: ptrs_h = hidden[n, s - T, k_offsets]
        x_tile = tl.zeros([BLOCK_K], dtype=tl.float32)
        if s < T:
            e_ptrs = encoder_hidden_states + n * stride_e_n + s * stride_e_s + k_offsets * stride_e_h
            x_tile = tl.load(e_ptrs, mask=k_mask, other=0.0)
        else:
            h_ptrs = hidden_states + n * stride_h_n + (s - T) * stride_h_s + k_offsets * stride_h_h
            x_tile = tl.load(h_ptrs, mask=k_mask, other=0.0)

        # Accumulate: out_vec += sum_k (x_tile[k] * w_tile[k, :])
        # Implement dot product for this tile: [BLOCK_H]
        # We'll do a manual sum across K for each H column
        for kk in range(BLOCK_K):
            # scale = x_tile[kk] if kk < H else 0
            scale = x_tile[kk]  # kk < H guaranteed by k_mask handling
            # Multiply w_tile[kk, :] by scale and add to out_vec
            # w_tile[kk, :] is a vector of length BLOCK_H
            out_vec += w_tile[kk, :] * scale

    # Store out_vec to processed_concat[n, s, :]
    # We write only for valid s in [0, total_seq). This grid ensures s < total_seq always.
    out_ptrs = processed_concat + n * stride_out_n + s * stride_out_s + tl.arange(0, BLOCK_H) * stride_out_h
    h_offsets = tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H
    tl.store(out_ptrs, out_vec, mask=h_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure dtype and contiguity
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B, T, H = encoder_hidden_states.shape
        B2, I, H2 = hidden_states.shape
        assert B == B2, "Batch size mismatch"
        assert H == H2, "Hidden dim mismatch"
        total_seq = T + I

        # Allocate output [B, total_seq, H]
        processed_concat = torch.empty((B, total_seq, H), device=hidden_states.device, dtype=torch.float32)

        # Strides
        stride_e_n, stride_e_s, stride_e_h = encoder_hidden_states.stride()
        stride_h_n, stride_h_s, stride_h_h = hidden_states.stride()
        stride_w_h, stride_w_k = process_weight.stride()  # [H, H]
        stride_out_n, stride_out_s, stride_out_h = processed_concat.stride()

        # Tiling parameters
        BLOCK_H = 128  # works well for H=128/256; masks handle tails like H=293 (tiles_h=3)
        BLOCK_K = 64   # iterate across K in tiles of 64

        tiles_h = (H + BLOCK_H - 1) // BLOCK_H
        grid = (B, total_seq)

        # Launch Triton kernel
        concat_linear_split_kernel[grid](
            encoder_hidden_states, hidden_states, process_weight, processed_concat,
            B, T, I, H, total_seq,
            stride_e_n, stride_e_s, stride_e_h,
            stride_h_n, stride_h_s, stride_h_h,
            stride_w_h, stride_w_k,
            stride_out_n, stride_out_s, stride_out_h,
            tiles_h,
            BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
            num_warps=4,  # balanced default
            num_stages=2,
        )

        # Split outputs
        processed_encoder = processed_concat[:, :T, :]
        processed_hidden = processed_concat[:, T:, :]

        return processed_encoder, processed_hidden