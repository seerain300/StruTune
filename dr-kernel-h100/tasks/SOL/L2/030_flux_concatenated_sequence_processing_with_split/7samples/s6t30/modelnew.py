import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_split_kernel(
    encoder_ptr,    # [B, T, H], float32, contiguous
    hidden_ptr,     # [B, I, H], float32, contiguous
    weight_ptr,     # [H, H], float32, contiguous
    out_ptr,        # [B, T+I, H], float32, contiguous

    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    total_seq: tl.int32,
    tiles_h: tl.int32,

    stride_e_n: tl.int32, stride_e_s: tl.int32, stride_e_h: tl.int32,
    stride_h_n: tl.int32, stride_h_s: tl.int32, stride_h_h: tl.int32,
    stride_w_h: tl.int32, stride_w_k: tl.int32,
    stride_out_n: tl.int32, stride_out_s: tl.int32, stride_out_h: tl.int32,

    BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
):
    n = tl.program_id(0)
    s = tl.program_id(1)
    tile_h = tl.program_id(2)

    # Compute output H indices for this tile
    h_off = tile_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_off < H

    # Decide source based on s: s < T comes from encoder, else from hidden
    # Note: grid ensures s in [0, total_seq), so s < T is valid.
    src_encoder = s < T

    # Pointer to input row (vector of length H)
    if src_encoder:
        in_row_ptr = encoder_ptr + n * stride_e_n + s * stride_e_s
    else:
        in_row_ptr = hidden_ptr + n * stride_h_n + (s - T) * stride_h_s

    # Accumulator for output vector
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Loop over input features K in tiles of BLOCK_K
    for k0 in range(0, H, BLOCK_K):
        k_off = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_off < H

        # Load input row segment: [BLOCK_K]
        in_vec = tl.load(in_row_ptr + k_off * stride_e_h, mask=k_mask, other=0.0)  # other rows use stride_h_h

        # Load weight segment: weight is [H, H], we need weight[k, h] -> w_ptr[k_off, h_off]
        # Note: for hidden_ptr, we use stride_h_h; for encoder, stride_e_h is used above for in_vec.
        w_tile = tl.load(
            weight_ptr + k_off[:, None] * stride_w_h + h_off[None, :] * stride_w_k,
            mask=k_mask[:, None] & h_mask[None, :],
            other=0.0,
        )  # [BLOCK_K, BLOCK_H]

        # Accumulate: acc[h] += sum_k w[k, h] * in_vec[k]
        acc += tl.sum(w_tile * in_vec[:, None], axis=0)

    # Store results into out[n, s, :]
    # out shape [B, total_seq, H] so out_ptr + n*stride_out_n + s*stride_out_s + h_off*stride_out_h
    out_row_ptr = out_ptr + n * stride_out_n + s * stride_out_s
    tl.store(out_row_ptr + h_off * stride_out_h, acc, mask=h_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of:
          concatenated = cat(encoder_hidden_states, hidden_states, dim=1)
          processed = concatenated @ process_weight.T
          processed_encoder = processed[:, :T, :]
          processed_hidden = processed[:, T:, :]
        """
        # Enforce dtype and contiguity for predictable Triton behavior
        # Original process_weight is [H, H]; we use its transpose in matmul (concatenated @ weight.T).
        # Here, weight is [H, H]; no need to transpose on host.
        B, T, H_e = encoder_hidden_states.shape
        B_h, I, H = hidden_states.shape
        assert B == B_h and H_e == H, "Encoder and hidden hidden_dim must match."

        # Ensure float32 and contiguous
        encoder = encoder_hidden_states.contiguous().to(torch.float32)
        hidden = hidden_states.contiguous().to(torch.float32)
        weight = process_weight.contiguous().to(torch.float32)  # [H, H]

        # Allocate output [B, T+I, H]
        total_seq = T + I
        processed_concat = torch.empty((B, total_seq, H), dtype=torch.float32, device=encoder.device)

        # Compute tiling parameters
        BLOCK_H = 256 if H >= 256 else 128
        BLOCK_K = 128 if H >= 128 else 64
        tiles_h = (H + BLOCK_H - 1) // BLOCK_H

        # Launch Triton kernel with 3D grid: (batch, total_seq, tiles over H)
        grid = (B, total_seq, tiles_h)
        concat_linear_split_kernel[grid](
            encoder, hidden, weight, processed_concat,
            B, T, I, H,
            total_seq,
            tiles_h,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            weight.stride(0), weight.stride(1),
            processed_concat.stride(0), processed_concat.stride(1), processed_concat.stride(2),
            BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
            num_warps=8,
            num_stages=2,
        )

        # Split outputs into encoder and hidden streams
        processed_encoder = processed_concat[:, :T, :]
        processed_hidden = processed_concat[:, T:, :]
        return processed_encoder, processed_hidden