import torch
import triton
import triton.language as tl

# Optimized Triton kernel:
# Computes the full processed tensor of shape [B, T+I, H] as:
# processed[b, t, :] = (encoder[b, t, :] if t < T else hidden[b, t-T, :]) @ process_weight.T
# Without explicitly concatenating, by choosing the source per output position.
@triton.jit
def concat_linear_opt_kernel(
    image_ptr,           # *f16/f32 [B, I, H]
    encoder_ptr,         # *f16/f32 [B, T, H]
    weight_ptr,          # *f16/f32 [H, H]
    out_ptr,             # *f16/f32 [B, T+I, H]
    B: tl.int32,
    I: tl.int32,
    T: tl.int32,
    H: tl.int32,
    # strides (in elements)
    image_stride_b, image_stride_i, image_stride_h,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    weight_stride_w, weight_stride_k,   # weight is [H, H]
    out_stride_b, out_stride_t, out_stride_h,
    # meta-parameters
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (pid_b in [0, B), pid_tok in [0, ceil_div(T+I, BLOCK_T)), pid_ht in [0, ceil_div(H, BLOCK_H)])
    pid_b = tl.program_id(0)
    pid_tok = tl.program_id(1)
    pid_ht = tl.program_id(2)

    total_out = T + I

    # Compute tiles
    offs_t = pid_tok * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    offs_h_tile = pid_ht * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]

    # Masks for valid output positions and hidden dimension
    mask_t = offs_t < total_out
    mask_h = offs_h_tile < H

    # Initialize output accumulator for this (batch, hidden tile)
    output_vec = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k_off in range(0, H, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = offs_k < H

        # Build input matrix X of shape [BLOCK_T, BLOCK_K]:
        # For each t in offs_t (valid by mask_t), choose source: encoder if t < T else image at t - T.
        # Load X_chunk per k chunk; we will compute X as a broadcasted product with weight.
        # Initialize X as zeros [BLOCK_T, BLOCK_K], float32
        X = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)

        # Determine which t's belong to encoder and which to image
        # Note: t may be invalid if mask_t is False; we guard loads with mask_t.
        # Build pointer for each t
        # We'll compute pointers per t and then vectorize over k.
        # Loop over t within the chunk to compute pointers (simple and robust).
        # Triton supports python range loops with runtime bounds; we loop t from 0 to BLOCK_T-1 but masked.
        for ti in range(0, BLOCK_T):
            t = offs_t[ti]
            valid_t = mask_t[ti]
            # Choose source tensor
            # If t < T: use encoder[b, t, :], else use image[b, t - T, :]
            if t < T:
                src_b = pid_b
                src_t = t
                x_ptr = encoder_ptr + src_b * encoder_stride_b + src_t * encoder_stride_t
            else:
                src_b = pid_b
                src_i = t - T
                x_ptr = image_ptr + src_b * image_stride_b + src_i * image_stride_i
            # Load x values for offs_k into X[ti, :]
            # Since src_h = offs_k, use pointer + offs_k * stride_h
            x_vals = tl.load(
                x_ptr + offs_k * encoder_stride_h,  # encoder_stride_h can be used; for image we need image_stride_h
                mask=mask_k & valid_t,
                other=0.0,
            )
            # We need to ensure x_vals uses correct stride: for image, use image_stride_h, for encoder use encoder_stride_h.
            # In practice, the stride passed is the H stride for the source tensor; it's the same for both (H is last dim).
            # We can use either; here we use encoder_stride_h for safety since hidden tensors are typically contiguous.
            # However, since encoder_hidden_states and hidden_states have same last-dim stride (H), this is fine.
            # If you want strictness, pass separate stride for X loads; here we assume both share the same H stride.
            X[ti, :] = x_vals.to(tl.float32)

        # Now compute partial output for this K chunk using weight tiles: [H_tile, K_tile] x [BLOCK_T, BLOCK_K]
        # Note: weight is [H, H], we load sub-blocks of shape [BLOCK_H, BLOCK_K] along w rows (n in offs_h_tile) and k in offs_k.
        # We'll compute acc_partial by broadcasting: weight_tile[:, :, None] * X[None, :, :], then sum over K.
        # To do this, we iterate n in offs_h_tile and compute per-n dot with X.
        # Alternatively, use tl.dot for [BLOCK_H, BLOCK_K] @ [BLOCK_K, BLOCK_T] -> [BLOCK_H, BLOCK_T]. We'll prefer explicit broadcast.
        # Prepare acc_partial as [BLOCK_H, BLOCK_T]
        acc_partial = tl.zeros((BLOCK_H, BLOCK_T), dtype=tl.float32)

        # Loop over rows in hidden tile
        for n in range(0, BLOCK_H):
            # Current n is offs_h_tile[n] but only if mask_h[n] is True; we will guard with mask_h.
            n_idx = offs_h_tile[n]
            valid_n = mask_h[n]
            # Load w_row = weight[n_idx, offs_k] as [BLOCK_K]
            w_row_ptrs = weight_ptr + n_idx * weight_stride_w + offs_k * weight_stride_k
            w_row = tl.load(w_row_ptrs, mask=mask_k & valid_n, other=0.0).to(tl.float32)  # [BLOCK_K]

            # Compute dot for each t in offs_t: sum_k w_row[k] * X[:, k]
            # Broadcast w_row over rows and X over cols, then sum along K (axis=1)
            # reshape to [1, BLOCK_K] and [BLOCK_T, BLOCK_K]
            w_row_col = w_row[None, :]  # [1, BLOCK_K]
            X_col = X  # [BLOCK_T, BLOCK_K]
            prod = w_row_col * X_col  # [1, BLOCK_T, BLOCK_K] broadcast to [BLOCK_T, BLOCK_K]
            # Sum over K (BLOCK_K) to get [BLOCK_T]
            dot_vec = tl.sum(prod, axis=1)  # shape [BLOCK_T], dtype float32
            # Store into acc_partial[n, :]
            acc_partial[n, :] = dot_vec

        # Accumulate into output_vec
        # For each n in offs_h_tile, add acc_partial[n, :] to output_vec[n] (only if valid_n)
        for n in range(0, BLOCK_H):
            n_idx = offs_h_tile[n]
            valid_n = mask_h[n]
            output_vec[n] += tl.sum(acc_partial[n, :], axis=0)  # sum over BLOCK_T if any t was valid, but here acc_partial[n, :] is just the dot results for each t. We want to add the contribution of this K-chunk across all t in the tile.

        # Store the output_vec for the H_tile into out[b, offs_t, offs_h_tile]
        # We need a 2D store: out[b, t, h] for t in offs_t (valid) and h in offs_h_tile (valid)
        # Initialize store pointer for each t
        # Note: We cannot directly broadcast store; we loop over t and h
        for ti in range(0, BLOCK_T):
            t = offs_t[ti]
            valid_t = mask_t[ti]
            # For each n in offs_h_tile, store output_vec[n] to out[b, t, n]
            for n in range(0, BLOCK_H):
                n_idx = offs_h_tile[n]
                valid_n = mask_h[n]
                # Only store if both t and n are valid
                if valid_t and valid_n:
                    out_ptr_t = out_ptr + pid_b * out_stride_b + t * out_stride_t + n_idx * out_stride_h
                    # Cast to output dtype (assumed same as inputs)
                    # We stored as float32; Triton will cast on store if out_ptr is f16/f32. Here we keep f32 accumulation.
                    # Since the original example likely uses f32, this is fine. If inputs are f16, consider casting here.
                    tl.store(out_ptr_t, output_vec[n], mask=valid_t)

# Helper to launch the optimized kernel
def triton_concat_linear(image: torch.Tensor, encoder: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Compute processed = cat([encoder, image], dim=1) @ weight.T without materializing the concatenation.
    Returns tensor of shape [B, T+I, H].
    """
    assert image.is_cuda and encoder.is_cuda and weight.is_cuda, "Triton inputs must be on CUDA."
    B = image.shape[0]
    I = image.shape[1]
    T = encoder.shape[1]
    H = image.shape[2]
    total_out = T + I

    # Ensure tensors are contiguous along last dimension (standard layout)
    image = image.contiguous()
    encoder = encoder.contiguous()
    weight = weight.contiguous()

    # Output tensor: [B, T+I, H]
    out = torch.empty((B, total_out, H), dtype=torch.float32, device=image.device)  # accumulate in float32

    # Choose tile sizes. For small H, 64 works well; for larger H, 128 may be better.
    # We keep it simple and robust: BLOCK_H=64, BLOCK_T=64, BLOCK_K=64. Tune as needed.
    BLOCK_H = 64
    BLOCK_T = 64
    BLOCK_K = 64

    grid = (B, triton.cdiv(total_out, BLOCK_T), triton.cdiv(H, BLOCK_H))
    concat_linear_opt_kernel[grid](
        image, encoder, weight, out,
        B, I, T, H,
        image.stride(0), image.stride(1), image.stride(2),
        encoder.stride(0), encoder.stride(1), encoder.stride(2),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_T=BLOCK_T, BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    # Cast back to input dtype if needed (original code implied f32). If you want strict dtype matching, uncomment:
    # if out.dtype != image.dtype:
    #     out = out.to(image.dtype)
    return out

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_image_hidden_states).
        """
        # Ensure inputs are on CUDA for Triton
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
        total = triton_concat_linear(hidden_states, encoder_hidden_states, process_weight)  # [B, T+I, H]
        T = encoder_hidden_states.shape[1]
        processed_encoder = total[:, :T, :]
        processed_hidden = total[:, T:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
