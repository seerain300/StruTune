import torch
import triton
import triton.language as tl

@triton.jit
def matvec_fused_tokens_kernel(
    encoder_ptr,      # *f32 [B, T, H]
    image_ptr,        # *f32 [B, I, H]
    weight_ptr,       # *f32 [H, H]
    out_ptr,          # *f32 [B, T+I, H]
    B: tl.int32,
    I: tl.int32,
    T: tl.int32,
    H: tl.int32,
    # strides in elements
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    image_stride_b, image_stride_i, image_stride_h,
    weight_stride_w, weight_stride_k,   # weight is [H, H]
    out_stride_b, out_stride_t, out_stride_h,
    # meta-parameters
    BLOCK_T: tl.constexpr,   # number of tokens per program
    BLOCK_H: tl.constexpr,   # hidden tile size
    BLOCK_K: tl.constexpr,   # input tile size
):
    # program ids
    pid_b = tl.program_id(0)         # batch
    pid_t_tile = tl.program_id(1)    # token tile
    pid_h_tile = tl.program_id(2)    # hidden tile

    # compute token offsets for this program
    t_start = pid_t_tile * BLOCK_T
    # vector of token indices within this tile
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    total_tokens = T + I
    # mask for valid tokens
    mask_tokens = t_offsets < total_tokens

    # compute hidden offsets for this program
    h_start = pid_h_tile * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    # Build input matrix X of shape [BLOCK_T, BLOCK_H]:
    # For each token t in t_offsets, if t < T: X[t, :] = encoder[b, t, :], else: X[t, :] = image[b, t - T, :]
    # Initialize X as zeros
    X = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)

    # We'll fill X row by row using masked loads
    for t_idx in range(0, BLOCK_T):
        t = t_offsets[t_idx]
        valid_token = mask_tokens[t_idx]  # scalar bool
        # Determine source: encoder if t < T, else image at t - T
        # Compute pointers for the chosen source
        # Use tl.where to form a pointer; Triton supports elementwise ops
        src_from_encoder = t < T
        # If src_from_encoder, use encoder[b, t, :], else use image[b, t - T, :]
        # Note: when src_from_encoder is False and t >= T, t - T is in [0, I)
        # Compute base pointer offsets for this row
        # For encoder: base = b*encoder_stride_b + t*encoder_stride_t
        # For image: base = b*image_stride_b + (t - T)*image_stride_i
        # We'll construct these using tl.where
        # offset components
        b_term = pid_b * encoder_stride_b  # same for both, since encoder_stride_b == image_stride_b for contiguous layout
        enc_row_ptr = encoder_ptr + b_term + (t * encoder_stride_t)
        # image index idx_i = t - T if valid else 0
        idx_i = tl.where(t < T, 0, t - T)  # if t < T, idx_i=0; else idx_i=t-T
        img_row_ptr = image_ptr + b_term + (idx_i * image_stride_i)

        # Build vector of addresses for H for this token:
        # For each h in BLOCK_H, load from enc_row_ptr + h*encoder_stride_h or img_row_ptr + h*image_stride_h
        # But since t is invalid if mask_tokens[t_idx] is False, we guard loads by valid_token
        # We'll load a [BLOCK_H] vector from each source and select using valid_token
        # Create h_vec for addressing
        h_vec = h_offsets * 0 + tl.arange(0, BLOCK_H)  # trick: h_vec = h_offsets (both are [BLOCK_H])
        # Build address vectors
        enc_addrs = enc_row_ptr + (h_vec * encoder_stride_h)
        img_addrs = img_row_ptr + (h_vec * image_stride_h)
        # Load with masks: when valid_token is True, use enc_addrs; else use zero (masked)
        # Triton doesn't support dynamic selection per element cleanly here; instead, compute based on valid_token scalar
        # If valid_token is True, load enc; else load zeros.
        # We implement by computing two loads and selecting via scalar mask:
        # However, Triton requires vector masks, so we use a dummy mask with valid_token broadcast:
        # Compute mask for loads
        # If valid_token: mask_load = mask_h; else mask_load = ~mask_h (this will be all False, but masked loads won't read)
        # In practice, we can just mask by valid_token scalar across the entire vector.
        # A common trick: construct mask based on valid_token scalar by broadcasting (valid_token & mask_h)
        # For masked loads, 'other' can be 0.0 regardless.
        mask_load = valid_token & mask_h

        # Load enc or zeros
        x_enc = tl.load(enc_addrs, mask=mask_load, other=0.0)
        # If valid_token False, we must set zeros for this row; otherwise x_enc is correct.
        # We'll assign X[t_idx, :] = x_enc for all rows; when valid_token False, x_enc is all zeros.
        # So directly assign:
        X[t_idx, :] = x_enc

    # Now compute output partials for this hidden tile:
    # Accumulator [BLOCK_T, BLOCK_H]
    out_acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)

    # Loop over K in tiles and accumulate
    for k_start in range(0, H, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # Load weight block W_block of shape [BLOCK_H, BLOCK_K]
        # For h in h_offsets and k in k_offsets: W[h, k]
        # Build address grid: w_addrs = weight_ptr + h_offsets[:, None] * weight_stride_w + k_offsets[None, :] * weight_stride_k
        w_addrs = weight_ptr + (h_offsets[:, None] * weight_stride_w) + (k_offsets[None, :] * weight_stride_k)
        W_block = tl.load(w_addrs, mask=(mask_h[:, None] & mask_k[None, :]), other=0.0)  # [BLOCK_H, BLOCK_K]

        # Load input chunks X_chunk of shape [BLOCK_K] for this tile
        # For k in k_offsets, X[:, k]
        # We need to gather X[:, k] across BLOCK_K. Construct pointer per k:
        X_chunk = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for kk in range(0, BLOCK_K):
            k = k_offsets[kk]
            valid_k = mask_k[kk]
            # For each row t, X[t, k] = X[t, :] at position k. We can form addresses as:
            # X_addr_row = encoder_ptr + b_term + (t_offsets * encoder_stride_t) + k * encoder_stride_h
            # But since X was already constructed as [BLOCK_T, BLOCK_H], we can simply index X[:, k] with masking.
            # Here we need to extract column; Triton supports indexing into tensors, but simpler is to compute per k from original X via broadcasting.
            # Better approach: recompute X_chunk[k] using masked loads from encoder/image as we did for full H, but for a single k.
            # To avoid recomputation, we rely on the fact that X is precomputed; thus we can load X[:, k] directly.
            # However, to ensure correctness without recomputation, we compute X_chunk[k] per k using masked load from either encoder or image.
            # This reintroduces some work but keeps code simple and robust.

            # Build scalar t loop to load per-k elements from X
            # Alternatively, since X was constructed as [BLOCK_T, BLOCK_H], we can gather columns via broadcasting:
            # We do this by constructing per-t pointer for kth column; but Triton doesn't easily allow that.
            # Instead, we recompute X_chunk[k] by loading from encoder or image:
            t_scan = 0
            while t_scan < BLOCK_T:
                t = t_offsets[t_scan]
                valid_token = mask_tokens[t_scan]
                src_from_encoder = t < T
                idx_i = tl.where(t < T, 0, t - T)
                # Compute row pointers
                enc_row_ptr_k = encoder_ptr + b_term + (t * encoder_stride_t)
                img_row_ptr_k = image_ptr + b_term + (idx_i * image_stride_i)
                # Address for this (t, k)
                addr_t_k = tl.where(src_from_encoder, enc_row_ptr_k + k * encoder_stride_h, img_row_ptr_k + k * image_stride_h)
                # Load with mask; since k is within H, we only need token validity
                x_val = tl.load(addr_t_k, mask=valid_token, other=0.0)
                # Assign to X_chunk[k]
                X_chunk[kk] = x_val
                t_scan += 1

            # Then we continue accumulation:
            # out_acc[:, :] += sum over kk of W_block[:, kk] * X_chunk[kk]
            # Note: X_chunk is [BLOCK_K]; we need to broadcast along hidden tile.
            # Compute contribution: W_block[:, kk] * X_chunk[kk] -> broadcast to [BLOCK_T, BLOCK_H]
            # Here we exploit that X was constructed previously; to avoid complexity, we recompute X_chunk per k via loads and add to out_acc directly.
            # Simpler and correct: we reload needed elements from encoder/image per k for accumulation.
            # This is acceptable for robustness and correctness. The earlier simpler kernel did this, and it was correct.

        # Now out_acc += sum over kk of W_block[:, kk] * X_chunk[kk] broadcast over rows
        # We cannot directly multiply W_block with X_chunk without broadcasting. So we add each kk contribution manually:
        # For each kk in BLOCK_K:
        # contrib = W_block[:, kk][:, None] * X_chunk[kk][None, :]
        # out_acc += contrib
        # Implement this loop:
        for kk in range(0, BLOCK_K):
            k = k_offsets[kk]
            valid_k = mask_k[kk]
            # If invalid k, skip (X_chunk[kk] is 0 by construction)
            # Gather W_col = W_block[:, kk] for valid_k
            W_col = W_block[:, kk]  # [BLOCK_H]
            # Multiply by X_chunk[kk] broadcast over hidden tile
            # X_chunk[kk] is a scalar; Triton will broadcast correctly
            contrib = W_col[:, None] * X_chunk[kk]
            out_acc += contrib

    # Store the accumulated results to out[b, t_offsets, h_offsets]
    # out[b, t, h] address = out_ptr + b*out_stride_b + t*out_stride_t + h*out_stride_h
    # We store a [BLOCK_T, BLOCK_H] tile
    # Build address matrix
    out_addrs = out_ptr + (pid_b * out_stride_b) + (t_offsets[:, None] * out_stride_t) + (h_offsets[None, :] * out_stride_h)
    # Mask for valid tokens and h
    store_mask = (mask_tokens[:, None]) & (mask_h[None, :])
    tl.store(out_addrs, out_acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version: compute processed = (cat([encoder, hidden], dim=1)) @ process_weight.T
        without materializing the concatenation, then split into encoder and hidden streams.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."

        # Ensure contiguous layout for predictable strides
        encoder = encoder_hidden_states.contiguous()  # [B, T, H]
        image = hidden_states.contiguous()           # [B, I, H]
        weight = process_weight.contiguous()         # [H, H]
        B, T, H = encoder.shape
        I = image.shape[1]

        # Allocate output [B, T+I, H] in float32
        out = torch.empty((B, T + I, H), device=encoder.device, dtype=torch.float32)

        # Launch Triton kernel with a 3D grid over (batch, token tiles, hidden tiles)
        grid = (B, triton.cdiv(T + I, 32), triton.cdiv(H, 64))  # tiles; can be tuned
        matvec_fused_tokens_kernel[grid](
            encoder, image, weight, out,
            B, I, T, H,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            image.stride(0), image.stride(1), image.stride(2),
            weight.stride(0), weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_T=32, BLOCK_H=64, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # Split outputs
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]
        return processed_encoder, processed_hidden