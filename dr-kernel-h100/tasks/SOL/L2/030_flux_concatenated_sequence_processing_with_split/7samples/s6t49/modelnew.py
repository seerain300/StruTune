import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_split_blocked_kernel(
    e, h, w, out,
    B, T, I, H, total_seq,
    stride_e_n, stride_e_s, stride_e_h,
    stride_h_n, stride_h_s, stride_h_h,
    stride_w_h, stride_w_k,  # weight is [H, H], we treat as W^T with k-index on rows, h-index on cols
    stride_out_n, stride_out_s, stride_out_h,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program ids
    pid_n = tl.program_id(0)
    pid_st = tl.program_id(1)  # tiles over sequence
    pid_ht = tl.program_id(2)  # tiles over hidden dim

    # offsets
    s_offsets = pid_st * BLOCK_S + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    h_offsets = pid_ht * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]

    # masks for bounds
    mask_s = s_offsets < total_seq
    mask_h = h_offsets < H

    # Determine input source for each s in the tile: s < T -> encoder, else hidden at s - T
    mask_encoder = (s_offsets[None, :] < T)  # shape [1, BLOCK_S]
    # Broadcast to [BLOCK_H, BLOCK_S]
    mask_encoder = mask_encoder  # we'll use it to select the input tensor below

    # Accumulator for output tile [BLOCK_H, BLOCK_S]
    acc = tl.zeros((BLOCK_H, BLOCK_S), dtype=tl.float32)

    # Loop over K (input features) in tiles
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < H

        # Build input vector for each s in the tile
        # If encoder path, input_vec = e[pid_n, s, k_offsets]; else input_vec = h[pid_n, s-T, k_offsets]
        # We'll compute via tl.where to select source.
        # Note: e shape [B, T, H], h shape [B, I, H].
        # Compute base pointers
        # e_ptr element: e + pid_n*stride_e_n + s_offsets*stride_e_s + k_offsets*stride_e_h
        # h_ptr element: h + pid_n*stride_h_n + (s_offsets-T)*stride_h_s + k_offsets*stride_h_h
        # But we need to guard by mask_encoder: if False, use h with s_offsets-T; else use e with s_offsets.
        # Compute s_src for both paths
        s_src_e = s_offsets[None, :]  # [1, BLOCK_S] -> broadcast to [BLOCK_K, BLOCK_S]
        s_src_h = s_offsets[None, :] - T  # [1, BLOCK_S]

        # We need to create pointers for both cases and select via tl.where
        # First, compute masks for s in [0, T-1] and [T, total_seq-1]
        # We'll compute input_vec as a masked load using a combined mask; however Triton requires pointers.
        # Instead, we'll compute pointers for both and then select per path. We can't branch per-s due to Triton's vectorization, so we load both and mask out invalid ones.
        # This is done by setting invalid pointers to some safe address; then masking the result. Triton allows masked tl.load.

        # Create broadcasted K and S grids for loads
        k_broadcast = k_offsets[:, None]  # [BLOCK_K, 1]
        s_broadcast = s_offsets[None, :]  # [1, BLOCK_S]

        # Compute input vector for encoder path: shape [BLOCK_K, BLOCK_S]
        # Pointer: e + pid_n*stride_e_n + s_src_e*stride_e_s + k_broadcast*stride_e_h
        e_ptrs = e + pid_n * stride_e_n + s_src_e * stride_e_s + k_broadcast * stride_e_h
        # Combine masks: valid only if s_offsets < T and mask_s
        mask_e = mask_encoder & mask_s[None, :]  # [1, BLOCK_S] -> broadcast to [BLOCK_K, BLOCK_S]
        # Load with mask: invalid elements get 0
        input_e = tl.load(e_ptrs, mask=mask_e, other=0.0)

        # Compute input vector for hidden path: shape [BLOCK_K, BLOCK_S]
        h_ptrs = h + pid_n * stride_h_n + s_src_h * stride_h_s + k_broadcast * stride_h_h
        # mask_h = s_offsets >= T and mask_s
        mask_h_path = (~mask_encoder) & mask_s[None, :]
        input_h = tl.load(h_ptrs, mask=mask_h_path, other=0.0)

        # Select the correct input vector per s: where mask_encoder True -> input_e, else input_h
        # Since mask_encoder is [1, BLOCK_S], we can broadcast over K dimension by combining with input_e's shape
        # However, Triton expects consistent shapes for tl.where. The straightforward way is to compute a selector and combine.
        # Here, we cannot directly index which branch to take per s in vector form; instead we rely on masks to zero out the wrong branch:
        # We can form a selector matrix sel_e = where(mask_encoder, 1.0, 0.0) -> [BLOCK_K, BLOCK_S], then combine.
        # But Triton’s tl.where works on tensors; to avoid mismatch, we can compute selector by broadcasting:
        # Create a [BLOCK_K, 1] selector and broadcast across S: sel_e = mask_encoder.to(tl.float32)[:, None]
        sel_e = mask_encoder.to(tl.float32)[:, None]
        sel_h = (1.0 - sel_e)
        input_vec = sel_e * input_e + sel_h * input_h  # [BLOCK_K, BLOCK_S]

        # Now, for each k in k_offsets, acc += input_vec[k, :] @ W[k, h_offsets]
        # We need to load W[k, h_offsets] as a vector [BLOCK_H]
        w_row_ptrs = w + k_offsets[:, None] * stride_w_k + h_offsets[None, :] * stride_w_h  # [BLOCK_K, BLOCK_H]
        # Mask for W: valid when k_offsets < H and h_offsets < H
        mask_w = (mask_k[:, None]) & (mask_h[None, :])
        w_tile = tl.load(w_row_ptrs, mask=mask_w, other=0.0)  # [BLOCK_K, BLOCK_H]

        # Accumulate: for each k in BLOCK_K, acc += w_tile[k, :] * input_vec[k, :]
        # We can implement this with a loop over k dimension
        for kk in range(BLOCK_K):
            w_vec = w_tile[kk, :]  # [BLOCK_H]
            val_vec = input_vec[kk, :]  # [BLOCK_S]
            # Broadcast val_vec to [BLOCK_H, BLOCK_S] and multiply, then reduce over S
            # But we want acc += w_vec[:, None] * val_vec[None, :]
            acc += w_vec[:, None] * val_vec[None, :]

    # Store the accumulated tile into out[n, s_offsets, h_offsets]
    # out pointer for a tile: out + pid_n*stride_out_n + s_offsets[None, :]*stride_out_s + h_offsets[:, None]*stride_out_h
    out_ptrs = out + pid_n * stride_out_n + s_offsets[None, :].to(tl.int32) * stride_out_s + h_offsets[:, None].to(tl.int32) * stride_out_h
    # Combine masks for store: valid only if s_offsets < total_seq and h_offsets < H
    mask_store = mask_s[None, :] & mask_h[:, None]
    tl.store(out_ptrs, acc, mask=mask_store)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        total_seq = T + I

        # Ensure dtype and contiguity
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors"
        e = encoder_hidden_states.contiguous()
        h = hidden_states.contiguous()
        w = process_weight.contiguous()  # [H, H]

        # Output tensor
        processed_concat = torch.empty((B, total_seq, H), device=e.device, dtype=e.dtype)

        # Strides (elements, not bytes)
        stride_e_n, stride_e_s, stride_e_h = e.stride()
        stride_h_n, stride_h_s, stride_h_h = h.stride()
        stride_w_h, stride_w_k = w.stride()  # w is [H, H]
        stride_out_n, stride_out_s, stride_out_h = processed_concat.stride()

        # Tiling parameters
        BLOCK_S = 64
        BLOCK_H = 128
        BLOCK_K = 64

        tiles_s = (total_seq + BLOCK_S - 1) // BLOCK_S
        tiles_h = (H + BLOCK_H - 1) // BLOCK_H

        # Launch Triton kernel
        grid = (B, tiles_s, tiles_h)
        concat_linear_split_blocked_kernel[grid](
            e, h, w, processed_concat,
            B, T, I, H, total_seq,
            stride_e_n, stride_e_s, stride_e_h,
            stride_h_n, stride_h_s, stride_h_h,
            stride_w_h, stride_w_k,
            stride_out_n, stride_out_s, stride_out_h,
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )

        # Split outputs
        processed_encoder = processed_concat[:, :T, :]
        processed_hidden = processed_concat[:, T:, :]
        return processed_encoder, processed_hidden