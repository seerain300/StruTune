import torch
import triton
import triton.language as tl


@triton.jit
def concat_encoder_image_kernel(
    e_ptr,                # *const T: [B, T, H]
    i_ptr,                # *const T: [B, I, H]
    out_ptr,              # *T: [B, T+I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2,     # strides for e_ptr
    i_s0, i_s1, i_s2,     # strides for i_ptr
    o_s0, o_s1, o_s2,     # strides for out_ptr
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l_offsets = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)            # [BLOCK_l]
    h_offsets = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)            # [BLOCK_h]

    mask_l = l_offsets < (T + I)
    mask_h = h_offsets < H

    # Broadcast to 2D tile
    L = l_offsets[:, None]                  # [BLOCK_l, 1]
    H = h_offsets[None, :]                  # [1, BLOCK_h]

    # Compute source pointers for encoder and image parts
    e_ptrs = e_ptr + pid_b * e_s0 + L * e_s1 + H * e_s2            # [BLOCK_l, BLOCK_h]
    i_ptrs = i_ptr + pid_b * i_s0 + (L - T) * i_s1 + H * i_s2      # [BLOCK_l, BLOCK_h]
    o_ptrs = out_ptr + pid_b * o_s0 + L * o_s1 + H * o_s2          # [BLOCK_l, BLOCK_h]

    mask = mask_l[:, None] & mask_h[None, :]

    # Load from encoder if l < T else from image part
    load_from_encoder = L < T
    # Create scalar 0/1 masks to select source
    # Note: Triton doesn't support dynamic where on pointer tensors; instead we do masked loads.
    # We'll perform two masked loads and store. However, Triton doesn't support two stores; instead
    # we do a conditional load: compute pointers for both and use mask to choose.
    # Better: load from appropriate source using mask selection.
    # Since Triton requires pointer tensors, we can't branch; but we can compute two tensors and
    # store by selecting from them using masks. Triton allows elementwise arithmetic.
    # Instead, we will load from encoder where mask & load_from_encoder, else load from image.
    e_mask = mask & (L < T)
    i_mask = mask & (L >= T)

    e_vals = tl.load(e_ptrs, mask=e_mask, other=0.0)
    i_vals = tl.load(i_ptrs, mask=i_mask, other=0.0)

    # Select source: where L < T -> e_vals, else -> i_vals
    out_vals = tl.where(L < T, e_vals, i_vals)

    # Store
    out_ptrs = out_ptr + pid_b * o_s0 + L * o_s1 + H * o_s2
    tl.store(out_ptrs, out_vals, mask=mask)


@triton.jit
def matmul_rowwise_kernel(
    A_ptr,                # *const T: concatenated [B, L, H]
    W_ptr,                # *const T: process_weight [H, H]
    C_ptr,                # *T: processed [B, L, H]
    B: tl.int32, L: tl.int32, H: tl.int32,
    A_s0, A_s1, A_s2,     # strides for A
    W_s0, W_s1, W_s2,     # strides for W
    C_s0, C_s1, C_s2,     # strides for C
    BLOCK_h: tl.constexpr, BLOCK_k: tl.constexpr,
):
    # One program per (batch, row) pair
    pid_b = tl.program_id(0)
    pid_row = tl.program_id(1)

    # row index in the concatenated sequence
    l = pid_row  # 0 <= l < L

    # iterate over hidden columns in tiles
    for h_start in range(0, H, BLOCK_h):
        h_offsets = h_start + tl.arange(0, BLOCK_h)
        mask_h = h_offsets < H

        # Accumulator for this (b, l, h_offsets)
        acc = tl.zeros([BLOCK_h], dtype=tl.float32)

        # Reduce over hidden dimension (K=H) in chunks
        for k_start in range(0, H, BLOCK_k):
            k_offsets = k_start + tl.arange(0, BLOCK_k)
            mask_k = k_offsets < H

            # Load A_row[k_offsets]: A[b, l, k_offsets]
            A_row_ptrs = A_ptr + pid_b * A_s0 + l * A_s1 + k_offsets * A_s2
            A_vals = tl.load(A_row_ptrs, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_k]

            # Load W[k_offsets, h_offsets]: [BLOCK_k, BLOCK_h]
            W_ptrs = W_ptr + k_offsets[:, None] * W_s0 + h_offsets[None, :] * W_s2
            W_vals = tl.load(W_ptrs, mask=mask_k[:, None] & mask_h[None, :], other=0.0).to(tl.float32)

            # Accumulate: sum over k chunk
            acc += tl.sum(W_vals * A_vals[:, None], axis=0)

        # Store result to C[b, l, h_offsets]
        C_ptrs = C_ptr + pid_b * C_s0 + l * C_s1 + h_offsets * C_s2
        tl.store(C_ptrs, acc, mask=mask_h)


@triton.jit
def copy_rows_kernel(
    src_ptr, dest_ptr,
    B: tl.int32, NUM_ROWS: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dest_s0, dest_s1, dest_s2,
    ROW_START: tl.int32,
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # Copy first NUM_ROWS rows starting at ROW_START from src to dest
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l_offsets = ROW_START + pid_l * BLOCK_l + tl.arange(0, BLOCK_l)    # [BLOCK_l]
    h_offsets = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)                # [BLOCK_h]

    mask_l = l_offsets < (ROW_START + NUM_ROWS)
    mask_h = h_offsets < H

    L = l_offsets[:, None]   # [BLOCK_l, 1]
    H = h_offsets[None, :]   # [1, BLOCK_h]

    src_ptrs = src_ptr + pid_b * src_s0 + L * src_s1 + H * src_s2
    dest_ptrs = dest_ptr + pid_b * dest_s0 + L * dest_s1 + H * dest_s2
    mask = mask_l[:, None] & mask_h[None, :]

    vals = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(dest_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        Concatenate -> linear projection -> split into encoder and image streams.
        All operations are performed by Triton kernels; no torch ops on tensors in host.
        """
        # Ensure inputs are contiguous and on the same device
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape == (B, T, H)
        assert hidden_states.shape == (B, I, H)
        assert process_weight.shape == (H, H)

        # 1) Concatenate [B, T, H] and [B, I, H] -> [B, T+I, H]
        concatenated = torch.empty((B, T + I, H), device=hidden_states.device, dtype=torch.float32)

        BLOCK_l = 64
        BLOCK_h = 64
        grid_concat = (B, triton.cdiv(T + I, BLOCK_l), triton.cdiv(H, BLOCK_h))
        concat_encoder_image_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_l=BLOCK_l, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: processed = concatenated @ process_weight.T
        # Concatenated: [B, L, H], W: [H, H] -> C: [B, L, H]
        processed = torch.empty((B, T + I, H), device=hidden_states.device, dtype=torch.float32)

        BLOCK_h = 64
        BLOCK_k = 32
        grid_matmul = (B, T + I)
        matmul_rowwise_kernel[grid_matmul](
            concatenated, process_weight, processed,
            B, T + I, H,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            process_weight.stride(0), process_weight.stride(1), process_weight.stride(2),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_h=BLOCK_h, BLOCK_k=BLOCK_k,
            num_warps=4, num_stages=2,
        )

        # 3) Split into processed_encoder [B, T, H] and processed_hidden [B, I, H]
        processed_encoder = torch.empty((B, T, H), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, H), device=hidden_states.device, dtype=torch.float32)

        BLOCK_l_split = 64
        BLOCK_h_split = 64

        grid_copy_encoder = (B, triton.cdiv(T, BLOCK_l_split), triton.cdiv(H, BLOCK_h_split))
        copy_rows_kernel[grid_copy_encoder](
            processed, processed_encoder,
            B, T, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0, BLOCK_l=BLOCK_l_split, BLOCK_h=BLOCK_h_split,
            num_warps=4, num_stages=2,
        )

        grid_copy_hidden = (B, triton.cdiv(I, BLOCK_l_split), triton.cdiv(H, BLOCK_h_split))
        copy_rows_kernel[grid_copy_hidden](
            processed, processed_hidden,
            B, I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            ROW_START=T, BLOCK_l=BLOCK_l_split, BLOCK_h=BLOCK_h_split,
            num_warps=4, num_stages=2,
        )

        # If original tensors were not float32, you may cast outputs back here:
        # processed_encoder = processed_encoder.to(...)
        # processed_hidden = processed_hidden.to(...)

        return processed_encoder, processed_hidden