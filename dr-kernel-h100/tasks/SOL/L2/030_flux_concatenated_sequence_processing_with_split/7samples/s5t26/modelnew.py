import torch
import triton
import triton.language as tl


@triton.jit
def concat_encoder_image_kernel(
    e_ptr, i_ptr, out_ptr,
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2,
    i_s0, i_s1, i_s2,
    o_s0, o_s1, o_s2,
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # sequence indices [0, T+I)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # hidden dim indices [0, H)

    # Bounds mask
    mask_l = l < (T + I)
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    # Determine source: encoder if l < T, else image at l - T
    is_encoder = l < T
    l_img = l - T

    # Pointers for source and destination
    e_off = pid_b * e_s0 + l[:, None] * e_s1 + h[None, :] * e_s2
    i_off = pid_b * i_s0 + l_img[:, None] * i_s1 + h[None, :] * i_s2
    o_off = pid_b * o_s0 + l[:, None] * o_s1 + h[None, :] * o_s2

    # Select source pointer
    src_ptr = tl.where(is_encoder[:, None], e_ptr + e_off, i_ptr + i_off)

    # Load, store with mask
    vals = tl.load(src_ptr, mask=mask, other=0.0)
    tl.store(out_ptr + o_off, vals, mask=mask)


@triton.jit
def matmul_kernel(
    A_ptr, W_ptr, C_ptr,
    B: tl.int32, L: tl.int32, H: tl.int32,
    A_s0, A_s1, A_s2,
    W_s0, W_s1, W_s2,
    C_s0, C_s1, C_s2,
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr, BLOCK_k: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)   # rows in [0, L)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)   # columns in [0, H)

    # Bounds masks
    mask_l = l < L
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    # Accumulator
    acc = tl.zeros((BLOCK_l, BLOCK_h), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, H, BLOCK_k):
        k = k0 + tl.arange(0, BLOCK_k)
        mask_k = k < H

        # Load A tile: [BLOCK_l, BLOCK_k]
        a_off = pid_b * A_s0 + l[:, None] * A_s1 + k[None, :] * A_s2
        A_tile = tl.load(A_ptr + a_off, mask=mask_l[:, None] & mask_k[None, :], other=0.0)

        # Load W^T tile: we need W[k, h] -> shape [BLOCK_k, BLOCK_h]
        w_off = k[:, None] * W_s0 + h[None, :] * W_s2
        Wt_tile = tl.load(W_ptr + w_off, mask=mask_k[:, None] & mask_h[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, Wt_tile)

    # Store result
    c_off = pid_b * C_s0 + l[:, None] * C_s1 + h[None, :] * C_s2
    tl.store(C_ptr + c_off, acc, mask=mask)


@triton.jit
def copy_rows_kernel(
    src_ptr, dst_ptr,
    B: tl.int32, L: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dst_s0, dst_s1, dst_s2,
    start_row: tl.constexpr,  # start row in src to copy
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    row = start_row + pid_l * BLOCK_l + tl.arange(0, BLOCK_l)   # rows to copy
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)                 # hidden dim indices

    mask_row = row < (start_row + L)
    mask_h = h < H
    mask = mask_row[:, None] & mask_h[None, :]

    # Source and destination offsets
    src_off = pid_b * src_s0 + row[:, None] * src_s1 + h[None, :] * src_s2
    dst_off = pid_b * dst_s0 + (row[:, None] - start_row) * dst_s1 + h[None, :] * dst_s2

    vals = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_off, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H]
        Returns (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton."
        B, T, H = encoder_hidden_states.shape
        Bi, I, Hw = hidden_states.shape
        assert B == Bi and H == Hw, "Batch and hidden_dim must match."

        # 1) Concatenate along sequence dimension: concatenated [B, T+I, H]
        concatenated = torch.empty((B, T + I, H), dtype=hidden_states.dtype, device=hidden_states.device)

        e_s0, e_s1, e_s2 = encoder_hidden_states.stride()
        i_s0, i_s1, i_s2 = hidden_states.stride()
        o_s0, o_s1, o_s2 = concatenated.stride()

        grid_concat = (B, triton.cdiv(T + I, 64), triton.cdiv(H, 64))
        concat_encoder_image_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, T, I, H,
            e_s0, e_s1, e_s2,
            i_s0, i_s1, i_s2,
            o_s0, o_s1, o_s2,
            BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: processed = concatenated @ process_weight.T
        processed = torch.empty((B, T + I, H), dtype=hidden_states.dtype, device=hidden_states.device)

        A_s0, A_s1, A_s2 = concatenated.stride()
        W_s0, W_s1, W_s2 = process_weight.stride()
        C_s0, C_s1, C_s2 = processed.stride()

        L = T + I
        grid_matmul = (B, triton.cdiv(L, 64), triton.cdiv(H, 64))
        # Note: process_weight should be float32 for stability; if not, cast inside kernel (here we assume float32)
        matmul_kernel[grid_matmul](
            concatenated, process_weight, processed,
            B, L, H,
            A_s0, A_s1, A_s2,
            W_s0, W_s1, W_s2,
            C_s0, C_s1, C_s2,
            BLOCK_l=64, BLOCK_h=64, BLOCK_k=64,
            num_warps=4, num_stages=2,
        )

        # 3) Split into two streams using Triton copy kernels
        processed_encoder = torch.empty((B, T, H), dtype=hidden_states.dtype, device=hidden_states.device)
        processed_hidden = torch.empty((B, I, H), dtype=hidden_states.dtype, device=hidden_states.device)

        # First T rows
        grid_copy_encoder = (B, triton.cdiv(T, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_encoder](
            processed, processed_encoder,
            B, T, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            start_row=0,
            BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # Next I rows
        grid_copy_image = (B, triton.cdiv(I, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_image](
            processed, processed_hidden,
            B, I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            start_row=T,
            BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden