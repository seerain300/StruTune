import torch
import triton
import triton.language as tl


@triton.jit
def concat_encoder_image_kernel(
    e_ptr,        # *T: [B, T, H]
    i_ptr,        # *T: [B, I, H]
    out_ptr,      # *T: [B, T+I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2,     # strides for e_ptr
    i_s0, i_s1, i_s2,     # strides for i_ptr
    o_s0, o_s1, o_s2,     # strides for out_ptr
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    L = T + I

    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # [BLOCK_l]
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # [BLOCK_h]

    # Compute base offsets
    # e: offset for encoder rows, i: offset for image rows, o: offset for output rows
    # Broadcast to 2D tile
    e_offsets = pid_b * e_s0 + l[:, None] * e_s1 + h[None, :] * e_s2
    i_offsets = pid_b * i_s0 + (l[:, None] - T) * i_s1 + h[None, :] * i_s2
    o_offsets = pid_b * o_s0 + l[:, None] * o_s1 + h[None, :] * o_s2

    # Mask for valid l (since l ranges over T+I)
    mask = (l[:, None] < L) & (h[None, :] < H)

    # Select source based on l
    mask_e = l[:, None] < T
    mask_i = l[:, None] >= T

    # Load from appropriate source
    e_vals = tl.load(e_ptr + e_offsets, mask=mask & mask_e, other=0.0)
    i_vals = tl.load(i_ptr + i_offsets, mask=mask & mask_i, other=0.0)

    # Combine
    out_vals = tl.where(mask_e, e_vals, i_vals)

    # Store to output
    tl.store(out_ptr + o_offsets, out_vals, mask=mask)


@triton.jit
def matmul_bLH_HH_kernel(
    A_ptr,       # *T: [B, L, H]
    W_ptr,       # *T: [H, H] (process_weight.T)
    C_ptr,       # *float32: [B, L, H] (we compute in fp32)
    B: tl.int32, L: tl.int32, H: tl.int32,
    A_s0, A_s1, A_s2,
    W_s0, W_s1, W_s2,
    C_s0, C_s1, C_s2,
    BLOCK_m: tl.constexpr, BLOCK_n: tl.constexpr, BLOCK_k: tl.constexpr,
):
    # Flatten M = B*L rows
    M = B * L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m = pid_m * BLOCK_m + tl.arange(0, BLOCK_m)  # [BLOCK_m]
    n = pid_n * BLOCK_n + tl.arange(0, BLOCK_n)  # [BLOCK_n]

    # Masks
    mask_m = m < M
    mask_n = n < H
    mask_mn = mask_m[:, None] & mask_n[None, :]

    # Accumulator
    acc = tl.zeros((BLOCK_m, BLOCK_n), dtype=tl.float32)

    # Loop over K = H in chunks
    for k0 in range(0, H, BLOCK_k):
        k = k0 + tl.arange(0, BLOCK_k)  # [BLOCK_k]
        mask_k = k < H

        # Compute b and l from flattened m
        b = m // L
        l = m % L

        # A offsets: [BLOCK_m, BLOCK_k] -> A[b, l, k]
        A_offsets = b[:, None] * A_s0 + l[:, None] * A_s1 + k[None, :] * A_s2
        # Load A tile
        A_tile = tl.load(A_ptr + A_offsets, mask=mask_m[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_m, BLOCK_k]

        # W offsets: [BLOCK_k, BLOCK_n] -> W[k, n]
        W_offsets = k[:, None] * W_s0 + n[None, :] * W_s1
        W_tile = tl.load(W_ptr + W_offsets, mask=mask_k[:, None] & mask_n[None, :], other=0.0)  # [BLOCK_k, BLOCK_n]

        # Accumulate
        acc += tl.dot(A_tile, W_tile)  # [BLOCK_m, BLOCK_n]

    # Store to C: [B, L, H]
    # c_m = m
    c_m = m  # m is 0..B*L-1
    # For each b in 0..B-1, l = c_m // L, but we can map c_m directly to (b, l) via integer ops:
    # Here we keep mapping via c_m and recover b, l
    b_out = c_m // L
    l_out = c_m % L

    C_offsets = b_out[:, None] * C_s0 + l_out[:, None] * C_s1 + n[None, :] * C_s2
    tl.store(C_ptr + C_offsets, acc, mask=mask_mn)


@triton.jit
def copy_rows_kernel(
    src_ptr, dst_ptr,
    B: tl.int32, NUM_ROWS: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dst_s0, dst_s1, dst_s2,
    ROW_START: tl.int32,
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l = ROW_START + pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # [BLOCK_l]
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)             # [BLOCK_h]

    mask = (l[:, None] < NUM_ROWS) & (h[None, :] < H)

    src_offsets = pid_b * src_s0 + l[:, None] * src_s1 + h[None, :] * src_s2
    dst_offsets = pid_b * dst_s0 + l[:, None] * dst_s1 + h[None, :] * dst_s2

    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dimension.
        2) Apply linear projection: concatenated @ process_weight.T
        3) Split back into separate encoder and image streams.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA device for Triton."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        L = T + I

        # Ensure contiguous tensors
        e = encoder_hidden_states.contiguous()
        i = hidden_states.contiguous()
        W = process_weight.contiguous()  # [H, H]

        # 1) Concatenation: out [B, L, H]
        out = torch.empty((B, L, H), dtype=e.dtype, device=e.device)
        grid_concat = (B, triton.cdiv(L, 64), triton.cdiv(H, 64))
        concat_encoder_image_kernel[grid_concat](
            e, i, out,
            B, T, I, H,
            e.stride(0), e.stride(1), e.stride(2),
            i.stride(0), i.stride(1), i.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: processed = out @ W.T, output in float32 for numerical stability
        processed = torch.empty((B, L, H), dtype=torch.float32, device=e.device)
        grid_mm = (triton.cdiv(B * L, 128), triton.cdiv(H, 128))
        matmul_bLH_HH_kernel[grid_mm](
            out, W, processed,
            B, L, H,
            out.stride(0), out.stride(1), out.stride(2),
            W.stride(0), W.stride(1), W.stride(2),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_m=128, BLOCK_n=128, BLOCK_k=64,
            num_warps=8, num_stages=2,
        )

        # 3) Split into two outputs
        processed_encoder = torch.empty((B, T, H), dtype=processed.dtype, device=processed.device)
        processed_hidden = torch.empty((B, I, H), dtype=processed.dtype, device=processed.device)

        grid_copy_encoder = (B, triton.cdiv(T, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_encoder](
            processed, processed_encoder,
            B, T, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        grid_copy_hidden = (B, triton.cdiv(I, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_hidden](
            processed, processed_hidden,
            B, I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            ROW_START=T, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
