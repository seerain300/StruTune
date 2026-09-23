import torch
import triton
import triton.language as tl


@triton.jit
def concatenate_sequences_kernel(
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

    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # [0, T+I)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # [0, H)

    L = T + I
    mask_l = l < L
    mask_h = h < H

    # Broadcast to 2D [BLOCK_l, BLOCK_h]
    l2d = l[:, None]
    h2d = h[None, :]

    # Compute offsets and load from encoder or image depending on l
    # Note: we do not write out-of-bound values due to masks
    mask_encoder = (l2d < T) & mask_l[:, None] & mask_h[None, :]
    mask_image = (l2d >= T) & mask_l[:, None] & mask_h[None, :]
    src_idx = l2d - T  # index into hidden for l >= T

    # Load from encoder for l < T
    e_offsets = pid_b * e_s0 + src_idx * e_s1 + h2d * e_s2
    e_vals = tl.load(e_ptr + e_offsets, mask=mask_encoder, other=0.0)

    # Load from hidden for l >= T
    i_offsets = pid_b * i_s0 + src_idx * i_s1 + h2d * i_s2
    i_vals = tl.load(i_ptr + i_offsets, mask=mask_image, other=0.0)

    # Select based on mask; both are zero where the other mask is False
    out_vals = tl.where(mask_encoder, e_vals, 0.0) + tl.where(mask_image, i_vals, 0.0)

    # Store to out
    out_offsets = pid_b * o_s0 + l2d * o_s1 + h2d * o_s2
    tl.store(out_ptr + out_offsets, out_vals, mask=mask_l[:, None] & mask_h[None, :])


@triton.jit
def matmul_kernel(
    A_ptr, W_ptr, C_ptr,
    B: tl.int32, L: tl.int32, H: tl.int32,
    a_s0, a_s1, a_s2,
    w_s0, w_s1, w_s2,
    c_s0, c_s1, c_s2,
    BLOCK_m: tl.constexpr, BLOCK_n: tl.constexpr, BLOCK_k: tl.constexpr,
):
    # We flatten M = B*L rows
    pid_m = tl.program_id(0)  # over rows
    pid_n = tl.program_id(1)  # over columns

    m = pid_m * BLOCK_m + tl.arange(0, BLOCK_m)  # [0, B*L)
    n = pid_n * BLOCK_n + tl.arange(0, BLOCK_n)  # [0, H)

    M = B * L
    mask_m = m < M
    mask_n = n < H

    # Recover b and l for each row m
    b = m // L
    l = m % L

    # Accumulator
    acc = tl.zeros((BLOCK_m, BLOCK_n), dtype=tl.float32)

    # Reduction over K=H
    for k0 in range(0, H, BLOCK_k):
        k = k0 + tl.arange(0, BLOCK_k)
        mask_k = k < H

        # Load A tile: A[b, l, k] -> shape [BLOCK_m, BLOCK_k]
        a_offsets = b[:, None] * a_s0 + l[:, None] * a_s1 + k[None, :] * a_s2
        a_vals = tl.load(A_ptr + a_offsets, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load W^T tile: W^T[k, n] means W(n, k) -> shape [BLOCK_k, BLOCK_n]
        w_offsets = n[None, :] * w_s0 + k[:, None] * w_s1
        w_vals = tl.load(W_ptr + w_offsets, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(a_vals.to(tl.float32), w_vals.to(tl.float32))

    # Store C[b, l, n]
    c_offsets = b[:, None] * c_s0 + l[:, None] * c_s1 + n[None, :] * c_s2
    tl.store(C_ptr + c_offsets, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def copy_rows_kernel(
    src_ptr, dst_ptr,
    B: tl.int32, NUM_ROWS: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dst_s0, dst_s1, dst_s2,
    ROW_START: tl.int32,
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # dst has shape [B, NUM_ROWS, H], src has shape [B, T+I, H]
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # [0, NUM_ROWS)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # [0, H)

    mask_l = l < NUM_ROWS
    mask_h = h < H

    # l2d and h2d for 2D addressing
    l2d = l[:, None]
    h2d = h[None, :]

    # src row index: l + ROW_START
    src_row = l2d + ROW_START

    # Build offsets
    src_offsets = pid_b * src_s0 + src_row * src_s1 + h2d * src_s2
    dst_offsets = pid_b * dst_s0 + l2d * dst_s1 + h2d * dst_s2

    mask = mask_l[:, None] & mask_h[None, :]

    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension.
        - Applies linear projection (concatenated @ process_weight.T).
        - Splits back into processed_encoder and processed_hidden.
        """
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        # Ensure inputs are on CUDA and contiguous
        device = hidden_states.device
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA device for Triton kernels."

        # 1) Concatenate sequences using Triton
        out = torch.empty((B, T + I, H), dtype=hidden_states.dtype, device=device)
        grid_concat = (B, triton.cdiv(T + I, 64), triton.cdiv(H, 64))
        concatenate_sequences_kernel[grid_concat](
            encoder_hidden_states, hidden_states, out,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: processed = out @ process_weight.T
        # process_weight is [H, H], we pass as-is; weight.T in kernel via strides.
        processed = torch.empty((B, T + I, H), dtype=hidden_states.dtype, device=device)
        grid_mm = (B * (T + I), triton.cdiv(H, 64), triton.cdiv(H, 64))
        matmul_kernel[grid_mm](
            out, process_weight, processed,
            B, T + I, H,
            out.stride(0), out.stride(1), out.stride(2),
            process_weight.stride(0), process_weight.stride(1), process_weight.stride(2),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_m=64, BLOCK_n=64, BLOCK_k=64,
            num_warps=4, num_stages=2,
        )

        # 3) Split: copy first T rows to processed_encoder, next I rows to processed_hidden
        processed_encoder = torch.empty((B, T, H), dtype=hidden_states.dtype, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=hidden_states.dtype, device=device)

        # Copy first T rows
        grid_first = (B, triton.cdiv(T, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_first](
            processed, processed_encoder,
            B, T, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # Copy next I rows
        grid_second = (B, triton.cdiv(I, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_second](
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
