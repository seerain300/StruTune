import torch
import triton
import triton.language as tl


@triton.jit
def concat_encoder_image_kernel(
    e_ptr,                # *const float32: [B, T, H]
    i_ptr,                # *const float32: [B, I, H]
    out_ptr,              # *float32: [B, T+I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2,     # strides for e_ptr (int64)
    i_s0, i_s1, i_s2,     # strides for i_ptr
    o_s0, o_s1, o_s2,     # strides for out_ptr
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # program ids
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    # tile indices
    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # [BLOCK_l]
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # [BLOCK_h]

    # masks
    mask_l = l < (T + I)
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    # compute source indices
    is_encoder = l < T  # [BLOCK_l]
    src_l_e = l
    src_l_i = l - T

    # broadcast strides to [BLOCK_l, BLOCK_h]
    # compute offsets
    # encoder offsets: e_off = b*e_s0 + l*e_s1 + h*e_s2
    # image  offsets:  i_off = b*i_s0 + (l - T)*i_s1 + h*i_s2
    e_off = pid_b * e_s0 + src_l_e[:, None] * e_s1 + h[None, :] * e_s2
    i_off = pid_b * i_s0 + src_l_i[:, None] * i_s1 + h[None, :] * i_s2

    # load with masks and select
    e_vals = tl.load(e_ptr + e_off, mask=mask & (is_encoder[:, None]), other=0.0)
    i_vals = tl.load(i_ptr + i_off, mask=mask & (~is_encoder[:, None]), other=0.0)

    # select: where l < T -> e_vals else i_vals
    # Triton allows where with tensors
    out_vals = tl.where(is_encoder[:, None], e_vals, i_vals)

    # store to output
    out_off = pid_b * o_s0 + l[:, None] * o_s1 + h[None, :] * o_s2
    tl.store(out_ptr + out_off, out_vals, mask=mask)


@triton.jit
def matmul_kernel(
    A_ptr,                # *const float32: [B*(T+I), H]
    W_ptr,                # *const float32: [H, H]  (process_weight)
    C_ptr,                # *float32:        [B*(T+I), H]
    B: tl.int32, L: tl.int32, H: tl.int32,  # L = T + I
    A_s0: tl.int64, A_s1: tl.int64,         # strides for A_ptr
    W_s0: tl.int64, W_s1: tl.int64,         # strides for W_ptr
    C_s0: tl.int64, C_s1: tl.int64,         # strides for C_ptr
    BLOCK_M: tl.constexpr,                   # rows tile
    BLOCK_N: tl.constexpr,                   # cols tile
    BLOCK_K: tl.constexpr,                   # reduction tile
):
    # Grid is 2D: (M_tiles, N_tiles). M = B*L rows, N = H columns.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # row indices in [0, B*L)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # column indices in [0, H)

    mask_m = m < (B * L)
    mask_n = n < H

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K = H in chunks of BLOCK_K
    for k0 in range(0, H, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < H

        # Load A tile: shape [BLOCK_M, BLOCK_K]
        # A[m, k] => offset = m*A_s0 + k*A_s1
        A_off = m[:, None] * A_s0 + k[None, :] * A_s1
        A_tile = tl.load(A_ptr + A_off, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        # Load W^T tile: we want [BLOCK_K, BLOCK_N] which corresponds to W[k, n]
        # W[k, n] => offset = k*W_s0 + n*W_s1
        W_off = k[:, None] * W_s0 + n[None, :] * W_s1
        W_tile = tl.load(W_ptr + W_off, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(A_tile, W_tile)

    # Store result C[m, n] => offset = m*C_s0 + n*C_s1
    C_off = m[:, None] * C_s0 + n[None, :] * C_s1
    tl.store(C_ptr + C_off, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def copy_rows_kernel(
    src_ptr,              # *const float32: [B, L, H]
    dst_ptr,              # *float32: [B, L_out, H]
    B: tl.int32, L: tl.int32, L_out: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,  # strides for src_ptr
    dst_s0, dst_s1, dst_s2,  # strides for dst_ptr
    ROW_START: tl.int32,     # starting row in src to copy
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # 3D grid: (B, tiles over L_out, tiles over H)
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)   # [BLOCK_l]
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)   # [BLOCK_h]

    mask_l = l < L_out
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    # Compute source row indices
    src_l = ROW_START + l  # [BLOCK_l]
    src_mask = (src_l < L) & mask_l

    # Compute offsets
    src_off = pid_b * src_s0 + src_l[:, None] * src_s1 + h[None, :] * src_s2
    dst_off = pid_b * dst_s0 + l[:, None] * dst_s1 + h[None, :] * dst_s2

    # Load from src and store to dst
    vals = tl.load(src_ptr + src_off, mask=src_mask[:, None] & mask, other=0.0)
    tl.store(dst_ptr + dst_off, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension using a Triton kernel.
        - Apply linear projection concatenated @ process_weight.T using a Triton matmul kernel.
        - Split processed tensor into processed_encoder and processed_hidden using Triton copy kernels.
        """
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        device = hidden_states.device
        # Ensure all tensors are float32 for consistent behavior. The original example likely uses float32.
        e = encoder_hidden_states
        i = hidden_states
        W = process_weight

        # Allocate output for concatenation
        L = T + I
        concatenated = torch.empty((B, L, H), dtype=torch.float32, device=device)

        # Launch concatenation kernel
        BLOCK_l = 64
        BLOCK_h = 64
        grid_concat = (B, triton.cdiv(L, BLOCK_l), triton.cdiv(H, BLOCK_h))
        concat_encoder_image_kernel[grid_concat](
            e, i, concatenated,
            B, T, I, H,
            e.stride(0), e.stride(1), e.stride(2),
            i.stride(0), i.stride(1), i.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_l=BLOCK_l, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        # Allocate processed output [B, L, H], float32
        processed = torch.empty((B, L, H), dtype=torch.float32, device=device)

        # Launch matmul kernel: concatenated @ W.T
        # A is [B*L, H], W is [H, H], C is [B*L, H]
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 64
        grid_mm = (triton.cdiv(B * L, BLOCK_M), triton.cdiv(H, BLOCK_N))
        # A_ptr: concatenated viewed as [B*L, H]; flatten strides: A_s0 = H, A_s1 = 1
        # Note: we pass actual strides of concatenated for generality
        A_s0 = concatenated.stride(0) * (T + I)
        A_s1 = concatenated.stride(2)
        C_s0 = processed.stride(0) * (T + I)
        C_s1 = processed.stride(2)
        # W strides
        W_s0 = W.stride(0)
        W_s1 = W.stride(1)

        matmul_kernel[grid_mm](
            concatenated, W, processed,
            B, L, H,
            A_s0, A_s1,
            W_s0, W_s1,
            C_s0, C_s1,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Allocate outputs
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=device)

        # Copy first T rows
        BLOCK_l_first = 64
        BLOCK_h_copy = 64
        grid_copy_first = (B, triton.cdiv(T, BLOCK_l_first), triton.cdiv(H, BLOCK_h_copy))
        copy_rows_kernel[grid_copy_first](
            processed, processed_encoder,
            B, L, T, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0, BLOCK_l=BLOCK_l_first, BLOCK_h=BLOCK_h_copy,
            num_warps=4, num_stages=2,
        )

        # Copy next I rows
        grid_copy_second = (B, triton.cdiv(I, BLOCK_l_first), triton.cdiv(H, BLOCK_h_copy))
        copy_rows_kernel[grid_copy_second](
            processed, processed_hidden,
            B, L, I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            ROW_START=T, BLOCK_l=BLOCK_l_first, BLOCK_h=BLOCK_h_copy,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden