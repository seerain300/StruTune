import torch
import triton
import triton.language as tl


@triton.jit
def _concat_encoder_image_kernel(
    e_ptr, i_ptr, out_ptr,
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2,
    i_s0, i_s1, i_s2,
    o_s0, o_s1, o_s2,
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # program ids
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)  # tiles over sequence length (T + I)
    pid_h = tl.program_id(2)  # tiles over hidden dim H

    # indices within the tile
    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # [BLOCK_l]
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # [BLOCK_h]

    # masks to guard bounds
    mask_l = l < (T + I)
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]  # [BLOCK_l, BLOCK_h]

    # compute which source tensor to use: if l < T -> encoder, else -> image
    use_encoder = l < T  # [BLOCK_l] boolean

    # broadcast strides and indices
    # out[b, l, h] = e[b, l, h] if l < T else i[b, l - T, h]
    # We build pointers for both and then select via mask.

    # pointer offsets for out
    out_offsets = pid_b * o_s0 + l[:, None] * o_s1 + h[None, :] * o_s2  # [BLOCK_l, BLOCK_h]

    # pointer offsets for e (only valid where use_encoder is True)
    e_offsets = pid_b * e_s0 + l[:, None] * e_s1 + h[None, :] * e_s2  # [BLOCK_l, BLOCK_h]

    # pointer offsets for i (only valid where not use_encoder and l >= T)
    # When l >= T, l - T is in [0, I), so safe to load from i.
    i_offsets = pid_b * i_s0 + (l - T)[:, None] * i_s1 + h[None, :] * i_s2  # [BLOCK_l, BLOCK_h]

    # load from appropriate source with masks
    # We need to compute vals_e and vals_i and then select via mask
    # However, Triton doesn't support masked selection directly here across 2D tile,
    # so we perform masked loads into e_vals and i_vals and then assemble out vals.

    e_vals = tl.load(e_ptr + e_offsets, mask=mask & use_encoder[:, None], other=0.0)
    i_vals = tl.load(i_ptr + i_offsets, mask=mask & (~use_encoder)[:, None], other=0.0)

    # assemble out vals: where use_encoder -> e_vals, else -> i_vals
    # We can use tl.where since masks are 2D.
    vals = tl.where(use_encoder[:, None], e_vals, i_vals)

    # store to out
    tl.store(out_ptr + out_offsets, vals, mask=mask)


@triton.jit
def _matmul_linear_kernel(
    A_ptr, W_ptr, C_ptr,
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,  # we'll set L = T+I in host
    A_s0, A_s1, A_s2,
    W_s0, W_s1, W_s2,
    C_s0, C_s1, C_s2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Flatten M = B * (T+I), N = H
    pid_m = tl.program_id(0)  # tile over rows
    pid_n = tl.program_id(1)  # tile over cols

    M = B * (T + I)
    L = T + I

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    k = tl.arange(0, BLOCK_K)                    # [BLOCK_K]

    mask_m = m < M
    mask_n = n < H
    mask = mask_m[:, None] & mask_n[None, :]

    # Map flattened m to (b, l)
    b_idx = m // L
    l_idx = m % L

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K = H in chunks
    for kk in range(0, H, BLOCK_K):
        k_idx = kk + k  # [BLOCK_K]
        mask_k = k_idx < H

        # Load A tile: A[b, l, k] -> shape (BLOCK_M, BLOCK_K)
        # A_s0 = stride for batch, A_s1 = stride for seq, A_s2 = stride for hidden
        a_ptrs = A_ptr + b_idx[:, None] * A_s0 + l_idx[:, None] * A_s1 + k_idx[None, :] * A_s2
        a_mask = mask_m[:, None] & mask_k[None, :]
        a_vals = tl.load(A_ptr + a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load W^T tile: W^T[k, n] = W[n, k] -> shape (BLOCK_K, BLOCK_N)
        w_ptrs = W_ptr + n[None, :] * W_s1 + k_idx[:, None] * W_s0
        w_mask = mask_n[None, :] & mask_k[:, None]
        w_vals = tl.load(W_ptr + w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(a_vals, w_vals)  # [BLOCK_M, BLOCK_N]

    # Store result to C[b, l, n]
    c_ptrs = C_ptr + b_idx[:, None] * C_s0 + l_idx[:, None] * C_s1 + n[None, :] * C_s2
    tl.store(C_ptr + c_ptrs, acc, mask=mask)


@triton.jit
def _copy_rows_kernel(
    src_ptr, dst_ptr,
    B: tl.int32, COUNT: tl.int32, H: tl.int32,
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

    mask_l = l < (ROW_START + COUNT)
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    # offsets for src: [b, l, h]
    src_offsets = pid_b * src_s0 + l[:, None] * src_s1 + h[None, :] * src_s2
    # offsets for dst: [b, l, h]
    dst_offsets = pid_b * dst_s0 + (l - ROW_START)[:, None] * dst_s1 + h[None, :] * dst_s2

    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Computes:
          concatenated = cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, H]
          processed = concatenated @ process_weight.T  # [B, T+I, H]
          processed_encoder = processed[:, :T, :]
          processed_hidden = processed[:, T:, :]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be CUDA tensors"
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D tensors"
        B, I, H = hidden_states.shape
        assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == H, "Shape mismatch"
        T = encoder_hidden_states.shape[1]
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"

        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1) Concatenate encoder_hidden_states and hidden_states into [B, T+I, H]
        concatenated = torch.empty((B, T + I, H), device=device, dtype=dtype)
        BLOCK_l = 128
        BLOCK_h = 128
        grid_concat = (B, triton.cdiv(T + I, BLOCK_l), triton.cdiv(H, BLOCK_h))
        _concat_encoder_image_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_l=BLOCK_l, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: processed = concatenated @ process_weight.T
        # Ensure process_weight is transposed to [H, H] and contiguous
        W_t = process_weight.t().contiguous()
        processed = torch.empty((B, T + I, H), device=device, dtype=torch.float32)  # compute in fp32 for stability
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64
        M = B * (T + I)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        _matmul_linear_kernel[grid](
            concatenated, W_t, processed,
            B, T, I, H,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            W_t.stride(0), W_t.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=2,
        )

        # 3) Split into two streams
        processed_encoder = torch.empty((B, T, H), device=device, dtype=dtype)
        BLOCK_e = 128
        grid_encoder = (B, triton.cdiv(T, BLOCK_e), triton.cdiv(H, 128))
        _copy_rows_kernel[grid_encoder](
            processed, processed_encoder,
            B, T, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0,
            BLOCK_l=BLOCK_e, BLOCK_h=128,
            num_warps=4, num_stages=2,
        )

        processed_hidden = torch.empty((B, I, H), device=device, dtype=dtype)
        BLOCK_i = 128
        grid_image = (B, triton.cdiv(I, BLOCK_i), triton.cdiv(H, 128))
        _copy_rows_kernel[grid_image](
            processed, processed_hidden,
            B, I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            ROW_START=T,
            BLOCK_l=BLOCK_i, BLOCK_h=128,
            num_warps=4, num_stages=2,
        )

        # Cast back to original dtype if needed
        if processed_encoder.dtype != dtype:
            processed_encoder = processed_encoder.to(dtype)
        if processed_hidden.dtype != dtype:
            processed_hidden = processed_hidden.to(dtype)

        return processed_encoder, processed_hidden