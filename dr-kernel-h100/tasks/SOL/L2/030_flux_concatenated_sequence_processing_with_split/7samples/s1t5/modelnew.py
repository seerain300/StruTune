import math
import torch
import triton
import triton.language as tl

@triton.jit
def _concatenate_sequences_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, I, D,
    se_b, se_t, se_d,
    sh_b, sh_i, sh_d,
    out_b, out_s, out_d,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    # pid_s in [0, T+I)
    if pid_s < T:
        # copy from encoder
        row_ptr = encoder_ptr + pid_b * se_b + pid_s * se_t
    else:
        # copy from hidden
        local_s = pid_s - T
        row_ptr = hidden_ptr + pid_b * sh_b + local_s * sh_i
    # copy entire D dimension in tiles
    for d_start in range(0, D, BLOCK_D):
        d_offsets = d_start + tl.arange(0, BLOCK_D)
        mask = d_offsets < D
        vals = tl.load(row_ptr + d_offsets * se_d, mask=mask, other=0.0)
        out_row_ptr = out_ptr + pid_b * out_b + pid_s * out_s
        tl.store(out_row_ptr + d_offsets * out_d, vals, mask=mask)


@triton.jit
def _batched_matmul_yxw_kernel(
    X_ptr, W_ptr, Y_ptr,
    B, S, D,
    X_b, X_s, X_d,
    W_d0, W_d1,
    Y_b, Y_s, Y_d,
    BLOCK_K: tl.constexpr,
):
    # program per (batch, output row)
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    # accumulate vector acc across hidden_dim D
    acc = tl.zeros([D], dtype=tl.float32)
    # iterate over K (hidden_dim) in tiles of BLOCK_K
    for k_start in range(0, D, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # load W_sub: shape [D, BLOCK_K]
        w_sub_ptr = W_ptr + k_offsets[None, :] * W_d0  # k axis -> columns
        w_sub = tl.load(
            w_sub_ptr,
            mask=(k_offsets[None, :] < D),
            other=0.0
        )  # shape [D, BLOCK_K]
        # load X_row: shape [D]
        x_row_ptr = X_ptr + pid_b * X_b + pid_s * X_s
        x_row = tl.load(x_row_ptr + tl.arange(0, D) * X_d, mask=(tl.arange(0, D) < D), other=0.0)
        # accumulate: acc += sum over k_tile of x_row[k] * w_sub[k, :]
        # We need to multiply x_row by each column of w_sub, then reduce over k axis.
        # Do it per k in the tile.
        for m in range(BLOCK_K):
            k_idx = k_start + m
            # if k_idx >= D, w_sub[:, m] is zero due to mask; still safe
            col = w_sub[:, m]  # [D]
            acc += x_row * col
    # store result
    y_row_ptr = Y_ptr + pid_b * Y_b + pid_s * Y_s
    tl.store(y_row_ptr + tl.arange(0, D) * Y_d, acc, mask=(tl.arange(0, D) < D))


@triton.jit
def _split_sequences_kernel(
    in_ptr, out1_ptr, out2_ptr,
    B, T, I, D,
    in_b, in_s, in_d,
    out1_b, out1_t, out1_d,
    out2_b, out2_i, out2_d,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    if pid_s < T:
        in_row_ptr = in_ptr + pid_b * in_b + pid_s * in_s
        out_row_ptr = out1_ptr + pid_b * out1_b + pid_s * out1_t
    else:
        in_row_ptr = in_ptr + pid_b * in_b + (pid_s - T) * in_s
        out_row_ptr = out2_ptr + pid_b * out2_b + (pid_s - T) * out2_i
    for d_start in range(0, D, 128):  # tile across D, 128 is fine for general
        d_offsets = d_start + tl.arange(0, 128)
        mask = d_offsets < D
        vals = tl.load(in_row_ptr + d_offsets * in_d, mask=mask, other=0.0)
        tl.store(out_row_ptr + d_offsets * out1_d, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, D]
        encoder_hidden_states: [B, T, D]
        process_weight: [D, D]
        returns (processed_encoder: [B, T, D], processed_hidden: [B, I, D])
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3 and process_weight.dim() == 2
        B, I, D = hidden_states.shape
        B_e, T, D_e = encoder_hidden_states.shape
        assert B_e == B and D == D_e, "Batch size or hidden dim mismatch"

        device = hidden_states.device
        # Ensure dtypes match
        # We'll compute in float32 for numerical stability; process_weight should be float32 typically
        dtype = torch.float32
        # Make inputs contiguous
        enc = encoder_hidden_states.contiguous().to(dtype)
        hs = hidden_states.contiguous().to(dtype)
        W = process_weight.contiguous().to(dtype)
        assert W.shape[1] == D and W.shape[0] == D, "process_weight must be [D, D]"

        # Step 1: Concatenate sequences along sequence dimension using Triton
        S = T + I
        concatenated = torch.empty((B, S, D), device=device, dtype=dtype)
        # Choose BLOCK_D for concatenation copy; 128 is fine
        BLOCK_D = 128
        grid_concat = (B, S)
        _concatenate_sequences_kernel[grid_concat](
            enc, hs, concatenated,
            B, T, I, D,
            enc.stride(0), enc.stride(1), enc.stride(2),
            hs.stride(0), hs.stride(1), hs.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_D=BLOCK_D,
            num_warps=1, num_stages=1,
        )

        # Step 2: Apply linear projection (no bias) using Triton batched matmul: Y = concatenated @ W
        Y = torch.empty((B, S, D), device=device, dtype=dtype)
        # Choose BLOCK_K: must divide D; pick largest power-of-two that divides D up to 4096.
        def largest_power_of_two_divisor(n):
            # returns the largest power-of-two <= n that divides n; else 1
            if n <= 1:
                return 1
            p = 1
            while (p << 1) <= n:
                p <<= 1
            # check divisibility
            return p if (n % p == 0) else 1

        BLOCK_K = largest_power_of_two_divisor(D)
        # Fallback to D if no power-of-two divisor found (rare)
        if BLOCK_K == 1 and D > 1:
            BLOCK_K = D

        grid_mm = (B, S)
        _batched_matmul_yxw_kernel[grid_mm](
            concatenated, W, Y,
            B, S, D,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            W.stride(0), W.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Step 3: Split back into separate streams using Triton
        processed_encoder = torch.empty((B, T, D), device=device, dtype=dtype)
        processed_hidden = torch.empty((B, I, D), device=device, dtype=dtype)
        grid_split = (B, T), (B, I)
        _split_sequences_kernel[grid_split](
            Y, processed_encoder, processed_hidden,
            B, T, I, D,
            Y.stride(0), Y.stride(1), Y.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
        )

        return processed_encoder, processed_hidden