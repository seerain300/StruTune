import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: Conv2d 3x3, stride=2, padding=1, generic IC -> OC
# X: [B, IC, F_in, T_in], W: [OC, IC, 3, 3], bias: [OC], Y: [B, OC, F_out, T_out]
@triton.jit
def conv3x3_s2_p1_gelu(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, IC, F_in, T_in, OC, F_out, T_out,
    x_sN, x_sC, x_sF, x_sT,
    w_sOC, w_sIC, w_sKH, w_sKW,
    y_sN, y_sOC, y_sF, y_sT,
    BLOCK_F: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # Launch grid: 1D over (b, oc, f_block, t_block)
    total_blocks = B * OC * ((F_out + BLOCK_F - 1) // BLOCK_F) * ((T_out + BLOCK_T - 1) // BLOCK_T)
    pid = tl.program_id(0)

    # Decode pid into (b, oc, f_block, t_block)
    grid_f_blocks = (F_out + BLOCK_F - 1) // BLOCK_F
    grid_t_blocks = (T_out + BLOCK_T - 1) // BLOCK_T
    oc_blocks = OC

    b = pid // (oc_blocks * grid_f_blocks * grid_t_blocks)
    rem = pid % (oc_blocks * grid_f_blocks * grid_t_blocks)
    oc = rem // (grid_f_blocks * grid_t_blocks)
    f_block = rem % (grid_f_blocks * grid_t_blocks) // grid_t_blocks
    t_block = rem % (grid_f_blocks * grid_t_blocks) % grid_t_blocks

    # Compute output indices for this tile
    f_out_start = f_block * BLOCK_F
    t_out_start = t_block * BLOCK_T
    f_out_idx = f_out_start + tl.arange(0, BLOCK_F)[:, None]  # [BF, 1]
    t_out_idx = t_out_start + tl.arange(0, BLOCK_T)[None, :]  # [1, BT]
    out_mask = (f_out_idx < F_out) & (t_out_idx < T_out)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_F, BLOCK_T), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel
    # Note: input padding=1, stride=2
    for ic in range(0, IC):
        for kh in range(3):
            for kw in range(3):
                # Input coordinates with padding=1
                f_in_idx = f_out_idx * 2 + (1 - kh)  # [BF, 1]
                t_in_idx = t_out_idx * 2 + (1 - kw)  # [1, BT]
                # Mask for in-bounds input
                in_mask = (f_in_idx >= 0) & (f_in_idx < F_in) & (t_in_idx >= 0) & (t_in_idx < T_in)
                # Compute input pointers
                x_ptrs = X_ptr + b * x_sN + ic * x_sC + f_in_idx * x_sF + t_in_idx * x_sT
                x_tile = tl.load(x_ptrs, mask=out_mask & in_mask, other=0.0)  # [BF, BT]

                # Load corresponding weights (scalar per (oc, ic, kh, kw))
                w_ptr = W_ptr + oc * w_sOC + ic * w_sIC + kh * w_sKH + kw * w_sKW
                w_val = tl.load(w_ptr)  # scalar

                acc += x_tile * w_val

    # Add bias
    b_val = tl.load(BIAS_ptr + oc)
    acc += b_val

    # GELU activation
    # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # Store result
    y_ptrs = Y_ptr + b * y_sN + oc * y_sOC + f_out_idx * y_sF + t_out_idx * y_sT
    tl.store(y_ptrs, acc, mask=out_mask)


# Triton elementwise kernel: Y = X * scale + P, where X: [B*T, D], P: [T, D] (broadcast over batch), Y: [B*T, D]
@triton.jit
def scale_add_pos_embed_1d(
    X_ptr, P_ptr, Y_ptr, NUMEL, D, scale, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < NUMEL

    # Load X
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # For each element at offset, compute its sequence position and channel index:
    # seq_pos = offset // D, channel = offset % D
    seq_pos = offsets // D
    channel = offsets % D

    # Compute flat index into P: row = seq_pos * D + channel
    row = seq_pos * D + channel

    p = tl.load(P_ptr + row, mask=mask, other=0.0)

    y = x * scale + p

    tl.store(Y_ptr + offsets, y, mask=mask)


# Triton matmul kernel: Y = X @ W, where X: [M, K], W: [K, N], Y: [M, N], no bias.
@triton.jit
def matmul_1d(X_ptr, W_ptr, Y_ptr, M, K, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # 2D grid: (pid_m, pid_n)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        off_k = k0 + tl.arange(0, BLOCK_K)

        a_ptrs = X_ptr + off_m[:, None] * K + off_k[None, :]
        b_ptrs = W_ptr + off_k[:, None] * N + off_n[None, :]

        a = tl.load(a_ptrs, mask=(off_m[:, None] < M) & (off_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(off_k[:, None] < K) & (off_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    y_ptrs = Y_ptr + off_m[:, None] * N + off_n[None, :]
    tl.store(y_ptrs, acc, mask=(off_m[:, None] < M) & (off_n[None, :] < N))


class ModelNew(nn.Module):
    def forward(self, input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        # Ensure device and Triton availability
        device = input_features.device
        assert TRITON_AVAILABLE, "Triton is not available."

        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        x = torch.nn.functional.conv2d(
            input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1
        )
        # Triton GELU
        B, IC, F_in, T_in = x.shape
        OC1 = conv2d1_weight.shape[0]
        F_out1 = (F_in + 2 * 1 - 3) // 2 + 1
        T_out1 = (T_in + 2 * 1 - 3) // 2 + 1
        y1 = torch.empty((B, OC1, F_out1, T_out1), device=device, dtype=torch.float32)
        x_fp32 = x.contiguous().float()
        w1_fp32 = conv2d1_weight.contiguous().float()
        b1_fp32 = conv2d1_bias.contiguous().float()
        grid1 = (B * OC1 * triton.cdiv(F_out1, 32) * triton.cdiv(T_out1, 32),)
        conv3x3_s2_p1_gelu[grid1](
            x_fp32, w1_fp32, b1_fp32, y1,
            B, 1, F_in, T_in, OC1, F_out1, T_out1,
            x_fp32.stride(0), x_fp32.stride(1), x_fp32.stride(2), x_fp32.stride(3),
            w1_fp32.stride(0), w1_fp32.stride(1), w1_fp32.stride(2), w1_fp32.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_F=32, BLOCK_T=32
        )

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        x2 = y1
        OC2 = conv2d2_weight.shape[0]
        F_in2 = F_out1
        T_in2 = T_out1
        F_out2 = (F_in2 + 2 * 1 - 3) // 2 + 1
        T_out2 = (T_in2 + 2 * 1 - 3) // 2 + 1
        y2 = torch.empty((B, OC2, F_out2, T_out2), device=device, dtype=torch.float32)
        w2_fp32 = conv2d2_weight.contiguous().float()
        b2_fp32 = conv2d2_bias.contiguous().float()
        x2_fp32 = x2.contiguous().float()
        grid2 = (B * OC2 * triton.cdiv(F_out2, 32) * triton.cdiv(T_out2, 32),)
        conv3x3_s2_p1_gelu[grid2](
            x2_fp32, w2_fp32, b2_fp32, y2,
            B, OC2, F_in2, T_in2, OC2, F_out2, T_out2,
            x2_fp32.stride(0), x2_fp32.stride(1), x2_fp32.stride(2), x2_fp32.stride(3),
            w2_fp32.stride(0), w2_fp32.stride(1), w2_fp32.stride(2), w2_fp32.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_F=32, BLOCK_T=32
        )

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        x3 = y2
        OC3 = conv2d3_weight.shape[0]
        F_in3 = F_out2
        T_in3 = T_out2
        F_out3 = (F_in3 + 2 * 1 - 3) // 2 + 1
        T_out3 = (T_in3 + 2 * 1 - 3) // 2 + 1
        y3 = torch.empty((B, OC3, F_out3, T_out3), device=device, dtype=torch.float32)
        w3_fp32 = conv2d3_weight.contiguous().float()
        b3_fp32 = conv2d3_bias.contiguous().float()
        x3_fp32 = x3.contiguous().float()
        grid3 = (B * OC3 * triton.cdiv(F_out3, 32) * triton.cdiv(T_out3, 32),)
        conv3x3_s2_p1_gelu[grid3](
            x3_fp32, w3_fp32, b3_fp32, y3,
            B, OC3, F_in3, T_in3, OC3, F_out3, T_out3,
            x3_fp32.stride(0), x3_fp32.stride(1), x3_fp32.stride(2), x3_fp32.stride(3),
            w3_fp32.stride(0), w3_fp32.stride(1), w3_fp32.stride(2), w3_fp32.stride(3),
            y3.stride(0), y3.stride(1), y3.stride(2), y3.stride(3),
            BLOCK_F=32, BLOCK_T=32
        )

        # Reshape: [B, channels, F, T] -> [B, T, channels*F]
        b, c, f, t = y3.shape
        y3 = y3.permute(0, 3, 1, 2).contiguous()  # [B, T, C, F]
        D = c * f  # 384 * 3 = 1152
        y3 = y3.view(b, t, D)  # [B, T, D]

        # Linear projection to d_model (1024) using Triton matmul
        Bsz, Tsz, Ksz = y3.shape  # Bsz=batch_size, Tsz=time_after_conv, Ksz=1152
        d_model = conv_out_weight.shape[0]  # 1024
        X_flat = y3.contiguous().view(Bsz * Tsz, Ksz).float()  # [M, K]
        W_flat = conv_out_weight.contiguous().float()          # [d_model, K]
        Y_matmul = torch.empty((Bsz * Tsz, d_model), device=device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(Bsz * Tsz, BLOCK_M), triton.cdiv(d_model, BLOCK_N))
        matmul_1d[grid](
            X_flat, W_flat, Y_matmul,
            Bsz * Tsz, Ksz, d_model,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

        # Reshape back to [B, T, d_model]
        y_linear = Y_matmul.view(Bsz, Tsz, d_model)

        # Scale
        y_scaled = y_linear * embed_scale  # embed_scale = sqrt(1024) = 32.0

        # Add positional embedding: [Tsz, d_model]
        pos_embed = positional_embedding[:Tsz, :].contiguous().float()  # [Tsz, d_model]
        NUMEL = Bsz * Tsz * d_model
        Ddim = d_model

        # Triton scale + add
        Y_out = torch.empty(NUMEL, device=device, dtype=torch.float32)
        X_out = torch.empty(NUMEL, device=device, dtype=torch.float32)
        # We need to copy y_scaled to X_out for elementwise op
        y_scaled_flat = y_scaled.contiguous().view(-1).float()
        X_out.copy_(y_scaled_flat)
        grid_scale = (triton.cdiv(NUMEL, 1024),)
        scale_add_pos_embed_1d[grid_scale](
            X_out, pos_embed, Y_out,
            NUMEL, Ddim, embed_scale, BLOCK=1024
        )
        y_final = Y_out.view(Bsz, Tsz, d_model)

        return y_final


def run(*args):
    return ModelNew()(*args)
