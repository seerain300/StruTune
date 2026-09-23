import torch
import triton
import triton.language as tl


# Triton kernel: Linear projection Y = X @ W + b
# X: (B, S, H) contiguous; W: (M, H) contiguous; bias: (M); Y: (B, S, H) contiguous
@triton.jit
def TritonLinearKernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    Bsz, S, H,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    # Accumulator for the [BLOCK_S, BLOCK_H] tile (float32)
    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # Loop over H dimension: Y[b, s, h] = sum_i X[b, s, i] * W[h, i] + bias[h]
    for i in range(0, H):
        x_ptrs = X_ptr + pid_b * (S * H) + s_offsets[:, None] * H + i  # [BLOCK_S, 1]
        w_ptrs = W_ptr + h_offsets[None, :] * H + i                    # [1, BLOCK_H]
        x_vals = tl.load(x_ptrs, mask=mask_s[:, None], other=0.0)      # [BLOCK_S, 1]
        w_vals = tl.load(w_ptrs, mask=mask_h[None, :], other=0.0)      # [1, BLOCK_H]
        # Broadcast multiply and accumulate
        acc += x_vals * w_vals

    # Add bias
    bias_vals = tl.load(BIAS_ptr + h_offsets, mask=mask_h, other=0.0)  # [BLOCK_H]
    acc = acc + bias_vals[None, :]

    # Store result to Y
    y_ptrs = Y_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(y_ptrs, acc, mask=mask_s[:, None] & mask_h[None, :])


# Triton kernel: element-wise gating Bx = B * X over (B, S, H)
@triton.jit
def TritonGateKernel(
    B_ptr, X_ptr, OUT_ptr,
    Bsz, S, H,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    b_ptrs = B_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    x_ptrs = X_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]

    b_vals = tl.load(b_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)
    x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)

    out_vals = b_vals * x_vals

    out_ptrs = OUT_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, out_vals, mask=mask_s[:, None] & mask_h[None, :])


# Triton kernel: left-pad along sequence by PAD for Bx, producing Bx_pad[B, H, S + PAD]
# Inputs: Bx [B, H, S] contiguous, output: out_pad [B, H, S + PAD] contiguous, PAD=3
@triton.jit
def TritonPadLeftKernel(
    Bx_ptr, out_ptr,
    B, H, S, PAD,  # sizes
    BLOCK_S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_sp = tl.program_id(2)  # tiling over S_out = S + PAD

    b = pid_b
    h = pid_h
    S_out = S + PAD

    s_out_start = pid_sp * BLOCK_S

    # write zeros at the first PAD columns
    for i in range(0, PAD):
        out_pos = out_ptr + b * (H * S_out) + h * S_out + i
        tl.store(out_pos, 0.0)

    # copy from Bx[:, :, :] into out[:, :, PAD:]
    for i in range(0, BLOCK_S):
        s_in = s_out_start + i
        if s_in < S:
            val = tl.load(Bx_ptr + b * (H * S) + h * S + s_in)
            tl.store(out_ptr + b * (H * S_out) + h * S_out + (s_in + PAD), val)


# Triton kernel: Grouped causal 1D convolution with groups=H, kernel_size=4
# Input: Bx_pad of shape (B, H, S+3); conv_weight of shape (H, 1, 4); conv_bias (H)
# Output: conv_out of shape (B, H, S)
@triton.jit
def TritonCausalConvKernel(
    Bx_pad_ptr, conv_weight_ptr, conv_bias_ptr, out_ptr,
    B, H, S, PAD,  # sizes
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    # Accumulator for conv result tile [BLOCK_S, BLOCK_H]
    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # For each s in tile, compute sum over k=0..3 of Bx_pad[b, h, s+k] * conv_weight[h, 0, k]
    for i in range(0, BLOCK_S):
        s_i = s_offsets[i]
        valid = s_i < S
        # k=0
        pos0 = Bx_pad_ptr + b * (H * (S + PAD)) + h_offsets[:, None] * (S + PAD) + (s_i + 0)
        val0 = tl.load(pos0, mask=mask_h[None, :] & valid, other=0.0)
        # k=1
        pos1 = Bx_pad_ptr + b * (H * (S + PAD)) + h_offsets[:, None] * (S + PAD) + (s_i + 1)
        val1 = tl.load(pos1, mask=mask_h[None, :] & valid, other=0.0)
        # k=2
        pos2 = Bx_pad_ptr + b * (H * (S + PAD)) + h_offsets[:, None] * (S + PAD) + (s_i + 2)
        val2 = tl.load(pos2, mask=mask_h[None, :] & valid, other=0.0)
        # k=3
        pos3 = Bx_pad_ptr + b * (H * (S + PAD)) + h_offsets[:, None] * (S + PAD) + (s_i + 3)
        val3 = tl.load(pos3, mask=mask_h[None, :] & valid, other=0.0)

        # Load conv weights for h tile
        w0 = tl.load(conv_weight_ptr + h_offsets[:, None] * 4 + 0)  # [BLOCK_H, 1]
        w1 = tl.load(conv_weight_ptr + h_offsets[:, None] * 4 + 1)  # [BLOCK_H, 1]
        w2 = tl.load(conv_weight_ptr + h_offsets[:, None] * 4 + 2)  # [BLOCK_H, 1]
        w3 = tl.load(conv_weight_ptr + h_offsets[:, None] * 4 + 3)  # [BLOCK_H, 1]

        # Convert to 1D for multiply and sum over H
        acc[i, :] += (val0 + val1 + val2 + val3) * (w0 + w1 + w2 + w3)  # broadcast over H

    # Add bias
    bias_vals = tl.load(conv_bias_ptr + h_offsets, mask=mask_h, other=0.0)  # [BLOCK_H]
    acc = acc + bias_vals[None, :]

    # Store conv_out[b, h, s]
    out_ptrs = out_ptr + pid_b * (H * S) + h_offsets[:, None] * S + s_offsets[None, :]
    tl.store(out_ptrs, acc, mask=mask_s[None, :] & mask_h[:, None])


# Triton kernel: final linear projection output = Y @ out_proj_weight.T + out_proj_bias
@triton.jit
def TritonFinalLinearKernel(
    Y_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    Bsz, S, H,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # Y[b, s, i] * W[h, i] + bias[h]
    for i in range(0, H):
        y_ptrs = Y_ptr + pid_b * (S * H) + s_offsets[:, None] * H + i
        w_ptrs = W_ptr + h_offsets[None, :] * H + i

        y_vals = tl.load(y_ptrs, mask=mask_s[:, None], other=0.0)
        w_vals = tl.load(w_ptrs, mask=mask_h[None, :], other=0.0)

        acc += (y_vals * w_vals)

    bias_vals = tl.load(BIAS_ptr + h_offsets, mask=mask_h, other=0.0)
    acc = acc + bias_vals[None, :]

    out_ptrs = OUT_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, acc, mask=mask_s[:, None] & mask_h[None, :])


class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
    ):
        # Ensure contiguity for safe 1D indexing in Triton
        x = x.contiguous()
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        conv_weight = conv_weight.contiguous()
        conv_bias = conv_bias.contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        Bsz, S, H = x.shape

        # 1) Three linear projections
        # First group: (H, H)
        W0 = in_proj_weight[:H, :].contiguous()
        b0 = in_proj_bias[:H].contiguous()
        B = torch.empty((Bsz, S, H), device=x.device, dtype=x.dtype)
        TritonLinearKernel[(Bsz, triton.cdiv(S, 64), triton.cdiv(H, 64))](x, W0, b0, B, Bsz, S, H, 64, 64)

        # Second group: (H, H)
        W1 = in_proj_weight[H:2 * H, :].contiguous()
        b1 = in_proj


def run(*args):
    return ModelNew()(*args)
