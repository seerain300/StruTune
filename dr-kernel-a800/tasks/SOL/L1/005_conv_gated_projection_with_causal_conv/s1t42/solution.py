import torch
import triton
import triton.language as tl


# Triton kernel: Linear projection Y = X @ W^T + bias
# X: (B, S, H), W: (M, H) (here M=H for each of the three groups), bias: (M)
# OUT: (B, S, M)
@triton.jit
def TritonLinearKernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    B, S, M, H,
    BLOCK_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    mask_s = s_offsets < S
    mask_m = m_offsets < M

    acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

    # loop over hidden dimension H to accumulate
    for i in range(0, H):
        x_ptrs = X_ptr + pid_b * (S * H) + s_offsets[:, None] * H + i  # (BLOCK_S, 1)
        x_vals = tl.load(x_ptrs, mask=mask_s[:, None], other=0.0).to(tl.float32)  # (BLOCK_S, 1)

        w_ptrs = W_ptr + m_offsets[None, :] * H + i  # (1, BLOCK_M)
        w_vals = tl.load(w_ptrs, mask=mask_m[None, :], other=0.0).to(tl.float32)  # (1, BLOCK_M)

        acc += x_vals * w_vals  # broadcast over columns

    bias_vals = tl.load(BIAS_ptr + m_offsets, mask=mask_m, other=0.0).to(tl.float32)
    acc += bias_vals[None, :]

    out_ptrs = OUT_ptr + pid_b * (S * M) + s_offsets[:, None] * M + m_offsets[None, :]
    tl.store(out_ptrs, acc, mask=mask_s[:, None] & mask_m[None, :])


# Triton kernel: element-wise gating Bx = B * x_proj over (B, S, H)
@triton.jit
def TritonGateKernel(
    B_ptr, X_ptr, OUT_ptr,
    Bsz, S, H,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
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

    b_vals = tl.load(b_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
    x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)

    out_vals = b_vals * x_vals

    out_ptrs = OUT_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, out_vals, mask=mask_s[:, None] & mask_h[None, :])


# Triton kernel: Grouped causal 1D convolution (groups=H) producing OUT_conv[b, h, s]
# X_conv: [B, S, H] (Bx), conv_weight: [H, 1, 4], conv_bias: [H]
# OUT_conv: [B, H, S]
@triton.jit
def TritonCausalConvKernel(
    Bx_ptr, CONV_W_ptr, CONV_BIAS_ptr, OUT_ptr,
    B, H, S,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # loop over kernel_size = 4, conv_weight[h, 0, k]
    for k in range(0, 4):
        # load Bx[b, s, h] for all s and h
        b_ptrs = Bx_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
        vals = tl.load(b_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)

        # conv_weight[h, 0, k] per h
        w_ptrs = CONV_W_ptr + h_offsets * 4 + k
        w_vals = tl.load(w_ptrs, mask=mask_h, other=0.0).to(tl.float32)

        acc += vals * w_vals[None, :]

    # add bias
    bias_ptrs = CONV_BIAS_ptr + h_offsets
    bias_vals = tl.load(bias_ptrs, mask=mask_h, other=0.0).to(tl.float32)
    acc += bias_vals[None, :]

    out_ptrs = OUT_ptr + pid_b * (H * S) + h_offsets[None, :] * S + s_offsets[:, None]
    tl.store(out_ptrs, acc, mask=mask_s[:, None] & mask_h[None, :])


# Triton kernel: Output gating producing y[b, h, s] = C[b, s, h] * conv_out[b, h, s]
# C: [B, S, H], conv_out: [B, H, S]
@triton.jit
def TritonGateOutKernel(
    C_ptr, CONVOUT_ptr, OUT_ptr,
    B, S, H,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    C_ptrs = C_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    CO_ptrs = CONVOUT_ptr + pid_b * (H * S) + h_offsets[None, :] * S + s_offsets[:, None]

    C_vals = tl.load(C_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
    CO_vals = tl.load(CO_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)

    out_vals = C_vals * CO_vals

    out_ptrs = OUT_ptr + pid_b * (H * S) + h_offsets[None, :] * S + s_offsets[:, None]
    tl.store(out_ptrs, out_vals, mask=mask_s[:, None] & mask_h[None, :])


# Triton kernel: Final linear projection OUT = Y @ OUT_W^T + OUT_BIAS
# Y: (B, H, S), OUT_W: (H, H), OUT_BIAS: (H), OUT: (B, S, H)
@triton.jit
def TritonFinalLinearKernel(
    Y_ptr, OUT_W_ptr, OUT_BIAS_ptr, OUT_ptr,
    B, S, H,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # Loop over hidden dimension H for accumulation
    for i in range(0, H):
        y_ptrs = Y_ptr + pid_b * (H * S) + h_offsets[None, :] * S + s_offsets[:, None]
        y_vals = tl.load(y_ptrs, mask=mask_h[None, :] & mask_s[:, None], other=0.0).to(tl.float32)

        outw_ptrs = OUT_W_ptr + h_offsets[None, :] * H + i  # (1, BLOCK_H)
        outw_vals = tl.load(outw_ptrs, mask=mask_h[None, :], other=0.0).to(tl.float32)

        acc += y_vals * outw_vals  # broadcast over columns

    bias_vals = tl.load(OUT_BIAS_ptr + h_offsets, mask=mask_h, other=0.0).to(tl.float32)
    acc += bias_vals[None, :]

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
        # Ensure contiguity
        x = x.contiguous()
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        conv_weight = conv_weight.contiguous()
        conv_bias = conv_bias.contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        Bsz, S, H = x.shape

        # 1) Three linear projections: B, C, x_proj
        W0 = in_proj_weight[:H, :].contiguous


def run(*args):
    return ModelNew()(*args)
