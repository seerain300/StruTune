import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton kernel: linear projection Y[B, S, M] = X[B, S, H] @ W[M, H].T + bias[M]
# X: (B, S, H), contiguous; W: (M, H), contiguous; bias: (M); Y: (B, S, M), contiguous
@triton.jit
def TritonLinearProjectionKernel(
    X_ptr, W_ptr, Bias_ptr, Y_ptr,
    B, S, M, H,
    BLOCK_S: tl.constexpr,  # tile size along S
    BLOCK_H: tl.constexpr,  # tile size along M
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    m_offsets = pid_m * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_m = m_offsets < M

    acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

    # Reduction over H (input feature dimension)
    for h in range(0, H):
        # load X[b, s, h] for all s in tile, shape (BLOCK_S, 1)
        x_ptrs = X_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h
        x_vals = tl.load(x_ptrs, mask=mask_s[:, None], other=0.0)

        # load W[m, h] for all m in tile, shape (1, BLOCK_M)
        w_ptrs = W_ptr + m_offsets[None, :] * H + h
        w_vals = tl.load(w_ptrs, mask=mask_m[None, :], other=0.0)

        # outer product accumulate
        acc += x_vals * w_vals

    # add bias
    bias_ptrs = Bias_ptr + m_offsets
    bias_vals = tl.load(bias_ptrs, mask=mask_m, other=0.0)
    acc += bias_vals[None, :]

    # store Y[b, s, m] tile
    y_ptrs = Y_ptr + pid_b * (S * M) + s_offsets[:, None] * M + m_offsets[None, :]
    tl.store(y_ptrs, acc, mask=mask_s[:, None] & mask_m[None, :])


# Triton kernel: element-wise gating out = B * X over (B, S, H)
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

    b_vals = tl.load(b_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)
    x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)

    out_vals = b_vals * x_vals

    out_ptrs = OUT_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, out_vals, mask=mask_s[:, None] & mask_h[None, :])


# Triton kernel: left-pad along sequence by PAD for input to conv
# Inputs: Bx [B, H, S], output: out_pad [B, H, S + PAD], PAD=3
@triton.jit
def TritonPadLeftKernel(
    Bx_ptr, out_ptr,
    B, H, S, PAD,
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_ob_b, stride_ob_h, stride_ob_s,
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
        tl.store(out_ptr + b * stride_ob_b + h * stride_ob_h + i * stride_ob_s, 0.0)

    # copy from Bx[:, :, :] into out[:, :, PAD:]
    for i in range(0, BLOCK_S):
        s_in = s_out_start + i
        if s_in < S:
            val = tl.load(Bx_ptr + b * stride_bx_b + h * stride_bx_h + s_in * stride_bx_s)
            tl.store(out_ptr + b * stride_ob_b + h * stride_ob_h + (s_in + PAD) * stride_ob_s, val)


def _launch_linear_projection(x, w, bias, out):
    """
    x: [B, S, H] contiguous float32
    w: [M, H] contiguous float32
    bias: [M] contiguous float32
    out: [B, S, M] contiguous float32
    """
    B, S, H = x.shape
    M = w.shape[0]
    out.zero_()
    BLOCK_S = 128
    BLOCK_M = 64
    grid = (B, triton.cdiv(S, BLOCK_S), triton.cdiv(M, BLOCK_M))
    TritonLinearProjectionKernel[grid](
        x, w, bias, out,
        B, S, M, H,
        BLOCK_S=BLOCK_S, BLOCK_M=BLOCK_M,
        num_warps=4,
    )


def _launch_gate(b, x_proj, out):
    """
    out = b * x_proj elementwise
    """
    B, S, H = b.shape
    BLOCK_S = 128
    BLOCK_H = 64
    grid = (B, triton.cdiv(S, BLOCK_S), triton.cdiv(H, BLOCK_H))
    TritonGateKernel[grid](
        b, x_proj, out,
        B, S, H,
        BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H,
        num_warps=4,
    )


def _launch_pad_left(bx, out_pad):
    """
    out_pad[:, :, :PAD] = 0, out_pad[:, :, PAD:] = bx
    """
    B, H, S = bx.shape
    PAD = 3
    S_out = S + PAD
    grid = (B, H, triton.cdiv(S_out, 128))
    TritonPadLeftKernel[grid](
        bx, out_pad,
        B, H, S, PAD,
        bx.stride(0), bx.stride(1), bx.stride(2),
        out_pad.stride(0), out_pad.stride(1), out_pad.stride(2),
        BLOCK_S=128,
        num_warps=4,
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        """
        x: (B, S, H), float32, CUDA
        in_proj_weight: (3H, H), float32, CUDA
        in_proj_bias: (3H), float32, CUDA
        conv_weight: (H, 1, 4), float32, CUDA
        conv_bias: (H), float32, CUDA
        out_proj_weight: (H, H), float32, CUDA
        out_proj_bias: (H), float32, CUDA
        """
        B, S, H = x.shape

        # 1) Three linear projections via Triton (M=H each)
        x = x.contiguous()
        # First projection: B
        w1 = in_proj_weight[:H, :].contiguous()
        b1 = in_proj_bias[:H].contiguous()
        B_t = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        _launch_linear_projection(x, w1, b1, B_t)

        # Second projection: C
        w2 = in_proj_weight[H:2 * H, :].contiguous()
        b2 = in_proj_bias[H:2 * H].contiguous()
        C_t = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        _launch_linear_projection(x, w2, b2, C_t)

        # Third projection: x_proj
        w3 = in_proj_weight[2 * H:3 * H, :].contiguous()
        b3 = in_proj_bias[2 * H:3 * H].contiguous()
        x_proj_t = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        _launch_linear_projection(x, w3, b3, x_proj_t)

        # 2) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        _launch_gate(B_t, x_proj_t, Bx)

        # 3) Left-pad along S by PAD=3 for causal conv
        Bx_pad = torch.empty((B, H, S + 3), device=x.device, dtype=torch.float32)
        _launch_pad_left(Bx, Bx_pad)

        # 4) Grouped causal 1D convolution with groups=H
        # Using PyTorch for robustness: pad already done as zeros left
        # Input: Bx_pad [B, H, S+3]; weight: (H, 1, 4); bias: (H); groups=H
        conv_out = F.conv1d(Bx_pad, conv_weight, conv_bias, stride=1, padding=0, dilation=1, groups=H)  # (B, H, S)
        # conv_out: (B, H, S)

        # 5) Output gating: y = C * conv_out, shapes (B, S, H) * (B, H, S)
        # We need conv_out (B, H, S) multiplied elementwise with C_t (B, S, H). Use broadcasting by permuting.
        conv_out_t = conv_out.permute(0, 2, 1).contiguous()  # (B, S, H)
        y = C_t * conv_out_t  # elementwise (B, S, H)

        # 6) Final output projection: F.linear(y, out_proj_weight, out_proj_bias)
        output = F.linear(y, out_proj_weight, out_proj_bias)

        return output


def run(*args):
    return ModelNew()(*args)
