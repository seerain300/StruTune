import torch
import triton
import triton.language as tl


# 1) Linear projection via Triton: Y[b, s, h] = sum_i X[b, s, i] * W[h, i] + bias[h]
# Shapes:
#   X: (B, S, H) contiguous
#   W: (M, H) contiguous, typically M=H for each of the three projections
#   bias: (M)
#   Y: (B, S, M)
@triton.jit
def TritonLinearKernel(
    X_ptr, W_ptr, Bias_ptr, Y_ptr,
    B, S, M, H,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_m, stride_w_h,
    stride_y_b, stride_y_s, stride_y_h,
    BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    mask_s = s_offsets < S
    mask_m = m_offsets < M

    acc = tl.zeros((BLOCK_S, BLOCK_M), dtype=tl.float32)

    # loop over hidden dimension H to compute the dot product
    for k in range(0, H):
        # X[b, s, k]
        x_ptrs = X_ptr + pid_b * stride_x_b + s_offsets[:, None] * stride_x_s + k * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask_s[:, None], other=0.0)
        # W[m, k]
        w_ptrs = W_ptr + m_offsets[None, :] * stride_w_m + k * stride_w_h
        w_vals = tl.load(w_ptrs, mask=mask_m[None, :], other=0.0)
        acc += x_vals * w_vals

    # add bias
    b_ptrs = Bias_ptr + m_offsets[None, :]
    bias_vals = tl.load(b_ptrs, mask=mask_m[None, :], other=0.0)
    acc += bias_vals[None, :]

    # store to Y[b, s, m]
    y_ptrs = Y_ptr + pid_b * stride_y_b + s_offsets[:, None] * stride_y_s + m_offsets[None, :] * stride_y_h
    tl.store(y_ptrs, acc, mask=mask_s[:, None] & mask_m[None, :])


# 2) Element-wise gating: OUT[b, s, h] = B[b, s, h] * X_proj[b, s, h]
@triton.jit
def TritonGateKernel(
    B_ptr, X_ptr, OUT_ptr,
    Bsz, S, H,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr,
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


# 3) Left-pad along sequence by PAD for causal conv: OUT[B, H, S+PAD]
# This kernel expects input as (B, H, S) and produces (B, H, S+PAD)
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
        out_ptr_pos = out_ptr + b * stride_ob_b + h * stride_ob_h + i * stride_ob_s
        tl.store(out_ptr_pos, 0.0)

    # copy from Bx[:, :, :] into out[:, :, PAD:]
    for i in range(0, BLOCK_S):
        s_in = s_out_start + i
        if s_in < S_out:
            val = tl.load(Bx_ptr + b * stride_bx_b + h * stride_bx_h + (s_in - PAD) * stride_bx_s)
            tl.store(out_ptr + b * stride_ob_b + h * stride_ob_h + s_in * stride_ob_s, val)


# 4) Grouped causal 1D convolution (depthwise groups=H) with kernel_size=KW=4:
# Input Bx_pad: (B, H, S+3), conv_weight: (H, 1, 4) flattened as (H, 4), conv_bias: (H)
# Output conv_out: (B, H, S)
@triton.jit
def TritonCausalConvKernel(
    Bx_pad_ptr, conv_w_ptr, conv_b_ptr, out_ptr,
    Bsz, S, H, PAD, KW,  # KW=4
    stride_bxp_b, stride_bxp_h, stride_bxp_s,
    stride_ow_h, stride_ow_s,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # Loop over kernel window k=0..KW-1
    for k in range(0, KW):
        s_pos = s_offsets + PAD - k  # output s corresponds to input s + k in padded Bx
        valid = (s_pos >= 0) & (s_pos < S)
        bx_ptrs = Bx_pad_ptr + pid_b * stride_bxp_b + h_offsets[None, :] * stride_bxp_h + s_pos[:, None] * stride_bxp_s
        bx_vals = tl.load(bx_ptrs, mask=mask_s[:, None] & valid[:, None], other=0.0)

        # conv_weight[h, k]
        cw_ptrs = conv_w_ptr + h_offsets[None, :] * stride_ow_h + k * stride_ow_s
        cw_vals = tl.load(cw_ptrs, mask=mask_h[None, :], other=0.0)
        acc += bx_vals * cw_vals[None, :]

    # add bias
    bias_ptrs = conv_b_ptr + h_offsets[None, :]
    bias_vals = tl.load(bias_ptrs, mask=mask_h[None, :], other=0.0)
    acc += bias_vals[None, :]

    # store conv_out[b, h, s]
    out_ptrs = out_ptr + pid_b * stride_ow_h + h_offsets[None, :] * stride_ow_h + s_offsets[:, None] * stride_ow_s
    tl.store(out_ptrs, acc, mask=mask_s[:, None] & mask_h[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        x: (B, S, H)
        in_proj_weight: (3H, H), in_proj_bias: (3H)
        conv_weight: (H, 1, 4), conv_bias: (H)
        out_proj_weight: (H, H), out_proj_bias: (H)
        Returns: (B, S, H)
        """

        B, S, H = x.shape
        device = x.device

        # Ensure contiguous for Triton kernels
        x_contig = x.contiguous()
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        conv_weight = conv_weight.contiguous()
        conv_bias = conv_bias.contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        # 1) Three linear projections using Triton: B, C, x_proj
        # First projection: B = linear(x, in_proj_weight[:H, :], in_proj_bias[:H])
        W1 = in_proj_weight[:H, :].to(torch.float32)  # (H, H)
        b1 = in_proj_bias[:H].to(torch.float32)      # (H)
        B = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid1 = (B, triton.cdiv(S, 128), triton.cdiv(H, 64))
        TritonLinearKernel[grid1](
            x_contig, W1, b1, B,
            B, S, H, H,
            x_contig.stride(0), x_contig.stride(1), x_contig.stride(2),
            W1.stride(0), W1.stride(1),
            B.stride(0), B.stride(1), B.stride(2),
            BLOCK_S=128, BLOCK_M=64,
        )

        # Second projection: C = linear(x, in_proj_weight[H:2H, :], in_proj_bias[H:2H])
        W2 = in_proj_weight[H:2 * H, :].to(torch.float32)  # (H, H)
        b2 = in_proj_bias[H:2 * H].to(torch.float32)       # (H)
        C = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid2 = (B, triton.cdiv(S, 128), triton.cdiv(H, 64))
        TritonLinearKernel[grid2](
            x_contig, W2, b2, C,
            B, S, H, H,
            x_contig.stride(0), x_contig.stride(1), x_contig.stride(2),
            W2.stride(0), W2.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_S=128, BLOCK_M=64,
        )

        # Third projection: x_proj = linear(x, in_proj_weight[2H:3H, :], in_proj_bias[2H:3H])
        W3 = in_proj_weight[2 * H:3 * H, :].to(torch.float32)  # (H, H)
        b3 = in_proj_bias[2 * H:3 * H].to(torch.float32)       # (H)
        X_proj = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid3 = (B, triton.cdiv(S, 128), triton.cdiv(H, 64))
        TritonLinearKernel[grid3](
            x_contig, W3, b3, X_proj,
            B, S, H, H,
            x_contig.stride(0), x_contig.stride(1), x_contig.stride(2),
            W3.stride(0), W3.stride(1),
            X_proj.stride(0), X_proj.stride(1), X_proj.stride(2),
            BLOCK_S=128, BLOCK_M=64,
        )

        # 2) Element-wise gating Bx = B * X_proj using Triton
        Bx = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid_gate = (B, triton.cdiv(S, 128), triton.cdiv(H, 64))
        TritonGateKernel[grid_gate](
            B, X_proj, Bx,
            B, S, H,
            BLOCK_S=128, BLOCK_H=64,
        )

        # 3) Pad Bx by 3 for causal conv (transpose to (B, H, S) for kernel)
        Bx_t = Bx.transpose(1, 2).contiguous()  # (B, H, S)
        Bx_pad = torch.empty((B, H, S + 3), dtype=torch.float32, device=device)
        grid_pad = (B, H, triton.cdiv(S + 3, 256))
        TritonPadLeftKernel[grid_pad](
            Bx_t, Bx_pad,
            B, H, S, 3,
            Bx_t.stride(0), Bx_t.stride(1), Bx_t.stride(2),
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            BLOCK_S=256,
        )

        # 4) Grouped causal conv: conv_out (B, H, S)
        conv_weight_flat = conv_weight.view(H, 4).contiguous().to(torch.float32)  # (H, 4)
        conv_bias_t = conv_bias.contiguous().to(torch.float32)                    # (H)
        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=device)
        grid_conv = (B, H, triton.cdiv(S, 128))
        TritonCausalConvKernel[grid_conv](
            Bx_pad, conv_weight_flat, conv_bias_t, conv_out,
            B, S, H, 3, 4,
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            1, 1,  # conv_weight_flat strides: (H, 4)
            BLOCK_S=128, BLOCK_H=64,
        )

        # 5) Output gating: y = C * conv_out, shapes C:(B,S,H), conv_out:(B,H,S)
        y = C * conv_out.transpose(1, 2)  # (B, S, H), elementwise multiply

        # 6) Final projection: out = linear(y, out_proj_weight, out_proj_bias)
        out = torch.nn.functional.linear(y, out_proj_weight, out_proj_bias)
        return out


def run(*args):
    return ModelNew()(*args)
