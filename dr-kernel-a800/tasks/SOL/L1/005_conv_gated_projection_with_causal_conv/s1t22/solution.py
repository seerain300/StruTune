import torch
import triton
import triton.language as tl


# 1) Linear projection via Triton: out[B, S, M] = x @ W[:M, :].T + bias[:M]
# Shapes:
#   x: (B, S, H) contiguous, row-major: [B, S, H]
#   W: (M, H) contiguous, row-major: [M, H]
#   bias: (M)
#   out: (B, S, M) contiguous
@triton.jit
def TritonLinearKernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    B, S, M, H,
    stride_x_b, stride_x_s, stride_x_h,    # strides for x
    stride_w_m, stride_w_h,                # strides for W
    stride_out_b, stride_out_s, stride_out_m,  # strides for out
    BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_b
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    m = pid_m

    acc = tl.zeros([BLOCK_S, 1], dtype=tl.float32)

    # Reduce over H (input features)
    for h in range(0, H):
        x_ptrs = x_ptr + b * stride_x_b + s_offsets * stride_x_s + h * stride_x_h
        x_vals = tl.load(x_ptrs, mask=(s_offsets < S), other=0.0).to(tl.float32)  # [BLOCK_S]
        w_val = tl.load(w_ptr + m * stride_w_m + h * stride_w_h).to(tl.float32)   # scalar
        acc += x_vals[:, None] * w_val

    # Add bias
    bias_val = tl.load(bias_ptr + m).to(tl.float32)
    acc += bias_val

    out_ptrs = out_ptr + b * stride_out_b + s_offsets[:, None] * stride_out_s + m * stride_out_m
    tl.store(out_ptrs, acc, mask=(s_offsets < S)[:, None])


# 2) Element-wise gating: Bx = B * x_proj over (B, S, H)
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

    b_vals = tl.load(b_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
    x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)

    out_vals = b_vals * x_vals

    out_ptrs = OUT_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, out_vals, mask=mask_s[:, None] & mask_h[None, :])


# 3) Left-pad along sequence by PAD for Bx, producing Bx_pad[B, H, S + PAD]
# Inputs: Bx [B, H, S], output: out_pad [B, H, S + PAD], PAD=3
@triton.jit
def TritonPadLeftKernel(
    Bx_ptr, out_ptr,
    B, H, S, PAD,
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_ob_b, stride_ob_h, stride_ob_s,
    BLOCK_S: tl.constexpr,
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


# 4) Grouped causal 1D convolution with groups=H (depthwise), kernel_size=4
# Input: Bx_pad of shape (B, H, S+PAD); conv_weight of shape (H, 1, 4); conv_bias (H)
# Output: conv_out of shape (B, H, S)
@triton.jit
def TritonGroupedCausalConvKernel(
    Bx_pad_ptr, conv_weight_ptr, conv_bias_ptr, out_ptr,
    B, H, S, PAD,  # sizes
    stride_bx_b, stride_bx_h, stride_bx_s,  # strides for Bx_pad
    stride_w_h, stride_w_k,                 # strides for conv_weight (layout assumed (H, 1, 4))
    stride_out_b, stride_out_h, stride_out_s,
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    b = pid_b
    h = pid_h
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    S_out = S + PAD

    acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    # Accumulate sum_{k=0..3} Bx_pad[b, h, s_out + k] * conv_weight[h, 0, k]
    for k in range(0, 4):
        s_out = s_offsets + k  # vector of output positions
        mask = (s_out < S_out) & (s_out >= PAD)  # causal condition: s_out >= PAD
        b_ptrs = Bx_pad_ptr + b * stride_bx_b + h * stride_bx_h + s_out * stride_bx_s
        b_vals = tl.load(b_ptrs, mask=mask, other=0.0).to(tl.float32)  # [BLOCK_S]
        w_ptrs = conv_weight_ptr + h * stride_w_h + 0 * stride_w_h + k * stride_w_k
        w_val = tl.load(w_ptrs).to(tl.float32)  # scalar
        acc += b_vals * w_val

    # Add bias
    bias_val = tl.load(conv_bias_ptr + h).to(tl.float32)
    acc += bias_val

    out_ptrs = out_ptr + b * stride_out_b + h * stride_out_h + s_offsets * stride_out_s
    tl.store(out_ptrs, acc, mask=(s_offsets < S))


# 5) Elementwise multiply: Y = C * conv_out, shapes (B, S, H) and (B, H, S)
# We implement Y[b, s, h] = C[b, s, h] * conv_out[b, h, s]
@triton.jit
def TritonMulKernel(
    C_ptr, conv_out_ptr, Y_ptr,
    B, S, H,
    stride_c_b, stride_c_s, stride_c_h,
    stride_co_b, stride_co_h, stride_co_s,
    stride_y_b, stride_y_s, stride_y_h,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    c_ptrs = C_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    co_ptrs = conv_out_ptr + pid_b * (H * S) + h_offsets[:, None] * H + s_offsets[None, :]

    c_vals = tl.load(c_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0).to(tl.float32)
    co_vals = tl.load(co_ptrs, mask=mask_s[None, :] & mask_h[:, None], other=0.0).to(tl.float32)

    y_vals = c_vals * co_vals

    y_ptrs = Y_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(y_ptrs, y_vals, mask=mask_s[:, None] & mask_h[None, :])


# 6) Final projection: OUT[b, s, m] = sum_h Y[b, s, h] * W[m, h] + bias[m]
# Y: [B, S, H], W: [M, H] (M=H), bias: [M], OUT: [B, S, M]
@triton.jit
def TritonLinearFinalKernel(
    Y_ptr, w_ptr, bias_ptr, out_ptr,
    B, S, M, H,
    stride_y_b, stride_y_s, stride_y_m,
    stride_w_m, stride_w_h,
    stride_out_b, stride_out_s, stride_out_m,
    BLOCK_S: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_b
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    m = pid_m

    acc = tl.zeros([BLOCK_S, 1], dtype=tl.float32)

    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        y_ptrs = Y_ptr + b * stride_y_b + s_offsets[:, None] * stride_y_s + h_offsets[None, :] * stride_y_m
        y_tile = tl.load(y_ptrs, mask=mask_h[None, :], other=0.0).to(tl.float32)  # [BLOCK_S, BLOCK_H]

        w_ptrs = w_ptr + m * stride_w_m + h_offsets * stride_w_h
        w_tile = tl.load(w_ptrs, mask=mask_h, other=0.0).to(tl.float32)  # [BLOCK_H]
        w_tile = w_tile[None, :]  # broadcast across s

        acc += tl.dot(y_tile, w_tile)

    bias_val = tl.load(bias_ptr + m).to(tl.float32)
    acc += bias_val

    out_ptrs = out_ptr + b * stride_out_b + s_offsets[:, None] * stride_out_s + m * stride_out_m
    tl.store(out_ptrs, acc, mask=(s_offsets < S)[:, None])


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
        B, S, H = x.shape
        device = x.device
        dtype = x.dtype

        # Ensure inputs are contiguous
        x = x.contiguous()
        # Compute 3 linear projections via Triton
        # Output shapes: (B, S, H)
        B_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        C_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        XPR_out = torch.empty((B, S, H), device=device, dtype=torch.float32)

        # Launch TritonLinearKernel three times with appropriate slices
        # BLOCK sizes tuned for typical sizes; masks prevent OOB
        BLOCK_S = 256
        BLOCK_H = 128

        # First projection M=H, weight=in_proj_weight[:H, :]
        w1 = in_proj_weight[:H, :].contiguous()
        b1 = in_proj_bias[:H].contiguous()
        TritonLinearKernel[(B, triton.cdiv(S, BLOCK_S), H)](
            x, w1, b1, B_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            w1.stride(0), w1.stride(1),
            B_out.stride(0), B_out.stride(1), B_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_M=1,
        )

        # Second projection M=H, weight=in_proj_weight[H:2H, :]
        w2 = in_proj_weight[H:2 * H, :].contiguous()
        b2 = in_proj_bias[H:2 * H].contiguous()
        TritonLinearKernel[(B, triton.cdiv(S, BLOCK_S), H)](
            x, w2, b2, C_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            w2.stride(0), w2.stride(1),
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_M=1,
        )

        # Third projection M=H, weight=in_proj_weight[2H:3H, :]
        w3 = in_proj_weight[2 * H:3 * H, :].contiguous()
        b3 = in_proj_bias[2 * H:3 * H].contiguous()
        TritonGateKernel[(B, triton.cdiv(S, BLOCK_S), triton.cdiv(H, BLOCK_H))](
            x, x, x,  # placeholder to ensure kernel exists; not used here
            0, S, H,
            BLOCK_S=BLOCK_S, BLOCK_H=1,  # dummy, will be overwritten
        )
        TritonLinearKernel[(B, triton.cdiv(S, BLOCK_S), H)](
            x, w3, b3, XPR_out,
            B, S, H, H,
            x.stride(0), x.stride(1), x.stride(2),
            w3.stride(0), w3.stride(1),
            XPR_out.stride(0), XPR_out.stride(1), XPR_out.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_M=1,
        )

        # 2) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)
        TritonGateKernel[(B, triton.cdiv(S, BLOCK_S), triton.cdiv(H, BLOCK_H))](
            B_out, XPR_out, Bx,
            B, S, H,
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H,
        )

        # 3) Left-pad along sequence by 3
        PAD = 3
        Bx_pad = torch.empty((B, H, S + PAD), device=device, dtype=torch.float32)
        TritonPadLeftKernel[(B, H, triton.cdiv(S + PAD, BLOCK_S))](
            Bx, Bx_pad,
            B, H, S, PAD,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            BLOCK_S=BLOCK_S,
        )

        # 4) Grouped causal 1D conv: conv_out[b, h, s] = sum_{k=0..3} Bx_pad[b, h, s + k] * conv_weight[h, 0, k] + conv_bias[h]
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)
        # conv_weight shape: (H, 1, 4) — ensure contiguous
        wgt = conv_weight.contiguous()
        bias = conv_bias.contiguous()
        TritonGroupedCausalConvKernel[(B, H, triton.cdiv(S, BLOCK_S))](
            Bx_pad, wgt, bias, conv_out,
            B, H, S, PAD,
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            wgt.stride(0), wgt.stride(2),  # wgt layout assumed (H, 1, 4)
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=BLOCK_S,
        )

        # 5) Output gating: y = C * conv_out
        Y = torch.empty((B, S, H), device=device, dtype=torch.float32)
        TritonMulKernel[(B, triton.cdiv(S, BLOCK_S), triton.cdiv(H, BLOCK_H))](
            C_out, conv_out, Y,
            B, S, H,
            C_out.stride(0), C_out.stride(1), C_out.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H,
        )

        # 6) Final projection: F.linear(Y, out_proj_weight, out_proj_bias)
        # Implement in Triton: out[b, s, m] = sum_h Y[b, s, h] * out_proj_weight[m, h] + bias[m]
        OUT = torch.empty((B, S, H), device=device, dtype=torch.float32)
        w_out = out_proj_weight.contiguous()  # (H, H)
        b_out = out_proj_bias.contiguous()    # (H)
        TritonLinearFinalKernel[(B, triton.cdiv(S, BLOCK_S), H)](
            Y, w_out, b_out, OUT,
            B, S, H, H,
            Y.stride(0), Y.stride(1), Y.stride(2),
            w_out.stride(0), w_out.stride(1),
            OUT.stride(0), OUT.stride(1), OUT.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_M=1, BLOCK_H=BLOCK_H,
        )

        # Return final output
        return OUT


def run(*args):
    return ModelNew()(*args)
