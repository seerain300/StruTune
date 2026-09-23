import torch
import triton
import triton.language as tl


# Triton kernel: Linear projection Y[b, s, h] = sum_i X[b, s, i] * W[h, i] + bias[h]
# X: (B, S, H), W: (H, H) slice for the group, bias: (H), Y: (B, S, H)
@triton.jit
def TritonLinearKernel(
    X_ptr, W_ptr, Bias_ptr, Y_ptr,
    B, S, H,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_h, stride_w_k,  # W is (H, H) but we use strides for h and k
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

    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # reduction over hidden dimension H
    for k in range(0, H):
        x_ptrs = X_ptr + pid_b * stride_x_b + s_offsets[:, None] * stride_x_s + k * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask_s[:, None], other=0.0)
        w_ptrs = W_ptr + h_offsets[None, :] * stride_w_h + k * stride_w_k
        w_vals = tl.load(w_ptrs, mask=mask_h[None, :], other=0.0)
        acc += x_vals * w_vals

    # add bias
    bias_vals = tl.load(Bias_ptr + h_offsets[None, :], mask=mask_h[None, :], other=0.0)
    acc += bias_vals[None, :]

    # store
    y_ptrs = Y_ptr + pid_b * stride_y_b + s_offsets[:, None] * stride_y_s + h_offsets[None, :] * stride_y_h
    tl.store(y_ptrs, acc, mask=mask_s[:, None] & mask_h[None, :])


# Triton kernel: Element-wise gating OUT[b, s, h] = B[b, s, h] * X_proj[b, s, h]
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


# Triton kernel: Pad along sequence by PAD for Bx, producing OUT[B, H, S + PAD]
# Input Bx: (B, H, S), Output OUT: (B, H, S + PAD), PAD=3
@triton.jit
def TritonPadLeftKernel(
    Bx_ptr, OUT_ptr,
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
        out_ptr_pos = OUT_ptr + b * stride_ob_b + h * stride_ob_h + i * stride_ob_s
        tl.store(out_ptr_pos, 0.0)

    # copy from Bx[:, :, :] into OUT[:, :, PAD:]
    for i in range(0, BLOCK_S):
        s_in = s_out_start + i
        if s_in < S:
            val = tl.load(Bx_ptr + b * stride_bx_b + h * stride_bx_h + s_in * stride_bx_s)
            tl.store(OUT_ptr + b * stride_ob_b + h * stride_ob_h + (s_in + PAD) * stride_ob_s, val)


# Triton kernel: Grouped causal 1D convolution (groups=H, kernel_size=4)
# Inputs:
#   Bx_pad: (B, H, S + PAD), conv_weight_flat: (H, 4), conv_bias: (H)
# Output: conv_out: (B, H, S)
@triton.jit
def TritonCausalConvKernel(
    Bx_pad_ptr, conv_weight_ptr, conv_bias_ptr, out_ptr,
    B, S, H, PAD, KW,
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_w_h, stride_w_k,  # conv_weight_flat strides (H, 4)
    stride_out_b, stride_out_h, stride_out_s,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # for each k in kernel, read Bx_pad[b, h, s + k] and weight[h, k], sum over k
    for k in range(0, KW):
        s_in_offsets = s_offsets + k  # causal input index
        mask_s_in = (s_in_offsets >= PAD) & (s_in_offsets < (S + PAD)) & mask_s
        bx_ptrs = Bx_pad_ptr + pid_b * stride_bx_b + h_offsets[None, :] * stride_bx_h + s_in_offsets[:, None] * stride_bx_s
        bx_vals = tl.load(bx_ptrs, mask=mask_s_in[ :, None] & mask_h[None, :], other=0.0)
        w_ptrs = conv_weight_ptr + h_offsets[None, :] * stride_w_h + k * stride_w_k
        w_vals = tl.load(w_ptrs, mask=mask_h[None, :], other=0.0)
        acc += bx_vals * w_vals

    # add bias
    bias_ptrs = conv_bias_ptr + h_offsets[None, :]
    bias_vals = tl.load(bias_ptrs, mask=mask_h[None, :], other=0.0)
    acc += bias_vals[None, :]

    # store conv_out[b, h, s]
    out_ptrs = out_ptr + pid_b * stride_out_b + s_offsets[:, None] * stride_out_s + h_offsets[None, :] * stride_out_h
    tl.store(out_ptrs, acc, mask=mask_s[:, None] & mask_h[None, :])


# Triton kernel: Final linear projection Y_out[b, s, h] = sum_k Y[b, s, k] * out_proj_weight[h, k] + out_proj_bias[h]
# Y: (B, S, H), out_proj_weight: (H, H), out_proj_bias: (H), Y_out: (B, S, H)
@triton.jit
def TritonLinearFinalKernel(
    Y_ptr, W_ptr, Bias_ptr, Out_ptr,
    B, S, H,
    stride_y_b, stride_y_s, stride_y_h,
    stride_w_h, stride_w_k,  # W is (H, H)
    stride_out_b, stride_out_s, stride_out_h,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # reduction over hidden dimension H (input Y has H features)
    for k in range(0, H):
        y_ptrs = Y_ptr + pid_b * stride_y_b + s_offsets[:, None] * stride_y_s + k * stride_y_h
        y_vals = tl.load(y_ptrs, mask=mask_s[:, None], other=0.0)
        w_ptrs = W_ptr + h_offsets[None, :] * stride_w_h + k * stride_w_k
        w_vals = tl.load(w_ptrs, mask=mask_h[None, :], other=0.0)
        acc += y_vals * w_vals

    # add bias
    bias_vals = tl.load(Bias_ptr + h_offsets[None, :], mask=mask_h[None, :], other=0.0)
    acc += bias_vals[None, :]

    # store output
    out_ptrs = Out_ptr + pid_b * stride_out_b + s_offsets[:, None] * stride_out_s + h_offsets[None, :] * stride_out_h
    tl.store(out_ptrs, acc, mask=mask_s[:, None] & mask_h[None, :])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

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

        device = x.device
        B, S, H = x.shape

        # Ensure inputs are contiguous
        x = x.contiguous()
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        conv_weight = conv_weight.contiguous()
        conv_bias = conv_bias.contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        # 1) Three linear projections using Triton
        # First projection: B = linear(x, in_proj_weight[:H, :], in_proj_bias[:H]) → (B, S, H)
        W1 = in_proj_weight[:H, :].contiguous()
        b1 = in_proj_bias[:H].contiguous()
        B = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid1 = (B, triton.cdiv(S, 128), triton.cdiv(H, 64))
        TritonLinearKernel[grid1](
            x, W1, b1, B,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W1.stride(0), 1,  # W1 is (H, H); stride along H=1, stride along K=1
            B.stride(0), B.stride(1), B.stride(2),
            BLOCK_S=128, BLOCK_H=64,
        )

        # Second projection: C = linear(x, in_proj_weight[H:2H, :], in_proj_bias[H:2H]) → (B, S, H)
        W2 = in_proj_weight[H:2 * H, :].contiguous()
        b2 = in_proj_bias[H:2 * H].contiguous()
        C = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid2 = (B, triton.cdiv(S, 128), triton.cdiv(H, 64))
        TritonLinearKernel[grid2](
            x, W2, b2, C,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W2.stride(0), 1,
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_S=128, BLOCK_H=64,
        )

        # Third projection: x_proj = linear(x, in_proj_weight[2H:3H, :], in_proj_bias[2H:3H]) → (B, S, H)
        W3 = in_proj_weight[2 * H:3 * H, :].contiguous()
        b3 = in_proj_bias[2 * H:3 * H].contiguous()
        X_proj = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid3 = (B, triton.cdiv(S, 128), triton.cdiv(H, 64))
        TritonLinearKernel[grid3](
            x, W3, b3, X_proj,
            B, S, H,
            x.stride(0), x.stride(1), x.stride(2),
            W3.stride(0), 1,
            X_proj.stride(0), X_proj.stride(1), X_proj.stride(2),
            BLOCK_S=128, BLOCK_H=64,
        )

        # 2) Element-wise gating Bx = B * X_proj
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
        conv_weight_flat = conv_weight.view(H, 4).contiguous()  # (H, 4)
        conv_bias_t = conv_bias.contiguous()                    # (H)
        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=device)
        grid_conv = (B, H, triton.cdiv(S, 128))
        TritonCausalConvKernel[grid_conv](
            Bx_pad, conv_weight_flat, conv_bias_t, conv_out,
            B, S, H, 3, 4,
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            conv_weight_flat.stride(0), conv_weight_flat.stride(1),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=128, BLOCK_H=64,
        )

        # 5) Output gating: y = C * conv_out, shapes C:(B,S,H), conv_out:(B,H,S)
        # We perform this elementwise multiply using PyTorch since Triton kernel was not defined previously.
        # Note: conv_out is (B, H, S); C is (B, S, H). Elementwise multiply across (B,S,H) per h:
        y = C * conv_out.transpose(1, 2)  # (B, S, H)

        # 6) Final projection: out = linear(y, out_proj_weight, out_proj_bias) → (B, S, H)
        out = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid_final = (B, triton.cdiv(S, 128), triton.cdiv(H, 64))
        TritonLinearFinalKernel[grid_final](
            y, out_proj_weight, out_proj_bias, out,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), 1,  # (H, H)
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_S=128, BLOCK_H=64,
        )

        return out


def run(*args):
    return ModelNew()(*args)
