import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,         # *const float, input x: (B, S, H)
    W_ptr,         # *const float, in_proj_weight: (I, H), I=3*H
    Bias_ptr,      # *const float, in_proj_bias: (I) or nullptr
    Out_ptr,       # *float, output BCx: (B, S, I)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    I: tl.int32,
    # strides for X
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    # strides for Out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    # strides for W
    w_i_stride: tl.int32, w_h_stride: tl.int32,
    has_bias: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # One program per (b, s)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    x_base = X_ptr + b * x_b_stride + s * x_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for i in range(0, I):
        acc = tl.zeros((), dtype=tl.float32)
        for h in range(0, H, BLOCK_H):
            h_offsets = h + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H
            x_vals = tl.load(
                x_base + h_offsets * x_h_stride,
                mask=h_mask,
                other=0.0
            ).to(tl.float32)
            w_vals = tl.load(
                W_ptr + i * w_i_stride + h_offsets * w_h_stride,
                mask=h_mask,
                other=0.0
            ).to(tl.float32)
            acc += tl.sum(x_vals * w_vals, axis=0)
        if has_bias:
            bval = tl.load(Bias_ptr + i).to(tl.float32)
            acc += bval
        # store as float32; Out_ptr dtype is float32 in our forward
        tl.store(out_base + i * out_i_stride, acc)


@triton.jit
def grouped_causal_conv1d_kernel(
    In_ptr,         # *const float, input Bx: (B, H, S), contiguous
    W_ptr,          # *const float, conv_weight: (H, 1, 4) contiguous
    Bias_ptr,       # *const float, conv_bias: (H) or nullptr
    Out_ptr,        # *float, output conv_out: (B, H, S)
    B: tl.int32,
    H: tl.int32,
    S: tl.int32,
    # strides for In
    in_b_stride: tl.int32, in_h_stride: tl.int32, in_s_stride: tl.int32,
    # strides for Out
    out_b_stride: tl.int32, out_h_stride: tl.int32, out_s_stride: tl.int32,
    # strides for W (W is (H, 1, 4))
    w_g_stride: tl.int32, w_c_stride: tl.int32, w_k_stride: tl.int32,
    has_bias: tl.int32,
    BLOCK_T: tl.constexpr,
):
    # grid = (B, H): one program per (b, g)
    pid_b = tl.program_id(axis=0)
    pid_g = tl.program_id(axis=1)
    b = pid_b
    g = pid_g

    in_base = In_ptr + b * in_b_stride + g * in_h_stride
    out_base = Out_ptr + b * out_b_stride + g * out_h_stride

    # We tile along S
    for t0 in range(0, S, BLOCK_T):
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < S

        # accumulator for this (b, g) tile
        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

        # reduce over kernel K=4 (causal)
        for k in range(0, 4):
            t_read = t_offsets + k - 1  # causal padding: we read t + k - 1
            # valid positions where t_read in [0, S-1]
            valid = (t_read >= 0) & (t_read < S) & t_mask
            vals = tl.load(in_base + t_read * in_s_stride, mask=valid, other=0.0)
            # weight for group g at kernel position k
            wval = tl.load(W_ptr + g * w_g_stride + 0 * w_c_stride + k * w_k_stride)
            acc += vals * wval

        if has_bias:
            bval = tl.load(Bias_ptr + g).to(tl.float32)
            acc += bval

        # store result
        tl.store(out_base + t_offsets * out_s_stride, acc, mask=t_mask)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,         # *const float, input y: (B, S, H)
    W_ptr,         # *const float, out_proj_weight: (H, H)
    Bias_ptr,      # *const float, out_proj_bias: (H) or nullptr
    Out_ptr,       # *float, output: (B, S, H)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    # strides
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    w_h_out_stride: tl.int32, w_h_in_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    has_bias: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # One program per (b, s)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for h_out in range(0, H):
        acc = tl.zeros((), dtype=tl.float32)
        for h_in in range(0, H, BLOCK_H):
            h_in_offsets = h_in + tl.arange(0, BLOCK_H)
            h_in_mask = h_in_offsets < H
            y_vals = tl.load(
                y_base + h_in_offsets * y_h_stride,
                mask=h_in_mask,
                other=0.0
            ).to(tl.float32)
            w_vals = tl.load(
                W_ptr + h_out * w_h_out_stride + h_in_offsets * w_h_in_stride,
                mask=h_in_mask,
                other=0.0
            ).to(tl.float32)
            acc += tl.sum(y_vals * w_vals, axis=0)
        if has_bias:
            bval = tl.load(Bias_ptr + h_out).to(tl.float32)
            acc += bval
        tl.store(out_base + h_out * out_h_stride, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        # Ensure float32 for numerical stability and to match typical reference
        # If inputs are not float32, cast here. The original run uses float32.
        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32) if in_proj_bias is not None else None
        conv_weight = conv_weight.contiguous().to(torch.float32)  # (H, 1, 4)
        conv_bias = conv_bias.contiguous().to(torch.float32) if conv_bias is not None else None
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32) if out_proj_bias is not None else None

        B, S, H = x.shape
        I = 3 * H

        # 1) in_proj: BCx = linear(x, in_proj_weight, in_proj_bias) -> (B, S, I)
        BCx = torch.empty((B, S, I), device=x.device, dtype=torch.float32)

        # Strides for X and Out
        x_b_stride, x_s_stride, x_h_stride = B * S, S, 1  # x is (B,S,H) contiguous -> strides (S*H, H, 1)
        # For X, since we pass pointer directly, Triton will use strides we compute logically.
        # Better: we'll pass actual strides via pointer arithmetic in kernel by treating X as 3D.
        # Implement by computing strides for X as:
        x_b_stride = S * H
        x_s_stride = H
        x_h_stride = 1

        # Out strides: (B,S,I) contiguous
        out_b_stride = S * I
        out_s_stride = I
        out_i_stride = 1

        w_i_stride, w_h_stride = I, 1  # W is (I, H) contiguous

        # Launch in_proj kernel
        grid_in = (B * S,)
        in_proj_linear_kernel[grid_in](
            x, in_proj_weight, in_proj_bias if in_proj_bias is not None else x,  # dummy ptr if no bias
            BCx, B, S, H, I,
            x_b_stride, x_s_stride, x_h_stride,
            out_b_stride, out_s_stride, out_i_stride,
            w_i_stride, w_h_stride,
            1 if in_proj_bias is not None else 0,
            BLOCK_H=64,
            num_warps=4,
        )

        # 2) Split BCx into B, C, x_proj
        # BCx: (B,S,I) where I=3H. chunk(3, dim=1) slices along I dimension
        B_tensor, C_tensor, x_proj_tensor = BCx.chunk(3, dim=-1)

        # 3) Elementwise gating: Bx = B * x_proj (PyTorch for simplicity)
        Bx = B_tensor * x_proj_tensor  # (B, S, H)

        # 4) Prepare for conv: transpose to (B, H, S) and apply grouped causal conv with groups=H
        Bx_trans = Bx.transpose(-1, -2).contiguous()  # (B, H, S)
        conv_out = torch.empty((B, H, S), device=x.device, dtype=torch.float32)

        # Strides for In and Out for conv
        in_b_stride, in_h_stride, in_s_stride = Bx_trans.stride()  # (H*S, S, 1)
        out_b_stride_conv, out_h_stride_conv, out_s_stride_conv = conv_out.stride()  # (H*S, 1, 1)

        # Strides for conv weight W: (H, 1, 4) contiguous
        w_g_stride, w_c_stride, w_k_stride = conv_weight.stride()  # (4, 1, 1)

        # Launch grouped causal conv kernel: grid = (B, H)
        grid_conv = (B, H)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_trans, conv_weight, conv_bias if conv_bias is not None else conv_weight,  # dummy ptr if no bias
            conv_out, B, H, S,
            in_b_stride, in_h_stride, in_s_stride,
            out_b_stride_conv, out_h_stride_conv, out_s_stride_conv,
            w_g_stride, w_c_stride, w_k_stride,
            1 if conv_bias is not None else 0,
            BLOCK_T=256,
            num_warps=4,
        )

        # 5) Output gating: y = C * conv_out (C: (B,S,H), conv_out: (B,H,S))
        # C.transpose(-1, -2) -> (B,S,H)
        y = C_tensor * conv_out.transpose(-1, -2)

        # 6) out_proj: final output = linear(y, out_proj_weight, out_proj_bias) -> (B,S,H)
        output = torch.empty((B, S, H), device=x.device, dtype=torch.float32)

        # Strides for Y and Output
        y_b_stride, y_s_stride, y_h_stride = y.stride()  # (S*H, H, 1)
        out_b_stride_out, out_s_stride_out, out_h_stride_out = output.stride()  # (S*H, H, 1)

        w_h_out_stride, w_h_in_stride = out_proj_weight.stride()  # (H, 1)

        grid_out = (B * S,)
        out_proj_linear_kernel[grid_out](
            y, out_proj_weight, out_proj_bias if out_proj_bias is not None else out_proj_weight,  # dummy ptr if no bias
            output, B, S, H,
            y_b_stride, y_s_stride, y_h_stride,
            w_h_out_stride, w_h_in_stride,
            out_b_stride_out, out_s_stride_out, out_h_stride_out,
            1 if out_proj_bias is not None else 0,
            BLOCK_H=64,
            num_warps=4,
        )

        return output


def run(*args):
    return ModelNew()(*args)
