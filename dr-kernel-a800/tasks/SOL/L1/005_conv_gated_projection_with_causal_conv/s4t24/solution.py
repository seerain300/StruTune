import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,        # *const float, input x: (B, S, H), float32
    W_ptr,        # *const float, in_proj_weight: (I, H), float32, I=3*H
    Bias_ptr,     # *const float, in_proj_bias: (I), float32
    Out_ptr,      # *float, output BCx: (B, S, I), float32
    B: tl.int32,  # batch size
    S: tl.int32,  # sequence length
    H: tl.int32,  # hidden size
    I: tl.int32,  # output channels = 3*H
    # strides
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    w_i_stride: tl.int32, w_h_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
):
    # One program per (b, s)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    x_base = X_ptr + b * x_b_stride + s * x_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for i in range(0, I):
        acc = tl.zeros((), dtype=tl.float32)
        for h in range(0, H):
            x_val = tl.load(x_base + h * x_h_stride)
            w_val = tl.load(W_ptr + i * w_i_stride + h * w_h_stride)
            acc += x_val * w_val
        bias_val = tl.load(Bias_ptr + i)
        acc += bias_val
        tl.store(out_base + i * out_i_stride, acc)


@triton.jit
def pad_left_kernel(
    In_ptr,       # *const float, input: (B, H, S), float32
    Out_ptr,      # *float, output: (B, H, S+pad), float32
    B: tl.int32,  # batch size
    H: tl.int32,  # channel count (groups)
    S: tl.int32,  # sequence length
    pad: tl.int32,  # padding on left
    # strides
    in_b_stride: tl.int32, in_h_stride: tl.int32, in_s_stride: tl.int32,
    out_b_stride: tl.int32, out_h_stride: tl.int32, out_s_stride: tl.int32,
):
    # One program per (b, h)
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    in_base = In_ptr + b * in_b_stride + h * in_h_stride
    out_base = Out_ptr + b * out_b_stride + h * out_h_stride

    S_plus = S + pad
    for s_out in range(0, S_plus):
        if s_out < pad:
            # write zeros
            tl.store(out_base + s_out * out_s_stride, 0.0)
        else:
            src = s_out - pad
            val = tl.load(in_base + src * in_s_stride)
            tl.store(out_base + s_out * out_s_stride, val)


@triton.jit
def grouped_causal_conv1d_kernel(
    Bx_pad_ptr,   # *const float, padded input: (B, H, S+pad), float32
    W_ptr,        # *const float, conv_weight: (H, 1, 4), float32
    Bias_ptr,     # *const float, conv_bias: (H), float32
    Out_ptr,      # *float, output conv_out: (B, H, S), float32
    B: tl.int32,  # batch size
    H: tl.int32,  # groups (channel count)
    S: tl.int32,  # output sequence length
    pad: tl.int32,  # causal left pad
    # strides
    bx_b_stride: tl.int32, bx_h_stride: tl.int32, bx_s_stride: tl.int32,
    out_b_stride: tl.int32, out_h_stride: tl.int32, out_s_stride: tl.int32,
):
    # One program per (b, g)
    pid = tl.program_id(axis=0)
    b = pid // H
    g = pid % H

    bx_base = Bx_pad_ptr + b * bx_b_stride + g * bx_h_stride
    out_base = Out_ptr + b * out_b_stride + g * out_h_stride

    for t in range(0, S):
        acc = tl.zeros((), dtype=tl.float32)
        # kernel_size = 4
        # k = 0
        t_in0 = t + 0 - pad
        if t_in0 >= 0:
            val0 = tl.load(bx_base + t_in0 * bx_s_stride)
            w0 = tl.load(W_ptr + g * tl.multiple_of(0, 1) + 0 * tl.multiple_of(1, 1) + 0 * tl.multiple_of(4, 1))
            acc += val0 * w0
        # k = 1
        t_in1 = t + 1 - pad
        if t_in1 >= 0:
            val1 = tl.load(bx_base + t_in1 * bx_s_stride)
            w1 = tl.load(W_ptr + g * tl.multiple_of(0, 1) + 0 * tl.multiple_of(1, 1) + 1 * tl.multiple_of(4, 1))
            acc += val1 * w1
        # k = 2
        t_in2 = t + 2 - pad
        if t_in2 >= 0:
            val2 = tl.load(bx_base + t_in2 * bx_s_stride)
            w2 = tl.load(W_ptr + g * tl.multiple_of(0, 1) + 0 * tl.multiple_of(1, 1) + 2 * tl.multiple_of(4, 1))
            acc += val2 * w2
        # k = 3
        t_in3 = t + 3 - pad
        if t_in3 >= 0:
            val3 = tl.load(bx_base + t_in3 * bx_s_stride)
            w3 = tl.load(W_ptr + g * tl.multiple_of(0, 1) + 0 * tl.multiple_of(1, 1) + 3 * tl.multiple_of(4, 1))
            acc += val3 * w3
        # add bias
        bias_g = tl.load(Bias_ptr + g)
        acc += bias_g
        tl.store(out_base + t * out_s_stride, acc)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,        # *const float, input y: (B, S, H), float32
    W_out_ptr,    # *const float, out_proj_weight: (H, H), float32
    Bias_out_ptr, # *const float, out_proj_bias: (H), float32
    Out_ptr,      # *float, output: (B, S, H), float32
    B: tl.int32,  # batch size
    S: tl.int32,  # sequence length
    H: tl.int32,  # hidden size
    # strides
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    w_out_hout_stride: tl.int32, w_out_hin_stride: tl.int32,
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
):
    # One program per (b, s)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for h_out in range(0, H):
        acc = tl.zeros((), dtype=tl.float32)
        for h_in in range(0, H):
            y_val = tl.load(y_base + h_in * y_h_stride)
            w_val = tl.load(W_out_ptr + h_out * w_out_hout_stride + h_in * w_out_hin_stride)
            acc += y_val * w_val
        bias_val = tl.load(Bias_out_ptr + h_out)
        acc += bias_val
        tl.store(out_base + h_out * out_h_stride, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; forward receives inputs as args

    def forward(
        self,
        x: torch.Tensor,                 # (B, S, H) float32
        in_proj_weight: torch.Tensor,    # (I, H) float32, I=3*H
        in_proj_bias: torch.Tensor,      # (I) float32
        conv_weight: torch.Tensor,       # (H, 1, 4) float32
        conv_bias: torch.Tensor,         # (H) float32
        out_proj_weight: torch.Tensor,   # (H, H) float32
        out_proj_bias: torch.Tensor,     # (H) float32
    ):
        B, S, H = x.shape
        I = 3 * H
        pad = conv_weight.shape[2] - 1  # kernel_size=4 -> pad=3

        # Ensure all tensors are contiguous float32
        x = x.contiguous().float()
        in_proj_weight = in_proj_weight.contiguous().float()
        in_proj_bias = in_proj_bias.contiguous().float()
        conv_weight = conv_weight.contiguous().float()
        conv_bias = conv_bias.contiguous().float()
        out_proj_weight = out_proj_weight.contiguous().float()
        out_proj_bias = out_proj_bias.contiguous().float()

        # 1) in_proj: BCx = F.linear(x, in_proj_weight, in_proj_bias)
        BCx = torch.empty((B, S, I), device=x.device, dtype=torch.float32)
        grid_in = (B * S,)
        in_proj_linear_kernel[grid_in](
            x, in_proj_weight, in_proj_bias, BCx,
            B, S, H, I,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
        )

        # 2) Slice BCx to get B_tensor, C_tensor, x_proj_tensor
        B_tensor = BCx[:, :, :H]
        C_tensor = BCx[:, :, H:2 * H]
        x_proj_tensor = BCx[:, :, 2 * H:]

        # 3) Elementwise gating: Bx = B_tensor * x_proj_tensor
        Bx = B_tensor * x_proj_tensor  # float32 elementwise multiply

        # 4) Pad Bx for causal conv: (B, H, S+pad)
        Bx_pad = torch.empty((B, H, S + pad), device=x.device, dtype=torch.float32)
        grid_pad = (B * H,)
        pad_left_kernel[grid_pad](
            Bx, Bx_pad,
            B, H, S, pad,
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
        )

        # 5) Grouped causal conv: conv_out (B, H, S)
        conv_out = torch.empty((B, H, S), device=x.device, dtype=torch.float32)
        grid_conv = (B * H,)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx_pad, conv_weight, conv_bias, conv_out,
            B, H, S, pad,
            Bx_pad.stride(0), Bx_pad.stride(1), Bx_pad.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        )

        # 6) Output gating: y = C_tensor * conv_out (C_tensor is (B, S, H), conv_out is (B, H, S))
        # We need y with shape (B, S, H); to do elementwise multiply, transpose conv_out to (B, S, H):
        # Note: C_tensor is (B, S, H); conv_out is (B, H, S). We can do C[:, :, h] * conv_out[:, h, :]
        # But for elementwise we need both (B, S, H). We transpose conv_out to (B, S, H).
        conv_out_T = conv_out.transpose(1, 2).contiguous()  # (B, S, H)
        y = C_tensor * conv_out_T  # elementwise multiply

        # 7) Final out_proj: output = F.linear(y, out_proj_weight, out_proj_bias)
        output = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        grid_out = (B * S,)
        out_proj_linear_kernel[grid_out](
            y, out_proj_weight, out_proj_bias, output,
            B, S, H,
            y.stride(0), y.stride(1), y.stride(2),
            out_proj_weight.stride(0), out_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
        )

        return output


def run(*args):
    return ModelNew()(*args)
