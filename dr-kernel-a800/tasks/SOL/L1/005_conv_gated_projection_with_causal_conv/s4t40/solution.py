import torch
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,         # *const float, input x: (B, S, H)
    W_ptr,         # *const float, in_proj_weight: (I, H), I=3*H
    Bias_ptr,      # *const float, in_proj_bias: (I,)
    Out_ptr,       # *float, output BCx: (B, S, I)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    I: tl.int32,
    # strides for x
    x_b_stride: tl.int32, x_s_stride: tl.int32, x_h_stride: tl.int32,
    # strides for w
    w_i_stride: tl.int32, w_h_stride: tl.int32,
    # strides for out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_i_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # grid: axis=0 over B*S
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    x_base = X_ptr + b * x_b_stride + s * x_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for i in range(0, I):
        acc = tl.zeros((), dtype=tl.float32)
        # reduce over H dimension in tiles
        for h in range(0, H, BLOCK_H):
            h_offsets = h + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < H
            x_vals = tl.load(x_base + h_offsets * x_h_stride, mask=h_mask, other=0.0)
            w_vals = tl.load(W_ptr + i * w_i_stride + h_offsets * w_h_stride, mask=h_mask, other=0.0)
            acc += tl.sum(x_vals * w_vals, axis=0)

        # add bias if provided
        if Bias_ptr != 0:
            bias_i = tl.load(Bias_ptr + i)
            acc += bias_i

        # store to Out at (b, s, i)
        tl.store(out_base + i * out_i_stride, acc)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,           # *const float, input y: (B, S, H)
    Wout_ptr,        # *const float, out_proj_weight: (H, H)
    Bout_ptr,        # *const float, out_proj_bias: (H,)
    Out_ptr,         # *float, output: (B, S, H)
    B: tl.int32,
    S: tl.int32,
    H: tl.int32,
    # strides for Y
    y_b_stride: tl.int32, y_s_stride: tl.int32, y_h_stride: tl.int32,
    # strides for Wout
    wout_hout_stride: tl.int32, wout_hin_stride: tl.int32,
    # strides for Out
    out_b_stride: tl.int32, out_s_stride: tl.int32, out_h_stride: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # grid: axis=0 over B*S
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * y_b_stride + s * y_s_stride
    out_base = Out_ptr + b * out_b_stride + s * out_s_stride

    for h_out in range(0, H, BLOCK_H):
        h_out_offsets = h_out + tl.arange(0, BLOCK_H)
        h_out_mask = h_out_offsets < H

        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        # reduce over H_in
        for h_in in range(0, H, BLOCK_H):
            h_in_offsets = h_in + tl.arange(0, BLOCK_H)
            h_in_mask = h_in_offsets < H

            y_vals = tl.load(y_base + h_in_offsets * y_h_stride, mask=h_in_mask, other=0.0)
            w_vals = tl.load(
                Wout_ptr + h_out_offsets[:, None] * wout_hout_stride + h_in_offsets[None, :] * wout_hin_stride,
                mask=h_out_mask[:, None] & h_in_mask[None, :],
                other=0.0
            )
            acc += tl.sum(y_vals[:, None] * w_vals, axis=1)

        # add bias
        if Bout_ptr != 0:
            bias_vals = tl.load(Bout_ptr + h_out_offsets, mask=h_out_mask, other=0.0)
            acc += bias_vals

        tl.store(out_base + h_out_offsets * out_h_stride, acc, mask=h_out_mask)


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # Shapes
        B, S, H = x.shape
        I = 3 * H  # in_proj output channels

        # 1) in_proj: y = F.linear(x, in_proj_weight, in_proj_bias)
        # x: (B, S, H), in_proj_weight: (I, H), bias: (I,)
        BCx = torch.empty((B, S, I), device=x.device, dtype=torch.float32)  # compute in fp32
        in_proj_linear_kernel[(B * S,)](
            x.contiguous(), in_proj_weight.contiguous(), (in_proj_bias if in_proj_bias is not None else 0),
            BCx, B, S, H, I,
            x_b_stride=x.stride(0), x_s_stride=x.stride(1), x_h_stride=x.stride(2),
            w_i_stride=in_proj_weight.stride(0), w_h_stride=in_proj_weight.stride(1),
            out_b_stride=BCx.stride(0), out_s_stride=BCx.stride(1), out_i_stride=BCx.stride(2),
            BLOCK_H=64, num_warps=4
        )

        # 2) Split BCx into B, C, x_proj
        # BCx: (B, S, 3H) with channel dimension at dim=1
        B_tensor = BCx[:, :, :H]
        C_tensor = BCx[:, :, H:2 * H]
        x_proj_tensor = BCx[:, :, 2 * H:]

        # Elementwise gating: Bx = B_tensor * x_proj_tensor
        Bx = B_tensor * x_proj_tensor  # (B, S, H)

        # 3) Grouped causal conv: conv_out = conv1d(Bx_padded, conv_weight, conv_bias, groups=H)
        # conv_weight is (H, 4), groups=H. Use PyTorch for correctness.
        Bx_conv = Bx.transpose(-1, -2).contiguous()  # (B, H, S)
        Bx_padded = torch.nn.functional.pad(Bx_conv, (3, 0))  # pad left by 3 for kernel_size=4
        conv_out = torch.nn.functional.conv1d(
            Bx_padded, conv_weight, conv_bias, stride=1, padding=0, dilation=1, groups=H
        )  # (B, H, S)

        # 4) Output gating: y = C_tensor * conv_out
        # C_tensor: (B, S, H)
        # conv_out: (B, H, S) but we need to match dims for elementwise multiply: transpose C to (B, H, S)
        C_for_gate = C_tensor.transpose(-1, -2).contiguous()  # (B, H, S)
        y = C_for_gate * conv_out  # (B, H, S)

        # 5) Final out_proj: output = F.linear(y, out_proj_weight, out_proj_bias)
        # y: (B, H, S) -> (B, S, H)
        y_t = y.transpose(-1, -2).contiguous()  # (B, S, H)
        output = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        out_proj_linear_kernel[(B * S,)](
            y_t, out_proj_weight.contiguous(), (out_proj_bias if out_proj_bias is not None else 0),
            output, B, S, H,
            y_b_stride=y_t.stride(0), y_s_stride=y_t.stride(1), y_h_stride=y_t.stride(2),
            wout_hout_stride=out_proj_weight.stride(0), wout_hin_stride=out_proj_weight.stride(1),
            out_b_stride=output.stride(0), out_s_stride=output.stride(1), out_h_stride=output.stride(2),
            BLOCK_H=64, num_warps=4
        )

        return output


def run(*args):
    return ModelNew()(*args)
