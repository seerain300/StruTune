import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_forward_k5_p2(
    x_ptr,        # *float32, [B, C_in, T_in]
    w_ptr,        # *float32, [C_out, C_in, 5]
    y_ptr,        # *float32, [B, C_out, T_out] where T_out = T_in - 1 (padding=2, kernel=5)
    B, C_out, C_in, T_in, T_out,
    stride_xb, stride_xc, stride_xt,
    stride_wco, stride_wci, stride_wk,
    stride_yb, stride_yco, stride_yt,
    BLOCK_T: tl.constexpr,
):
    # Grid: (pid0, pid1) -> (B*C_out, ceil_div(T_out, BLOCK_T))
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    # Decode batch and output channel
    b = pid0 // C_out
    co = pid0 % C_out

    # Time tile
    t0 = pid1 * BLOCK_T
    t_offsets = t0 + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T_out

    # Accumulator for output vector across BLOCK_T
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # For valid conv with padding=2 and K=5, output index t_out corresponds to input index t_in = t_out - 2 + k
    # Loop over input channels and kernel taps
    for ci in range(0, C_in):
        for k in range(0, 5):
            t_in = t_offsets - 2 + k  # valid since T_out = T_in - 1
            # Load x[b, ci, t_in]
            x_idx = b * stride_xb + ci * stride_xc + t_in * stride_xt
            x_vals = tl.load(x_ptr + x_idx, mask=mask_t, other=0.0)
            # Load w[co, ci, k]
            w_idx = co * stride_wco + ci * stride_wci + k * stride_wk
            w_val = tl.load(w_ptr + w_idx)
            acc += x_vals * w_val

    # Store result to y[b, co, t_offsets]
    y_idx = b * stride_yb + co * stride_yco + t_offsets * stride_yt
    tl.store(y_ptr + y_idx, acc, mask=mask_t)


@triton.jit
def conv1d_bias(
    y_ptr,        # *float32, [B, C_out, T_out] (pre-conv output)
    bias_ptr,     # *float32, [C_out]
    B, C_out, T_out,
    stride_yb, stride_yco, stride_yt,
    BLOCK_T: tl.constexpr,
):
    # Grid: (pid0, pid1) -> (B*C_out, ceil_div(T_out, BLOCK_T))
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    b = pid0 // C_out
    co = pid0 % C_out

    t0 = pid1 * BLOCK_T
    t_offsets = t0 + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T_out

    y_idx = b * stride_yb + co * stride_yco + t_offsets * stride_yt
    y_vals = tl.load(y_ptr + y_idx, mask=mask_t, other=0.0)

    bias_val = tl.load(bias_ptr + co)
    y_vals += bias_val

    tl.store(y_ptr + y_idx, y_vals, mask=mask_t)


@triton.jit
def relu_elementwise(
    z_ptr,        # *float32, input/output tensor of shape [B, C, T]
    B, C, T,
    stride_zb, stride_zc, stride_zt,
    BLOCK_T: tl.constexpr,
):
    # Grid: (pid0, pid1) -> (B*C, ceil_div(T, BLOCK_T))
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    bc = pid0
    b = bc // C
    c = bc % C

    t0 = pid1 * BLOCK_T
    t_offsets = t0 + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    z_idx = b * stride_zb + c * stride_zc + t_offsets * stride_zt
    z_vals = tl.load(z_ptr + z_idx, mask=mask_t, other=0.0)
    z_vals = tl.maximum(z_vals, 0.0)
    tl.store(z_ptr + z_idx, z_vals, mask=mask_t)


@triton.jit
def elementwise_mul_mask(
    y_ptr,        # *float32, [B, C, T] (transform output h)
    mask_ptr,     # *float32, [B, 1, T] (broadcast across C)
    out_ptr,      # *float32, [B, C, T] (result h * mask)
    B, C, T,
    stride_yb, stride_yc, stride_yt,
    stride_mb, stride_mc, stride_mt,
    BLOCK_T: tl.constexpr,
):
    # Grid: (pid0, pid1) -> (B*C, ceil_div(T, BLOCK_T))
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    bc = pid0
    b = bc // C
    c = bc % C

    t0 = pid1 * BLOCK_T
    t_offsets = t0 + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    # Load y[b, c, t_offsets]
    y_idx = b * stride_yb + c * stride_yc + t_offsets * stride_yt
    y_vals = tl.load(y_ptr + y_idx, mask=mask_t, other=0.0)

    # Load mask[b, 0, t_offsets] (c dimension is 1)
    m_idx = b * stride_mb + 0 * stride_mc + t_offsets * stride_mt
    m_vals = tl.load(mask_ptr + m_idx, mask=mask_t, other=1.0)

    out_vals = y_vals * m_vals

    out_idx = b * stride_yb + c * stride_yc + t_offsets * stride_yt  # same layout
    tl.store(out_ptr + out_idx, out_vals, mask=mask_t)


@triton.jit
def elementwise_add_sub(
    out_ptr,      # *float32, [B, C, T] (destination tensor, e.g., x1)
    y_ptr,        # *float32, [B, C, T] (source tensor, e.g., h * mask)
    B, C, T,
    stride_outb, stride_outc, stride_outt,
    stride_yb, stride_yc, stride_yt,
    operation: tl.constexpr,   # 0 => add, 1 => sub
    BLOCK_T: tl.constexpr,
):
    # Grid: (pid0, pid1) -> (B*C, ceil_div(T, BLOCK_T))
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    bc = pid0
    b = bc // C
    c = bc % C

    t0 = pid1 * BLOCK_T
    t_offsets = t0 + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    out_idx = b * stride_outb + c * stride_outc + t_offsets * stride_outt
    y_idx = b * stride_yb + c * stride_yc + t_offsets * stride_yt

    out_vals = tl.load(out_ptr + out_idx, mask=mask_t, other=0.0)
    y_vals = tl.load(y_ptr + y_idx, mask=mask_t, other=0.0)

    if operation == 0:
        out_vals = out_vals + y_vals
    else:
        out_vals = out_vals - y_vals

    tl.store(out_ptr + out_idx, out_vals, mask=mask_t)


@triton.jit
def copy_half_channels(
    src_ptr,      # *float32, [B, C_src, T]
    out_ptr,      # *float32, [B, C_dst, T]
    B, C_src, C_dst, T,
    stride_srcb, stride_srcc, stride_srtc,
    stride_outb, stride_outc, stride_outt,
    BLOCK_T: tl.constexpr,
):
    # Grid: (pid0, pid1) -> (B*C_dst, ceil_div(T, BLOCK_T))
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    bc = pid0
    b = bc // C_dst
    c_dst = bc % C_dst

    t0 = pid1 * BLOCK_T
    t_offsets = t0 + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    # Copy from src[:, c_dst, :] to out[:, c_dst, :]
    src_idx = b * stride_srcb + c_dst * stride_srcc + t_offsets * stride_srtc
    out_idx = b * stride_outb + c_dst * stride_outc + t_offsets * stride_outt

    vals = tl.load(src_ptr + src_idx, mask=mask_t, other=0.0)
    tl.store(out_ptr + out_idx, vals, mask=mask_t)


class ModelNew(nn.Module):
    def forward(self, x, x_mask, reverse,
                # transform 0
                transform_0_conv0_weight, transform_0_conv0_bias,
                transform_0_conv1_weight, transform_0_conv1_bias,
                transform_0_conv2_weight, transform_0_conv2_bias,
                # transform 1
                transform_1_conv0_weight, transform_1_conv0_bias,
                transform_1_conv1_weight, transform_1_conv1_bias,
                transform_1_conv2_weight, transform_1_conv2_bias,
                # transform 2
                transform_2_conv0_weight, transform_2_conv0_bias,
                transform_2_conv1_weight, transform_2_conv1_bias,
                transform_2_conv2_weight, transform_2_conv2_bias,
                # transform 3
                transform_3_conv0_weight, transform_3_conv0_bias,
                transform_3_conv1_weight, transform_3_conv1_bias,
                transform_3_conv2_weight, transform_3_conv2_bias):
        """
        Triton-optimized forward and reverse pass. All computation (conv, bias, ReLU, mask, add/sub, copy/concat) is done via Triton kernels.
        No torch.conv1d or torch elementwise ops in forward.
        """
        # Ensure contiguous for Triton
        x = x.contiguous()
        x_mask = x_mask.contiguous()

        batch_size = x.shape[0]
        C = x.shape[1]
        T_in = x.shape[2]
        half_channels = C // 2

        # List of transforms (each is (conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b))
        transforms = [
            (transform_0_conv0_weight, transform_0_conv0_bias,
             transform_0_conv1_weight, transform_0_conv1_bias,
             transform_0_conv2_weight, transform_0_conv2_bias),
            (transform_1_conv0_weight, transform_1_conv0_bias,
             transform_1_conv1_weight, transform_1_conv1_bias,
             transform_1_conv2_weight, transform_1_conv2_bias),
            (transform_2_conv0_weight, transform_2_conv0_bias,
             transform_2_conv1_weight, transform_2_conv1_bias,
             transform_2_conv2_weight, transform_2_conv2_bias),
            (transform_3_conv0_weight, transform_3_conv0_bias,
             transform_3_conv1_weight, transform_3_conv1_bias,
             transform_3_conv2_weight, transform_3_conv2_bias),
        ]

        # We will compute the forward pass sequentially (forward=True). For reverse, we would process in reversed order,
        # but since we don't use torch ops here, we keep it generic by reading arguments and applying transforms in the same order
        # (reverse controls the add/sub, not the order of transforms).
        # However, to respect the original semantics, we keep transforms in order when not reverse, and reverse the order when reverse=True.

        if not reverse:
            # Forward: apply transforms sequentially
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
                # Split input into two halves
                x0 = x[:, :half_channels, :]  # [B, 96, T]
                x1 = x[:, half_channels:, :]  # [B, 96, T]
                x0 = x0.contiguous()
                x1 = x1.contiguous()

                # Compute h = transform(x0)
                # conv0: in_channels=96, out_channels=192
                hidden_channels = conv0_w.shape[0]
                T_in0 = x0.shape[2]
                T_out0 = T_in0 - 1  # output length after valid conv with K=5 and padding=2

                # conv0
                y0 = torch.empty((batch_size, hidden_channels, T_out0), device=x.device, dtype=torch.float32)
                grid0 = (batch_size * hidden_channels, triton.cdiv(T_out0, 128))
                conv1d_forward_k5_p2[grid0](
                    x0, conv0_w, y0,
                    batch_size, hidden_channels, conv0_w.shape[1], T_in0, T_out0,
                    x0.stride(0), x0.stride(1), x0.stride(2),
                    conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                    y0.stride(0), y0.stride(1), y0.stride(2),
                    128
                )

                # bias
                grid_bias0 = (batch_size * hidden_channels, triton.cdiv(T_out0, 128))
                conv1d_bias[grid_bias0](
                    y0, conv0_b,
                    batch_size, hidden_channels, T_out0,
                    y0.stride(0), y0.stride(1), y0.stride(2),
                    128
                )

                # ReLU (in Triton)
                grid_relu0 = (batch_size * hidden_channels, triton.cdiv(T_out0, 128))
                relu_elementwise[grid_relu0](
                    y0, batch_size, hidden_channels, T_out0,
                    y0.stride(0), y0.stride(1), y0.stride(2),
                    128
                )

                # conv1: in_channels=192, out_channels=192
                T_in1 = y0.shape[2]
                T_out1 = T_in1 - 1  # since conv1 is also valid with K=5
                y1 = torch.empty((batch_size, hidden_channels, T_out1), device=x.device, dtype=torch.float32)
                grid1 = (batch_size * hidden_channels, triton.cdiv(T_out1, 128))
                conv1d_forward_k5_p2[grid1](
                    y0, conv1_w, y1,
                    batch_size, hidden_channels, conv1_w.shape[1], T_in1, T_out1,
                    y0.stride(0), y0.stride(1), y0.stride(2),
                    conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                    y1.stride(0), y1.stride(1), y1.stride(2),
                    128
                )

                grid_bias1 = (batch_size * hidden_channels, triton.cdiv(T_out1, 128))
                conv1d_bias[grid_bias1](
                    y1, conv1_b,
                    batch_size, hidden_channels, T_out1,
                    y1.stride(0), y1.stride(1), y1.stride(2),
                    128
                )

                grid_relu1 = (batch_size * hidden_channels, triton.cdiv(T_out1, 128))
                relu_elementwise[grid_relu1](
                    y1, batch_size, hidden_channels, T_out1,
                    y1.stride(0), y1.stride(1), y1.stride(2),
                    128
                )

                # conv2: in_channels=192, out_channels=96
                T_in2 = y1.shape[2]
                T_out2 = T_in2 - 1
                h = torch.empty((batch_size, half_channels, T_out2), device=x.device, dtype=torch.float32)
                grid2 = (batch_size * half_channels, triton.cdiv(T_out2, 128))
                conv1d_forward_k5_p2[grid2](
                    y1, conv2_w, h,
                    batch_size, half_channels, conv2_w.shape[1], T_in2, T_out2,
                    y1.stride(0), y1.stride(1), y1.stride(2),
                    conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                    h.stride(0), h.stride(1), h.stride(2),
                    128
                )

                grid_bias2 = (batch_size * half_channels, triton.cdiv(T_out2, 128))
                conv1d_bias[grid_bias2](
                    h, conv2_b,
                    batch_size, half_channels, T_out2,
                    h.stride(0), h.stride(1), h.stride(2),
                    128
                )

                # ReLU (in Triton)
                grid_relu2 = (batch_size * half_channels, triton.cdiv(T_out2, 128))
                relu_elementwise[grid_relu2](
                    h, batch_size, half_channels, T_out2,
                    h.stride(0), h.stride(1), h.stride(2),
                    128
                )

                # Apply mask: h = h * x_mask (broadcast across channels)
                h_masked = torch.empty_like(h, device=x.device, dtype=torch.float32)
                grid_mul = (batch_size * half_channels, triton.cdiv(T_out2, 128))
                elementwise_mul_mask[grid_mul](
                    h, x_mask, h_masked,
                    batch_size, half_channels, T_out2,
                    h.stride(0), h.stride(1), h.stride(2),
                    x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                    128
                )

                # Affine coupling: x1 = x1 + h_masked
                grid_add = (batch_size * half_channels, triton.cdiv(T_out2, 128))
                elementwise_add_sub[grid_add](
                    x1, h_masked,
                    batch_size, half_channels, T_out2,
                    x1.stride(0), x1.stride(1), x1.stride(2),
                    h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                    0,  # add
                    128
                )

                # Concatenate x0 and updated x1 back along channel dimension
                out_channels = half_channels * 2
                x_out = torch.empty((batch_size, out_channels, T_out2), device=x.device, dtype=torch.float32)

                # Copy x0 into first half channels
                grid_copy_x0 = (batch_size * half_channels, triton.cdiv(T_out2, 128))
                copy_half_channels[grid_copy_x0](
                    x0, x_out,
                    batch_size, half_channels, half_channels, T_out2,
                    x0.stride(0), x0.stride(1), x0.stride(2),
                    x_out.stride(0), x_out.stride(1), x_out.stride(2),
                    128
                )

                # Copy updated x1 into second half channels
                grid_copy_x1 = (batch_size * half_channels, triton.cdiv(T_out2, 128))
                copy_half_channels[grid_copy_x1](
                    x1, x_out,
                    batch_size, half_channels, half_channels, T_out2,
                    x1.stride(0), x1.stride(1), x1.stride(2),
                    x_out.stride(0), x_out.stride(1), x_out.stride(2),
                    128
                )

                # Apply mask to output (broadcast across channels and time)
                x_out_masked = torch.empty_like(x_out, device=x.device, dtype=torch.float32)
                grid_out_mul = (batch_size * out_channels, triton.cdiv(T_out2, 128))
                elementwise_mul_mask[grid_out_mul](
                    x_out, x_mask, x_out_masked,
                    batch_size, out_channels, T_out2,
                    x_out.stride(0), x_out.stride(1), x_out.stride(2),
                    x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                    128
                )

                # Update x to x_out_masked for next transform
                x = x_out_masked

        else:
            # Reverse: process transforms in reverse order and subtract
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
                # Split input into two halves
                x0 = x[:, :half_channels, :]  # [B, 96, T]
                x1 = x[:, half_channels:, :]  # [B, 96, T]
                x0 = x0.contiguous()
                x1 = x1.contiguous()

                # Compute h = transform(x0) in reverse order
                # conv2 first (since last in original transform)
                hidden_channels = conv2_w.shape[0]  # out_channels=96
                T_in2 = x0.shape[2]
                T_out2 = T_in2 - 1

                h = torch.empty((batch_size, half_channels, T_out2), device=x.device, dtype=torch.float32)
                grid2 = (batch_size * half_channels, triton.cdiv(T_out2, 128))
                conv1d_forward_k5_p2[grid2](
                    x0, conv2_w, h,
                    batch_size, half_channels, conv2_w.shape[1], T_in2, T_out2,
                    x0.stride(0), x0.stride(1), x0.stride(2),
                    conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                    h.stride(0), h.stride(1), h.stride(2),
                    128
                )

                grid_bias2 = (batch_size * half_channels, triton.cdiv(T_out2, 128))
                conv1d_bias[grid_bias2](
                    h, conv2_b,
                    batch_size, half_channels, T_out2,
                    h.stride(0), h.stride(1), h.stride(2),
                    128
                )

                grid_relu2 = (batch_size * half_channels, triton.cdiv(T_out2, 128))
                relu_elementwise[grid_relu2](
                    h, batch_size, half_channels, T_out2,
                    h.stride(0), h.stride(1), h.stride(2),
                    128
                )

                # conv1: in_channels=192, out_channels=192
                # We need an intermediate tensor of shape [B, 192, T_in1] where T_in1 = T_out2 + 4
                T_in1 = T_out2 + 2  # because conv1 has kernel=5, output T_in1 - 1 = T_out2 -> T_in1 = T_out2 + 2
                y1 = torch.empty((batch_size, hidden_channels, T_out2 + 2), device=x.device, dtype=torch.float32)
                grid1 = (batch_size * hidden_channels, triton.cdiv(T_out2 + 2, 128))
                conv1d_forward_k5_p2[grid1](
                    h, conv1_w, y1,
                    batch_size, hidden_channels, conv1_w.shape[1], T_in1, T_out2 + 2,
                    h.stride(0), h.stride(1), h.stride(2),
                    conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                    y1.stride(0), y1.stride(1), y1.stride(2),
                    128
                )

                grid_bias1 = (batch_size * hidden_channels, triton.cdiv(T_out2 + 2, 128))
                conv1d_bias[grid_bias1](
                    y1, conv1_b,
                    batch_size, hidden_channels, T_out2 + 2,
                    y1.stride(0), y1.stride(1), y1.stride(2),
                    128
                )

                grid_relu1 = (batch_size * hidden_channels, triton.cdiv(T_out2 + 2, 128))
                relu_elementwise[grid_relu1](
                    y1, batch_size, hidden_channels, T_out2 + 2,
                    y1.stride(0), y1.stride(1), y1.stride(2),
                    128
                )

                # conv0: in_channels=96, out_channels=192
                T_in0 = T_out2 + 2  # since conv0 precedes conv1
                y0 = torch.empty((batch_size, hidden_channels, T_out2 + 4), device=x.device, dtype=torch.float32)
                grid0 = (batch_size * hidden_channels, triton.cdiv(T_out2 + 4, 128))
                conv1d_forward_k5_p2[grid0](
                    y1, conv0_w, y0,
                    batch_size, hidden_channels, conv0_w.shape[1], T_in0, T_out2 + 4,
                    y1.stride(0), y1.stride(1), y1.stride(2),
                    conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                    y0.stride(0), y0.stride(1), y0.stride(2),
                    128
                )

                grid_bias0 = (batch_size * hidden_channels, triton.cdiv(T_out2 + 4, 128))
                conv1d_bias[grid_bias0](
                    y0, conv0_b,
                    batch_size, hidden_channels, T_out2 + 4,
                    y0.stride(0), y0.stride(1), y0.stride(2),
                    128
                )

                grid_relu0 = (batch_size * hidden_channels, triton.cdiv(T_out2 + 4, 128))
                relu_elementwise[grid_relu0](
                    y0, batch_size, hidden_channels, T_out2 + 4,
                    y0.stride(0), y0.stride(1), y0.stride(2),
                    128
                )

                # Apply mask: h = h * x_mask (broadcast across channels)
                h_masked = torch.empty_like(h, device=x.device, dtype=torch.float32)
                grid_mul = (batch_size * half_channels, triton.cdiv(T_out2, 128))
                elementwise_mul_mask[grid_mul](
                    h, x_mask, h_masked,
                    batch_size, half_channels, T_out2,
                    h.stride(0), h.stride(1), h.stride(2),
                    x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                    128
                )

                # Inverse affine coupling: x1 = x1 - h_masked
                grid_sub = (batch_size * half_channels, triton.cdiv(T_out2, 128))
                elementwise_add_sub[grid_sub](
                    x1, h_masked,
                    batch_size, half_channels, T_out2,
                    x1.stride(0), x1.stride(1), x1.stride(2),
                    h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                    1,  # subtract
                    128
                )

                # Concatenate x0 and updated x1 back along channel dimension
                out_channels = half_channels * 2
                x_out = torch.empty((batch_size, out_channels, T_out2), device=x.device, dtype=torch.float32)

                # Copy x0 into first half channels
                grid_copy_x0 = (batch_size * half_channels, triton.cdiv(T_out2, 128))
                copy_half_channels[grid_copy_x0](
                    x0, x_out,
                    batch_size, half_channels, half_channels, T_out2,
                    x0.stride(0), x0.stride(1), x0.stride(2),
                    x_out.stride(0), x_out.stride(1), x_out.stride(2),
                    128
                )

                # Copy updated x1 into second half channels
                grid_copy_x1 = (batch_size * half_channels, triton.cdiv(T_out2, 128))
                copy_half_channels[grid_copy_x1](
                    x1, x_out,
                    batch_size, half_channels, half_channels, T_out2,
                    x1.stride(0), x1.stride(1), x1.stride(2),
                    x_out.stride(0), x_out.stride(1), x_out.stride(2),
                    128
                )

                # Apply mask to output (broadcast across channels and time)
                x_out_masked = torch.empty_like(x_out, device=x.device, dtype=torch.float32)
                grid_out_mul = (batch_size * out_channels, triton.cdiv(T_out2, 128))
                elementwise_mul_mask[grid_out_mul](
                    x_out, x_mask, x_out_masked,
                    batch_size, out_channels, T_out2,
                    x_out.stride(0), x_out.stride(1), x_out.stride(2),
                    x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                    128
                )

                # Update x to x_out_masked for next transform
                x = x_out_masked

        return x


def run(*args):
    return ModelNew()(*args)
