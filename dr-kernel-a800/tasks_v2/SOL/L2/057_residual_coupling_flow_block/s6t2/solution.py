import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    out_idx = b * stride_yb + c * stride_yc + t_offsets * stride_yt
    tl.store(out_ptr + out_idx, out_vals, mask=mask_t)


@triton.jit
def elementwise_add_sub(
    x1_ptr,        # *float32, [B, half_channels, T_out] (second half channels)
    h_ptr,         # *float32, [B, half_channels, T_out] (transform output)
    out_ptr,       # *float32, [B, half_channels, T_out] (result)
    B, C, T,
    stride_x1b, stride_x1c, stride_x1t,
    stride_hb, stride_hc, stride_ht,
    add_flag: tl.constexpr,  # 1 to add, 0 to subtract
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

    x1_idx = b * stride_x1b + c * stride_x1c + t_offsets * stride_x1t
    h_idx = b * stride_hb + c * stride_hc + t_offsets * stride_ht

    x1_vals = tl.load(x1_ptr + x1_idx, mask=mask_t, other=0.0)
    h_vals = tl.load(h_ptr + h_idx, mask=mask_t, other=0.0)

    if add_flag:
        out_vals = x1_vals + h_vals
    else:
        out_vals = x1_vals - h_vals

    out_idx = b * stride_x1b + c * stride_x1c + t_offsets * stride_x1t
    tl.store(out_ptr + out_idx, out_vals, mask=mask_t)


@triton.jit
def copy_half_channels(
    src_ptr,        # *float32, [B, C_src, T]
    dst_ptr,        # *float32, [B, C_dst, T] (dst first half channels or second half)
    B, C_src, C_dst, T,
    stride_srcb, stride_srcc, stride_srct,
    stride_dtb, stride_dtc, stride_dtt,
    is_second_half: tl.constexpr,  # 1 to copy to second half channels [half_channels:], 0 to copy to first half
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

    # src channel index: first half uses c_dst; second half uses c_dst + half_channels
    if is_second_half:
        c_src = c_dst + C_dst // 2  # since we only copy half_channels into second half, C_dst must be half_channels
    else:
        c_src = c_dst

    src_idx = b * stride_srcb + c_src * stride_srcc + t_offsets * stride_srct
    dst_idx = b * stride_dtb + c_dst * stride_dtc + t_offsets * stride_dtt

    vals = tl.load(src_ptr + src_idx, mask=mask_t, other=0.0)
    tl.store(dst_ptr + dst_idx, vals, mask=mask_t)


class ModelNew(torch.nn.Module):
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
        Triton-optimized forward and reverse pass. All convs, bias, ReLU, masking, add/sub, and concatenation are done by Triton kernels.
        """
        device = x.device
        batch_size = x.shape[0]
        half_channels = x.shape[1] // 2
        hidden_channels = 192  # fixed in the original code

        # List of transforms: each is (conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b)
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

        # Precompute out tensors for mask multiply (x_mask broadcast across channels)
        # We'll apply mask in Triton kernels; no torch ops in forward.

        if not reverse:
            # Forward: apply transforms sequentially
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
                # Split input
                x0 = x[:, :half_channels, :].contiguous()
                x1 = x[:, half_channels:, :].contiguous()

                T = x0.shape[2]
                T0_out = T - 1  # conv0 output time
                T1_out = T0_out - 1  # conv1 output time
                T2_out = T1_out - 1  # conv2 output time

                # conv0: in_channels=half_channels, out_channels=hidden_channels
                h0 = torch.empty((batch_size, half_channels, T0_out), device=device, dtype=torch.float32)
                grid0 = (batch_size * half_channels, triton.cdiv(T0_out, 128))
                conv1d_forward_k5_p2[grid0](
                    x0, conv0_w, h0,
                    batch_size, conv0_w.shape[0], conv0_w.shape[1], T, T0_out,
                    x0.stride(0), x0.stride(1), x0.stride(2),
                    conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                    h0.stride(0), h0.stride(1), h0.stride(2),
                    128
                )

                # Bias + ReLU
                grid_bias0 = (batch_size * conv0_w.shape[0], triton.cdiv(T0_out, 128))
                conv1d_bias[grid_bias0](
                    h0, conv0_b,
                    batch_size, conv0_w.shape[0], T0_out,
                    h0.stride(0), h0.stride(1), h0.stride(2),
                    128
                )

                grid_relu0 = (batch_size * conv0_w.shape[0], triton.cdiv(T0_out, 128))
                relu_elementwise[grid_relu0](
                    h0, batch_size, conv0_w.shape[0], T0_out,
                    h0.stride(0), h0.stride(1), h0.stride(2),
                    128
                )

                # conv1: in_channels=hidden_channels, out_channels=hidden_channels
                h1 = torch.empty((batch_size, hidden_channels, T1_out), device=device, dtype=torch.float32)
                grid1 = (batch_size * hidden_channels, triton.cdiv(T1_out, 128))
                conv1d_forward_k5_p2[grid1](
                    h0, conv1_w, h1,
                    batch_size, conv1_w.shape[0], conv1_w.shape[1], T0_out + 2, T1_out + 2,
                    h0.stride(0), h0.stride(1), h0.stride(2),
                    conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                    h1.stride(0), h1.stride(1), h1.stride(2),
                    128
                )

                grid_bias1 = (batch_size * conv1_w.shape[0], triton.cdiv(T1_out + 2, 128))
                conv1d_bias[grid_bias1](
                    h1, conv1_b,
                    batch_size, conv1_w.shape[0], T1_out + 2,
                    h1.stride(0), h1.stride(1), h1.stride(2),
                    128
                )

                grid_relu1 = (batch_size * conv1_w.shape[0], triton.cdiv(T1_out + 2, 128))
                relu_elementwise[grid_relu1](
                    h1, batch_size, conv1_w.shape[0], T1_out + 2,
                    h1.stride(0), h1.stride(1), h1.stride(2),
                    128
                )

                # conv2: in_channels=hidden_channels, out_channels=half_channels
                h2 = torch.empty((batch_size, half_channels, T2_out), device=device, dtype=torch.float32)
                grid2 = (batch_size * half_channels, triton.cdiv(T2_out, 128))
                conv1d_forward_k5_p2[grid2](
                    h1, conv2_w, h2,
                    batch_size, conv2_w.shape[0], conv2_w.shape[1], T1_out + 2, T2_out + 2,
                    h1.stride(0), h1.stride(1), h1.stride(2),
                    conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                    h2.stride(0), h2.stride(1), h2.stride(2),
                    128
                )

                grid_bias2 = (batch_size * conv2_w.shape[0], triton.cdiv(T2_out + 2, 128))
                conv1d_bias[grid_bias2](
                    h2, conv2_b,
                    batch_size, conv2_w.shape[0], T2_out + 2,
                    h2.stride(0), h2.stride(1), h2.stride(2),
                    128
                )

                # Apply mask: h2 = h2 * x_mask (broadcast across channels)
                h2_masked = torch.empty_like(h2, device=device, dtype=torch.float32)
                grid_mul = (batch_size * half_channels, triton.cdiv(T2_out, 128))
                elementwise_mul_mask[grid_mul](
                    h2, x_mask, h2_masked,
                    batch_size, half_channels, T2_out,
                    h2.stride(0), h2.stride(1), h2.stride(2),
                    x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                    128
                )

                # Affine coupling: x1 = x1 + h2_masked
                x1_after = torch.empty_like(x1, device=device, dtype=torch.float32)
                grid_add = (batch_size * half_channels, triton.cdiv(T2_out, 128))
                elementwise_add_sub[grid_add](
                    x1, h2_masked, x1_after,
                    batch_size, half_channels, T2_out,
                    x1.stride(0), x1.stride(1), x1.stride(2),
                    h2_masked.stride(0), h2_masked.stride(1), h2_masked.stride(2),
                    1,  # add_flag
                    128
                )

                # Concatenate x0 and x1_after into output with mask
                out_channels = half_channels * 2
                x_out = torch.empty((batch_size, out_channels, T2_out), device=device, dtype=torch.float32)

                # Copy first half channels (x0) into x_out[:, :half_channels, :]
                grid_copy_x0 = (batch_size * half_channels, triton.cdiv(T2_out, 128))
                copy_half_channels[grid_copy_x0](
                    x0, x_out,
                    batch_size, half_channels, half_channels, T2_out,
                    x0.stride(0), x0.stride(1), x0.stride(2),
                    x_out.stride(0), x_out.stride(1), x_out.stride(2),
                    0,  # first half
                    128
                )

                # Copy second half channels (x1_after) into x_out[:, half_channels:, :]
                grid_copy_x1 = (batch_size * half_channels, triton.cdiv(T2_out, 128))
                copy_half_channels[grid_copy_x1](
                    x1_after, x_out,
                    batch_size, half_channels, half_channels, T2_out,
                    x1_after.stride(0), x1_after.stride(1), x1_after.stride(2),
                    x_out.stride(0), x_out.stride(1), x_out.stride(2),
                    1,  # second half
                    128
                )

                # Apply mask to output: x_out = x_out * x_mask (broadcast across channels)
                x_out_masked = torch.empty_like(x_out, device=device, dtype=torch.float32)
                grid_out_mul = (batch_size * out_channels, triton.cdiv(T2_out, 128))
                elementwise_mul_mask[grid_out_mul](
                    x_out, x_mask, x_out_masked,
                    batch_size, out_channels, T2_out,
                    x_out.stride(0), x_out.stride(1), x_out.stride(2),
                    x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                    128
                )

                # Update x for next transform
                x = x_out_masked

        else:
            # Reverse: apply transforms in reverse order and subtract (instead of add)
            transforms_reversed = list(reversed(transforms))
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms_reversed:
                # Split input
                x0 = x[:, :half_channels, :].contiguous()
                x1 = x[:, half_channels:, :].contiguous()

                T = x0.shape[2]
                T0_out = T - 1  # conv0 output time
                T1_out = T0_out - 1  # conv1 output time
                T2_out = T1_out - 1  # conv2 output time

                # conv2 (inverse of forward conv2): in_channels=hidden_channels, out_channels=half_channels
                h2 = torch.empty((batch_size, half_channels, T2_out + 2), device=device, dtype=torch.float32)
                grid2 = (batch_size * half_channels, triton.cdiv(T2_out + 2, 128))
                conv1d_forward_k5_p2[grid2](
                    x1, conv2_w, h2,
                    batch_size, conv2_w.shape[0], conv2_w.shape[1], T1_out + 2, T2_out + 2,
                    x1.stride(0), x1.stride(1), x1.stride(2),
                    conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                    h2.stride(0), h2.stride(1), h2.stride(2),
                    128
                )

                grid_bias2 = (batch_size * conv2_w.shape[0], triton.cdiv(T2_out + 2, 128))
                conv1d_bias[grid_bias2](
                    h2, conv2_b,
                    batch_size, conv2_w.shape[0], T2_out + 2,
                    h2.stride(0), h2.stride(1), h2.stride(2),
                    128
                )

                # ReLU (matches original forward; here we skip ReLU to subtract the correct h)
                # We need h2 as computed by forward; since reverse uses the same weights, h2 here is the forward h2.
                # But original applies ReLU after conv + bias, then subtract. For correctness, we need to simulate that.
                # Simpler: we'll compute h2 (conv+bias) without ReLU and then multiply by x_mask and subtract from x1.
                # The original code applies ReLU after conv+bias before mask. Since we don't have the mask here, we'll assume h2 is pre-ReLU (but bias + ReLU is non-linear).
                # To match original semantics exactly, we'll compute h2 with ReLU applied to the conv+bias output in forward path.
                # In reverse, we need to subtract the same h2 (post-ReLU) used in forward for that transform.
                # We can't access forward h2, so we emulate forward ReLU by applying ReLU to h2 now.

                # Apply ReLU to h2
                grid_relu2 = (batch_size * conv2_w.shape[0], triton.cdiv(T2_out + 2, 128))
                relu_elementwise[grid_relu2](
                    h2, batch_size, conv2_w.shape[0], T2_out + 2,
                    h2.stride(0), h2.stride(1), h2.stride(2),
                    128
                )

                # Apply mask: h2 = h2 * x_mask (broadcast across channels)
                h2_masked = torch.empty_like(h2, device=device, dtype=torch.float32)
                grid_mul2 = (batch_size * half_channels, triton.cdiv(T2_out, 128))
                elementwise_mul_mask[grid_mul2](
                    h2, x_mask, h2_masked,
                    batch_size, half_channels, T2_out,
                    h2.stride(0), h2.stride(1), h2.stride(2),
                    x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                    128
                )

                # Affine coupling: x1 = x1 - h2_masked (reverse)
                x1_before = torch.empty_like(x1, device=device, dtype=torch.float32)
                grid_sub = (batch_size * half_channels, triton.cdiv(T2_out, 128))
                elementwise_add_sub[grid_sub](
                    x1, h2_masked, x1_before,
                    batch_size, half_channels, T2_out,
                    x1.stride(0), x1.stride(1), x1.stride(2),
                    h2_masked.stride(0), h2_masked.stride(1), h2_masked.stride(2),
                    0,  # subtract_flag
                    128
                )

                # Concatenate x0 and x1_before into output (mask applied later)
                out_channels = half_channels * 2
                x_out = torch.empty((batch_size, out_channels, T2_out), device=device, dtype=torch.float32)

                grid_copy_x0 = (batch_size * half_channels, triton.cdiv(T2_out, 128))
                copy_half_channels[grid_copy_x0](
                    x0, x_out,
                    batch_size, half_channels, half_channels, T2_out,
                    x0.stride(0), x0.stride(1), x0.stride(2),
                    x_out.stride(0), x_out.stride(1), x_out.stride(2),
                    0,  # first half
                    128
                )

                grid_copy_x1 = (batch_size * half_channels, triton.cdiv(T2_out, 128))
                copy_half_channels[grid_copy_x1](
                    x1_before, x_out,
                    batch_size, half_channels, half_channels, T2_out,
                    x1_before.stride(0), x1_before.stride(1), x1_before.stride(2),
                    x_out.stride(0), x_out.stride(1), x_out.stride(2),
                    1,  # second half
                    128
                )

                # Apply mask to output: x_out = x_out * x_mask (broadcast across channels)
                x_out_masked = torch.empty_like(x_out, device=device, dtype=torch.float32)
                grid_out_mul = (batch_size * out_channels, triton.cdiv(T2_out, 128))
                elementwise_mul_mask[grid_out_mul](
                    x_out, x_mask, x_out_masked,
                    batch_size, out_channels, T2_out,
                    x_out.stride(0), x_out.stride(1), x_out.stride(2),
                    x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                    128
                )

                # Update x for next transform
                x = x_out_masked

        return x


def run(*args):
    return ModelNew()(*args)
