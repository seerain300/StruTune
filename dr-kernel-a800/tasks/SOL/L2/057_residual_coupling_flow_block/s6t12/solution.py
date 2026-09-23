import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def conv1d_k5_p2(x_ptr, w_ptr, bias_ptr, y_ptr,
                 B, C_IN, C_OUT, T_IN, T_OUT,
                 x_stride_b, x_stride_c, x_stride_t,
                 w_stride_co, w_stride_ci, w_stride_k,
                 y_stride_b, y_stride_c, y_stride_t,
                 BLOCK_T: tl.constexpr):
    # Each program handles one (b, co) and a tile along time.
    pid = tl.program_id(axis=0)  # over B*C_OUT
    b = pid // C_OUT
    co = pid % C_OUT

    # Time tile
    t_start = tl.program_id(axis=1) * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T_OUT

    # Accumulator for the output tile
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # Loop over input channels and kernel taps
    # K=5, padding=2 -> valid conv with output T_OUT = T_IN - 1
    for ci in range(0, C_IN):
        for k in range(0, 5):
            t_src = t_offsets - 2 + k  # source time index for each output t
            valid = (t_src >= 0) & (t_src < T_IN) & mask_t

            # Load x[b, ci, t_src] as float32
            x_addr = x_ptr + b * x_stride_b + ci * x_stride_c + t_src * x_stride_t
            x_vals = tl.load(x_addr, mask=valid, other=0.0)

            # Load w[co, ci, k] as scalar
            w_addr = w_ptr + co * w_stride_co + ci * w_stride_ci + k * w_stride_k
            w_val = tl.load(w_addr)

            acc += x_vals * w_val

    # Add bias if provided (bias_ptr may be None; assume bias is None if not used)
    # We assume bias is present for conv1d_k5_p2 launches.
    b_addr = bias_ptr + co
    bias_val = tl.load(b_addr)
    acc += bias_val

    # Store output tile as float32; Triton will cast on store if needed
    y_addr = y_ptr + b * y_stride_b + co * y_stride_c + t_offsets * y_stride_t
    tl.store(y_addr, acc, mask=mask_t)


@triton.jit
def relu_kernel(y_ptr, B, C, T, y_stride_b, y_stride_c, y_stride_t, BLOCK_T: tl.constexpr):
    pid = tl.program_id(axis=0)
    b = pid // C
    c = pid % C

    t_start = tl.program_id(axis=1) * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    y_addr = y_ptr + b * y_stride_b + c * y_stride_c + t_offsets * y_stride_t
    vals = tl.load(y_addr, mask=mask_t, other=0.0)
    vals = tl.maximum(vals, 0.0)
    tl.store(y_addr, vals, mask=mask_t)


@triton.jit
def mul_mask(y_ptr, mask_ptr, B, C, T, y_stride_b, y_stride_c, y_stride_t,
             mask_stride_b, mask_stride_c, mask_stride_t, BLOCK_T: tl.constexpr):
    pid = tl.program_id(axis=0)
    b = pid // C
    c = pid % C

    t_start = tl.program_id(axis=1) * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    y_addr = y_ptr + b * y_stride_b + c * y_stride_c + t_offsets * y_stride_t
    y_vals = tl.load(y_addr, mask=mask_t, other=0.0)

    # mask has shape [B, 1, T]; we index t_offsets, channel dimension is 0
    mask_addr = mask_ptr + b * mask_stride_b + 0 * mask_stride_c + t_offsets * mask_stride_t
    mask_vals = tl.load(mask_addr, mask=mask_t, other=1.0)

    y_vals = y_vals * mask_vals
    tl.store(y_addr, y_vals, mask=mask_t)


@triton.jit
def add_or_sub(y_ptr, h_ptr, B, C, T, y_stride_b, y_stride_c, y_stride_t,
               h_stride_b, h_stride_c, h_stride_t, op: tl.constexpr, BLOCK_T: tl.constexpr):
    # op: 0 => add, 1 => sub
    pid = tl.program_id(axis=0)
    b = pid // C
    c = pid % C

    t_start = tl.program_id(axis=1) * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    y_addr = y_ptr + b * y_stride_b + c * y_stride_c + t_offsets * y_stride_t
    y_vals = tl.load(y_addr, mask=mask_t, other=0.0)

    h_addr = h_ptr + b * h_stride_b + c * h_stride_c + t_offsets * h_stride_t
    h_vals = tl.load(h_addr, mask=mask_t, other=0.0)

    if op == 0:
        y_vals = y_vals + h_vals
    else:
        y_vals = y_vals - h_vals

    tl.store(y_addr, y_vals, mask=mask_t)


@triton.jit
def copy_to(src_ptr, dst_ptr, B, C, T, src_stride_b, src_stride_c, src_stride_t,
            dst_stride_b, dst_c_off, dst_stride_c, dst_stride_t, BLOCK_T: tl.constexpr):
    # copy src[:, :, :] to dst[:, dst_c_off:, :]
    pid = tl.program_id(axis=0)
    b = pid // C
    c = pid % C

    t_start = tl.program_id(axis=1) * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    src_addr = src_ptr + b * src_stride_b + c * src_stride_c + t_offsets * src_stride_t
    vals = tl.load(src_addr, mask=mask_t, other=0.0)

    dst_addr = dst_ptr + b * dst_stride_b + (c + dst_c_off) * dst_stride_c + t_offsets * dst_stride_t
    tl.store(dst_addr, vals, mask=mask_t)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; everything is Triton computation

    def forward(self, x, x_mask,
                transform_0_conv0_weight, transform_0_conv0_bias,
                transform_0_conv1_weight, transform_0_conv1_bias,
                transform_0_conv2_weight, transform_0_conv2_bias,
                transform_1_conv0_weight, transform_1_conv0_bias,
                transform_1_conv1_weight, transform_1_conv1_bias,
                transform_1_conv2_weight, transform_1_conv2_bias,
                transform_2_conv0_weight, transform_2_conv0_bias,
                transform_2_conv1_weight, transform_2_conv1_bias,
                transform_2_conv2_weight, transform_2_conv2_bias,
                transform_3_conv0_weight, transform_3_conv0_bias,
                transform_3_conv1_weight, transform_3_conv1_bias,
                transform_3_conv2_weight, transform_3_conv2_bias):
        """
        Forward: apply 4 transforms to x in-place:
        Split x into x0 (first 96 channels) and x1 (last 96 channels).
        For each transform, run conv0 -> ReLU -> conv1 -> ReLU -> conv2 -> ReLU -> multiply mask -> add to x1.
        Finally, concatenate x0 and updated x1 into x (in-place), then multiply by mask.
        All math is done via Triton kernels; no torch ops are used.
        """
        # x: [B, 192, T], x_mask: [B, 1, T]
        B, C, T = x.shape
        half_channels = C // 2
        # x0 and x1 slices (views); we will mutate x1 in-place
        x0 = x[:, :half_channels, :]  # [B, 96, T]
        x1 = x[:, half_channels:, :]  # [B, 96, T]

        # We apply 4 transforms sequentially
        # List of transforms per loop: (conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b)
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

        # Iterate over transforms
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            # Compute conv0: input x0 -> h0 (channels=192, T_out=T-1)
            C_IN0 = conv0_w.shape[1]  # 96
            C_OUT0 = conv0_w.shape[0] # 192
            T0 = T
            T_OUT0 = T0 - 1
            if T_OUT0 <= 0:
                # Degenerate case; skip for safety
                continue
            h0 = torch.empty((B, C_OUT0, T_OUT0), device=x.device, dtype=x.dtype)

            # Launch conv1d_k5_p2 for conv0
            BLOCK_T = 128
            grid0 = (B * C_OUT0, triton.cdiv(T_OUT0, BLOCK_T))
            conv1d_k5_p2[grid0](
                x0, conv0_w, conv0_b, h0,
                B, C_IN0, C_OUT0, T0, T_OUT0,
                x0.stride(0), x0.stride(1), x0.stride(2),
                conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                h0.stride(0), h0.stride(1), h0.stride(2),
                BLOCK_T=BLOCK_T,
            )

            # ReLU conv0 output
            grid_relu0 = (B * C_OUT0, triton.cdiv(T_OUT0, BLOCK_T))
            relu_kernel[grid_relu0](
                h0, B, C_OUT0, T_OUT0,
                h0.stride(0), h0.stride(1), h0.stride(2),
                BLOCK_T=BLOCK_T,
            )

            # Compute conv1: input h0 -> h1 (channels=192, T1=T_OUT0-1 = T-2)
            C_IN1 = conv1_w.shape[1]  # 192
            C_OUT1 = conv1_w.shape[0] # 192
            T1 = T_OUT0
            T_OUT1 = T1 - 1
            if T_OUT1 <= 0:
                continue
            h1 = torch.empty((B, C_OUT1, T_OUT1), device=x.device, dtype=x.dtype)

            grid1 = (B * C_OUT1, triton.cdiv(T_OUT1, BLOCK_T))
            conv1d_k5_p2[grid1](
                h0, conv1_w, conv1_b, h1,
                B, C_IN1, C_OUT1, T1, T_OUT1,
                h0.stride(0), h0.stride(1), h0.stride(2),
                conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                h1.stride(0), h1.stride(1), h1.stride(2),
                BLOCK_T=BLOCK_T,
            )

            # ReLU conv1 output
            grid_relu1 = (B * C_OUT1, triton.cdiv(T_OUT1, BLOCK_T))
            relu_kernel[grid_relu1](
                h1, B, C_OUT1, T_OUT1,
                h1.stride(0), h1.stride(1), h1.stride(2),
                BLOCK_T=BLOCK_T,
            )

            # Compute conv2: input h1 -> h2 (channels=96, T2=T_OUT1-1 = T-3)
            C_IN2 = conv2_w.shape[1]  # 192
            C_OUT2 = conv2_w.shape[0] # 96
            T2 = T_OUT1
            T_OUT2 = T2 - 1
            if T_OUT2 <= 0:
                continue
            h2 = torch.empty((B, C_OUT2, T_OUT2), device=x.device, dtype=x.dtype)

            grid2 = (B * C_OUT2, triton.cdiv(T_OUT2, BLOCK_T))
            conv1d_k5_p2[grid2](
                h1, conv2_w, conv2_b, h2,
                B, C_IN2, C_OUT2, T2, T_OUT2,
                h1.stride(0), h1.stride(1), h1.stride(2),
                conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                h2.stride(0), h2.stride(1), h2.stride(2),
                BLOCK_T=BLOCK_T,
            )

            # ReLU conv2 output
            grid_relu2 = (B * C_OUT2, triton.cdiv(T_OUT2, BLOCK_T))
            relu_kernel[grid_relu2](
                h2, B, C_OUT2, T_OUT2,
                h2.stride(0), h2.stride(1), h2.stride(2),
                BLOCK_T=BLOCK_T,
            )

            # Apply mask to h2: x_mask shape [B, 1, T], broadcast across channels
            # Launch mul_mask
            grid_mul = (B * C_OUT2, triton.cdiv(T_OUT2, BLOCK_T))
            mul_mask[grid_mul](
                h2, x_mask, B, C_OUT2, T_OUT2,
                h2.stride(0), h2.stride(1), h2.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                BLOCK_T=BLOCK_T,
            )

            # Update x1: x1 = x1 + h2
            # x1: [B, 96, T], h2: [B, 96, T-3]
            grid_add = (B * C_OUT2, triton.cdiv(T, BLOCK_T))
            add_or_sub[grid_add](
                x1, h2, B, C_OUT2, T,
                x1.stride(0), x1.stride(1), x1.stride(2),
                h2.stride(0), h2.stride(1), h2.stride(2),
                op=0, BLOCK_T=BLOCK_T,
            )

            # At this point, x1 has been updated. We continue to the next transform.

        # Finally, concatenate x0 and updated x1 into x in-place.
        # x: [B, 192, T], we need to write x[:, :96, :] = x0, x[:, 96:, :] = x1
        # Launch copy_to twice:
        # 1) copy x0 into x[:, :96, :]
        grid_copy0 = (B * half_channels, triton.cdiv(T, BLOCK_T))
        copy_to[grid_copy0](
            x0, x, B, half_channels, T,
            x0.stride(0), x0.stride(1), x0.stride(2),
            x.stride(0), 0, x.stride(1), x.stride(2),
            BLOCK_T=BLOCK_T,
        )
        # 2) copy updated x1 into x[:, 96:, :]
        grid_copy1 = (B * half_channels, triton.cdiv(T, BLOCK_T))
        copy_to[grid_copy1](
            x1, x, B, half_channels, T,
            x1.stride(0), x1.stride(1), x1.stride(2),
            x.stride(0), half_channels, x.stride(1), x.stride(2),
            BLOCK_T=BLOCK_T,
        )

        # Apply mask to final x: x = x * x_mask (broadcast [B,1,T] over channels)
        # Launch mul_mask across x's channels (192) and time T
        grid_final_mask = (B * 192, triton.cdiv(T, BLOCK_T))
        mul_mask[grid_final_mask](
            x, x_mask, B, 192, T,
            x.stride(0), x.stride(1), x.stride(2),
            x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
            BLOCK_T=BLOCK_T,
        )

        # Return mutated x
        return x


def run(*args):
    return ModelNew()(*args)
