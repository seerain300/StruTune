import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_triton_fused_bias_relu(
    x_ptr,  # *f32, [B, Cin, T]
    w_ptr,  # *f32, [Cout, Cin*K]
    b_ptr,  # *f32, [Cout]
    out_ptr,  # *f32, [B, Cout, T_out]
    B: tl.constexpr, Cin: tl.constexpr, T: tl.constexpr,
    Cout: tl.constexpr, K: tl.constexpr,  # kernel size
    STRIDE: tl.constexpr,  # stride of conv
    PADDING: tl.constexpr,  # padding
    x_stride_b, x_stride_c, x_stride_t,
    w_stride_oc, w_stride_ic,  # w has shape [Cout, Cin*K] with ic = k*Cin + ci
    out_stride_b, out_stride_c, out_stride_t,
    T_out: tl.constexpr,
    BLOCK_CO: tl.constexpr, BLOCK_T: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_t = tl.program_id(2)

    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    t_out_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_co = co_offsets < Cout
    mask_to = t_out_offsets < T_out

    acc = tl.zeros((BLOCK_CO, BLOCK_T), dtype=tl.float32)

    # Accumulate over input channels and kernel positions
    for ci in range(0, Cin):
        for k in range(0, K):
            t_in = t_out_offsets * STRIDE - PADDING + k  # vector of length BLOCK_T
            # valid positions
            mask_t = (t_in >= 0) & (t_in < T) & mask_to
            # load x[b, ci, t_in]
            x_ptrs = x_ptr + pid_b * x_stride_b + ci * x_stride_c + t_in * x_stride_t
            x_vals = tl.load(x_ptrs, mask=mask_t, other=0.0)  # shape [BLOCK_T]

            # load w[co, ci*K + k] for all co in this block
            w_ptrs = w_ptr + co_offsets * w_stride_oc + (ci * K + k) * w_stride_ic
            w_vals = tl.load(w_ptrs, mask=mask_co, other=0.0)  # shape [BLOCK_CO]

            # outer product accumulate: [BLOCK_CO, 1] * [1, BLOCK_T]
            acc += w_vals[:, None] * x_vals[None, :]

    # add bias
    b_vals = tl.load(b_ptr + co_offsets, mask=mask_co, other=0.0)  # [BLOCK_CO]
    acc = acc + b_vals[:, None]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # store to out[b, co, t_out]
    out_ptrs = out_ptr + pid_b * out_stride_b + co_offsets[:, None] * out_stride_c + t_out_offsets[None, :] * out_stride_t
    store_mask = mask_co[:, None] & mask_to[None, :]
    tl.store(out_ptrs, acc, mask=store_mask)


@triton.jit
def apply_mask_to_h_triton(
    h_ptr,  # *f32, [B, C1, T]
    mask_ptr,  # *f32, [B, 1, T]
    out_ptr,  # *f32, [B, C1, T]
    B: tl.constexpr, C1: tl.constexpr, T: tl.constexpr,
    h_stride_b, h_stride_c, h_stride_t,
    mask_stride_b, mask_stride_c, mask_stride_t,
    out_stride_b, out_stride_c, out_stride_t,
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_c = c_offsets < C1
    mask_t = t_offsets < T

    # load h
    h_ptrs = h_ptr + pid_b * h_stride_b + c_offsets[:, None] * h_stride_c + t_offsets[None, :] * h_stride_t
    h_vals = tl.load(h_ptrs, mask=mask_c[:, None] & mask_t[None, :], other=0.0)

    # load mask (shape [B, 1, T]) => index c=0
    mask_ptrs = mask_ptr + pid_b * mask_stride_b + 0 * mask_stride_c + t_offsets[None, :] * mask_stride_t
    mask_vals = tl.load(mask_ptrs, mask=mask_t[None, :], other=1.0)  # [1, BLOCK_T], broadcast across channels

    out_vals = h_vals * mask_vals

    out_ptrs = out_ptr + pid_b * out_stride_b + c_offsets[:, None] * out_stride_c + t_offsets[None, :] * out_stride_t
    tl.store(out_ptrs, out_vals, mask=mask_c[:, None] & mask_t[None, :])


@triton.jit
def add_h_to_x1_triton(
    x1_ptr,  # *f32, [B, C1, T]
    h_ptr,   # *f32, [B, C1, T]
    out_ptr, # *f32, [B, C1, T]
    B: tl.constexpr, C1: tl.constexpr, T: tl.constexpr,
    x1_stride_b, x1_stride_c, x1_stride_t,
    h_stride_b, h_stride_c, h_stride_t,
    out_stride_b, out_stride_c, out_stride_t,
    ADD: tl.constexpr,  # True: out = x1 + h; False: out = x1 - h
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_c = c_offsets < C1
    mask_t = t_offsets < T

    x1_ptrs = x1_ptr + pid_b * x1_stride_b + c_offsets[:, None] * x1_stride_c + t_offsets[None, :] * x1_stride_t
    h_ptrs = h_ptr + pid_b * h_stride_b + c_offsets[:, None] * h_stride_c + t_offsets[None, :] * h_stride_t

    x1_vals = tl.load(x1_ptrs, mask=mask_c[:, None] & mask_t[None, :], other=0.0)
    h_vals = tl.load(h_ptrs, mask=mask_c[:, None] & mask_t[None, :], other=0.0)

    if ADD:
        out_vals = x1_vals + h_vals
    else:
        out_vals = x1_vals - h_vals

    out_ptrs = out_ptr + pid_b * out_stride_b + c_offsets[:, None] * out_stride_c + t_offsets[None, :] * out_stride_t
    tl.store(out_ptrs, out_vals, mask=mask_c[:, None] & mask_t[None, :])


@triton.jit
def concat_copy_half(
    out_ptr,  # *f32, destination [B, Cout, T], but we only write the half
    src_ptr,  # *f32, source [B, C_half, T]
    B: tl.constexpr, C_half: tl.constexpr, T: tl.constexpr,
    out_stride_b, out_stride_c, out_stride_t,
    src_stride_b, src_stride_c, src_stride_t,
    start_c_in_out: tl.constexpr,  # starting channel index in output (0 or half_channels)
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_c = c_offsets < C_half
    mask_t = t_offsets < T

    # src uses [0:C_half], out uses [start_c_in_out:start_c_in_out+C_half]
    src_ptrs = src_ptr + pid_b * src_stride_b + c_offsets[:, None] * src_stride_c + t_offsets[None, :] * src_stride_t
    src_vals = tl.load(src_ptrs, mask=mask_c[:, None] & mask_t[None, :], other=0.0)

    out_ptrs = out_ptr + pid_b * out_stride_b + (c_offsets[:, None] + start_c_in_out) * out_stride_c + t_offsets[None, :] * out_stride_t
    tl.store(out_ptrs, src_vals, mask=mask_c[:, None] & mask_t[None, :])


@triton.jit
def concat_add_v1(
    x0_ptr, h_ptr, x1_ptr, out_ptr,
    B: tl.constexpr, half_channels: tl.constexpr, C1: tl.constexpr, T: tl.constexpr,
    x0_stride_b, x0_stride_c, x0_stride_t,
    h_stride_b, h_stride_c, h_stride_t,
    x1_stride_b, x1_stride_c, x1_stride_t,
    out_stride_b, out_stride_c, out_stride_t,
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr
):
    # Launch grid over (B, half_channels, T)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_c = c_offsets < half_channels
    mask_t = t_offsets < T

    # copy x0 -> out[:, :half_channels, :]
    x0_ptrs = x0_ptr + pid_b * x0_stride_b + c_offsets[:, None] * x0_stride_c + t_offsets[None, :] * x0_stride_t
    out_ptrs0 = out_ptr + pid_b * out_stride_b + c_offsets[:, None] * out_stride_c + t_offsets[None, :] * out_stride_t
    tl.store(out_ptrs0, tl.load(x0_ptrs, mask=mask_c[:, None] & mask_t[None, :], other=0.0), mask=mask_c[:, None] & mask_t[None, :])

    # update x1 = x1 + h and write to out[:, half_channels:, :]
    x1_ptrs = x1_ptr + pid_b * x1_stride_b + c_offsets[:, None] * x1_stride_c + t_offsets[None, :] * x1_stride_t
    h_ptrs = h_ptr + pid_b * h_stride_b + c_offsets[:, None] * h_stride_c + t_offsets[None, :] * h_stride_t
    x1_vals = tl.load(x1_ptrs, mask=mask_c[:, None] & mask_t[None, :], other=0.0)
    h_vals = tl.load(h_ptrs, mask=mask_c[:, None] & mask_t[None, :], other=0.0)
    out1_vals = x1_vals + h_vals

    out_ptrs1 = out_ptr + pid_b * out_stride_b + (c_offsets[:, None] + half_channels) * out_stride_c + t_offsets[None, :] * out_stride_t
    tl.store(out_ptrs1, out1_vals, mask=mask_c[:, None] & mask_t[None, :])


@triton.jit
def concat_add_v2(
    x0_ptr, h_ptr, x1_ptr, out_ptr,
    B: tl.constexpr, half_channels: tl.constexpr, C1: tl.constexpr, T: tl.constexpr,
    x0_stride_b, x0_stride_c, x0_stride_t,
    h_stride_b, h_stride_c, h_stride_t,
    x1_stride_b, x1_stride_c, x1_stride_t,
    out_stride_b, out_stride_c, out_stride_t,
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr
):
    # Launch grid over (B, half_channels, T)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_c = c_offsets < half_channels
    mask_t = t_offsets < T

    # copy x0 -> out[:, :half_channels, :]
    x0_ptrs = x0_ptr + pid_b * x0_stride_b + c_offsets[:, None] * x0_stride_c + t_offsets[None, :] * x0_stride_t
    out_ptrs0 = out_ptr + pid_b * out_stride_b + c_offsets[:, None] * out_stride_c + t_offsets[None, :] * out_stride_t
    tl.store(out_ptrs0, tl.load(x0_ptrs, mask=mask_c[:, None] & mask_t[None, :], other=0.0), mask=mask_c[:, None] & mask_t[None, :])

    # update x1 = x1 - h and write to out[:, half_channels:, :]
    x1_ptrs = x1_ptr + pid_b * x1_stride_b + c_offsets[:, None] * x1_stride_c + t_offsets[None, :] * x1_stride_t
    h_ptrs = h_ptr + pid_b * h_stride_b + c_offsets[:, None] * h_stride_c + t_offsets[None, :] * h_stride_t
    x1_vals = tl.load(x1_ptrs, mask=mask_c[:, None] & mask_t[None, :], other=0.0)
    h_vals = tl.load(h_ptrs, mask=mask_c[:, None] & mask_t[None, :], other=0.0)
    out1_vals = x1_vals - h_vals

    out_ptrs1 = out_ptr + pid_b * out_stride_b + (c_offsets[:, None] + half_channels) * out_stride_c + t_offsets[None, :] * out_stride_t
    tl.store(out_ptrs1, out1_vals, mask=mask_c[:, None] & mask_t[None, :])


@triton.jit
def apply_transform_triton(
    x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b,
    x_mask,  # [B, 1, T]
    B: tl.constexpr, half_channels: tl.constexpr, C1: tl.constexpr, T: tl.constexpr,
    # strides for x0
    x0_stride_b, x0_stride_c, x0_stride_t,
    # strides for conv weights (shape [Cout, Cin*K])
    w0_stride_oc, w0_stride_ic, w1_stride_oc, w1_stride_ic, w2_stride_oc, w2_stride_ic,
    # strides for biases
    b0_stride, b1_stride, b2_stride,
    # strides for h
    h_stride_b, h_stride_c, h_stride_t,
    # strides for x1
    x1_stride_b, x1_stride_c, x1_stride_t,
    # conv params
    K: tl.constexpr, PADDING: tl.constexpr,
    # block sizes
    BLOCK_CO: tl.constexpr, BLOCK_T: tl.constexpr,
    ADD: tl.constexpr  # True for forward (+), False for reverse (-)
):
    # Launch conv0
    h0 = torch.empty((B, C1, T), device=x0.device, dtype=x0.dtype)
    conv1d_triton_fused_bias_relu(
        x0, conv0_w, conv0_b, h0, B, half_channels, T, C1, K, 1, PADDING,
        x0_stride_b, x0_stride_c, x0_stride_t,
        w0_stride_oc, w0_stride_ic,
        h0.stride(0), h0.stride(1), h0.stride(2),
        C1, BLOCK_CO, BLOCK_T
    )

    # ReLU
    h0 = torch.maximum(h0, torch.tensor(0.0, dtype=h0.dtype, device=h0.device))

    # conv1
    h1 = torch.empty((B, C1, T), device=x0.device, dtype=x0.dtype)
    conv1d_triton_fused_bias_relu(
        h0, conv1_w, conv1_b, h1, B, C1, T, C1, K, 1, PADDING,
        h0.stride(0), h0.stride(1), h0.stride(2),
        w1_stride_oc, w1_stride_ic,
        h1.stride(0), h1.stride(1), h1.stride(2),
        C1, BLOCK_CO, BLOCK_T
    )
    h1 = torch.maximum(h1, torch.tensor(0.0, dtype=h1.dtype, device=h1.device))

    # conv2
    h2 = torch.empty((B, C1, T), device=x0.device, dtype=x0.dtype)
    conv1d_triton_fused_bias_relu(
        h1, conv2_w, conv2_b, h2, B, C1, T, C1, K, 1, PADDING,
        h1.stride(0), h1.stride(1), h1.stride(2),
        w2_stride_oc, w2_stride_ic,
        h2.stride(0), h2.stride(1), h2.stride(2),
        C1, BLOCK_CO, BLOCK_T
    )

    # apply mask
    masked_h = torch.empty_like(h2)
    apply_mask_to_h_triton(
        h2, x_mask, masked_h, B, C1, T,
        h2.stride(0), h2.stride(1), h2.stride(2),
        x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
        masked_h.stride(0), masked_h.stride(1), masked_h.stride(2),
        C1, BLOCK_CO, BLOCK_T
    )

    # update x1
    x1 = x0[:, half_channels:, :]  # dummy for kernel; we pass real x1 below
    # Create a temporary x1 tensor (we don't have it here; use x0 dummy for addresses). We will instead launch concat_add_v* with actual x1.
    # Note: This function signature is used in ModelNew.apply_one_transform to provide actual x1_ptr.

    # We need to return (h_masked, updated_x1). Since we can't store to caller's x1 directly, we will compute updated_x1 and use concat_add kernels in the caller.

    # The caller will handle out tensor and concat. We return nothing; the caller will pass actual x1_ptr for update.


# Helper to launch apply_transform_triton and concat
def apply_one_transform_triton(x, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b, x_mask, half_channels, C1, T,
                               x0, x1, out,
                               K=5, PADDING=2,
                               BLOCK_CO=128, BLOCK_T=128, ADD=True):
    # Compute conv and updates via Triton
    # Prepare strides
    x0_stride_b, x0_stride_c, x0_stride_t = x0.stride()
    x1_stride_b, x1_stride_c, x1_stride_t = x1.stride()
    out_stride_b, out_stride_c, out_stride_t = out.stride()

    w0_stride_oc, w0_stride_ic = conv0_w.stride()
    w1_stride_oc, w1_stride_ic = conv1_w.stride()
    w2_stride_oc, w2_stride_ic = conv2_w.stride()

    b0_stride = conv0_b.stride()[0]
    b1_stride = conv1_b.stride()[0]
    b2_stride = conv2_b.stride()[0]

    # conv0 -> ReLU
    h0 = torch.empty((x.shape[0], C1, T), device=x.device, dtype=x.dtype)
    conv1d_triton_fused_bias_relu(
        x, conv0_w, conv0_b, h0, x.shape[0], half_channels, T, C1, K, 1, PADDING,
        x.stride(0), x.stride(1), x.stride(2),
        w0_stride_oc, w0_stride_ic,
        h0.stride(0), h0.stride(1), h0.stride(2),
        C1, BLOCK_CO, BLOCK_T
    )
    # ReLU
    # We don't have a Triton ReLU kernel; apply in-place via PyTorch for correctness
    h0.clamp_min_(0)

    # conv1
    h1 = torch.empty((x.shape[0], C1, T), device=x.device, dtype=x.dtype)
    conv1d_triton_fused_bias_relu(
        h0, conv1_w, conv1_b, h1, x.shape[0], C1, T, C1, K, 1, PADDING,
        h0.stride(0), h0.stride(1), h0.stride(2),
        w1_stride_oc, w1_stride_ic,
        h1.stride(0), h1.stride(1), h1.stride(2),
        C1, BLOCK_CO, BLOCK_T
    )
    h1.clamp_min_(0)

    # conv2
    h2 = torch.empty((x.shape[0], C1, T), device=x.device, dtype=x.dtype)
    conv1d_triton_fused_bias_relu(
        h1, conv2_w, conv2_b, h2, x.shape[0], C1, T, C1, K, 1, PADDING,
        h1.stride(0), h1.stride(1), h1.stride(2),
        w2_stride_oc, w2_stride_ic,
        h2.stride(0), h2.stride(1), h2.stride(2),
        C1, BLOCK_CO, BLOCK_T
    )

    # apply mask
    masked_h = torch.empty_like(h2)
    apply_mask_to_h_triton(
        h2, x_mask, masked_h, x.shape[0], C1, T,
        h2.stride(0), h2.stride(1), h2.stride(2),
        x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
        masked_h.stride(0), masked_h.stride(1), masked_h.stride(2),
        C1, BLOCK_CO, BLOCK_T
    )

    # update x1
    if ADD:
        out_x1 = torch.empty_like(x1)
        add_h_to_x1_triton(
            x1, masked_h, out_x1, x.shape[0], C1, T,
            x1.stride(0), x1.stride(1), x1.stride(2),
            masked_h.stride(0), masked_h.stride(1), masked_h.stride(2),
            out_x1.stride(0), out_x1.stride(1), out_x1.stride(2),
            True, 128, 128
        )
    else:
        out_x1 = torch.empty_like(x1)
        add_h_to_x1_triton(
            x1, masked_h, out_x1, x.shape[0], C1, T,
            x1.stride(0), x1.stride(1), x1.stride(2),
            masked_h.stride(0), masked_h.stride(1), masked_h.stride(2),
            out_x1.stride(0), out_x1.stride(1), out_x1.stride(2),
            False, 128, 128
        )

    # concatenate [x0, out_x1] to out
    # out has shape [B, 2*half_channels, T]
    concat_add_v1(
        x0, masked_h, out_x1, out, x.shape[0], half_channels, C1, T,
        x0.stride(0), x0.stride(1), x0.stride(2),
        masked_h.stride(0), masked_h.stride(1), masked_h.stride(2),
        out_x1.stride(0), out_x1.stride(1), out_x1.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        128, 128
    ) if ADD else concat_add_v2(
        x0, masked_h, out_x1, out, x.shape[0], half_channels, C1, T,
        x0.stride(0), x0.stride(1), x0.stride(2),
        masked_h.stride(0), masked_h.stride(1), masked_h.stride(2),
        out_x1.stride(0), out_x1.stride(1), out_x1.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        128, 128
    )

    return out


class ModelNew(nn.Module):
    def forward(self, x, x_mask, reverse, transform_0_conv0_weight, transform_0_conv0_bias,
                transform_0_conv1_weight, transform_0_conv1_bias, transform_0_conv2_weight, transform_0_conv2_bias,
                transform_1_conv0_weight, transform_1_conv0_bias, transform_1_conv1_weight, transform_1_conv1_bias, transform_1_conv2_weight, transform_1_conv2_bias,
                transform_2_conv0_weight, transform_2_conv0_bias, transform_2_conv1_weight, transform_2_conv1_bias, transform_2_conv2_weight, transform_2_conv2_bias,
                transform_3_conv0_weight, transform_3_conv0_bias, transform_3_conv1_weight, transform_3_conv1_bias, transform_3_conv2_weight, transform_3_conv2_bias):
        B, C, T = x.shape
        half_channels = C // 2
        hidden_channels = 192
        C1 = hidden_channels  # number of channels in the second half per transform

        # Prepare transforms list (4 transforms)
        transforms = [
            (transform_0_conv0_weight, transform_0_conv0_bias, transform_0_conv1_weight, transform_0_conv1_bias, transform_0_conv2_weight, transform_0_conv2_bias),
            (transform_1_conv0_weight, transform_1_conv0_bias, transform_1_conv1_weight, transform_1_conv1_bias, transform_1_conv2_weight, transform_1_conv2_bias),
            (transform_2_conv0_weight, transform_2_conv0_bias, transform_2_conv1_weight, transform_2_conv1_bias, transform_2_conv2_weight, transform_2_conv2_bias),
            (transform_3_conv0_weight, transform_3_conv0_bias, transform_3_conv1_weight, transform_3_conv1_bias, transform_3_conv2_weight, transform_3_conv2_bias),
        ]

        # Initial x0, x1 views (not slices), but we need contiguous for stride
        x0 = x[:, :half_channels, :].contiguous()
        x1 = x[:, half_channels:, :].contiguous()

        if not reverse:
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
                # Allocate out for concatenation [B, C, T]
                out = torch.empty((B, C, T), device=x.device, dtype=x.dtype)
                # Apply transform and concatenate
                apply_one_transform_triton(
                    x, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b, x_mask,
                    half_channels, C1, T,
                    x0, x1, out,
                    K=5, PADDING=2,
                    BLOCK_CO=128, BLOCK_T=128, ADD=True
                )
                # Update x for next transform: x = out
                x = out
                # Recompute x0, x1 from updated x
                x0 = x[:, :half_channels, :].contiguous()
                x1 = x[:, half_channels:, :].contiguous()
        else:
            # Reverse order
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
                out = torch.empty((B, C, T), device=x.device, dtype=x.dtype)
                apply_one_transform_triton(
                    x, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b, x_mask,
                    half_channels, C1, T,
                    x0, x1, out,
                    K=5, PADDING=2,
                    BLOCK_CO=128, BLOCK_T=128, ADD=False
                )
                x = out
                x0 = x[:, :half_channels, :].contiguous()
                x1 = x[:, half_channels:, :].contiguous()

        # Apply final x_mask along channel dimension: broadcast [B,1,T] -> [B,C,T]
        # This matches original behavior. Note: in provided inputs, x_mask is ones so it's a no-op.
        x = x * x_mask

        return x


def run(*args):
    return ModelNew()(*args)
