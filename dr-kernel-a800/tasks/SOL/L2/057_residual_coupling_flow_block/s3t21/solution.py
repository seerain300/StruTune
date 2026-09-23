import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Conv1d for stride=1, padding=0, kernel_size=5, bias=True
# x: [N, Cin, L_in], w: [Cout, Cin, 5], b: [Cout], output: [N, Cout, L_out], L_out = L_in - 4
@triton.jit
def conv1d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, Cin, Cout, L_in, L_out,
    # strides for x: sN, sCin, sL
    x_sN, x_sCin, x_sL,
    # strides for w: sCout, sCin, sK
    w_sCout, w_sCin, w_sK,
    # strides for out: sN, sCout, sL_out
    out_sN, out_sCout, out_sL,
    BLOCK_T: tl.constexpr,
):
    # program id: along (N*Cout, tiles of L_out)
    pid_nc = tl.program_id(0)
    pid_t = tl.program_id(1)

    n = pid_nc // Cout
    oc = pid_nc % Cout

    # offsets for time tile
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L_out

    # accumulate in float32
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # loop over input channels and kernel taps
    for cin in range(0, Cin):
        # for each tap k, compute input l_in = t_offsets - k, check validity (padding=0 implies t_offsets >= k and t_offsets < L_in)
        for k in range(0, 5):
            l_in_offsets = t_offsets - k
            valid = (l_in_offsets >= 0) & (l_in_offsets < L_in) & mask_t

            # compute input pointers: x[n, cin, l_in]
            x_ptrs = x_ptr + n * x_sN + cin * x_sCin + l_in_offsets * x_sL
            # masked load; other=0.0
            x_vals = tl.load(x_ptrs, mask=valid, other=0.0)
            # cast to float32
            x_vals = x_vals.to(tl.float32)

            # load weight scalar w[oc, cin, k]
            w_ptr_scalar = w_ptr + oc * w_sCout + cin * w_sCin + k * w_sK
            w_val = tl.load(w_ptr_scalar)
            w_val = w_val.to(tl.float32)

            acc += x_vals * w_val

    # add bias
    b_val = tl.load(b_ptr + oc)
    b_val = b_val.to(tl.float32)
    acc += b_val

    # store to output: out[n, oc, t_offsets]
    out_ptrs = out_ptr + n * out_sN + oc * out_sCout + t_offsets * out_sL
    tl.store(out_ptrs, acc, mask=mask_t)


# ReLU elementwise
@triton.jit
def relu_kernel(x_ptr, out_ptr, N, C, L, sN, sC, sL, BLOCK_T: tl.constexpr):
    pid_nc = tl.program_id(0)
    pid_t = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L

    x_ptrs = x_ptr + n * sN + c * sC + t_offsets * sL
    x_vals = tl.load(x_ptrs, mask=mask_t, other=0.0).to(tl.float32)
    x_vals = tl.maximum(x_vals, 0.0)
    out_ptrs = out_ptr + n * sN + c * sC + t_offsets * sL
    tl.store(out_ptrs, x_vals, mask=mask_t)


# Multiply by mask: x: [N, C, L], mask: [N, 1, L]
@triton.jit
def multiply_mask_kernel(x_ptr, mask_ptr, out_ptr, N, C, L,
                          x_sN, x_sC, x_sL,
                          mask_sN, mask_sL,  # mask has Cdim=1, so we ignore sC for mask
                          out_sN, out_sC, out_sL,
                          BLOCK_T: tl.constexpr):
    pid_nc = tl.program_id(0)
    pid_t = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L

    x_ptrs = x_ptr + n * x_sN + c * x_sC + t_offsets * x_sL
    x_vals = tl.load(x_ptrs, mask=mask_t, other=0.0).to(tl.float32)

    # mask is [N, 1, L] => we can index with C=0 safely as it's ignored
    mask_ptrs = mask_ptr + n * mask_sN + 0 * mask_sL + t_offsets * mask_sL
    mask_vals = tl.load(mask_ptrs, mask=mask_t, other=1.0).to(tl.float32)

    out_vals = x_vals * mask_vals

    out_ptrs = out_ptr + n * out_sN + c * out_sC + t_offsets * out_sL
    tl.store(out_ptrs, out_vals, mask=mask_t)


# Add/subtract masked: out = x + h if reverse == False else out = x - h
@triton.jit
def add_masked_kernel(x_ptr, h_ptr, out_ptr, N, C, L, x_sN, x_sC, x_sL, h_sN, h_sC, h_sL, reverse: tl.constexpr, BLOCK_T: tl.constexpr):
    pid_nc = tl.program_id(0)
    pid_t = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C

    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L

    x_ptrs = x_ptr + n * x_sN + c * x_sC + t_offsets * x_sL
    h_ptrs = h_ptr + n * h_sN + c * h_sC + t_offsets * h_sL
    x_vals = tl.load(x_ptrs, mask=mask_t, other=0.0).to(tl.float32)
    h_vals = tl.load(h_ptrs, mask=mask_t, other=0.0).to(tl.float32)

    if reverse:
        out_vals = x_vals - h_vals
    else:
        out_vals = x_vals + h_vals

    out_ptrs = out_ptr + n * x_sN + c * x_sC + t_offsets * x_sL  # reuse x strides for out
    tl.store(out_ptrs, out_vals, mask=mask_t)


# Concatenate along channels: x0: [N, C0, L], x1: [N, C1, L] -> y: [N, C0+C1, L]
@triton.jit
def concatenate_channels_kernel(x0_ptr, x1_ptr, y_ptr, N, C0, C1, L,
                                 x0_sN, x0_sC, x0_sL,
                                 x1_sN, x1_sC, x1_sL,
                                 y_sN, y_sC, y_sL,
                                 BLOCK_T: tl.constexpr):
    pid_nc = tl.program_id(0)
    pid_t = tl.program_id(1)

    n = pid_nc // (C0 + C1)
    # for first part
    c0 = pid_nc % (C0 + C1)
    if c0 < C0:
        c = c0
        in0 = True
    else:
        c = c0 - C0
        in0 = False

    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < L

    if in0:
        src_ptr = x0_ptr + n * x0_sN + c * x0_sC + t_offsets * x0_sL
        vals = tl.load(src_ptr, mask=mask_t, other=0.0).to(tl.float32)
        dst_ptr = y_ptr + n * y_sN + c * y_sC + t_offsets * y_sL
        tl.store(dst_ptr, vals, mask=mask_t)
    else:
        src_ptr = x1_ptr + n * x1_sN + c * x1_sC + t_offsets * x1_sL
        vals = tl.load(src_ptr, mask=mask_t, other=0.0).to(tl.float32)
        dst_ptr = y_ptr + n * y_sN + (c + C0) * y_sC + t_offsets * y_sL
        tl.store(dst_ptr, vals, mask=mask_t)


@torch.no_grad()
def run(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    reverse: bool,
    transform_0_conv0_weight: torch.Tensor,
    transform_0_conv0_bias: torch.Tensor,
    transform_0_conv1_weight: torch.Tensor,
    transform_0_conv1_bias: torch.Tensor,
    transform_0_conv2_weight: torch.Tensor,
    transform_0_conv2_bias: torch.Tensor,
    transform_1_conv0_weight: torch.Tensor,
    transform_1_conv0_bias: torch.Tensor,
    transform_1_conv1_weight: torch.Tensor,
    transform_1_conv1_bias: torch.Tensor,
    transform_1_conv2_weight: torch.Tensor,
    transform_1_conv2_bias: torch.Tensor,
    transform_2_conv0_weight: torch.Tensor,
    transform_2_conv0_bias: torch.Tensor,
    transform_2_conv1_weight: torch.Tensor,
    transform_2_conv1_bias: torch.Tensor,
    transform_2_conv2_weight: torch.Tensor,
    transform_2_conv2_bias: torch.Tensor,
    transform_3_conv0_weight: torch.Tensor,
    transform_3_conv0_bias: torch.Tensor,
    transform_3_conv1_weight: torch.Tensor,
    transform_3_conv1_bias: torch.Tensor,
    transform_3_conv2_weight: torch.Tensor,
    transform_3_conv2_bias: torch.Tensor,
):
    """
    Residual coupling flow block.
    Forward: x1 = x1 + transform(x0) for each layer
    Reverse: x1 = x1 - transform(x0) for each layer (in reverse order)
    """
    half_channels = x.shape[1] // 2

    # List of transforms (4 of them)
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

    if not reverse:
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            # Split into two halves
            x0 = x[:, :half_channels, :]
            x1 = x[:, half_channels:, :]

            # conv0: x0 -> h0 [N, 192, L_out0] where L_out0 = L - 4
            w0 = conv0_w.contiguous()
            b0 = conv0_b.contiguous()
            N, Cin0, Cout0, K = x0.shape[0], w0.shape[1], w0.shape[0], 5
            L_in0 = x0.shape[2]
            L_out0 = L_in0 - 4  # padding=0
            h0 = torch.empty((x0.shape[0], w0.shape[0], L_out0), device=x.device, dtype=torch.float32)

            BLOCK_T0 = min(128, L_out0)
            grid0 = (x0.shape[0] * w0.shape[0], triton.cdiv(L_out0, BLOCK_T0))
            conv1d_kernel[grid0](
                x0, w0, b0, h0,
                x0.shape[0], Cin0, Cout0, L_in0, L_out0,
                x0.stride(0), x0.stride(1), x0.stride(2),
                w0.stride(0), w0.stride(1), w0.stride(2),
                h0.stride(0), h0.stride(1), h0.stride(2),
                BLOCK_T=BLOCK_T0, num_warps=4
            )

            # ReLU
            h0 = torch.empty_like(h0)
            grid_relu0 = (x0.shape[0] * w0.shape[0], triton.cdiv(L_out0, 128))
            relu_kernel[grid_relu0](
                h0, h0, x0.shape[0], w0.shape[0], L_out0, h0.stride(0), h0.stride(1), h0.stride(2),
                BLOCK_T=128, num_warps=4
            )

            # conv1: h0 -> h1 [N, 192, L_out1] where L_out1 = L_out0 - 4 = L - 8
            w1 = conv1_w.contiguous()
            b1 = conv1_b.contiguous()
            N2, Cin1, Cout1, K = h0.shape[0], w1.shape[1], w1.shape[0], 5
            L_in1 = h0.shape[2]
            L_out1 = L_in1 - 4
            h1 = torch.empty((h0.shape[0], w1.shape[0], L_out1), device=x.device, dtype=torch.float32)

            BLOCK_T1 = min(128, L_out1)
            grid1 = (h0.shape[0] * w1.shape[0], triton.cdiv(L_out1, BLOCK_T1))
            conv1d_kernel[grid1](
                h0, w1, b1, h1,
                h0.shape[0], Cin1, Cout1, L_in1, L_out1,
                h0.stride(0), h0.stride(1), h0.stride(2),
                w1.stride(0), w1.stride(1), w1.stride(2),
                h1.stride(0), h1.stride(1), h1.stride(2),
                BLOCK_T=BLOCK_T1, num_warps=4
            )

            # ReLU
            h1 = torch.empty_like(h1)
            grid_relu1 = (h0.shape[0] * w1.shape[0], triton.cdiv(L_out1, 128))
            relu_kernel[grid_relu1](
                h1, h1, h0.shape[0], w1.shape[0], L_out1, h1.stride(0), h1.stride(1), h1.stride(2),
                BLOCK_T=128, num_warps=4
            )

            # conv2: h1 -> h2 [N, 96, L_out2] where L_out2 = L_out1 - 4 = L - 12
            w2 = conv2_w.contiguous()
            L_in2 = h1.shape[2]
            L_out2 = L_in2 - 4  # pass zero bias (conv2 has no bias in original)
            h2 = torch.empty((h1.shape[0], w2.shape[0], L_out2), device=x.device, dtype=torch.float32)

            BLOCK_T2 = min(128, L_out2)
            grid2 = (h1.shape[0] * w2.shape[0], triton.cdiv(L_out2, BLOCK_T2))
            conv1d_bias_stride1_kernel[grid2](
                h1, w2, torch.zeros(1, device=x.device, dtype=torch.float32), h2,
                h1.shape[0], w2.shape[1], w2.shape[0], h1.shape[2], L_out2, 5,
                h1.stride(0), h1.stride(1), h1.stride(2),
                w2.stride(0), w2.stride(1), w2.stride(2),
                h2.stride(0), h2.stride(1), h2.stride(2),
                BLOCK_T=BLOCK_T2, num_warps=4
            )

            # Multiply by mask
            h2 = torch.empty_like(h2)
            grid_mul = (x.shape[0], triton.cdiv(L_out2, 128))
            multiply_mask_kernel[grid_mul](
                h2, x_mask, h2, x.shape[0], h2.shape[1], L_out2,
                h2.stride(0), h2.stride(1), h2.stride(2),
                x_mask.stride(0), x_mask.stride(2),
                h2.stride(0), h2.stride(1), h2.stride(2),
                BLOCK_T=128, num_warps=4
            )

            # Affine coupling
            x1_new = torch.empty_like(x1)
            grid_add = (x1.shape[0] * x1.shape[1], triton.cdiv(L_out2, 128))
            add_masked_kernel[grid_add](
                x1, h2, x1_new, x1.shape[0], x1.shape[1], L_out2,
                x1.stride(0), x1.stride(1), x1.stride(2),
                h2.stride(0), h2.stride(1), h2.stride(2),
                reverse=False,  # forward pass adds
                BLOCK_T=128, num_warps=4
            )

            # Concatenate back
            y_half = torch.empty((x.shape[0], half_channels + half_channels, L_out2), device=x.device, dtype=torch.float32)
            grid_concat = (x.shape[0], triton.cdiv(L_out2, 128))
            concatenate_channels_kernel[grid_concat](
                x0, x1_new, y_half, x.shape[0], half_channels, half_channels, L_out2,
                x0.stride(0), x0.stride(1), x0.stride(2),
                x1_new.stride(0), x1_new.stride(1), x1_new.stride(2),
                y_half.stride(0), y_half.stride(1), y_half.stride(2),
                BLOCK_T=128, num_warps=4
            )

            # Multiply by mask again
            y_half = torch.empty_like(y_half)
            grid_mul2 = (x.shape[0], triton.cdiv(L_out2, 128))
            multiply_mask_kernel[grid_mul2](
                y_half, x_mask, y_half, x.shape[0], y_half.shape[1], L_out2,
                y_half.stride(0), y_half.stride(1), y_half.stride(2),
                x_mask.stride(0), x_mask.stride(2),
                y_half.stride(0), y_half.stride(1), y_half.stride(2),
                BLOCK_T=128, num_warps=4
            )

            # Update x for next iteration
            x = y_half

    else:
        # Reverse pass: apply transformations in reverse order (same structure as forward, but subtract h)
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
            x0 = x[:, :half_channels, :]
            x1 = x[:, half_channels:, :]

            # conv0: x0 -> h0 [N, 192, L_out0] where L_out0 = L - 4
            w0 = conv0_w.contiguous()
            b0 = conv0_b.contiguous()
            N, Cin0, Cout0, K = x0.shape[0], w0.shape[1], w0.shape[0], 5
            L_in0 = x0.shape[2]
            L_out0 = L_in0 - 4
            h0 = torch.empty((x0.shape[0], w0.shape[0], L_out0), device=x.device, dtype=torch.float32)

            BLOCK_T0 = min(128, L_out0)
            grid0 = (x0.shape[0] * w0.shape[0], triton.cdiv(L_out0, BLOCK_T0))
            conv1d_kernel[grid0](
                x0, w0, b0, h0,
                x0.shape[0], Cin0, Cout0, L_in0, L_out0,
                x0.stride(0), x0.stride(1), x0.stride(2),
                w0.stride(0), w0.stride(1), w0.stride(2),
                h0.stride(0), h0.stride(1), h0.stride(2),
                BLOCK_T=BLOCK_T0, num_warps=4
            )

            # ReLU
            h0 = torch.empty_like(h0)
            grid_relu0 = (x0.shape[0] * w0.shape[0], triton.cdiv(L_out0, 128))
            relu_kernel[grid_relu0](
                h0, h0, x0.shape[0], w0.shape[0], L_out0, h0.stride(0), h0.stride(1), h0.stride(2),
                BLOCK_T=128, num_warps=4
            )

            # conv1: h0 -> h1 [N, 192, L_out1] where L_out1 = L_out0 - 4 = L - 8
            w1 = conv1_w.contiguous()
            b1 = conv1_b.contiguous()
            N2, Cin1, Cout1, K = h0.shape[0], w1.shape[1], w1.shape[0], 5
            L_in1 = h0.shape[2]
            L_out1 = L_in1 - 4
            h1 = torch.empty((h0.shape[0], w1.shape[0], L_out1), device=x.device, dtype=torch.float32)

            BLOCK_T1 = min(128, L_out1)
            grid1 = (h0.shape[0] * w1.shape[0], triton.cdiv(L_out1, BLOCK_T1))
            conv1d_kernel[grid1](
                h0, w1, b1, h1,
                h0.shape[0], Cin1, Cout1, L_in1, L_out1,
                h0.stride(0), h0.stride(1), h0.stride(2),
                w1.stride(0), w1.stride(1), w1.stride(2),
                h1.stride(0), h1.stride(1), h1.stride(2),
                BLOCK_T=BLOCK_T1, num_warps=4
            )

            # ReLU
            h1 = torch.empty_like(h1)
            grid_relu1 = (h0.shape[0] * w1.shape[0], triton.cdiv(L_out1, 128))
            relu_kernel[grid_relu1](
                h1, h1, h0.shape[0], w1.shape[0], L_out1, h1.stride(0), h1.stride(1), h1.stride(2),
                BLOCK_T=128, num_warps=4
            )

            # conv2: h1 -> h2 [N, 96, L_out2] where L_out2 = L_out1 - 4 = L - 12
            w2 = conv2_w.contiguous()
            L_in2 = h1.shape[2]
            L_out2 = L_in2 - 4
            h2 = torch.empty((h1.shape[0], w2.shape[0], L_out2), device=x.device, dtype=torch.float32)

            BLOCK_T2 = min(128, L_out2)
            grid2 = (h1.shape[0] * w2.shape[0], triton.cdiv(L_out2, BLOCK_T2))
            conv1d_bias_stride1_kernel[grid2](
                h1, w2, torch.zeros(1, device=x.device, dtype=torch.float32), h2,
                h1.shape[0], w2.shape[1], w2.shape[0], h1.shape[2], L_out2, 5,
                h1.stride(0), h1.stride(1), h1.stride(2),
                w2.stride(0), w2.stride(1), w2.stride(2),
                h2.stride(0), h2.stride(1), h2.stride(2),
                BLOCK_T=BLOCK_T2, num_warps=4
            )

            # Multiply by mask
            h2 = torch.empty_like(h2)
            grid_mul = (x.shape[0], triton.cdiv(L_out2, 128))
            multiply_mask_kernel[grid_mul](
                h2, x_mask, h2, x.shape[0], h2.shape[1], L_out2,
                h2.stride(0), h2.stride(1), h2.stride(2),
                x_mask.stride(0), x_mask.stride(2),
                h2.stride(0), h2.stride(1), h2.stride(2),
                BLOCK_T=128, num_warps=4
            )

            # Affine coupling in reverse: subtract
            x1_new = torch.empty_like(x1)
            grid_add = (x1.shape[0] * x1.shape[1], triton.cdiv(L_out2, 128))
            add_masked_kernel[grid_add](
                x1, h2, x1_new, x1.shape[0], x1.shape[1], L_out2,
                x1.stride(0), x1.stride(1), x1.stride(2),
                h2.stride(0), h2.stride(1), h2.stride(2),
                reverse=True,
                BLOCK_T=128, num_warps=4
            )

            # Concatenate back
            y_half = torch.empty((x.shape[0], half_channels + half_channels, L_out2), device=x.device, dtype=torch.float32)
            grid_concat = (x.shape[0], triton.cdiv(L_out2, 128))
            concatenate_channels_kernel[grid_concat](
                x0, x1_new, y_half, x.shape[0], half_channels, half_channels, L_out2,
                x0.stride(0), x0.stride(1), x0.stride(2),
                x1_new.stride(0), x1_new.stride(1), x1_new.stride(2),
                y_half.stride(0), y_half.stride(1), y_half.stride(2),
                BLOCK_T=128, num_warps=4
            )

            # Multiply by mask again
            y_half = torch.empty_like(y_half)
            grid_mul2 = (x.shape[0], triton.cdiv(L_out2, 128))
            multiply_mask_kernel[grid_mul2](
                y_half, x_mask, y_half, x.shape[0], y_half.shape[1], L_out2,
                y_half.stride(0), y_half.stride(1), y_half.stride(2),
                x_mask.stride(0), x_mask.stride(2),
                y_half.stride(0), y_half.stride(1), y_half.stride(2),
                BLOCK_T=128, num_warps=4
            )

            # Update x for next iteration
            x = y_half

    return x


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The run function expects the same signature as the original forward.
        # It will apply Triton kernels for convs, ReLU, mask multiply, add/sub, and concatenation.
        return run(*args)


def run(*args):
    return ModelNew()(*args)
