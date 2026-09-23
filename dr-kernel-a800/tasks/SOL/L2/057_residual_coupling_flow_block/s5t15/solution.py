import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: Conv1d with padding, stride=1, groups=1, dilation=1, ReLU
# Computes y[b, co, t] = relu( sum_{ci,k} x[b, ci, t - k + pad] * w[co, ci, k] + bias[co] )
@triton.jit
def conv1d_relu_triton(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    B, Cin, Cout, L_in, L_out, K, pad,
    x_stride_b, x_stride_c, x_stride_l,
    w_stride_co, w_stride_ci, w_stride_k,
    y_stride_b, y_stride_co, y_stride_l,
    BLOCK_CO: tl.constexpr, BLOCK_POS: tl.constexpr,
):
    b = tl.program_id(0)
    co_block = tl.program_id(1)
    pos_block = tl.program_id(2)

    co_offsets = co_block * BLOCK_CO + tl.arange(0, BLOCK_CO)
    pos_offsets = pos_block * BLOCK_POS + tl.arange(0, BLOCK_POS)

    co_mask = co_offsets < Cout
    pos_mask = pos_offsets < L_out

    # accumulator for [BLOCK_CO, BLOCK_POS]
    acc = tl.zeros((BLOCK_CO, BLOCK_POS), dtype=tl.float32)

    # loop over input channels and kernel elements
    for ci in range(0, Cin):
        for kk in range(0, K):
            # positions we need: pos_offsets - pad - kk
            in_pos = pos_offsets - pad - kk
            valid_in = pos_mask & (in_pos >= 0) & (in_pos < L_in)

            # load x for this (b, ci, in_pos)
            x_idx = b * x_stride_b + ci * x_stride_c + in_pos * x_stride_l
            # masked load: for invalid positions, treat as 0
            x_vals = tl.load(x_ptr + x_idx, mask=valid_in, other=0.0)  # [BLOCK_POS]

            # load weights for this (co, ci, kk)
            w_idx = co_offsets * w_stride_co + ci * w_stride_ci + kk * w_stride_k
            w_vals = tl.load(w_ptr + w_idx, mask=co_mask, other=0.0)  # [BLOCK_CO]

            # outer product accumulate
            # acc[co, pos] += w_vals[co] * x_vals[pos]
            acc += w_vals[:, None] * x_vals[None, :]

    # add bias
    bias_vals = tl.load(bias_ptr + co_offsets, mask=co_mask, other=0.0)  # [BLOCK_CO]
    acc += bias_vals[:, None]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # store
    y_idx = b * y_stride_b + co_offsets[:, None] * y_stride_co + pos_offsets[None, :] * y_stride_l
    store_mask = co_mask[:, None] & pos_mask[None, :]
    tl.store(y_ptr + y_idx, acc, mask=store_mask)


# Triton kernel: apply mask h[b, co, t] *= x_mask[b, 1, t], write masked_h
@triton.jit
def apply_mask_to_h_triton(
    h_ptr, mask_ptr, masked_ptr,
    B, Cout, L,
    h_stride_b, h_stride_c, h_stride_l,
    mask_stride_b, mask_stride_c, mask_stride_l,
    masked_stride_b, masked_stride_c, masked_stride_l,
    BLOCK_CO: tl.constexpr, BLOCK_POS: tl.constexpr,
):
    b = tl.program_id(0)
    co_block = tl.program_id(1)
    pos_block = tl.program_id(2)

    co_offsets = co_block * BLOCK_CO + tl.arange(0, BLOCK_CO)
    pos_offsets = pos_block * BLOCK_POS + tl.arange(0, BLOCK_POS)

    co_mask = co_offsets < Cout
    pos_mask = pos_offsets < L

    h_idx = b * h_stride_b + co_offsets[:, None] * h_stride_c + pos_offsets[None, :] * h_stride_l
    h_vals = tl.load(h_ptr + h_idx, mask=(co_mask[:, None] & pos_mask[None, :]), other=0.0)

    # load mask: mask has shape [B, 1, L], we index c=0 (since it's size-1 along channel)
    mask_idx = b * mask_stride_b + 0 * mask_stride_c + pos_offsets[None, :] * mask_stride_l
    mask_vals = tl.load(mask_ptr + mask_idx, mask=pos_mask[None, :], other=1.0)  # [1, BLOCK_POS]

    masked_vals = h_vals * mask_vals

    masked_idx = b * masked_stride_b + co_offsets[:, None] * masked_stride_c + pos_offsets[None, :] * masked_stride_l
    tl.store(masked_ptr + masked_idx, masked_vals, mask=(co_mask[:, None] & pos_mask[None, :]))


# Triton kernel: update x1 = x1 + masked_h (or x1 - masked_h for reverse)
@triton.jit
def add_h_to_x1_triton(
    x1_ptr, h_masked_ptr, out_ptr,
    B, Cout, L,
    x1_stride_b, x1_stride_c, x1_stride_l,
    h_stride_b, h_stride_c, h_stride_l,
    out_stride_b, out_stride_c, out_stride_l,
    ADD: tl.constexpr,
    BLOCK_CO: tl.constexpr, BLOCK_POS: tl.constexpr,
):
    b = tl.program_id(0)
    co_block = tl.program_id(1)
    pos_block = tl.program_id(2)

    co_offsets = co_block * BLOCK_CO + tl.arange(0, BLOCK_CO)
    pos_offsets = pos_block * BLOCK_POS + tl.arange(0, BLOCK_POS)

    co_mask = co_offsets < Cout
    pos_mask = pos_offsets < L

    x1_idx = b * x1_stride_b + co_offsets[:, None] * x1_stride_c + pos_offsets[None, :] * x1_stride_l
    x1_vals = tl.load(x1_ptr + x1_idx, mask=(co_mask[:, None] & pos_mask[None, :]), other=0.0)

    h_idx = b * h_stride_b + co_offsets[:, None] * h_stride_c + pos_offsets[None, :] * h_stride_l
    h_vals = tl.load(h_ptr + h_idx, mask=(co_mask[:, None] & pos_mask[None, :]), other=0.0)

    if ADD:
        out_vals = x1_vals + h_vals
    else:
        out_vals = x1_vals - h_vals

    out_idx = b * out_stride_b + co_offsets[:, None] * out_stride_c + pos_offsets[None, :] * out_stride_l
    tl.store(out_ptr + out_idx, out_vals, mask=(co_mask[:, None] & pos_mask[None, :]))


# Triton kernel: write final output [B, C0, L] and [B, C1, L] into [B, C0+C1, L]
@triton.jit
def concat_and_add_v1(
    x0_ptr, x1_ptr, h_ptr, out_ptr,
    B, C0, C1, L,
    x0_stride_b, x0_stride_c, x0_stride_l,
    x1_stride_b, x1_stride_c, x1_stride_l,
    h_stride_b, h_stride_c, h_stride_l,
    out_stride_b, out_stride_c, out_stride_l,
    BLOCK_C0: tl.constexpr, BLOCK_POS: tl.constexpr,
):
    # Writes out[b, c, l] where c in [0..C0-1] comes from x0, c in [C0..C0+C1-1] comes from x1 + h
    b = tl.program_id(0)
    c0_block = tl.program_id(1)
    pos_block = tl.program_id(2)

    c0_offsets = c0_block * BLOCK_C0 + tl.arange(0, BLOCK_C0)
    pos_offsets = pos_block * BLOCK_POS + tl.arange(0, BLOCK_POS)

    c0_mask = c0_offsets < C0
    pos_mask = pos_offsets < L

    # First half: c0
    x0_idx = b * x0_stride_b + c0_offsets[:, None] * x0_stride_c + pos_offsets[None, :] * x0_stride_l
    x0_vals = tl.load(x0_ptr + x0_idx, mask=(c0_mask[:, None] & pos_mask[None, :]), other=0.0)
    out_idx0 = b * out_stride_b + c0_offsets[:, None] * out_stride_c + pos_offsets[None, :] * out_stride_l
    tl.store(out_ptr + out_idx0, x0_vals, mask=(c0_mask[:, None] & pos_mask[None, :]))

    # Second half: c in [C0, C0+C1-1]
    c1_offsets = c0_offsets  # we’ll compute indices relative to C0
    # For x1: c1_offsets corresponds to out_c = c0_offsets + C0
    x1_idx = b * x1_stride_b + (c1_offsets[:, None] + C0) * x1_stride_c + pos_offsets[None, :] * x1_stride_l
    h_idx = b * h_stride_b + (c1_offsets[:, None] + C0) * h_stride_c + pos_offsets[None, :] * h_stride_l
    x1_vals = tl.load(x1_ptr + x1_idx, mask=(c0_mask[:, None] & pos_mask[None, :]), other=0.0)
    h_vals = tl.load(h_ptr + h_idx, mask=(c0_mask[:, None] & pos_mask[None, :]), other=0.0)
    sum_vals = x1_vals + h_vals
    out_idx1 = b * out_stride_b + (c1_offsets[:, None] + C0) * out_stride_c + pos_offsets[None, :] * out_stride_l
    tl.store(out_ptr + out_idx1, sum_vals, mask=(c0_mask[:, None] & pos_mask[None, :]))


@triton.jit
def concat_and_add_v2(
    x0_ptr, x1_ptr, h_ptr, out_ptr,
    B, C0, C1, L,
    x0_stride_b, x0_stride_c, x0_stride_l,
    x1_stride_b, x1_stride_c, x1_stride_l,
    h_stride_b, h_stride_c, h_stride_l,
    out_stride_b, out_stride_c, out_stride_l,
    BLOCK_C0: tl.constexpr, BLOCK_POS: tl.constexpr,
):
    # Writes out[b, c, l] where c in [0..C0-1] comes from x0, c in [C0..C0+C1-1] comes from x1 - h
    b = tl.program_id(0)
    c0_block = tl.program_id(1)
    pos_block = tl.program_id(2)

    c0_offsets = c0_block * BLOCK_C0 + tl.arange(0, BLOCK_C0)
    pos_offsets = pos_block * BLOCK_POS + tl.arange(0, BLOCK_POS)

    c0_mask = c0_offsets < C0
    pos_mask = pos_offsets < L

    x0_idx = b * x0_stride_b + c0_offsets[:, None] * x0_stride_c + pos_offsets[None, :] * x0_stride_l
    x0_vals = tl.load(x0_ptr + x0_idx, mask=(c0_mask[:, None] & pos_mask[None, :]), other=0.0)
    out_idx0 = b * out_stride_b + c0_offsets[:, None] * out_stride_c + pos_offsets[None, :] * out_stride_l
    tl.store(out_ptr + out_idx0, x0_vals, mask=(c0_mask[:, None] & pos_mask[None, :]))

    # Second half: c in [C0, C0+C1-1]
    c1_offsets = c0_offsets  # corresponds to out_c = c0_offsets + C0
    x1_idx = b * x1_stride_b + (c1_offsets[:, None] + C0) * x1_stride_c + pos_offsets[None, :] * x1_stride_l
    h_idx = b * h_stride_b + (c1_offsets[:, None] + C0) * h_stride_c + pos_offsets[None, :] * h_stride_l
    x1_vals = tl.load(x1_ptr + x1_idx, mask=(c0_mask[:, None] & pos_mask[None, :]), other=0.0)
    h_vals = tl.load(h_ptr + h_idx, mask=(c0_mask[:, None] & pos_mask[None, :]), other=0.0)
    diff_vals = x1_vals - h_vals
    out_idx1 = b * out_stride_b + (c1_offsets[:, None] + C0) * out_stride_c + pos_offsets[None, :] * out_stride_l
    tl.store(out_ptr + out_idx1, diff_vals, mask=(c0_mask[:, None] & pos_mask[None, :]))


def _launch_grid(B, size, BLOCK):
    # returns grid dim for given size using BLOCK
    return (B, triton.cdiv(size, BLOCK))


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
    Triton-optimized residual coupling flow:
    - Forward: for each transform, compute h = conv0(x0) -> ReLU -> conv1 -> ReLU -> conv2
      apply mask h *= x_mask, then update x1 = x1 + h; concatenate [x0, x1] and apply x_mask along channels.
    - Reverse: apply transforms in reverse order and subtract h.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    device = x.device
    B, C, T = x.shape
    half_channels = C // 2

    # Ensure tensors are on CUDA and contiguous
    x = x.contiguous()
    x_mask = x_mask.contiguous()

    # We will keep all computations in Triton, no torch conv or torch.cat.
    # Prepare transforms as tuples (w0, b0, w1, b1, w2, b2)

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

    # Initial x split
    x0 = x[:, :half_channels, :].contiguous()
    x1 = x[:, half_channels:, :].contiguous()

    if not reverse:
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            # Conv0: [B, half_channels, T] -> [B, hidden_channels, T]
            L_in = T
            L_out = T
            Cout0 = conv0_w.shape[0]
            Cin0 = conv0_w.shape[1]
            K0 = conv0_w.shape[2]
            pad0 = (K0 - 1) // 2

            y0 = torch.empty((B, Cout0, L_out), dtype=torch.float32, device=device)
            conv1d_relu_triton[_launch_grid(B, Cout0, 64), _launch_grid(L_out, 128, 128)](
                x0, conv0_w, conv0_b, y0,
                B, Cin0, Cout0, L_in, L_out, K0, pad0,
                x0.stride(0), x0.stride(1), x0.stride(2),
                conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                y0.stride(0), y0.stride(1), y0.stride(2),
                BLOCK_CO=64, BLOCK_POS=128,
            )

            # ReLU applied inside kernel; here nothing needed

            # Conv1: y0 -> [B, hidden_channels, T]
            Cout1 = conv1_w.shape[0]
            Cin1 = conv1_w.shape[1]
            K1 = conv1_w.shape[2]
            pad1 = (K1 - 1) // 2
            y1 = torch.empty((B, Cout1, L_out), dtype=torch.float32, device=device)
            conv1d_relu_triton[_launch_grid(B, Cout1, 64), _launch_grid(L_out, 128, 128)](
                y0, conv1_w, conv1_b, y1,
                B, Cin1, Cout1, L_in, L_out, K1, pad1,
                y0.stride(0), y0.stride(1), y0.stride(2),
                conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                y1.stride(0), y1.stride(1), y1.stride(2),
                BLOCK_CO=64, BLOCK_POS=128,
            )

            # ReLU applied inside kernel

            # Conv2: y1 -> [B, half_channels, T]
            Cout2 = conv2_w.shape[0]
            Cin2 = conv2_w.shape[1]
            K2 = conv2_w.shape[2]
            pad2 = (K2 - 1) // 2
            h = torch.empty((B, Cout2, L_out), dtype=torch.float32, device=device)
            conv1d_relu_triton[_launch_grid(B, Cout2, 64), _launch_grid(L_out, 128, 128)](
                y1, conv2_w, conv2_b, h,
                B, Cin2, Cout2, L_in, L_out, K2, pad2,
                y1.stride(0), y1.stride(1), y1.stride(2),
                conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                h.stride(0), h.stride(1), h.stride(2),
                BLOCK_CO=64, BLOCK_POS=128,
            )

            # Apply mask: h *= x_mask (broadcast along channel)
            masked_h = torch.empty_like(h)
            apply_mask_to_h_triton[_launch_grid(B, Cout2, 64), _launch_grid(L_out, 128, 128)](
                h, x_mask, masked_h,
                B, Cout2, L_out,
                h.stride(0), h.stride(1), h.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                masked_h.stride(0), masked_h.stride(1), masked_h.stride(2),
                BLOCK_CO=64, BLOCK_POS=128,
            )

            # Update x1: x1 = x1 + masked_h (since forward adds)
            x1_out = torch.empty_like(x1)
            add_h_to_x1_triton[_launch_grid(B, half_channels, 64), _launch_grid(L_out, 128, 128)](
                x1, masked_h, x1_out,
                B, half_channels, L_out,
                x1.stride(0), x1.stride(1), x1.stride(2),
                masked_h.stride(0), masked_h.stride(1), masked_h.stride(2),
                x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                ADD=True, BLOCK_CO=64, BLOCK_POS=128,
            )

            # Concatenate [x0, x1_out] into out [B, C, T]
            out = torch.empty((B, C, T), dtype=torch.float32, device=device)
            concat_and_add_v1[_launch_grid(B, half_channels, 64), _launch_grid(T, 128, 128)](
                x0, x1_out, masked_h, out,
                B, half_channels, half_channels, T,
                x0.stride(0), x0.stride(1), x0.stride(2),
                x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                masked_h.stride(0), masked_h.stride(1), masked_h.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_C0=64, BLOCK_POS=128,
            )

            # Apply x_mask along channels: out *= x_mask (broadcast over channels)
            out_masked = torch.empty_like(out)
            # out_masked = out * x_mask (broadcast along channels)
            # Implement elementwise mul in Triton
            @triton.jit
            def mul_broadcast_mask_triton(
                out_ptr, mask_ptr, res_ptr,
                B, C_out, L,
                out_stride_b, out_stride_c, out_stride_l,
                mask_stride_b, mask_stride_c, mask_stride_l,
                res_stride_b, res_stride_c, res_stride_l,
                BLOCK_C: tl.constexpr, BLOCK_POS: tl.constexpr,
            ):
                b = tl.program_id(0)
                c_block = tl.program_id(1)
                pos_block = tl.program_id(2)

                c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
                pos_offsets = pos_block * BLOCK_POS + tl.arange(0, BLOCK_POS)

                c_mask = c_offsets < C_out
                pos_mask = pos_offsets < L

                out_idx = b * out_stride_b + c_offsets[:, None] * out_stride_c + pos_offsets[None, :] * out_stride_l
                out_vals = tl.load(out_ptr + out_idx, mask=(c_mask[:, None] & pos_mask[None, :]), other=0.0)

                mask_idx = b * mask_stride_b + 0 * mask_stride_c + pos_offsets[None, :] * mask_stride_l
                mask_vals = tl.load(mask_ptr + mask_idx, mask=pos_mask[None, :], other=1.0)  # [1, BLOCK_POS]

                res_vals = out_vals * mask_vals

                res_idx = b * res_stride_b + c_offsets[:, None] * res_stride_c + pos_offsets[None, :] * res_stride_l
                tl.store(res_ptr + res_idx, res_vals, mask=(c_mask[:, None] & pos_mask[None, :]))

            mul_broadcast_mask_triton[_launch_grid(B, C, 64), _launch_grid(T, 128, 128)](
                out, x_mask, out_masked,
                B, C, T,
                out.stride(0), out.stride(1), out.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                out_masked.stride(0), out_masked.stride(1), out_masked.stride(2),
                BLOCK_C=64, BLOCK_POS=128,
            )

            # Update x for next layer
            x = out_masked

    else:
        # Reverse pass: subtract h in reverse order of transforms
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
            # Conv0: [B, half_channels, T] -> [B, hidden_channels, T]
            L_in = T
            L_out = T
            Cout0 = conv0_w.shape[0]
            Cin0 = conv0_w.shape[1]
            K0 = conv0_w.shape[2]
            pad0 = (K0 - 1) // 2

            y0 = torch.empty((B, Cout0, L_out), dtype=torch.float32, device=device)
            conv1d_relu_triton[_launch_grid(B, Cout0, 64), _launch_grid(L_out, 128, 128)](
                x0, conv0_w, conv0_b, y0,
                B, Cin0, Cout0, L_in, L_out, K0, pad0,
                x0.stride(0), x0.stride(1), x0.stride(2),
                conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                y0.stride(0), y0.stride(1), y0.stride(2),
                BLOCK_CO=64, BLOCK_POS=128,
            )

            # Conv1: y0 -> [B, hidden_channels, T]
            Cout1 = conv1_w.shape[0]
            Cin1 = conv1_w.shape[1]
            K1 = conv1_w.shape[2]
            pad1 = (K1 - 1) // 2
            y1 = torch.empty((B, Cout1, L_out), dtype=torch.float32, device=device)
            conv1d_relu_triton[_launch_grid(B, Cout1, 64), _launch_grid(L_out, 128, 128)](
                y0, conv1_w, conv1_b, y1,
                B, Cin1, Cout1, L_in, L_out, K1, pad1,
                y0.stride(0), y0.stride(1), y0.stride(2),
                conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                y1.stride(0), y1.stride(1), y1.stride(2),
                BLOCK_CO=64, BLOCK_POS=128,
            )

            # Conv2: y1 -> [B, half_channels, T]
            Cout2 = conv2_w.shape[0]
            Cin2 = conv2_w.shape[1]
            K2 = conv2_w.shape[2]
            pad2 = (K2 - 1) // 2
            h = torch.empty((B, Cout2, L_out), dtype=torch.float32, device=device)
            conv1d_relu_triton[_launch_grid(B, Cout2, 64), _launch_grid(L_out, 128, 128)](
                y1, conv2_w, conv2_b, h,
                B, Cin2, Cout2, L_in, L_out, K2, pad2,
                y1.stride(0), y1.stride(1), y1.stride(2),
                conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                h.stride(0), h.stride(1), h.stride(2),
                BLOCK_CO=64, BLOCK_POS=128,
            )

            # Apply mask: h *= x_mask (broadcast along channel)
            masked_h = torch.empty_like(h)
            apply_mask_to_h_triton[_launch_grid(B, Cout2, 64), _launch_grid(L_out, 128, 128)](
                h, x_mask, masked_h,
                B, Cout2, L_out,
                h.stride(0), h.stride(1), h.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                masked_h.stride(0), masked_h.stride(1), masked_h.stride(2),
                BLOCK_CO=64, BLOCK_POS=128,
            )

            # Update x1: x1 = x1 - masked_h (reverse subtract)
            x1_out = torch.empty_like(x1)
            add_h_to_x1_triton[_launch_grid(B, half_channels, 64), _launch_grid(L_out, 128, 128)](
                x1, masked_h, x1_out,
                B, half_channels, L_out,
                x1.stride(0), x1.stride(1), x1.stride(2),
                masked_h.stride(0), masked_h.stride(1), masked_h.stride(2),
                x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                ADD=False, BLOCK_CO=64, BLOCK_POS=128,
            )

            # Concatenate [x0, x1_out] into out [B, C, T]
            out = torch.empty((B, C, T), dtype=torch.float32, device=device)
            concat_and_add_v2[_launch_grid(B, half_channels, 64), _launch_grid(T, 128, 128)](
                x0, x1_out, masked_h, out,
                B, half_channels, half_channels, T,
                x0.stride(0), x0.stride(1), x0.stride(2),
                x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                masked_h.stride(0), masked_h.stride(1), masked_h.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_C0=64, BLOCK_POS=128,
            )

            # Apply x_mask along channels: out *= x_mask (broadcast over channels)
            out_masked = torch.empty_like(out)
            mul_broadcast_mask_triton[_launch_grid(B, C, 64), _launch_grid(T, 128, 128)](
                out, x_mask, out_masked,
                B, C, T,
                out.stride(0), out.stride(1), out.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                out_masked.stride(0), out_masked.stride(1), out_masked.stride(2),
                BLOCK_C=64, BLOCK_POS=128,
            )

            # Update x for next layer
            x = out_masked

    return x


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args order: x, x_mask, reverse, then weights for 4 transforms
        x = args[0]
        x_mask = args[1]
        reverse = args[2] if len(args) > 2 else False
        return run(*args)


def run(*args):
    return ModelNew()(*args)
