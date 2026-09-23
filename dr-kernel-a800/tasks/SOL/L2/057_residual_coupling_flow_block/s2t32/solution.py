import math
import torch
import triton
import triton.language as tl


def _ceil_div(a, b):
    return (a + b - 1) // b


# Triton Conv1d forward: y[n, co, lo] = bias[co] + sum_{ci,k} x[n, ci, li] * w[co, ci, k], with li = lo + P - k
@triton.jit
def conv1d_forward_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, C_out, L_in, L_out, K,
    stride_x_n, stride_x_c, stride_x_l,
    stride_w_co, stride_w_ci, stride_w_k,
    stride_y_n, stride_y_c, stride_y_l,
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    co = tl.program_id(1)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L_out

    # initialize accumulator with bias
    b_val = tl.load(b_ptr + co)
    acc = tl.full([BLOCK_L], b_val, tl.float32)

    P = K // 2
    # loop over input channels and kernel taps
    for ci in range(0, C_in):
        for k in range(0, K):
            li = l_offsets + P - k  # vector
            in_range = (li >= 0) & (li < L_in) & mask_out
            x_ptrs = x_ptr + n * stride_x_n + ci * stride_x_c + li * stride_x_l
            x_vals = tl.load(x_ptrs, mask=in_range, other=0.0)
            x_vals = x_vals.to(tl.float32)
            w_val = tl.load(w_ptr + co * stride_w_co + ci * stride_w_ci + k * stride_w_k)
            w_val = w_val.to(tl.float32)
            acc += x_vals * w_val

    # store result
    y_ptrs = y_ptr + n * stride_y_n + co * stride_y_c + l_offsets * stride_y_l
    tl.store(y_ptrs, acc, mask=mask_out)


# Triton ReLU: y = max(y, 0)
@triton.jit
def relu_kernel(
    x_ptr, y_ptr,
    N, C, L,
    stride_x_n, stride_x_c, stride_x_l,
    stride_y_n, stride_y_c, stride_y_l,
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    x_ptrs = x_ptr + n * stride_x_n + c * stride_x_c + l_offsets * stride_x_l
    y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l

    x_vals = tl.load(x_ptrs, mask=mask_out, other=0.0).to(tl.float32)
    y_vals = tl.maximum(x_vals, 0.0)  # ReLU
    tl.store(y_ptrs, y_vals, mask=mask_out)


# Triton split halves: read y_full[:, :C_half, :] into y0, y_full[:, C_half:, :] into y1
@triton.jit
def split_halves_forward(
    y_full_ptr, y0_ptr, y1_ptr,
    N, C_half, L,
    stride_y_full_n, stride_y_full_c, stride_y_full_l,
    stride_y0_n, stride_y0_c, stride_y0_l,
    stride_y1_n, stride_y1_c, stride_y1_l,
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    ch = tl.program_id(1)  # channel in [0, C_half)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    src0_ptrs = y_full_ptr + n * stride_y_full_n + ch * stride_y_full_c + l_offsets * stride_y_full_l
    src1_ptrs = y_full_ptr + n * stride_y_full_n + (ch + C_half) * stride_y_full_c + l_offsets * stride_y_full_l
    out0_ptrs = y0_ptr + n * stride_y0_n + ch * stride_y0_c + l_offsets * stride_y0_l
    out1_ptrs = y1_ptr + n * stride_y1_n + ch * stride_y1_c + l_offsets * stride_y1_l

    y0_vals = tl.load(src0_ptrs, mask=mask_out, other=0.0).to(tl.float32)
    y1_vals = tl.load(src1_ptrs, mask=mask_out, other=0.0).to(tl.float32)
    tl.store(out0_ptrs, y0_vals, mask=mask_out)
    tl.store(out1_ptrs, y1_vals, mask=mask_out)


# Triton concat halves: write y0 into first half and y1 into second half in y_full
@triton.jit
def concat_halves_forward(
    y0_ptr, y1_ptr, y_full_ptr,
    N, C_half, L,
    stride_y0_n, stride_y0_c, stride_y0_l,
    stride_y1_n, stride_y1_c, stride_y1_l,
    stride_y_full_n, stride_y_full_c, stride_y_full_l,
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    ch = tl.program_id(1)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    in0_ptrs = y0_ptr + n * stride_y0_n + ch * stride_y0_c + l_offsets * stride_y0_l
    in1_ptrs = y1_ptr + n * stride_y1_n + ch * stride_y1_c + l_offsets * stride_y1_l
    out0_ptrs = y_full_ptr + n * stride_y_full_n + ch * stride_y_full_c + l_offsets * stride_y_full_l
    out1_ptrs = y_full_ptr + n * stride_y_full_n + (ch + C_half) * stride_y_full_c + l_offsets * stride_y_full_l

    y0_vals = tl.load(in0_ptrs, mask=mask_out, other=0.0).to(tl.float32)
    y1_vals = tl.load(in1_ptrs, mask=mask_out, other=0.0).to(tl.float32)
    tl.store(out0_ptrs, y0_vals, mask=mask_out)
    tl.store(out1_ptrs, y1_vals, mask=mask_out)


# Triton multiply mask: y[i, j, k] *= mask[i, 0, k]
@triton.jit
def mul_mask_kernel(
    x_ptr, mask_ptr, y_ptr,
    N, C, L,
    stride_x_n, stride_x_c, stride_x_l,
    stride_mask_n, stride_mask_c, stride_mask_l,
    stride_y_n, stride_y_c, stride_y_l,
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    x_ptrs = x_ptr + n * stride_x_n + c * stride_x_c + l_offsets * stride_x_l
    y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l
    mask_ptrs = mask_ptr + n * stride_mask_n + 0 * stride_mask_c + l_offsets * stride_mask_l

    x_vals = tl.load(x_ptrs, mask=mask_out, other=0.0).to(tl.float32)
    m_vals = tl.load(mask_ptrs, mask=mask_out, other=1.0).to(tl.float32)
    y_vals = x_vals * m_vals
    tl.store(y_ptrs, y_vals, mask=mask_out)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        reverse: bool,
        # transform 0
        transform_0_conv0_weight: torch.Tensor,
        transform_0_conv0_bias: torch.Tensor,
        transform_0_conv1_weight: torch.Tensor,
        transform_0_conv1_bias: torch.Tensor,
        transform_0_conv2_weight: torch.Tensor,
        transform_0_conv2_bias: torch.Tensor,
        # transform 1
        transform_1_conv0_weight: torch.Tensor,
        transform_1_conv0_bias: torch.Tensor,
        transform_1_conv1_weight: torch.Tensor,
        transform_1_conv1_bias: torch.Tensor,
        transform_1_conv2_weight: torch.Tensor,
        transform_1_conv2_bias: torch.Tensor,
        # transform 2
        transform_2_conv0_weight: torch.Tensor,
        transform_2_conv0_bias: torch.Tensor,
        transform_2_conv1_weight: torch.Tensor,
        transform_2_conv1_bias: torch.Tensor,
        transform_2_conv2_weight: torch.Tensor,
        transform_2_conv2_bias: torch.Tensor,
        # transform 3
        transform_3_conv0_weight: torch.Tensor,
        transform_3_conv0_bias: torch.Tensor,
        transform_3_conv1_weight: torch.Tensor,
        transform_3_conv1_bias: torch.Tensor,
        transform_3_conv2_weight: torch.Tensor,
        transform_3_conv2_bias: torch.Tensor,
    ):
        # Shapes
        N, C, L = x.shape
        assert C == 192, "Input channels must be 192"
        C_half = C // 2
        K = 5
        P = K // 2

        # Prepare tensors
        x = x.contiguous()
        # x_mask is [N, 1, L], ensure contiguous
        x_mask = x_mask.contiguous()

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
            # Forward path: apply transforms sequentially
            for i, (conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b) in enumerate(transforms):
                # Ensure contiguity for weights and biases
                conv0_w = conv0_w.contiguous()
                conv0_b = conv0_b.contiguous()
                conv1_w = conv1_w.contiguous()
                conv1_b = conv1_b.contiguous()
                conv2_w = conv2_w.contiguous()
                conv2_b = conv2_b.contiguous()

                # Split x into two halves
                y0 = torch.empty((N, C_half, L), device=x.device, dtype=x.dtype)
                y1 = torch.empty((N, C_half, L), device=x.device, dtype=x.dtype)

                # Launch split
                BLOCK_L = 128  # tile along time dimension
                grid_split = (N, C_half, _ceil_div(L, BLOCK_L))
                split_halves_forward[grid_split](
                    x, y0, y1, N, C_half, L,
                    x.stride(0), x.stride(1), x.stride(2),
                    y0.stride(0), y0.stride(1), y0.stride(2),
                    y1.stride(0), y1.stride(1), y1.stride(2),
                    BLOCK_L, num_warps=4
                )

                # conv0: (C_half -> 192), padding=P=2
                y0_conv = torch.empty((N, 192, L), device=x.device, dtype=x.dtype)
                L_in0 = L
                L_out0 = L_in0 - 2 * P + (K - 1) + 1  # since P=2, K=5 => L_out=L
                grid0 = (N, 192, _ceil_div(L_out0, BLOCK_L))
                conv1d_forward_kernel[grid0](
                    y0, conv0_w, conv0_b, y0_conv,
                    N, C_half, 192, L_in0, L_out0, K,
                    y0.stride(0), y0.stride(1), y0.stride(2),
                    conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                    y0_conv.stride(0), y0_conv.stride(1), y0_conv.stride(2),
                    BLOCK_L, num_warps=4
                )

                # ReLU after conv0
                y0_conv_relu = torch.empty_like(y0_conv)
                grid_relu = (N, 192, _ceil_div(L_out0, BLOCK_L))
                relu_kernel[grid_relu](
                    y0_conv, y0_conv_relu,
                    N, 192, L_out0,
                    y0_conv.stride(0), y0_conv.stride(1), y0_conv.stride(2),
                    y0_conv_relu.stride(0), y0_conv_relu.stride(1), y0_conv_relu.stride(2),
                    BLOCK_L, num_warps=4
                )

                # conv1: (192 -> 192), padding=P=2
                y_mid = torch.empty((N, 192, L), device=x.device, dtype=x.dtype)
                L_in1 = L_out0
                L_out1 = L_in1 - 2 * P + (K - 1) + 1  # still L
                grid1 = (N, 192, _ceil_div(L_out1, BLOCK_L))
                conv1d_forward_kernel[grid1](
                    y0_conv_relu, conv1_w, conv1_b, y_mid,
                    N, 192, 192, L_in1, L_out1, K,
                    y0_conv_relu.stride(0), y0_conv_relu.stride(1), y0_conv_relu.stride(2),
                    conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                    y_mid.stride(0), y_mid.stride(1), y_mid.stride(2),
                    BLOCK_L, num_warps=4
                )

                # ReLU after conv1
                y_mid_relu = torch.empty_like(y_mid)
                grid_relu1 = (N, 192, _ceil_div(L_out1, BLOCK_L))
                relu_kernel[grid_relu1](
                    y_mid, y_mid_relu,
                    N, 192, L_out1,
                    y_mid.stride(0), y_mid.stride(1), y_mid.stride(2),
                    y_mid_relu.stride(0), y_mid_relu.stride(1), y_mid_relu.stride(2),
                    BLOCK_L, num_warps=4
                )

                # conv2: (192 -> C_half), padding=P=2
                y_update = torch.empty((N, C_half, L), device=x.device, dtype=x.dtype)
                L_in2 = L_out1
                L_out2 = L_in2 - 2 * P + (K - 1) + 1  # still L
                grid2 = (N, C_half, _ceil_div(L_out2, BLOCK_L))
                conv1d_forward_kernel[grid2](
                    y_mid_relu, conv2_w, conv2_b, y_update,
                    N, 192, C_half, L_in2, L_out2, K,
                    y_mid_relu.stride(0), y_mid_relu.stride(1), y_mid_relu.stride(2),
                    conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                    y_update.stride(0), y_update.stride(1), y_update.stride(2),
                    BLOCK_L, num_warps=4
                )

                # Update x1: x1 = x1 + y_update
                y1 = y1 + y_update

                # Concatenate [y0, y1] back into full channels
                y_full = torch.empty((N, C, L), device=x.device, dtype=x.dtype)
                grid_cat = (N, C_half, _ceil_div(L, BLOCK_L))
                concat_halves_forward[grid_cat](
                    y0, y1, y_full,
                    N, C_half, L,
                    y0.stride(0), y0.stride(1), y0.stride(2),
                    y1.stride(0), y1.stride(1), y1.stride(2),
                    y_full.stride(0), y_full.stride(1), y_full.stride(2),
                    BLOCK_L, num_warps=4
                )

                # Apply mask
                y_full_masked = torch.empty_like(y_full, device=x.device, dtype=x.dtype)
                grid_mask = (N, C, _ceil_div(L, BLOCK_L))
                mul_mask_kernel[grid_mask](
                    y_full, x_mask, y_full_masked,
                    N, C, L,
                    y_full.stride(0), y_full.stride(1), y_full.stride(2),
                    x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                    y_full_masked.stride(0), y_full_masked.stride(1), y_full_masked.stride(2),
                    BLOCK_L, num_warps=4
                )

                # Update x
                x = y_full_masked

        else:
            # Reverse path: apply transforms in reverse order
            for i, (conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b) in enumerate(reversed(transforms)):
                conv0_w = conv0_w.contiguous()
                conv0_b = conv0_b.contiguous()
                conv1_w = conv1_w.contiguous()
                conv1_b = conv1_b.contiguous()
                conv2_w = conv2_w.contiguous()
                conv2_b = conv2_b.contiguous()

                # Split x into two halves
                y0 = torch.empty((N, C_half, L), device=x.device, dtype=x.dtype)
                y1 = torch.empty((N, C_half, L), device=x.device, dtype=x.dtype)

                grid_split = (N, C_half, _ceil_div(L, BLOCK_L))
                split_halves_forward[grid_split](
                    x, y0, y1, N, C_half, L,
                    x.stride(0), x.stride(1), x.stride(2),
                    y0.stride(0), y0.stride(1), y0.stride(2),
                    y1.stride(0), y1.stride(1), y1.stride(2),
                    BLOCK_L, num_warps=4
                )

                # conv0 backward in forward: h = y0_conv; need to compute h = conv1d(y0, conv0_w)
                y0_conv = torch.empty((N, 192, L), device=x.device, dtype=x.dtype)
                L_in0 = L
                L_out0 = L_in0 - 2 * P + (K - 1) + 1  # L
                grid0 = (N, 192, _ceil_div(L_out0, BLOCK_L))
                conv1d_forward_kernel[grid0](
                    y0, conv0_w, conv0_b, y0_conv,
                    N, C_half, 192, L_in0, L_out0, K,
                    y0.stride(0), y0.stride(1), y0.stride(2),
                    conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                    y0_conv.stride(0), y0_conv.stride(1), y0_conv.stride(2),
                    BLOCK_L, num_warps=4
                )

                # ReLU after conv0 (needed for correctness)
                y0_conv_relu = torch.empty_like(y0_conv)
                grid_relu = (N, 192, _ceil_div(L_out0, BLOCK_L))
                relu_kernel[grid_relu](
                    y0_conv, y0_conv_relu,
                    N, 192, L_out0,
                    y0_conv.stride(0), y0_conv.stride(1), y0_conv.stride(2),
                    y0_conv_relu.stride(0), y0_conv_relu.stride(1), y0_conv_relu.stride(2),
                    BLOCK_L, num_warps=4
                )

                # conv1 backward in forward: h = y_mid; compute y_mid = conv1d(y0_conv_relu, conv1_w)
                y_mid = torch.empty((N, 192, L), device=x.device, dtype=x.dtype)
                L_in1 = L_out0
                L_out1 = L_in1 - 2 * P + (K - 1) + 1  # L
                grid1 = (N, 192, _ceil_div(L_out1, BLOCK_L))
                conv1d_forward_kernel[grid1](
                    y0_conv_relu, conv1_w, conv1_b, y_mid,
                    N, 192, 192, L_in1, L_out1, K,
                    y0_conv_relu.stride(0), y0_conv_relu.stride(1), y0_conv_relu.stride(2),
                    conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                    y_mid.stride(0), y_mid.stride(1), y_mid.stride(2),
                    BLOCK_L, num_warps=4
                )

                # ReLU after conv1
                y_mid_relu = torch.empty_like(y_mid)
                grid_relu1 = (N, 192, _ceil_div(L_out1, BLOCK_L))
                relu_kernel[grid_relu1](
                    y_mid, y_mid_relu,
                    N, 192, L_out1,
                    y_mid.stride(0), y_mid.stride(1), y_mid.stride(2),
                    y_mid_relu.stride(0), y_mid_relu.stride(1), y_mid_relu.stride(2),
                    BLOCK_L, num_warps=4
                )

                # conv2 backward in forward: h = y_update; compute y_update = conv1d(y_mid_relu, conv2_w)
                y_update = torch.empty((N, C_half, L), device=x.device, dtype=x.dtype)
                L_in2 = L_out1
                L_out2 = L_in2 - 2 * P + (K - 1) + 1  # L
                grid2 = (N, C_half, _ceil_div(L_out2, BLOCK_L))
                conv1d_forward_kernel[grid2](
                    y_mid_relu, conv2_w, conv2_b, y_update,
                    N, 192, C_half, L_in2, L_out2, K,
                    y_mid_relu.stride(0), y_mid_relu.stride(1), y_mid_relu.stride(2),
                    conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                    y_update.stride(0), y_update.stride(1), y_update.stride(2),
                    BLOCK_L, num_warps=4
                )

                # Update x1: x1 = x1 - y_update
                y1 = y1 - y_update

                # Concatenate [y0, y1] back into full channels
                y_full = torch.empty((N, C, L), device=x.device, dtype=x.dtype)
                grid_cat = (N, C_half, _ceil_div(L, BLOCK_L))
                concat_halves_forward[grid_cat](
                    y0, y1, y_full,
                    N, C_half, L,
                    y0.stride(0), y0.stride(1), y0.stride(2),
                    y1.stride(0), y1.stride(1), y1.stride(2),
                    y_full.stride(0), y_full.stride(1), y_full.stride(2),
                    BLOCK_L, num_warps=4
                )

                # Apply mask
                y_full_masked = torch.empty_like(y_full, device=x.device, dtype=x.dtype)
                grid_mask = (N, C, _ceil_div(L, BLOCK_L))
                mul_mask_kernel[grid_mask](
                    y_full, x_mask, y_full_masked,
                    N, C, L,
                    y_full.stride(0), y_full.stride(1), y_full.stride(2),
                    x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                    y_full_masked.stride(0), y_full_masked.stride(1), y_full_masked.stride(2),
                    BLOCK_L, num_warps=4
                )

                # Update x
                x = y_full_masked

        return x


def run(*args):
    return ModelNew()(*args)
