import math
import torch
import torch.nn.functional as F

# Triton kernels
@triton.jit
def conv1d_kernel(
    x_ptr,            # *const T, input [N, C_in, L_in]
    w_ptr,            # *const T, weights [C_out, C_in, K]
    b_ptr,            # *const T, bias [C_out]
    y_ptr,            # *T, output [N, C_out, L_out]
    N, C_in, C_out, L_in, L_out, K, P,  # int32
    stride_x_n, stride_x_c, stride_x_l,  # int64 strides for x
    stride_w_co, stride_w_ci, stride_w_k,  # int64 strides for w
    stride_y_n, stride_y_c, stride_y_l,  # int64 strides for y
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    co = tl.program_id(1)
    tile = tl.program_id(2)

    l_out_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_out_offsets < L_out

    # Load bias for this output channel
    b_val = tl.load(b_ptr + co)
    acc = b_val

    # Accumulate over input channels and kernel taps
    for ci in range(0, C_in):
        for k in range(0, K):
            li = l_out_offsets + P - k  # padding P = K//2
            mask_in = (li >= 0) & (li < L_in) & mask_out

            x_ptrs = x_ptr + n * stride_x_n + ci * stride_x_c + li * stride_x_l
            w_ptrs = w_ptr + co * stride_w_co + ci * stride_w_ci + k * stride_w_k

            x_val = tl.load(x_ptrs, mask=mask_in, other=0.0)
            w_val = tl.load(w_ptrs)
            acc += x_val * w_val

    # Store result
    y_ptrs = y_ptr + n * stride_y_n + co * stride_y_c + l_out_offsets * stride_y_l
    tl.store(y_ptrs, acc, mask=mask_out)


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
    x_vals = tl.load(x_ptrs, mask=mask_out, other=0.0)
    y_vals = tl.maximum(x_vals, 0.0)
    tl.store(y_ptrs, y_vals, mask=mask_out)


@triton.jit
def concat_halves_backward(
    y2c_ptr, y0_ptr, y1_ptr,
    N, C_half, L,
    stride_y2c_n, stride_y2c_c, stride_y2c_l,
    stride_y0_n, stride_y0_c, stride_y0_l,
    stride_y1_n, stride_y1_c, stride_y1_l,
    BLOCK_L: tl.constexpr,
):
    # y2c has shape [N, 2*C_half, L]; split along channel dimension
    n = tl.program_id(0)
    ch = tl.program_id(1)  # channel index in [0, C_half)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    y0_ptrs = y2c_ptr + n * stride_y2c_n + ch * stride_y2c_c + l_offsets * stride_y2c_l
    y1_ptrs = y2c_ptr + n * stride_y2c_n + (ch + C_half) * stride_y2c_c + l_offsets * stride_y2c_l

    out0_ptrs = y0_ptr + n * stride_y0_n + ch * stride_y0_c + l_offsets * stride_y0_l
    out1_ptrs = y1_ptr + n * stride_y1_n + ch * stride_y1_c + l_offsets * stride_y1_l

    y0_vals = tl.load(y0_ptrs, mask=mask_out, other=0.0)
    y1_vals = tl.load(y1_ptrs, mask=mask_out, other=0.0)
    tl.store(out0_ptrs, y0_vals, mask=mask_out)
    tl.store(out1_ptrs, y1_vals, mask=mask_out)


@triton.jit
def concat_halves_forward(
    y0_ptr, y1_ptr, y2c_ptr,
    N, C_half, L,
    stride_y0_n, stride_y0_c, stride_y0_l,
    stride_y1_n, stride_y1_c, stride_y1_l,
    stride_y2c_n, stride_y2c_c, stride_y2c_l,
    BLOCK_L: tl.constexpr,
):
    # y0: [N, C_half, L], y1: [N, C_half, L]; write y2c: [N, 2*C_half, L]
    n = tl.program_id(0)
    ch = tl.program_id(1)  # in [0, C_half)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    out0_ptrs = y0_ptr + n * stride_y0_n + ch * stride_y0_c + l_offsets * stride_y0_l
    out1_ptrs = y1_ptr + n * stride_y1_n + ch * stride_y1_c + l_offsets * stride_y1_l
    y2c_ptrs0 = y2c_ptr + n * stride_y2c_n + ch * stride_y2c_c + l_offsets * stride_y2c_l
    y2c_ptrs1 = y2c_ptr + n * stride_y2c_n + (ch + C_half) * stride_y2c_c + l_offsets * stride_y2c_l

    y0_vals = tl.load(out0_ptrs, mask=mask_out, other=0.0)
    y1_vals = tl.load(out1_ptrs, mask=mask_out, other=0.0)
    tl.store(y2c_ptrs0, y0_vals, mask=mask_out)
    tl.store(y2c_ptrs1, y1_vals, mask=mask_out)


@triton.jit
def mul_mask_kernel(
    y_ptr, mask_ptr,
    N, C, L,
    stride_y_n, stride_y_c, stride_y_l,
    stride_m_n, stride_m_c, stride_m_l,  # mask shape [N, 1, L]
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L

    y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l
    m_ptrs = mask_ptr + n * stride_m_n + 0 * stride_m_c + l_offsets * stride_m_l

    y_vals = tl.load(y_ptrs, mask=mask_out, other=0.0)
    m_vals = tl.load(m_ptrs, mask=mask_out, other=1.0)
    y_vals = y_vals * m_vals
    tl.store(y_ptrs, y_vals, mask=mask_out)


# Forward function using Triton kernels (no torch.conv1d on host)
@torch.no_grad()
def run_triton_only(
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
    # Ensure tensors are contiguous for predictable strides
    x = x.contiguous()
    half_channels = x.shape[1] // 2

    # Collect all transforms
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

    for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
        N, Cin, L_in = x.shape  # Cin = half_channels
        # Split into halves (use Triton split kernel)
        x0 = torch.empty((N, Cin, L_in), device=x.device, dtype=x.dtype)
        x1 = torch.empty((N, Cin, L_in), device=x.device, dtype=x.dtype)

        grid_split = (N, Cin, triton.cdiv(L_in, 128))
        concat_halves_backward[grid_split](
            x, x0, x1,
            N, Cin, L_in,
            x.stride(0), x.stride(1), x.stride(2),
            x0.stride(0), x0.stride(1), x0.stride(2),
            x1.stride(0), x1.stride(1), x1.stride(2),
            BLOCK_L=128, num_warps=4
        )

        # conv0: h0 = conv1d(x0, conv0_w, conv0_b) with padding=P
        Cout0 = conv0_w.shape[0]
        K0 = conv0_w.shape[2]
        P0 = K0 // 2
        h0 = torch.empty((N, Cout0, L_in), device=x.device, dtype=x.dtype)
        grid_c0 = (N, Cout0, triton.cdiv(L_in, 128))
        conv1d_kernel[grid_c0](
            x0, conv0_w, conv0_b, h0,
            N, x0.shape[1], Cout0, x0.shape[2], L_in, K0, P0,
            x0.stride(0), x0.stride(1), x0.stride(2),
            conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
            h0.stride(0), h0.stride(1), h0.stride(2),
            BLOCK_L=128, num_warps=4
        )
        # ReLU after conv0
        relu_kernel[grid_c0](
            h0, h0,
            N, Cout0, L_in,
            h0.stride(0), h0.stride(1), h0.stride(2),
            h0.stride(0), h0.stride(1), h0.stride(2),
            BLOCK_L=128, num_warps=4
        )

        # conv1: h1 = conv1d(h0, conv1_w, conv1_b) with padding=P1
        Cout1 = conv1_w.shape[0]
        K1 = conv1_w.shape[2]
        P1 = K1 // 2
        h1 = torch.empty((N, Cout1, L_in), device=x.device, dtype=x.dtype)
        grid_c1 = (N, Cout1, triton.cdiv(L_in, 128))
        conv1d_kernel[grid_c1](
            h0, conv1_w, conv1_b, h1,
            N, h0.shape[1], Cout1, h0.shape[2], L_in, K1, P1,
            h0.stride(0), h0.stride(1), h0.stride(2),
            conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
            h1.stride(0), h1.stride(1), h1.stride(2),
            BLOCK_L=128, num_warps=4
        )
        # ReLU after conv1
        relu_kernel[grid_c1](
            h1, h1,
            N, Cout1, L_in,
            h1.stride(0), h1.stride(1), h1.stride(2),
            h1.stride(0), h1.stride(1), h1.stride(2),
            BLOCK_L=128, num_warps=4
        )

        # conv2: h2 = conv1d(h1, conv2_w, conv2_b) with padding=P2
        Cout2 = conv2_w.shape[0]
        K2 = conv2_w.shape[2]
        P2 = K2 // 2
        h2 = torch.empty((N, Cout2, L_in), device=x.device, dtype=x.dtype)
        grid_c2 = (N, Cout2, triton.cdiv(L_in, 128))
        conv1d_kernel[grid_c2](
            h1, conv2_w, conv2_b, h2,
            N, h1.shape[1], Cout2, h1.shape[2], L_in, K2, P2,
            h1.stride(0), h1.stride(1), h1.stride(2),
            conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
            h2.stride(0), h2.stride(1), h2.stride(2),
            BLOCK_L=128, num_warps=4
        )

        # Update x1: forward add, reverse subtract
        h2 = h2.contiguous()
        x1 = x1.contiguous()
        if not reverse:
            x1 = x1 + h2
        else:
            x1 = x1 - h2

        # Concatenate back along channel dimension
        y_full = torch.empty((N, 2 * Cin, L_in), device=x.device, dtype=x.dtype)
        grid_concat = (N, Cin, triton.cdiv(L_in, 128))
        concat_halves_forward[grid_concat](
            x0, x1, y_full,
            N, Cin, L_in,
            x0.stride(0), x0.stride(1), x0.stride(2),
            x1.stride(0), x1.stride(1), x1.stride(2),
            y_full.stride(0), y_full.stride(1), y_full.stride(2),
            BLOCK_L=128, num_warps=4
        )

        # Apply mask to output (elementwise multiply)
        y_full = y_full.contiguous()
        x_mask = x_mask.contiguous()
        grid_mask = (N, 2 * Cin, triton.cdiv(L_in, 128))
        mul_mask_kernel[grid_mask](
            y_full, x_mask,
            N, 2 * Cin, L_in,
            y_full.stride(0), y_full.stride(1), y_full.stride(2),
            x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
            BLOCK_L=128, num_warps=4
        )

        # Update x for next transform
        x = y_full.clone()

    return x


# Entry point required by evaluator
class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run_triton_only(*args)


def run(*args):
    return ModelNew()(*args)
