import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv1d_forward_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, C_out, L_in, L_out, K,
    stride_x_n, stride_x_c, stride_x_l,
    stride_w_co, stride_w_ci, stride_w_k,
    stride_y_n, stride_y_c, stride_y_l,
    BLOCK_L: tl.constexpr,
):
    # Grid: (N, C_out, tiles along L_out)
    n = tl.program_id(0)
    co = tl.program_id(1)
    tile = tl.program_id(2)
    # Output positions this program handles
    l_out_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_out_offsets < L_out

    # Accumulator in float32
    acc = tl.zeros([BLOCK_L], dtype=tl.float32)
    # Add bias
    b_val = tl.load(b_ptr + co).to(tl.float32)
    acc += b_val

    # Padding P
    P = K // 2

    # Loop over input channels and kernel taps
    for ci in range(0, C_in):
        for k in range(0, K):
            li = l_out_offsets + P - k  # vector of length BLOCK_L
            mask_in = (li >= 0) & (li < L_in) & mask_out
            x_ptrs = x_ptr + n * stride_x_n + ci * stride_x_c + li * stride_x_l
            x_vals = tl.load(x_ptrs, mask=mask_in, other=0.0).to(tl.float32)
            w_ptrs = w_ptr + co * stride_w_co + ci * stride_w_ci + k * stride_w_k
            w_vals = tl.load(w_ptrs).to(tl.float32)  # scalar
            acc += x_vals * w_vals

    # Store results to y[n, co, l_out_offsets]
    y_ptrs = y_ptr + n * stride_y_n + co * stride_y_c + l_out_offsets * stride_y_l
    # acc is float32; cast to original dtype of y if needed. Here we assume y is float32.
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
    # Loop over time positions in tiles
    for lo_base in range(0, L, BLOCK_L):
        lo_offsets = lo_base + tl.arange(0, BLOCK_L)
        mask = lo_offsets < L
        x_ptrs = x_ptr + n * stride_x_n + c * stride_x_c + lo_offsets * stride_x_l
        y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + lo_offsets * stride_y_l
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
        y_vals = tl.maximum(x_vals, 0.0)
        tl.store(y_ptrs, y_vals, mask=mask)


@triton.jit
def split_halves_forward(
    y_full_ptr, y0_ptr, y1_ptr,
    N, C_half, L,
    stride_full_n, stride_full_c, stride_full_l,
    stride_y0_n, stride_y0_c, stride_y0_l,
    stride_y1_n, stride_y1_c, stride_y1_l,
    BLOCK_L: tl.constexpr,
):
    # y_full: [N, 2*C_half, L]; write y0 = y_full[:, :C_half, :], y1 = y_full[:, C_half:, :]
    n = tl.program_id(0)
    ch = tl.program_id(1)
    for lo_base in range(0, L, BLOCK_L):
        lo_offsets = lo_base + tl.arange(0, BLOCK_L)
        mask = lo_offsets < L
        full0_ptrs = y_full_ptr + n * stride_full_n + ch * stride_full_c + lo_offsets * stride_full_l
        full1_ptrs = y_full_ptr + n * stride_full_n + (ch + C_half) * stride_full_c + lo_offsets * stride_full_l
        out0_ptrs = y0_ptr + n * stride_y0_n + ch * stride_y0_c + lo_offsets * stride_y0_l
        out1_ptrs = y1_ptr + n * stride_y1_n + ch * stride_y1_c + lo_offsets * stride_y1_l
        v0 = tl.load(full0_ptrs, mask=mask, other=0.0)
        v1 = tl.load(full1_ptrs, mask=mask, other=0.0)
        tl.store(out0_ptrs, v0, mask=mask)
        tl.store(out1_ptrs, v1, mask=mask)


@triton.jit
def concat_halves_forward(
    y0_ptr, y1_ptr, y_full_ptr,
    N, C_half, L,
    stride_y0_n, stride_y0_c, stride_y0_l,
    stride_y1_n, stride_y1_c, stride_y1_l,
    stride_full_n, stride_full_c, stride_full_l,
    BLOCK_L: tl.constexpr,
):
    # y0: [N, C_half, L], y1: [N, C_half, L], y_full: [N, 2*C_half, L]
    n = tl.program_id(0)
    ch = tl.program_id(1)
    for lo_base in range(0, L, BLOCK_L):
        lo_offsets = lo_base + tl.arange(0, BLOCK_L)
        mask = lo_offsets < L
        in0_ptrs = y0_ptr + n * stride_y0_n + ch * stride_y0_c + lo_offsets * stride_y0_l
        in1_ptrs = y1_ptr + n * stride_y1_n + ch * stride_y1_c + lo_offsets * stride_y1_l
        out0_ptrs = y_full_ptr + n * stride_full_n + ch * stride_full_c + lo_offsets * stride_full_l
        out1_ptrs = y_full_ptr + n * stride_full_n + (ch + C_half) * stride_full_c + lo_offsets * stride_full_l
        v0 = tl.load(in0_ptrs, mask=mask, other=0.0)
        v1 = tl.load(in1_ptrs, mask=mask, other=0.0)
        tl.store(out0_ptrs, v0, mask=mask)
        tl.store(out1_ptrs, v1, mask=mask)


@triton.jit
def mul_mask_kernel(
    y_ptr, mask_ptr, out_ptr,
    N, C, L,
    stride_y_n, stride_y_c, stride_y_l,
    stride_mask_n, stride_mask_c, stride_mask_l,
    BLOCK_L: tl.constexpr,
):
    # y has shape [N, C, L], mask has shape [N, 1, L]; out = y * mask
    n = tl.program_id(0)
    c = tl.program_id(1)
    for lo_base in range(0, L, BLOCK_L):
        lo_offsets = lo_base + tl.arange(0, BLOCK_L)
        mask_l = lo_offsets < L
        y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + lo_offsets * stride_y_l
        mask_ptrs = mask_ptr + n * stride_mask_n + 0 * stride_mask_c + lo_offsets * stride_mask_l
        y_vals = tl.load(y_ptrs, mask=mask_l, other=0.0)
        m_vals = tl.load(mask_ptrs, mask=mask_l, other=1.0)  # broadcast along channel
        out_vals = y_vals * m_vals
        tl.store(out_ptr + n * stride_y_n + c * stride_y_c + lo_offsets * stride_y_l, out_vals, mask=mask_l)


@torch.no_grad()
def run(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    reverse: bool,
    # transform 0 weights
    transform_0_conv0_weight: torch.Tensor,
    transform_0_conv0_bias: torch.Tensor,
    transform_0_conv1_weight: torch.Tensor,
    transform_0_conv1_bias: torch.Tensor,
    transform_0_conv2_weight: torch.Tensor,
    transform_0_conv2_bias: torch.Tensor,
    # transform 1 weights
    transform_1_conv0_weight: torch.Tensor,
    transform_1_conv0_bias: torch.Tensor,
    transform_1_conv1_weight: torch.Tensor,
    transform_1_conv1_bias: torch.Tensor,
    transform_1_conv2_weight: torch.Tensor,
    transform_1_conv2_bias: torch.Tensor,
    # transform 2 weights
    transform_2_conv0_weight: torch.Tensor,
    transform_2_conv0_bias: torch.Tensor,
    transform_2_conv1_weight: torch.Tensor,
    transform_2_conv1_bias: torch.Tensor,
    transform_2_conv2_weight: torch.Tensor,
    transform_2_conv2_bias: torch.Tensor,
    # transform 3 weights
    transform_3_conv0_weight: torch.Tensor,
    transform_3_conv0_bias: torch.Tensor,
    transform_3_conv1_weight: torch.Tensor,
    transform_3_conv1_bias: torch.Tensor,
    transform_3_conv2_weight: torch.Tensor,
    transform_3_conv2_bias: torch.Tensor,
):
    """
    Triton-optimized residual coupling flow block.
    - forward: x1 = x1 + transform(x0) per layer
    - reverse: x1 = x1 - transform(x0) per layer (in reverse order)
    """
    N, C, L = x.shape
    C_half = C // 2  # 96

    # Ensure inputs are contiguous and on CUDA
    assert x.is_cuda and x_mask.is_cuda, "Triton kernels require CUDA tensors"
    x = x.contiguous()
    x_mask = x_mask.contiguous()

    # Prepare transforms as a list of (conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b)
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

    BLOCK_L = 128  # tile size along time

    if not reverse:
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            # Split into halves
            x0 = x[:, :C_half, :].contiguous()
            x1 = x[:, C_half:, :].contiguous()

            # conv0: x0 -> h0
            h0 = torch.empty((N, conv0_w.shape[0], L), device=x.device, dtype=torch.float32)
            conv1d_forward_kernel[(N, conv0_w.shape[0], triton.cdiv(L, BLOCK_L))](  # grid over (N, C_out, tiles)
                x0, conv0_w, conv0_b, h0,
                N, conv0_w.shape[1], conv0_w.shape[0], L, L, conv0_w.shape[2],
                x0.stride(0), x0.stride(1), x0.stride(2),
                conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                h0.stride(0), h0.stride(1), h0.stride(2),
                BLOCK_L=BLOCK_L,
                num_warps=4,
            )
            # ReLU
            h0_relu = torch.empty_like(h0)
            relu_kernel[(N, h0.shape[1], L)](
                h0, h0_relu,
                N, h0.shape[1], L,
                h0.stride(0), h0.shape(1), h0.shape(2),
                h0_relu.stride(0), h0_relu.shape(1), h0_relu.shape(2),
                BLOCK_L=BLOCK_L,
                num_warps=1,
            )
            h0 = h0_relu

            # conv1: h0 -> h1
            h1 = torch.empty((N, conv1_w.shape[0], L), device=x.device, dtype=torch.float32)
            conv1d_forward_kernel[(N, conv1_w.shape[0], triton.cdiv(L, BLOCK_L))](
                h0, conv1_w, conv1_b, h1,
                N, conv1_w.shape[1], conv1_w.shape[0], L, L, conv1_w.shape[2],
                h0.stride(0), h0.stride(1), h0.stride(2),
                conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                h1.stride(0), h1.stride(1), h1.stride(2),
                BLOCK_L=BLOCK_L,
                num_warps=4,
            )
            # ReLU
            h1_relu = torch.empty_like(h1)
            relu_kernel[(N, h1.shape[1], L)](
                h1, h1_relu,
                N, h1.shape[1], L,
                h1.stride(0), h1.shape(1), h1.shape(2),
                h1_relu.stride(0), h1_relu.shape(1), h1_relu.shape(2),
                BLOCK_L=BLOCK_L,
                num_warps=1,
            )
            h1 = h1_relu

            # conv2: h1 -> h2 (no ReLU)
            h2 = torch.empty((N, conv2_w.shape[0], L), device=x.device, dtype=torch.float32)
            conv1d_forward_kernel[(N, conv2_w.shape[0], triton.cdiv(L, BLOCK_L))](
                h1, conv2_w, conv2_b, h2,
                N, conv2_w.shape[1], conv2_w.shape[0], L, L, conv2_w.shape[2],
                h1.stride(0), h1.stride(1), h1.stride(2),
                conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                h2.stride(0), h2.stride(1), h2.stride(2),
                BLOCK_L=BLOCK_L,
                num_warps=4,
            )

            # Affine coupling: x1 = x1 + h2
            x1 = x1 + h2

            # Concatenate [x0, x1] back along channels
            y_full = torch.empty((N, C, L), device=x.device, dtype=torch.float32)
            concat_halves_forward[(N, C_half, triton.cdiv(L, BLOCK_L))](  # grid over (N, C_half, tiles)
                x0, x1, y_full,
                N, C_half, L,
                x0.stride(0), x0.stride(1), x0.stride(2),
                x1.stride(0), x1.stride(1), x1.stride(2),
                y_full.stride(0), y_full.stride(1), y_full.stride(2),
                BLOCK_L=BLOCK_L,
                num_warps=4,
            )

            # Apply mask: y_full = y_full * x_mask (broadcast along channels)
            y_masked = torch.empty_like(y_full)
            mul_mask_kernel[(N, C, L)](
                y_full, x_mask, y_masked,
                N, C, L,
                y_full.stride(0), y_full.stride(1), y_full.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                BLOCK_L=BLOCK_L,
                num_warps=1,
            )
            x = y_masked  # update x for next transform
    else:
        # Reverse pass: apply transformations in reverse order
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
            # Split into halves
            x0 = x[:, :C_half, :].contiguous()
            x1 = x[:, C_half:, :].contiguous()

            # conv0: x0 -> h0
            h0 = torch.empty((N, conv0_w.shape[0], L), device=x.device, dtype=torch.float32)
            conv1d_forward_kernel[(N, conv0_w.shape[0], triton.cdiv(L, BLOCK_L))](
                x0, conv0_w, conv0_b, h0,
                N, conv0_w.shape[1], conv0_w.shape[0], L, L, conv0_w.shape[2],
                x0.stride(0), x0.stride(1), x0.stride(2),
                conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                h0.stride(0), h0.stride(1), h0.stride(2),
                BLOCK_L=BLOCK_L,
                num_warps=4,
            )
            # ReLU
            h0_relu = torch.empty_like(h0)
            relu_kernel[(N, h0.shape[1], L)](
                h0, h0_relu,
                N, h0.shape[1], L,
                h0.stride(0), h0.shape(1), h0.shape(2),
                h0_relu.stride(0), h0_relu.shape(1), h0_relu.shape(2),
                BLOCK_L=BLOCK_L,
                num_warps=1,
            )
            h0 = h0_relu

            # conv1: h0 -> h1
            h1 = torch.empty((N, conv1_w.shape[0], L), device=x.device, dtype=torch.float32)
            conv1d_forward_kernel[(N, conv1_w.shape[0], triton.cdiv(L, BLOCK_L))](
                h0, conv1_w, conv1_b, h1,
                N, conv1_w.shape[1], conv1_w.shape[0], L, L, conv1_w.shape[2],
                h0.stride(0), h0.stride(1), h0.stride(2),
                conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                h1.stride(0), h1.stride(1), h1.stride(2),
                BLOCK_L=BLOCK_L,
                num_warps=4,
            )
            # ReLU
            h1_relu = torch.empty_like(h1)
            relu_kernel[(N, h1.shape[1], L)](
                h1, h1_relu,
                N, h1.shape[1], L,
                h1.stride(0), h1.shape(1), h1.shape(2),
                h1_relu.stride(0), h1_relu.shape(1), h1_relu.shape(2),
                BLOCK_L=BLOCK_L,
                num_warps=1,
            )
            h1 = h1_relu

            # conv2: h1 -> h2 (no ReLU)
            h2 = torch.empty((N, conv2_w.shape[0], L), device=x.device, dtype=torch.float32)
            conv1d_forward_kernel[(N, conv2_w.shape[0], triton.cdiv(L, BLOCK_L))](
                h1, conv2_w, conv2_b, h2,
                N, conv2_w.shape[1], conv2_w.shape[0], L, L, conv2_w.shape[2],
                h1.stride(0), h1.stride(1), h1.stride(2),
                conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                h2.stride(0), h2.stride(1), h2.stride(2),
                BLOCK_L=BLOCK_L,
                num_warps=4,
            )

            # Affine coupling: x1 = x1 - h2
            x1 = x1 - h2

            # Concatenate [x0, x1] back along channels
            y_full = torch.empty((N, C, L), device=x.device, dtype=torch.float32)
            concat_halves_forward[(N, C_half, triton.cdiv(L, BLOCK_L))](
                x0, x1, y_full,
                N, C_half, L,
                x0.stride(0), x0.stride(1), x0.stride(2),
                x1.stride(0), x1.stride(1), x1.stride(2),
                y_full.stride(0), y_full.stride(1), y_full.stride(2),
                BLOCK_L=BLOCK_L,
                num_warps=4,
            )

            # Apply mask
            y_masked = torch.empty_like(y_full)
            mul_mask_kernel[(N, C, L)](
                y_full, x_mask, y_masked,
                N, C, L,
                y_full.stride(0), y_full.stride(1), y_full.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                BLOCK_L=BLOCK_L,
                num_warps=1,
            )
            x = y_masked

    return x


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
