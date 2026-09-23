import math
import torch
import torch.nn as nn
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
    # program ids: over (n, co, tile along l_out)
    n = tl.program_id(0)
    co = tl.program_id(1)
    tile = tl.program_id(2)

    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L_out

    # Accumulate in float32
    acc = tl.zeros([BLOCK_L], dtype=tl.float32)

    # Bias for this output channel
    b_val = tl.load(b_ptr + co).to(tl.float32)
    acc += b_val

    P = K // 2
    # Loop over input channels and kernel taps
    for ci in range(0, C_in):
        for k in range(0, K):
            li = l_offsets + P - k
            in_range = (li >= 0) & (li < L_in) & mask_out
            # Compute pointers for x[n, ci, li]
            x_ptrs = x_ptr + n * stride_x_n + ci * stride_x_c + li * stride_x_l
            # Load with mask; other=0 ensures safe values for out-of-range
            x_vals = tl.load(x_ptrs, mask=in_range, other=0.0).to(tl.float32)

            # Load weight w[co, ci, k] (scalar)
            w_ptrs = w_ptr + co * stride_w_co + ci * stride_w_ci + k * stride_w_k
            w_val = tl.load(w_ptrs).to(tl.float32)

            # FMA accumulate
            acc += x_vals * w_val

    # Store results: y[n, co, l_offsets]
    y_ptrs = y_ptr + n * stride_y_n + co * stride_y_c + l_offsets * stride_y_l
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
    mask = l_offsets < L

    x_ptrs = x_ptr + n * stride_x_n + c * stride_x_c + l_offsets * stride_x_l
    y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l

    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
    y_vals = tl.maximum(x_vals, 0.0)  # PyTorch ReLU
    tl.store(y_ptrs, y_vals, mask=mask)


@triton.jit
def split_halves_forward(
    x_full_ptr, x0_ptr, x1_ptr,
    N, C_half, L,
    stride_x_full_n, stride_x_full_c, stride_x_full_l,
    stride_x0_n, stride_x0_c, stride_x0_l,
    stride_x1_n, stride_x1_c, stride_x1_l,
    BLOCK_L: tl.constexpr,
):
    # x_full: [N, 2*C_half, L]; write y0: [N, C_half, L], y1: [N, C_half, L]
    n = tl.program_id(0)
    ch = tl.program_id(1)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask = l_offsets < L

    # First half
    x0_ptrs = x_full_ptr + n * stride_x_full_n + ch * stride_x_full_c + l_offsets * stride_x_full_l
    x1_ptrs = x_full_ptr + n * stride_x_full_n + (ch + C_half) * stride_x_full_c + l_offsets * stride_x_full_l
    out0_ptrs = x0_ptr + n * stride_x0_n + ch * stride_x0_c + l_offsets * stride_x0_l
    out1_ptrs = x1_ptr + n * stride_x1_n + ch * stride_x1_c + l_offsets * stride_x1_l

    x0_vals = tl.load(x0_ptrs, mask=mask, other=0.0)
    x1_vals = tl.load(x1_ptrs, mask=mask, other=0.0)
    tl.store(out0_ptrs, x0_vals, mask=mask)
    tl.store(out1_ptrs, x1_vals, mask=mask)


@triton.jit
def concat_halves_forward(
    x0_ptr, x1_ptr, x_full_ptr,
    N, C_half, L,
    stride_x0_n, stride_x0_c, stride_x0_l,
    stride_x1_n, stride_x1_c, stride_x1_l,
    stride_x_full_n, stride_x_full_c, stride_x_full_l,
    BLOCK_L: tl.constexpr,
):
    # Reads x0: [N, C_half, L], x1: [N, C_half, L]; writes x_full: [N, 2*C_half, L]
    n = tl.program_id(0)
    ch = tl.program_id(1)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask = l_offsets < L

    x0_ptrs = x0_ptr + n * stride_x0_n + ch * stride_x0_c + l_offsets * stride_x0_l
    x1_ptrs = x1_ptr + n * stride_x1_n + ch * stride_x1_c + l_offsets * stride_x1_l
    x_full0_ptrs = x_full_ptr + n * stride_x_full_n + ch * stride_x_full_c + l_offsets * stride_x_full_l
    x_full1_ptrs = x_full_ptr + n * stride_x_full_n + (ch + C_half) * stride_x_full_c + l_offsets * stride_x_full_l

    x0_vals = tl.load(x0_ptrs, mask=mask, other=0.0)
    x1_vals = tl.load(x1_ptrs, mask=mask, other=0.0)
    tl.store(x_full0_ptrs, x0_vals, mask=mask)
    tl.store(x_full1_ptrs, x1_vals, mask=mask)


@triton.jit
def mul_mask_kernel(
    y_ptr, mask_ptr,
    N, C, L,
    stride_y_n, stride_y_c, stride_y_l,
    stride_mask_n, stride_mask_c, stride_mask_l,
    BLOCK_L: tl.constexpr,
):
    # Elementwise: y[n, c, l] *= mask[n, 0, l]
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask = l_offsets < L

    y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l
    mask_ptrs = mask_ptr + n * stride_mask_n + 0 * stride_mask_c + l_offsets * stride_mask_l

    y_vals = tl.load(y_ptrs, mask=mask, other=0.0)
    m_vals = tl.load(mask_ptrs, mask=mask, other=1.0)
    y_vals = y_vals * m_vals
    tl.store(y_ptrs, y_vals, mask=mask)


@triton.jit
def conv1d_forward_kernel_fp16(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, C_in, C_out, L_in, L_out, K,
    stride_x_n, stride_x_c, stride_x_l,
    stride_w_co, stride_w_ci, stride_w_k,
    stride_y_n, stride_y_c, stride_y_l,
    BLOCK_L: tl.constexpr,
):
    # Same as above but pointers are fp16; accumulation in float32; store to float32 output and caller casts
    n = tl.program_id(0)
    co = tl.program_id(1)
    tile = tl.program_id(2)

    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_out = l_offsets < L_out

    acc = tl.zeros([BLOCK_L], dtype=tl.float32)

    P = K // 2
    for ci in range(0, C_in):
        for k in range(0, K):
            li = l_offsets + P - k
            in_range = (li >= 0) & (li < L_in) & mask_out
            x_ptrs = x_ptr + n * stride_x_n + ci * stride_x_c + li * stride_x_l
            x_vals = tl.load(x_ptrs, mask=in_range, other=0.0).to(tl.float32)
            w_ptrs = w_ptr + co * stride_w_co + ci * stride_w_ci + k * stride_w_k
            w_val = tl.load(w_ptrs).to(tl.float32)
            acc += x_vals * w_val

    b_val = tl.load(b_ptr + co).to(tl.float32)
    acc += b_val

    y_ptrs = y_ptr + n * stride_y_n + co * stride_y_c + l_offsets * stride_y_l
    tl.store(y_ptrs, acc, mask=mask_out)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor,
                reverse: bool,
                # transforms weights/biases
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
                transform_3_conv2_bias: torch.Tensor):
        N, C, L = x.shape
        assert C == 192, "This implementation assumes C=192"
        half = C // 2
        assert x_mask.shape == (N, 1, L), "x_mask must be [N, 1, L]"
        # Ensure contiguity
        x = x.contiguous()
        x_mask = x_mask.contiguous()

        # Choose BLOCK_L based on L_out (L) to minimize masks and maximize performance
        if L >= 1024:
            BLOCK_L = 1024
            num_warps = 8
        elif L >= 512:
            BLOCK_L = 512
            num_warps = 8
        elif L >= 256:
            BLOCK_L = 256
            num_warps = 8
        elif L >= 128:
            BLOCK_L = 128
            num_warps = 4
        else:
            BLOCK_L = 64
            num_warps = 4

        grid_1d = (N,)

        # Iterate transforms
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
                # Split into halves
                x0 = x[:, :half, :].contiguous()
                x1 = x[:, half:, :].contiguous()

                # Conv0 on x0
                x0_full = torch.empty((N, conv0_w.shape[0], L), dtype=x.dtype, device=x.device)
                conv1d_forward_kernel[(N, conv0_w.shape[0], triton.cdiv(L, BLOCK_L))](
                    x0, conv0_w, conv0_b, x0_full,
                    N, x0.shape[1], conv0_w.shape[0], x0.shape[2], L, conv0_w.shape[2],
                    x0.stride(0), x0.stride(1), x0.stride(2),
                    conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                    x0_full.stride(0), x0_full.stride(1), x0_full.stride(2),
                    BLOCK_L=BLOCK_L,
                    num_warps=num_warps,
                )

                # ReLU after conv0
                x0_relu = torch.empty_like(x0_full)
                relu_kernel[(N, conv0_w.shape[0], triton.cdiv(L, BLOCK_L))](
                    x0_full, x0_relu,
                    N, conv0_w.shape[0], L,
                    x0_full.stride(0), x0_full.stride(1), x0_full.stride(2),
                    x0_relu.stride(0), x0_relu.stride(1), x0_relu.stride(2),
                    BLOCK_L=BLOCK_L,
                    num_warps=num_warps,
                )

                # Conv1 on x0_relu
                x1_out = torch.empty((N, conv1_w.shape[0], L), dtype=x.dtype, device=x.device)
                conv1d_forward_kernel[(N, conv1_w.shape[0], triton.cdiv(L, BLOCK_L))](
                    x0_relu, conv1_w, conv1_b, x1_out,
                    N, conv1_w.shape[1], conv1_w.shape[0], x0_relu.shape[2], L, conv1_w.shape[2],
                    x0_relu.stride(0), x0_relu.stride(1), x0_relu.stride(2),
                    conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                    x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                    BLOCK_L=BLOCK_L,
                    num_warps=num_warps,
                )

                # ReLU after conv1
                x1_relu = torch.empty_like(x1_out)
                relu_kernel[(N, conv1_w.shape[0], triton.cdiv(L, BLOCK_L))](
                    x1_out, x1_relu,
                    N, conv1_w.shape[0], L,
                    x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                    x1_relu.stride(0), x1_relu.stride(1), x1_relu.stride(2),
                    BLOCK_L=BLOCK_L,
                    num_warps=num_warps,
                )

                # Conv2 on x1_relu
                h = torch.empty((N, conv2_w.shape[0], L), dtype=x.dtype, device=x.device)
                conv1d_forward_kernel[(N, conv2_w.shape[0], triton.cdiv(L, BLOCK_L))](
                    x1_relu, conv2_w, conv2_b, h,
                    N, conv2_w.shape[1], conv2_w.shape[0], x1_relu.shape[2], L, conv2_w.shape[2],
                    x1_relu.stride(0), x1_relu.stride(1), x1_relu.stride(2),
                    conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                    h.stride(0), h.stride(1), h.stride(2),
                    BLOCK_L=BLOCK_L,
                    num_warps=num_warps,
                )

                # Apply mask: h = h * x_mask
                # x_mask is [N, 1, L]; we implement elementwise multiplication
                h_masked = torch.empty_like(h)
                # Launch mul_mask_kernel
                mul_mask_kernel[(N, h.shape[1], triton.cdiv(L, BLOCK_L))](
                    h, x_mask,
                    N, h.shape[1], L,
                    h.stride(0), h.stride(1), h.stride(2),
                    x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                    BLOCK_L=BLOCK_L,
                    num_warps=num_warps,
                )

                # Forward: x1 = x1 + h
                x1 = x1 + h_masked

                # Merge halves back: [x0, x1]
                x_full = torch.empty((N, C, L), dtype=x.dtype, device=x.device)
                concat_halves_forward[(N, half, triton.cdiv(L, BLOCK_L))](
                    x0, x1,
                    x_full,
                    N, half, L,
                    x0.stride(0), x0.stride(1), x0.stride(2),
                    x1.stride(0), x1.stride(1), x1.stride(2),
                    x_full.stride(0), x_full.stride(1), x_full.stride(2),
                    BLOCK_L=BLOCK_L,
                    num_warps=num_warps,
                )

                # Apply mask to output: x_full *= x_mask
                x_masked = torch.empty_like(x_full)
                mul_mask_kernel[(N, C, triton.cdiv(L, BLOCK_L))](
                    x_full, x_mask,
                    N, C, L,
                    x_full.stride(0), x_full.stride(1), x_full.stride(2),
                    x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                    BLOCK_L=BLOCK_L,
                    num_warps=num_warps,
                )
                x = x_masked

        else:
            # Reverse pass: apply transformations in reverse order
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
                # Split into halves
                x0 = x[:, :half, :].contiguous()
                x1 = x[:, half:, :].contiguous()

                # Conv0 on x0
                x0_full = torch.empty((N, conv0_w.shape[0], L), dtype=x.dtype, device=x.device)
                conv1d_forward_kernel[(N, conv0_w.shape[0], triton.cdiv(L, BLOCK_L))](
                    x0, conv0_w, conv0_b, x0_full,
                    N, x0.shape[1], conv0_w.shape[0], x0.shape[2], L, conv0_w.shape[2],
                    x0.stride(0), x0.stride(1), x0.stride(2),
                    conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
                    x0_full.stride(0), x0_full.stride(1), x0_full.stride(2),
                    BLOCK_L=BLOCK_L,
                    num_warps=num_warps,
                )

                # ReLU after conv0
                x0_relu = torch.empty_like(x0_full)
                relu_kernel[(N, conv0_w.shape[0], triton.cdiv(L, BLOCK_L))](
                    x0_full, x0_relu,
                    N, conv0_w.shape[0], L,
                    x0_full.stride(0), x0_full.stride(1), x0_full.stride(2),
                    x0_relu.stride(0), x0_relu.stride(1), x0_relu.stride(2),
                    BLOCK_L=BLOCK_L,
                    num_warps=num_warps,
                )

                # Conv1 on x0_relu
                x1_out = torch.empty((N, conv1_w.shape[0], L), dtype=x.dtype, device=x.device)
                conv1d_forward_kernel[(N, conv1_w.shape[0], triton.cdiv(L, BLOCK_L))](
                    x0_relu, conv1_w, conv1_b, x1_out,
                    N, conv1_w.shape[1], conv1_w.shape[0], x0_relu.shape[2], L, conv1_w.shape[2],
                    x0_relu.stride(0), x0_relu.stride(1), x0_relu.stride(2),
                    conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
                    x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                    BLOCK_L=BLOCK_L,
                    num_warps=num_warps,
                )

                # ReLU after conv1
                x1_relu = torch.empty_like(x1_out)
                relu_kernel[(N, conv1_w.shape[0], triton.cdiv(L, BLOCK_L))](
                    x1_out, x1_relu,
                    N, conv1_w.shape[0], L,
                    x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                    x1_relu.stride(0), x1_relu.stride(1), x1_relu.stride(2),
                    BLOCK_L=BLOCK_L,
                    num_warps=num_warps,
                )

                # Conv2 on x1_relu
                h = torch.empty((N, conv2_w.shape[0], L), dtype=x.dtype, device=x.device)
                conv1d_forward_kernel[(N, conv2_w.shape[0], triton.cdiv(L, BLOCK_L))](
                    x1_relu, conv2_w, conv2_b, h,
                    N, conv2_w.shape[1], conv2_w.shape[0], x1_relu.shape[2], L, conv2_w.shape[2],
                    x1_relu.stride(0), x1_relu.stride(1), x1_relu.stride(2),
                    conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
                    h.stride(0), h.stride(1), h.stride(2),
                    BLOCK_L=BLOCK_L,
                    num_warps=num_warps,
                )

                # Apply mask: h = h * x_mask
                h_masked = torch.empty_like(h)
                mul_mask_kernel[(N, h.shape[1], triton.cdiv(L, BLOCK_L))](
                    h, x_mask,
                    N, h.shape[1], L,
                    h.stride(0), h.stride(1), h.stride(2),
                    x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                    BLOCK_L=BLOCK_L,
                    num_warps=num_warps,
                )

                # Reverse: x1 = x1 - h
                x1 = x1 - h_masked

                # Merge halves back: [x0, x1]
                x_full = torch.empty((N, C, L), dtype=x.dtype, device=x.device)
                concat_halves_forward[(N, half, triton.cdiv(L, BLOCK_L))](
                    x0, x1,
                    x_full,
                    N, half, L,
                    x0.stride(0), x0.stride(1), x0.stride(2),
                    x1.stride(0), x1.stride(1), x1.stride(2),
                    x_full.stride(0), x_full.stride(1), x_full.stride(2),
                    BLOCK_L=BLOCK_L,
                    num_warps=num_warps,
                )

                # Apply mask to output: x_full *= x_mask
                x_masked = torch.empty_like(x_full)
                mul_mask_kernel[(N, C, triton.cdiv(L, BLOCK_L))](
                    x_full, x_mask,
                    N, C, L,
                    x_full.stride(0), x_full.stride(1), x_full.stride(2),
                    x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                    BLOCK_L=BLOCK_L,
                    num_warps=num_warps,
                )
                x = x_masked

        return x


# Original helpers remain the same; ModelNew is the required entry point.


def run(*args):
    return ModelNew()(*args)
