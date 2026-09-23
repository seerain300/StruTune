import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


# Triton Conv1d forward kernel: y = conv1d(x, w) + b
# x: [N, C_in, L_in], w: [C_out, C_in, K], b: [C_out], y: [N, C_out, L_out]
# Padding P = K//2, stride=1, dilation=1, L_out = L_in - P + 1
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

    # Accumulator in fp32
    acc = tl.zeros([BLOCK_L], dtype=tl.float32)

    # Add bias
    b_val = tl.load(b_ptr + co).to(tl.float32)
    acc += b_val

    # Loop over input channels and kernel taps
    for ci in range(0, C_in):
        for k in range(0, K):
            P = K // 2
            li = l_offsets + P - k
            in_range = (li >= 0) & (li < L_in)
            x_ptrs = x_ptr + n * stride_x_n + ci * stride_x_c + li * stride_x_l
            x_vals = tl.load(x_ptrs, mask=mask_out & in_range, other=0.0).to(tl.float32)

            w_ptrs = w_ptr + co * stride_w_co + ci * stride_w_ci + k * stride_w_k
            w_vals = tl.load(w_ptrs).to(tl.float32)

            acc += x_vals * w_vals

    # Store results
    y_ptrs = y_ptr + n * stride_y_n + co * stride_y_c + l_offsets * stride_y_l
    tl.store(y_ptrs, acc, mask=mask_out)


# Triton ReLU kernel: y = max(y, 0)
@triton.jit
def relu_kernel(
    y_ptr,
    N, C, L,
    stride_y_n, stride_y_c, stride_y_l,
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask = l_offsets < L
    y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l
    y_vals = tl.load(y_ptrs, mask=mask, other=0.0).to(tl.float32)
    y_vals = tl.maximum(y_vals, 0.0)
    tl.store(y_ptrs, y_vals, mask=mask)


# Triton split halves forward: y_full[:, :C_half, :] -> y0, y_full[:, C_half:, :] -> y1
@triton.jit
def split_halves_forward(
    y_full_ptr, y0_ptr, y1_ptr,
    N, C_half, L,
    stride_full_n, stride_full_c, stride_full_l,
    stride0_n, stride0_c, stride0_l,
    stride1_n, stride1_c, stride1_l,
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    ch = tl.program_id(1)  # channel in [0, C_half)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask = l_offsets < L

    full_ptrs0 = y_full_ptr + n * stride_full_n + ch * stride_full_c + l_offsets * stride_full_l
    full_ptrs1 = y_full_ptr + n * stride_full_n + (ch + C_half) * stride_full_c + l_offsets * stride_full_l
    out0_ptrs = y0_ptr + n * stride0_n + ch * stride0_c + l_offsets * stride0_l
    out1_ptrs = y1_ptr + n * stride1_n + ch * stride1_c + l_offsets * stride1_l

    v0 = tl.load(full_ptrs0, mask=mask, other=0.0).to(tl.float32)
    v1 = tl.load(full_ptrs1, mask=mask, other=0.0).to(tl.float32)
    tl.store(out0_ptrs, v0, mask=mask)
    tl.store(out1_ptrs, v1, mask=mask)


# Triton concat halves forward: y0[:, :C_half, :] -> y_full[:, :C_half, :], y1[:, :C_half, :] -> y_full[:, C_half:, :]
@triton.jit
def concat_halves_forward(
    y0_ptr, y1_ptr, y_full_ptr,
    N, C_half, L,
    stride0_n, stride0_c, stride0_l,
    stride1_n, stride1_c, stride1_l,
    stridefull_n, stridefull_c, stridefull_l,
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    ch = tl.program_id(1)  # channel in [0, C_half)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask = l_offsets < L

    in0_ptrs = y0_ptr + n * stride0_n + ch * stride0_c + l_offsets * stride0_l
    in1_ptrs = y1_ptr + n * stride1_n + ch * stride1_c + l_offsets * stride1_l
    out0_ptrs = y_full_ptr + n * stridefull_n + ch * stridefull_c + l_offsets * stridefull_l
    out1_ptrs = y_full_ptr + n * stridefull_n + (ch + C_half) * stridefull_c + l_offsets * stridefull_l

    v0 = tl.load(in0_ptrs, mask=mask, other=0.0).to(tl.float32)
    v1 = tl.load(in1_ptrs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out0_ptrs, v0, mask=mask)
    tl.store(out1_ptrs, v1, mask=mask)


# Triton add/sub kernel: y1 = y1 +/- h (h is already loaded from y0)
@triton.jit
def add_or_sub_kernel(
    y1_ptr, h_ptr,
    N, C, L,
    stride_y1_n, stride_y1_c, stride_y1_l,
    stride_h_n, stride_h_c, stride_h_l,
    add_flag: tl.constexpr,  # 1 for add, 0 for sub
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask = l_offsets < L

    y1_ptrs = y1_ptr + n * stride_y1_n + c * stride_y1_c + l_offsets * stride_y1_l
    h_ptrs = h_ptr + n * stride_h_n + c * stride_h_c + l_offsets * stride_h_l
    y1_vals = tl.load(y1_ptrs, mask=mask, other=0.0).to(tl.float32)
    h_vals = tl.load(h_ptrs, mask=mask, other=0.0).to(tl.float32)
    if add_flag:
        y1_vals = y1_vals + h_vals
    else:
        y1_vals = y1_vals - h_vals
    tl.store(y1_ptrs, y1_vals, mask=mask)


# Triton mask multiply: y *= mask (mask: [N, 1, L], broadcast across C)
@triton.jit
def mul_mask_kernel(
    y_ptr, mask_ptr,
    N, C, L,
    stride_y_n, stride_y_c, stride_y_l,
    stride_m_n, stride_m_c, stride_m_l,
    BLOCK_L: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)
    l_offsets = tile * BLOCK_L + tl.arange(0, BLOCK_L)
    mask = l_offsets < L

    y_ptrs = y_ptr + n * stride_y_n + c * stride_y_c + l_offsets * stride_y_l
    m_ptrs = mask_ptr + n * stride_m_n + 0 * stride_m_c + l_offsets * stride_m_l  # mask has C_dim=1
    y_vals = tl.load(y_ptrs, mask=mask, other=0.0).to(tl.float32)
    m_vals = tl.load(m_ptrs, mask=mask, other=1.0).to(tl.float32)  # assume mask is 1s
    y_vals = y_vals * m_vals
    tl.store(y_ptrs, y_vals, mask=mask)


@torch.no_grad()
def run_triton(
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
    Triton-only implementation of the residual coupling flow block.
    - x: [N, 192, L], x_mask: [N, 1, L]
    - For each transform, compute h = conv0(ReLU) -> conv1(ReLU) -> conv2; then update x1, concatenate halves, apply mask.
    """
    assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
    # Ensure contiguity
    x = x.contiguous()
    x_mask = x_mask.contiguous()

    N = x.shape[0]
    C = x.shape[1]
    L = x.shape[2]
    C_half = C // 2
    K = 5
    P = K // 2
    # We will perform 4 transforms sequentially. Final output is after the 4th transform.
    # Initialize output as x (we'll modify it per transform).
    out = x

    # Constants for kernels
    BLOCK_L = 128  # tile size along time; adjust if needed

    # Process 4 transforms
    # Note: We do not use torch ops; all are Triton kernels.
    for i in range(4):
        # Prepare inputs for current transform. We take the current 'out' as x0/x1.
        # Split halves
        y_full = out  # [N, 2*C_half, L]
        y0 = torch.empty((N, C_half, L), device=x.device, dtype=x.dtype)
        y1 = torch.empty((N, C_half, L), device=x.device, dtype=x.dtype)

        # Launch split halves forward
        grid_split = (N, C_half, triton.cdiv(L, BLOCK_L))
        split_halves_forward[grid_split](
            y_full, y0, y1,
            N, C_half, L,
            y_full.stride(0), y_full.stride(1), y_full.stride(2),
            y0.stride(0), y0.stride(1), y0.stride(2),
            y1.stride(0), y1.stride(1), y1.stride(2),
            BLOCK_L=BLOCK_L, num_warps=4
        )

        # conv0
        C_in = C_half
        C_out = transform_0_conv0_weight.shape[0] if i == 0 else (transform_1_conv0_weight.shape[0] if i == 1 else (transform_2_conv0_weight.shape[0] if i == 2 else transform_3_conv0_weight.shape[0]))
        conv_w = (transform_0_conv0_weight if i == 0 else transform_1_conv0_weight if i == 1 else transform_2_conv0_weight if i == 2 else transform_3_conv0_weight)
        conv_b = (transform_0_conv0_bias if i == 0 else transform_1_conv0_bias if i == 1 else transform_2_conv0_bias if i == 2 else transform_3_conv0_bias)
        h0 = torch.empty((N, C_out, L - P + 1), device=x.device, dtype=x.dtype)
        grid_conv0 = (N, C_out, triton.cdiv(L - P + 1, BLOCK_L))
        conv1d_forward_kernel[grid_conv0](
            y0, conv_w, conv_b, h0,
            N, C_in, C_out, L, L - P + 1, K,
            y0.stride(0), y0.stride(1), y0.stride(2),
            conv_w.stride(0), conv_w.stride(1), conv_w.stride(2),
            h0.stride(0), h0.stride(1), h0.stride(2),
            BLOCK_L=BLOCK_L, num_warps=4
        )
        # ReLU
        grid_relu = (N, C_out, triton.cdiv(L - P + 1, BLOCK_L))
        h0_relu = torch.empty_like(h0)
        relu_kernel[grid_relu](
            h0, N, C_out, L - P + 1,
            h0.stride(0), h0.stride(1), h0.stride(2),
            BLOCK_L=BLOCK_L, num_warps=4
        )

        # conv1
        C_in = C_out
        C_out_next = transform_0_conv1_weight.shape[0] if i == 0 else (transform_1_conv1_weight.shape[0] if i == 1 else (transform_2_conv1_weight.shape[0] if i == 2 else transform_3_conv1_weight.shape[0]))
        conv_w1 = (transform_0_conv1_weight if i == 0 else transform_1_conv1_weight if i == 1 else transform_2_conv1_weight if i == 2 else transform_3_conv1_weight)
        conv_b1 = (transform_0_conv1_bias if i == 0 else transform_1_conv1_bias if i == 1 else transform_2_conv1_bias if i == 2 else transform_3_conv1_bias)
        h1 = torch.empty((N, C_out_next, L - P + 1), device=x.device, dtype=x.dtype)
        grid_conv1 = (N, C_out_next, triton.cdiv(L - P + 1, BLOCK_L))
        conv1d_forward_kernel[grid_conv1](
            h0_relu, conv_w1, conv_b1, h1,
            N, C_in, C_out_next, L - P + 1, L - P + 1, K,
            h0_relu.stride(0), h0_relu.stride(1), h0_relu.stride(2),
            conv_w1.stride(0), conv_w1.stride(1), conv_w1.stride(2),
            h1.stride(0), h1.stride(1), h1.stride(2),
            BLOCK_L=BLOCK_L, num_warps=4
        )
        # ReLU
        grid_relu1 = (N, C_out_next, triton.cdiv(L - P + 1, BLOCK_L))
        h1_relu = torch.empty_like(h1)
        relu_kernel[grid_relu1](
            h1, N, C_out_next, L - P + 1,
            h1.stride(0), h1.stride(1), h1.stride(2),
            BLOCK_L=BLOCK_L, num_warps=4
        )

        # conv2 (no ReLU)
        C_in = C_out_next
        C_out_final = transform_0_conv2_weight.shape[0] if i == 0 else (transform_1_conv2_weight.shape[0] if i == 1 else (transform_2_conv2_weight.shape[0] if i == 2 else transform_3_conv2_weight.shape[0]))
        conv_w2 = (transform_0_conv2_weight if i == 0 else transform_1_conv2_weight if i == 1 else transform_2_conv2_weight if i == 2 else transform_3_conv2_weight)
        conv_b2 = (transform_0_conv2_bias if i == 0 else transform_1_conv2_bias if i == 1 else transform_2_conv2_bias if i == 2 else transform_3_conv2_bias)
        h = torch.empty((N, C_out_final, L - P + 1), device=x.device, dtype=x.dtype)
        grid_conv2 = (N, C_out_final, triton.cdiv(L - P + 1, BLOCK_L))
        conv1d_forward_kernel[grid_conv2](
            h1_relu, conv_w2, conv_b2, h,
            N, C_in, C_out_final, L - P + 1, L - P + 1, K,
            h1_relu.stride(0), h1_relu.stride(1), h1_relu.stride(2),
            conv_w2.stride(0), conv_w2.stride(1), conv_w2.stride(2),
            h.stride(0), h.stride(1), h.stride(2),
            BLOCK_L=BLOCK_L, num_warps=4
        )

        # Apply mask to h
        h_masked = torch.empty_like(h)
        grid_mask = (N, C_out_final, triton.cdiv(L - P + 1, BLOCK_L))
        mul_mask_kernel[grid_mask](
            h, x_mask,
            N, C_out_final, L - P + 1,
            h.stride(0), h.stride(1), h.stride(2),
            x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
            BLOCK_L=BLOCK_L, num_warps=4
        )
        # Update y1: forward add, reverse subtract
        grid_add = (N, C_out_final, triton.cdiv(L - P + 1, BLOCK_L))
        # We need to pass current y1 (which is x1 slice). In our case, y1 was initialized from out, but we need to update it.
        # We will perform in-place update by launching add_or_sub_kernel against y1's pointer (we pass y1 as input/output buffer).
        # Note: Triton allows passing the same tensor as input and output.
        add_flag = 1 if not reverse else 0
        add_or_sub_kernel[grid_add](
            y1, h_masked,
            N, C_out_final, L - P + 1,
            y1.stride(0), y1.stride(1), y1.stride(2),
            h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
            add_flag=add_flag, BLOCK_L=BLOCK_L, num_warps=4
        )

        # Concatenate halves back
        out_tmp = torch.empty((N, 2 * C_half, L), device=x.device, dtype=x.dtype)
        grid_concat = (N, C_half, triton.cdiv(L, BLOCK_L))
        concat_halves_forward[grid_concat](
            y0, y1, out_tmp,
            N, C_half, L,
            y0.stride(0), y0.stride(1), y0.stride(2),
            y1.stride(0), y1.stride(1), y1.stride(2),
            out_tmp.stride(0), out_tmp.stride(1), out_tmp.stride(2),
            BLOCK_L=BLOCK_L, num_warps=4
        )

        # Apply mask to out_tmp
        out_tmp_masked = torch.empty_like(out_tmp)
        grid_mask2 = (N, 2 * C_half, triton.cdiv(L, BLOCK_L))
        mul_mask_kernel[grid_mask2](
            out_tmp, x_mask,
            N, 2 * C_half, L,
            out_tmp.stride(0), out_tmp.stride(1), out_tmp.stride(2),
            x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
            BLOCK_L=BLOCK_L, num_warps=4
        )

        # Update out for next transform
        out = out_tmp_masked

    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # args layout as in the original: (x, x_mask, reverse, then all weights/biases for 4 transforms)
        x = args[0]
        x_mask = args[1]
        reverse = args[2]
        # Extract the 4 transforms' weights/biases
        # The order is: transform_0..., transform_1..., transform_2..., transform_3...
        t0_w0, t0_b0 = args[3], args[4]
        t0_w1, t0_b1 = args[5], args[6]
        t0_w2, t0_b2 = args[7], args[8]

        t1_w0, t1_b0 = args[9], args[10]
        t1_w1, t1_b1 = args[11], args[12]
        t1_w2, t1_b2 = args[13], args[14]

        t2_w0, t2_b0 = args[15], args[16]
        t2_w1, t2_b1 = args[17], args[18]
        t2_w2, t2_b2 = args[19], args[20]

        t3_w0, t3_b0 = args[21], args[22]
        t3_w1, t3_b1 = args[23], args[24]
        t3_w2, t3_b2 = args[25], args[26]

        # Ensure CUDA device and dtypes
        # The get_inputs() uses torch.randn on the given device, so tensors are already on the right device.
        # We still enforce contiguity and float32 (default).
        # Call Triton run
        return run_triton(
            x, x_mask, reverse,
            t0_w0, t0_b0, t0_w1, t0_b1, t0_w2, t0_b2,
            t1_w0, t1_b0, t1_w1, t1_b1, t1_w2, t1_b2,
            t2_w0, t2_b0, t2_w1, t2_b1, t2_w2, t2_b2,
            t3_w0, t3_b0, t3_w1, t3_b1, t3_w2, t3_b2,
        )


def run(*args):
    return ModelNew()(*args)
