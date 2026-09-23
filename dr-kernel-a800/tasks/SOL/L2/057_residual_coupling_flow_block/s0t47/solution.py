import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Conv1d forward (padding=0) with tiling over output channels.
@triton.jit
def conv1d_forward_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, T_in, C_out, T_out, K,
    x_stride_n, x_stride_c, x_stride_t,
    w_stride_co, w_stride_ci, w_stride_k,
    out_stride_n, out_stride_c, out_stride_t,
    BLOCK_C: tl.constexpr,
):
    """
    Compute y[n, co, t_out] = bias[co] + sum_ci sum_{k=0..K-1} w[co, ci, k] * x[n, ci, t_out - k]
    where t_in = t_out - k must satisfy 0 <= t_in < T_in (zero-padding semantics).
    """
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)  # output time index
    pid_cblk = tl.program_id(2)  # block over output channels

    co_start = pid_cblk * BLOCK_C
    co_offsets = co_start + tl.arange(0, BLOCK_C)
    co_mask = co_offsets < C_out

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    ci = 0
    while ci < C_in:
        k = 0
        while k < K:
            t_in = pid_t - k  # corresponding input time index for this kernel tap
            in_bounds = (t_in >= 0) & (t_in < T_in)

            # Load x[n, ci, t_in] for all co in this block (co are broadcast across ci/k)
            x_offsets = pid_n * x_stride_n + ci * x_stride_c + t_in * x_stride_t
            co_vec_offsets = co_offsets * x_stride_c
            x_ptrs = x_ptr + x_offsets + co_vec_offsets
            x_vals = tl.load(x_ptrs, mask=co_mask & in_bounds, other=0.0)

            # Load w[co, ci, k] for all co in this block
            w_ptrs = w_ptr + co_offsets * w_stride_co + ci * w_stride_ci + k * w_stride_k
            w_vals = tl.load(w_ptrs, mask=co_mask, other=0.0)

            acc += x_vals * w_vals
            k += 1
        ci += 1

    # Add bias
    b_vals = tl.load(b_ptr + co_offsets, mask=co_mask, other=0.0)
    acc += b_vals

    # Store result
    out_offsets = pid_n * out_stride_n + co_offsets * out_stride_c + pid_t * out_stride_t
    tl.store(out_ptr + out_offsets, acc, mask=co_mask)


# ReLU elementwise kernel
@triton.jit
def relu_kernel(
    in_ptr, out_ptr,
    N, C, T,
    in_stride_n, in_stride_c, in_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    val = tl.load(in_ptr + pid_n * in_stride_n + pid_c * in_stride_c + pid_t * in_stride_t)
    val = tl.maximum(val, 0.0)
    tl.store(out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t, val)


# Split halves: from x [N, 2*C_half, T] -> x0 [N, C_half, T], x1 [N, C_half, T]
@triton.jit
def split_halves_kernel(
    x_ptr, x0_ptr, x1_ptr,
    N, C_half, T,
    x_stride_n, x_stride_c, x_stride_t,
    x0_stride_n, x0_stride_c, x0_stride_t,
    x1_stride_n, x1_stride_c, x1_stride_t,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    # First half
    val = tl.load(x_ptr + pid_n * x_stride_n + pid_c * x_stride_c + pid_t * x_stride_t)
    tl.store(x0_ptr + pid_n * x0_stride_n + pid_c * x0_stride_c + pid_t * x0_stride_t, val)

    # Second half
    val = tl.load(x_ptr + pid_n * x_stride_n + (pid_c + C_half) * x_stride_c + pid_t * x_stride_t)
    tl.store(x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + pid_t * x1_stride_t, val)


# Add/subtract coupling: x1 = x1 + h (forward) or x1 = x1 - h (reverse)
@triton.jit
def add_halves_kernel(
    x1_ptr, h_ptr, out_ptr,
    N, C, T,
    x1_stride_n, x1_stride_c, x1_stride_t,
    h_stride_n, h_stride_c, h_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
    ADD: tl.constexpr,  # True for forward (add), False for reverse (subtract)
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    x1_val = tl.load(x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + pid_t * x1_stride_t)
    h_val = tl.load(h_ptr + pid_n * h_stride_n + pid_c * h_stride_c + pid_t * h_stride_t)
    if ADD:
        res = x1_val + h_val
    else:
        res = x1_val - h_val
    tl.store(out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t, res)


# Concatenate halves back: x0 [N, C_half, T] and x1 [N, C_half, T] -> x [N, 2*C_half, T]
@triton.jit
def cat_halves_kernel(
    x0_ptr, x1_ptr, out_ptr,
    N, C_half, T,
    x0_stride_n, x0_stride_c, x0_stride_t,
    x1_stride_n, x1_stride_c, x1_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    # First half
    val0 = tl.load(x0_ptr + pid_n * x0_stride_n + pid_c * x0_stride_c + pid_t * x0_stride_t)
    tl.store(out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t, val0)

    # Second half: channel index shifted by C_half
    val1 = tl.load(x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + pid_t * x1_stride_t)
    tl.store(out_ptr + pid_n * out_stride_n + (pid_c + C_half) * out_stride_c + pid_t * out_stride_t, val1)


# Mask multiplication: out = x * mask
@triton.jit
def mask_mul_kernel(
    x_ptr, mask_ptr, out_ptr,
    N, C, T,
    x_stride_n, x_stride_c, x_stride_t,
    mask_stride_n, mask_stride_c, mask_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    x_val = tl.load(x_ptr + pid_n * x_stride_n + pid_c * x_stride_c + pid_t * x_stride_t)
    m_val = tl.load(mask_ptr + pid_n * mask_stride_n + pid_c * mask_stride_c + pid_t * mask_stride_t)
    out_val = x_val * m_val
    tl.store(out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t, out_val)


@torch.no_grad()
def run(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    reverse: bool,
    # 4 transforms each with 3 convs and biases
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
    transform_3_conv2_weight, transform_3_conv2_bias,
):
    """
    Residual coupling flow block.
    Forward: x1 = x1 + transform(x0) for each layer
    Reverse: x1 = x1 - transform(x0) for each layer (in reverse order)
    Note: This implementation computes everything via Triton kernels.
    """
    N, C, T = x.shape
    half_channels = C // 2

    # Prepare output buffer for forward; we will write x_out and use x for intermediate
    # We need to compute h for each transform and update x1. We'll use Triton kernels for everything.

    # Loop over transforms
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

    # We'll work in-place on x as temporary; for coupling we need x1 only. We'll create x0, x1, and h tensors.
    # But conv outputs should go into separate buffers. To avoid tensor reallocations within loop, we create them here.
    # However, conv output depends on previous conv. So better strategy:
    # For each transform, allocate x0, x1, and h, compute convs and ReLU in Triton, and update x1.
    # Since original code applies convs to x0 (first 96 channels), we need to handle splitting per transform.

    # Initialize x0, x1 from x
    x0 = x[:, :half_channels, :]
    x1 = x[:, half_channels:, :]

    # Helper: compute one transform and apply coupling on x1
    for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
        # Compute h = conv0 -> ReLU -> conv1 -> ReLU -> conv2
        # h shape: [N, C_out_conv2, T_out_conv2] where C_out_conv2 = conv2 out channels, T_out = T_in - K + 1 for each conv
        # First conv: conv0, out_channels = conv0_w.shape[0] (192), T0_out = T_in - 4
        N, C_in_conv0 = conv0_w.shape[1], conv0_w.shape[0]  # in_channels, out_channels
        K = conv0_w.shape[2]
        T0_out = T - K + 1

        h = torch.empty((N, C_in_conv0, T0_out), device=x.device, dtype=torch.float32)
        # Launch Triton conv1d forward
        grid_conv0 = (N, T0_out, triton.cdiv(C_in_conv0, 128))
        conv1d_forward_kernel[grid_conv0](
            x0, conv0_w, conv0_b, h,
            N, x0.shape[1], T, C_in_conv0, T0_out, K,
            x0.stride(0), x0.stride(1), x0.stride(2),
            conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
            h.stride(0), h.stride(1), h.stride(2),
            BLOCK_C=128,
        )

        # ReLU after conv0
        h_relu0 = torch.empty_like(h)
        grid_relu0 = (N, C_in_conv0, T0_out)
        relu_kernel[grid_relu0](
            h, h_relu0,
            N, C_in_conv0, T0_out,
            h.stride(0), h.stride(1), h.stride(2),
            h_relu0.stride(0), h_relu0.stride(1), h_relu0.stride(2),
        )
        h = h_relu0  # after ReLU conv0

        # Second conv: conv1, in_channels=C_in_conv0 (192), out_channels=192, T1_out = T0_out - 4
        C_in_conv1 = C_in_conv0
        C_out_conv1 = conv1_w.shape[0]
        K1 = conv1_w.shape[2]
        T1_out = T0_out - K1 + 1

        h_relu = torch.empty((N, C_out_conv1, T1_out), device=x.device, dtype=torch.float32)
        grid_conv1 = (N, T1_out, triton.cdiv(C_out_conv1, 128))
        conv1d_forward_kernel[grid_conv1](
            h, conv1_w, conv1_b, h_relu,
            N, C_in_conv1, T0_out, C_out_conv1, T1_out, K1,
            h.stride(0), h.stride(1), h.stride(2),
            conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
            h_relu.stride(0), h_relu.stride(1), h_relu.stride(2),
            BLOCK_C=128,
        )

        # ReLU after conv1
        h_relu2 = torch.empty_like(h_relu)
        grid_relu1 = (N, C_out_conv1, T1_out)
        relu_kernel[grid_relu1](
            h_relu, h_relu2,
            N, C_out_conv1, T1_out,
            h_relu.stride(0), h_relu.stride(1), h_relu.stride(2),
            h_relu2.stride(0), h_relu2.stride(1), h_relu2.stride(2),
        )
        h = h_relu2  # after ReLU conv1

        # Third conv: conv2, in_channels=C_out_conv1 (192), out_channels=conv2_w.shape[0] (96), T2_out = T1_out - 4
        C_in_conv2 = C_out_conv1
        C_out_conv2 = conv2_w.shape[0]
        K2 = conv2_w.shape[2]
        T2_out = T1_out - K2 + 1

        h_final = torch.empty((N, C_out_conv2, T2_out), device=x.device, dtype=torch.float32)
        grid_conv2 = (N, T2_out, triton.cdiv(C_out_conv2, 128))
        conv1d_forward_kernel[grid_conv2](
            h, conv2_w, conv2_b, h_final,
            N, C_in_conv2, T1_out, C_out_conv2, T2_out, K2,
            h.stride(0), h.stride(1), h.stride(2),
            conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
            h_final.stride(0), h_final.stride(1), h_final.stride(2),
            BLOCK_C=128,
        )

        # Mask multiply: apply x_mask (all ones in provided setup, but we keep generality)
        # Create mask tensor of same shape as h_final
        mask_h = x_mask  # [N,1,T2_out] broadcast over channels
        # Expand mask to [N, C_out_conv2, T2_out]
        mask_h_expanded = mask_h.expand(N, C_out_conv2, T2_out).contiguous()
        h_masked = torch.empty_like(h_final)
        grid_mask = (N, C_out_conv2, T2_out)
        mask_mul_kernel[grid_mask](
            h_final, mask_h_expanded, h_masked,
            N, C_out_conv2, T2_out,
            h_final.stride(0), h_final.stride(1), h_final.stride(2),
            mask_h_expanded.stride(0), mask_h_expanded.stride(1), mask_h_expanded.stride(2),
            h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
        )
        h = h_masked

        # Update x1 via coupling: x1 = x1 + h (forward) or x1 = x1 - h (reverse)
        out_x1 = torch.empty_like(x1)
        grid_add = (N, half_channels, T)
        add_halves_kernel[grid_add](
            x1, h, out_x1,
            N, half_channels, T,
            x1.stride(0), x1.stride(1), x1.stride(2),
            h.stride(0), h.stride(1), h.stride(2),
            out_x1.stride(0), out_x1.stride(1), out_x1.stride(2),
            ADD=True if not reverse else False,
        )
        x1 = out_x1

        # Concatenate back to form x for next transform: x = [x0, x1]
        x_out = torch.empty((N, C, T), device=x.device, dtype=torch.float32)
        grid_cat = (N, half_channels, T)
        cat_halves_kernel[grid_cat](
            x0, x1, x_out,
            N, half_channels, T,
            x0.stride(0), x0.stride(1), x0.stride(2),
            x1.stride(0), x1.stride(1), x1.stride(2),
            x_out.stride(0), x_out.stride(1), x_out.stride(2),
        )
        x = x_out

        # Split again for next transform
        x0 = x[:, :half_channels, :]
        x1 = x[:, half_channels:, :]

    return x

# Entry point for the evaluator
class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The signature matches the original run function's arguments.
        # We expect: x, x_mask, reverse, then 4 sets of weights and biases.
        # Use Triton kernels for all heavy ops.
        return run(*args)


def run(*args):
    return ModelNew()(*args)
