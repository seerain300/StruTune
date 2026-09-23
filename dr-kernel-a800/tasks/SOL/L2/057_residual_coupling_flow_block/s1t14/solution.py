import math
import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def conv1d_triton(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, OC, T, K, pad,
    x_sN, x_sC, x_sT,
    w_sO, w_sI, w_sK,
    out_sN, out_sC, out_sT,
    BLOCK_T: tl.constexpr,
):
    """
    Triton Conv1d (cross-correlation), stride=1, padding=pad, dilation=1.
    x: [N, IC, T] (float32)
    w: [OC, IC, K] (float32)
    b: [OC] (float32)
    out: [N, OC, T] (float32)
    Grid: (N, OC, ceil_div(T, BLOCK_T))
    """
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_block = tl.program_id(2)

    t_out_start = pid_block * BLOCK_T
    t_out_idx = t_out_start + tl.arange(0, BLOCK_T)
    valid_t = t_out_idx < T

    # Accumulator for output vector of length BLOCK_T
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)

    # Loop over input channels and kernel taps
    for ic in range(0, IC):
        for k in range(0, K):
            t_in = t_out_idx + k - pad  # vector of positions to read from x
            valid_in = valid_t & (t_in >= 0) & (t_in < T)

            # Compute offsets for x[n, ic, t_in]
            x_offsets = pid_n * x_sN + ic * x_sC + t_in * x_sT
            x_vals = tl.load(x_ptr + x_offsets, mask=valid_in, other=0.0)

            # Load weight w[oc, ic, k] as scalar
            w_off = pid_oc * w_sO + ic * w_sI + k * w_sK
            w_val = tl.load(w_ptr + w_off)

            # Accumulate
            acc += x_vals * w_val

    # Add bias
    b_val = tl.load(b_ptr + pid_oc)
    acc += b_val

    # Store to out[n, oc, t_out_idx]
    out_offsets = pid_n * out_sN + pid_oc * out_sC + t_out_idx * out_sT
    tl.store(out_ptr + out_offsets, acc, mask=valid_t)


@triton.jit
def relu_triton(in_ptr, out_ptr, N, C, T, in_sN, in_sC, in_sT, out_sN, out_sC, out_sT, BLOCK: tl.constexpr):
    """
    Elementwise ReLU over tensor of shape [N, C, T], flattened in blocks of BLOCK.
    """
    pid = tl.program_id(0)
    total = N * C * T
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    CT = C * T
    n = offs // CT
    rem = offs % CT
    c = rem // T
    t = rem % T

    in_offs = n * in_sN + c * in_sC + t * in_sT
    val = tl.load(in_ptr + in_offs, mask=mask, other=0.0)
    val = tl.maximum(val, 0.0)
    out_offs = n * out_sN + c * out_sC + t * out_sT
    tl.store(out_ptr + out_offs, val, mask=mask)


@triton.jit
def mask_apply_triton(in_ptr, mask_ptr, out_ptr, N, C, T,
                      mask_sN, mask_sC, mask_sT, in_sN, in_sC, in_sT, out_sN, out_sC, out_sT, BLOCK: tl.constexpr):
    """
    Elementwise multiply of tensor [N, C, T] by mask [N, 1, T] (broadcast over C),
    writing to out.
    """
    pid = tl.program_id(0)
    total = N * C * T
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    CT = C * T
    n = offs // CT
    rem = offs % CT
    c = rem // T
    t = rem % T

    in_offs = n * in_sN + c * in_sC + t * in_sT
    val = tl.load(in_ptr + in_offs, mask=mask, other=0.0)

    mask_offs = n * mask_sN + 0 * mask_sC + t * mask_sT
    mval = tl.load(mask_ptr + mask_offs, mask=mask, other=1.0)
    val = val * mval

    out_offs = n * out_sN + c * out_sC + t * out_sT
    tl.store(out_ptr + out_offs, val, mask=mask)


@triton.jit
def add_to_slice_triton(in_ptr, add_ptr, out_ptr, N, C1, C2, T,
                        in_sN, in_sC, in_sT, add_sN, add_sC, add_sT, out_sN, out_sC, out_sT, BLOCK: tl.constexpr):
    """
    Add tensor [N, C1, T] to tensor [N, C2, T] and store to out [N, C1+C2, T].
    Each program handles one (n, c1) and a block of t.
    """
    pid_n = tl.program_id(0)
    pid_c1 = tl.program_id(1)
    pid_block = tl.program_id(2)

    t_start = pid_block * BLOCK
    t_idx = t_start + tl.arange(0, BLOCK)
    valid_t = t_idx < T

    in_offs = pid_n * in_sN + pid_c1 * in_sC + t_idx * in_sT
    add_offs = pid_n * add_sN + pid_c1 * add_sC + t_idx * add_sT
    out_offs = pid_n * out_sN + pid_c1 * out_sC + t_idx * out_sT

    a = tl.load(in_ptr + in_offs, mask=valid_t, other=0.0)
    b = tl.load(add_ptr + add_offs, mask=valid_t, other=0.0)
    c = a + b
    tl.store(out_ptr + out_offs, c, mask=valid_t)


@torch.no_grad()
def run(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    reverse: bool,
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
    Triton-only implementation:
    - All conv1d, ReLU, mask application, and affine coupling happen in Triton kernels.
    - No torch.conv1d or torch.cat in the forward path.
    """
    N, C, T = x.shape
    assert C == 192, "This implementation assumes C=192 for half_channels=96."
    half_channels = C // 2

    # We will construct final output as [N, C, T] by writing into slices via Triton.
    # Start with zeros (we'll write into it as we go, but keep an allocated out for convenience).
    # However, since we need to return the final x, we'll allocate out_final and write into it.
    out_final = torch.zeros((N, C, T), dtype=x.dtype, device=x.device)

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
        # Forward: x1 = x1 + transform(x0) per layer
        for w0, b0, w1, b1, w2, b2 in transforms:
            # Split x into halves
            x0 = x[:, :half_channels, :].contiguous()  # [N, 96, T]
            x1 = x[:, half_channels:, :].contiguous()  # [N, 96, T]

            # conv0: [N, 192, T]
            y0 = torch.empty((N, 192, T), dtype=x.dtype, device=x.device)
            grid0 = (N, 192, triton.cdiv(T, 128))
            conv1d_triton[grid0](
                x0, w0, b0, y0, N, 96, 192, T, 5, 2,
                x0.stride(0), x0.stride(1), x0.stride(2),
                w0.stride(0), w0.stride(1), w0.stride(2),
                y0.stride(0), y0.stride(1), y0.stride(2),
                BLOCK_T=128, num_warps=4
            )

            # ReLU and mask over full [N, 192, T]
            y0_relu = torch.empty_like(y0)
            relu_triton[grid0](
                y0, y0_relu, N, 192, T,
                y0.stride(0), y0.stride(1), y0.stride(2),
                y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
                BLOCK=128, num_warps=4
            )
            y0_masked = torch.empty_like(y0_relu)
            mask_apply_triton[(N, 192, triton.cdiv(T, 128))](
                y0_relu, x_mask, y0_masked, N, 192, T,
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                y0_masked.stride(0), y0_masked.stride(1), y0_masked.stride(2),
                BLOCK=128, num_warps=4
            )

            # conv1: [N, 192, T]
            y1 = torch.empty((N, 192, T), dtype=x.dtype, device=x.device)
            grid1 = (N, 192, triton.cdiv(T, 128))
            conv1d_triton[grid1](
                y0_masked, w1, b1, y1, N, 192, 192, T, 5, 2,
                y0_masked.stride(0), y0_masked.stride(1), y0_masked.stride(2),
                w1.stride(0), w1.stride(1), w1.stride(2),
                y1.stride(0), y1.stride(1), y1.stride(2),
                BLOCK_T=128, num_warps=4
            )

            # ReLU and mask over full [N, 192, T]
            y1_relu = torch.empty_like(y1)
            relu_triton[grid1](
                y1, y1_relu, N, 192, T,
                y1.stride(0), y1.stride(1), y1.stride(2),
                y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
                BLOCK=128, num_warps=4
            )
            y1_masked = torch.empty_like(y1_relu)
            mask_apply_triton[(N, 192, triton.cdiv(T, 128))](
                y1_relu, x_mask, y1_masked, N, 192, T,
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                y1_masked.stride(0), y1_masked.stride(1), y1_masked.stride(2),
                BLOCK=128, num_warps=4
            )

            # conv2: [N, 96, T]
            h = torch.empty((N, 96, T), dtype=x.dtype, device=x.device)
            grid2 = (N, 96, triton.cdiv(T, 128))
            conv1d_triton[grid2](
                y1_masked, w2, b2, h, N, 192, 96, T, 5, 2,
                y1_masked.stride(0), y1_masked.stride(1), y1_masked.stride(2),
                w2.stride(0), w2.stride(1), w2.stride(2),
                h.stride(0), h.stride(1), h.stride(2),
                BLOCK_T=128, num_warps=4
            )

            # Mask h (broadcast)
            h_masked = torch.empty_like(h)
            mask_apply_triton[(N, 96, triton.cdiv(T, 128))](
                h, x_mask, h_masked, N, 96, T,
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                BLOCK=128, num_warps=4
            )

            # Affine coupling: update x1
            # out_final[:, half_channels:, :] = x1 + h_masked
            # First write x0 (unchanged) into out_final[:, :half_channels, :]
            # Then add h to x1.
            # We'll allocate out slices and write via add_to_slice_triton.
            out_x0 = out_final[:, :half_channels, :].contiguous()  # already zero-initialized
            add_to_slice_triton[(N, half_channels, triton.cdiv(T, 128))](
                x0, x0, out_x0, N, 96, 96, T,
                x0.stride(0), x0.stride(1), x0.stride(2),
                x0.stride(0), x0.stride(1), x0.stride(2),
                out_x0.stride(0), out_x0.stride(1), out_x0.stride(2),
                BLOCK=128, num_warps=4
            )

            out_x1 = out_final[:, half_channels:, :].contiguous()
            add_to_slice_triton[(N, 96, triton.cdiv(T, 128))](
                x1, h_masked, out_x1, N, 96, 96, T,
                x1.stride(0), x1.stride(1), x1.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                out_x1.stride(0), out_x1.stride(1), out_x1.stride(2),
                BLOCK=128, num_warps=4
            )

            # Update x for the next transform
            x = out_final

    else:
        # Reverse: x1 = x1 - transform(x0) per layer (in reverse order)
        for w0, b0, w1, b1, w2, b2 in reversed(transforms):
            # Split x into halves
            x0 = x[:, :half_channels, :].contiguous()  # [N, 96, T]
            x1 = x[:, half_channels:, :].contiguous()  # [N, 96, T]

            # conv0: [N, 192, T]
            y0 = torch.empty((N, 192, T), dtype=x.dtype, device=x.device)
            grid0 = (N, 192, triton.cdiv(T, 128))
            conv1d_triton[grid0](
                x0, w0, b0, y0, N, 96, 192, T, 5, 2,
                x0.stride(0), x0.stride(1), x0.stride(2),
                w0.stride(0), w0.stride(1), w0.stride(2),
                y0.stride(0), y0.stride(1), y0.stride(2),
                BLOCK_T=128, num_warps=4
            )

            # ReLU and mask over full [N, 192, T]
            y0_relu = torch.empty_like(y0)
            relu_triton[grid0](
                y0, y0_relu, N, 192, T,
                y0.stride(0), y0.stride(1), y0.stride(2),
                y0_relu.stride(0), y0_relu.stride(1), y0_relu.stride(2),
                BLOCK=128, num_warps=4
            )
            y0_masked = torch.empty_like(y0_relu)
            mask_apply_triton[(N, 192, triton.cdiv(T, 128))](
                y0_relu, x_mask, y0_masked, N, 192, T,
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                y0_masked.stride(0), y0_masked.stride(1), y0_masked.stride(2),
                BLOCK=128, num_warps=4
            )

            # conv1: [N, 192, T]
            y1 = torch.empty((N, 192, T), dtype=x.dtype, device=x.device)
            grid1 = (N, 192, triton.cdiv(T, 128))
            conv1d_triton[grid1](
                y0_masked, w1, b1, y1, N, 192, 192, T, 5, 2,
                y0_masked.stride(0), y0_masked.stride(1), y0_masked.stride(2),
                w1.stride(0), w1.stride(1), w1.stride(2),
                y1.stride(0), y1.stride(1), y1.stride(2),
                BLOCK_T=128, num_warps=4
            )

            # ReLU and mask over full [N, 192, T]
            y1_relu = torch.empty_like(y1)
            relu_triton[grid1](
                y1, y1_relu, N, 192, T,
                y1.stride(0), y1.stride(1), y1.stride(2),
                y1_relu.stride(0), y1_relu.stride(1), y1_relu.stride(2),
                BLOCK=128, num_warps=4
            )
            y1_masked = torch.empty_like(y1_relu)
            mask_apply_triton[(N, 192, triton.cdiv(T, 128))](
                y1_relu, x_mask, y1_masked, N, 192, T,
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                y1_masked.stride(0), y1_masked.stride(1), y1_masked.stride(2),
                BLOCK=128, num_warps=4
            )

            # conv2: [N, 96, T]
            h = torch.empty((N, 96, T), dtype=x.dtype, device=x.device)
            grid2 = (N, 96, triton.cdiv(T, 128))
            conv1d_triton[grid2](
                y1_masked, w2, b2, h, N, 192, 96, T, 5, 2,
                y1_masked.stride(0), y1_masked.stride(1), y1_masked.stride(2),
                w2.stride(0), w2.stride(1), w2.stride(2),
                h.stride(0), h.stride(1), h.stride(2),
                BLOCK_T=128, num_warps=4
            )

            # Mask h (broadcast)
            h_masked = torch.empty_like(h)
            mask_apply_triton[(N, 96, triton.cdiv(T, 128))](
                h, x_mask, h_masked, N, 96, T,
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                BLOCK=128, num_warps=4
            )

            # Affine coupling: update x1
            out_x0 = out_final[:, :half_channels, :].contiguous()
            add_to_slice_triton[(N, half_channels, triton.cdiv(T, 128))](
                x0, x0, out_x0, N, 96, 96, T,
                x0.stride(0), x0.stride(1), x0.stride(2),
                x0.stride(0), x0.stride(1), x0.stride(2),
                out_x0.stride(0), out_x0.stride(1), out_x0.stride(2),
                BLOCK=128, num_warps=4
            )

            out_x1 = out_final[:, half_channels:, :].contiguous()
            # Subtract h since we're in reverse mode
            # We implement x1_minus_h = x1 - h by loading both and subtracting in kernel.
            # However, the add_to_slice_triton currently adds, not subtracts. To subtract,
            # we can pass add_ptr as x1 but note that Triton will add, so instead we compute:
            # We need a Triton kernel that does subtraction. Modify grid to subtract.
            # Create a temporary tensor for x1_minus_h.
            x1_minus_h = torch.empty_like(x1)
            # Launch a simple elementwise subtraction kernel if needed. For simplicity, we can do:
            # x1_minus_h = x1 - h_masked. But Triton kernel add_to_slice_triton adds, so we cannot reuse it.
            # Implement elementwise subtraction via Triton: we'll write our own.
            # Define a subtract kernel (minor addition below).
            pass

    # Note: The above 'pass' is a placeholder. In practice, we should implement elementwise subtraction via Triton.
    # To keep correctness, we can implement a simple elementwise Triton kernel for subtraction:
    # Implement elementwise subtraction of h_masked from x1 and write to out_x1.

    # Elementwise subtract Triton kernel
    @triton.jit
    def sub_triton(in_ptr, sub_ptr, out_ptr, N, C, T, in_sN, in_sC, in_sT, sub_sN, sub_sC, sub_sT, out_sN, out_sC, out_sT, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        total = N * C * T
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total

        CT = C * T
        n = offs // CT
        rem = offs % CT
        c = rem // T
        t = rem % T

        in_offs = n * in_sN + c * in_sC + t * in_sT
        a = tl.load(in_ptr + in_offs, mask=mask, other=0.0)
        sub_offs = n * sub_sN + c * sub_sC + t * sub_sT
        b = tl.load(sub_ptr + sub_offs, mask=mask, other=0.0)
        c = a - b
        out_offs = n * out_sN + c * out_sC + t * out_sT
        tl.store(out_ptr + out_offs, c, mask=mask)

    # Now perform x1_minus_h = x1 - h_masked
    x1_minus_h = torch.empty_like(x1)
    grid_sub = (N, 96, triton.cdiv(T, 128))
    sub_triton[grid_sub](
        x1, h_masked, x1_minus_h, N, 96, T,
        x1.stride(0), x1.stride(1), x1.stride(2),
        h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
        x1_minus_h.stride(0), x1_minus_h.stride(1), x1_minus_h.stride(2),
        BLOCK=128, num_warps=4
    )

    # Write results into out_final slices
    out_x0 = out_final[:, :half_channels, :].contiguous()
    add_to_slice_triton[(N, half_channels, triton.cdiv(T, 128))](
        x0, x0, out_x0, N, 96, 96, T,
        x0.stride(0), x0.stride(1), x0.stride(2),
        x0.stride(0), x0.stride(1), x0.stride(2),
        out_x0.stride(0), out_x0.stride(1), out_x0.stride(2),
        BLOCK=128, num_warps=4
    )

    out_x1 = out_final[:, half_channels:, :].contiguous()
    add_to_slice_triton[(N, 96, triton.cdiv(T, 128))](
        x1_minus_h, x1_minus_h, out_x1, N, 96, 96, T,
        x1_minus_h.stride(0), x1_minus_h.stride(1), x1_minus_h.stride(2),
        x1_minus_h.stride(0), x1_minus_h.stride(1), x1_minus_h.stride(2),
        out_x1.stride(0), out_x1.stride(1), out_x1.stride(2),
        BLOCK=128, num_warps=4
    )

    # Update x for the next transform
    x = out_final

    return x


class ModelNew(nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
