import math
import torch
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
    # Each program handles one (n, co) pair and computes all output time positions
    n = tl.program_id(0)
    co = tl.program_id(1)
    # Bias for this output channel
    b_val = tl.load(b_ptr + co)

    # Iterate over output time positions
    # We'll process in chunks of BLOCK_L
    for lo_base in range(0, L_out, BLOCK_L):
        # Vector of output time indices for this chunk
        lo_offsets = lo_base + tl.arange(0, BLOCK_L)
        mask_out = lo_offsets < L_out

        # Accumulator in fp32
        acc = tl.full((BLOCK_L,), b_val, tl.float32)

        # Loop over input channels and kernel taps
        # Note: K is small (5), loop is fine
        for ci in range(0, C_in):
            for k in range(0, K):
                # Compute corresponding input positions with padding P = K//2
                P = K // 2
                li = lo_offsets + P - k  # scalar vector
                in_bounds = (li >= 0) & (li < L_in) & mask_out

                # Load x[n, ci, li]
                x_ptrs = x_ptr + n * stride_x_n + ci * stride_x_c + li * stride_x_l
                # Cast to fp32 for accumulation
                x_vals = tl.load(x_ptrs, mask=in_bounds, other=0.0).to(tl.float32)

                # Load weight w[co, ci, k]
                w_ptrs = w_ptr + co * stride_w_co + ci * stride_w_ci + k * stride_w_k
                w_val = tl.load(w_ptrs).to(tl.float32)

                # FMA accumulate
                acc += x_vals * w_val

        # Store results to y[n, co, lo]
        y_ptrs = y_ptr + n * stride_y_n + co * stride_y_c + lo_offsets * stride_y_l
        # Cast back to output dtype (assume float32 for outputs here; inputs are float32 in this benchmark)
        tl.store(y_ptrs, acc, mask=mask_out)


@triton.jit
def relu_kernel(
    x_ptr, y_ptr,
    N, C, L,
    stride_x_n, stride_x_c, stride_x_l,
    stride_y_n, stride_y_c, stride_y_l,
    BLOCK_L: tl.constexpr,
):
    # Grid over (N, C) and process time in chunks
    n = tl.program_id(0)
    c = tl.program_id(1)
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
    stride_y_full_n, stride_y_full_c, stride_y_full_l,
    stride_y0_n, stride_y0_c, stride_y0_l,
    stride_y1_n, stride_y1_c, stride_y1_l,
    BLOCK_L: tl.constexpr,
):
    # y_full shape [N, 2*C_half, L]; write first half to y0, second half to y1
    n = tl.program_id(0)
    ch = tl.program_id(1)  # in [0, C_half)
    for lo_base in range(0, L, BLOCK_L):
        lo_offsets = lo_base + tl.arange(0, BLOCK_L)
        mask = lo_offsets < L

        full_ptrs = y_full_ptr + n * stride_y_full_n + ch * stride_y_full_c + lo_offsets * stride_y_full_l
        out0_ptrs = y0_ptr + n * stride_y0_n + ch * stride_y0_c + lo_offsets * stride_y0_l
        out1_ptrs = y1_ptr + n * stride_y1_n + ch * stride_y1_c + lo_offsets * stride_y1_l

        vals = tl.load(full_ptrs, mask=mask, other=0.0)
        tl.store(out0_ptrs, vals, mask=mask)
        tl.store(out1_ptrs, vals, mask=mask)


@triton.jit
def concat_halves_forward(
    y0_ptr, y1_ptr, y_full_ptr,
    N, C_half, L,
    stride_y0_n, stride_y0_c, stride_y0_l,
    stride_y1_n, stride_y1_c, stride_y1_l,
    stride_y_full_n, stride_y_full_c, stride_y_full_l,
    BLOCK_L: tl.constexpr,
):
    # y0: [N, C_half, L], y1: [N, C_half, L], y_full: [N, 2*C_half, L]
    n = tl.program_id(0)
    ch = tl.program_id(1)  # in [0, C_half)
    for lo_base in range(0, L, BLOCK_L):
        lo_offsets = lo_base + tl.arange(0, BLOCK_L)
        mask = lo_offsets < L

        in0_ptrs = y0_ptr + n * stride_y0_n + ch * stride_y0_c + lo_offsets * stride_y0_l
        in1_ptrs = y1_ptr + n * stride_y1_n + ch * stride_y1_c + lo_offsets * stride_y1_l
        out0_ptrs = y_full_ptr + n * stride_y_full_n + ch * stride_y_full_c + lo_offsets * stride_y_full_l
        out1_ptrs = y_full_ptr + n * stride_y_full_n + (ch + C_half) * stride_y_full_c + lo_offsets * stride_y_full_l

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
        # mask is [N, 1, L], so stride_mask_c == L for channel dimension; we index c=0
        mask_ptrs = mask_ptr + n * stride_mask_n + 0 * stride_mask_c + lo_offsets * stride_mask_l

        y_vals = tl.load(y_ptrs, mask=mask_l, other=0.0)
        m_vals = tl.load(mask_ptrs, mask=mask_l, other=1.0)
        out_vals = y_vals * m_vals
        out_ptrs = out_ptr + n * stride_y_n + c * stride_y_c + lo_offsets * stride_y_l  # same layout as y
        tl.store(out_ptrs, out_vals, mask=mask_l)


def triton_conv1d(x, w, b, out_shape):
    """
    x: [N, C_in, L_in], w: [C_out, C_in, K], b: [C_out]
    Returns y: [out_shape] = [N, C_out, L_out], where L_out = L_in - K + 1
    All tensors must be contiguous.
    """
    assert x.is_cuda and w.is_cuda and b.is_cuda, "Triton conv1d requires CUDA tensors"
    N, C_in, L_in = x.shape
    C_out = w.shape[0]
    K = w.shape[2]
    # PyTorch conv1d padding=K//2, output length = L_in - K + 1 (assuming no padding effect change)
    L_out = L_in - K + 1

    y = torch.empty(out_shape, device=x.device, dtype=torch.float32)  # accumulate/store in float32
    grid = (N, C_out)
    conv1d_forward_kernel[grid](
        x, w, b, y,
        N, C_in, C_out, L_in, L_out, K,
        x.stride(0), x.stride(1), x.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_L=128,
        num_warps=4,
    )
    return y  # float32


def split_halves_forward_launch(y_full, y0, y1):
    N = y_full.shape[0]
    C_half = y0.shape[1]
    L = y_full.shape[2]
    grid = (N, C_half)
    split_halves_forward[grid](
        y_full, y0, y1,
        N, C_half, L,
        y_full.stride(0), y_full.stride(1), y_full.stride(2),
        y0.stride(0), y0.stride(1), y0.stride(2),
        y1.stride(0), y1.stride(1), y1.stride(2),
        BLOCK_L=128,
        num_warps=2,
    )


def concat_halves_forward_launch(y0, y1, y_full):
    N = y0.shape[0]
    C_half = y0.shape[1]
    L = y0.shape[2]
    grid = (N, C_half)
    concat_halves_forward[grid](
        y0, y1, y_full,
        N, C_half, L,
        y0.stride(0), y0.stride(1), y0.stride(2),
        y1.stride(0), y1.stride(1), y1.stride(2),
        y_full.stride(0), y_full.stride(1), y_full.stride(2),
        BLOCK_L=128,
        num_warps=2,
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
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
        Triton-only forward: implements the original logic using Triton kernels.
        """
        # Ensure device is CUDA for Triton
        assert x.is_cuda, "Input x must be on CUDA device for Triton kernels"

        N = x.shape[0]
        C = x.shape[1]
        half_channels = C // 2
        L = x.shape[2]

        # Prepare tensors (assume float32 throughout for correctness)
        x = x.contiguous()
        x_mask = x_mask.contiguous()

        # List of transforms' weights/biases
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
                x0 = x[:, :half_channels, :].contiguous()
                x1 = x[:, half_channels:, :].contiguous()

                # conv0: x0 -> h0
                h0 = triton_conv1d(x0, conv0_w, conv0_b, (N, conv0_w.shape[0], L))
                # ReLU in Triton
                h0_relu = torch.empty_like(h0)
                relu_kernel[(N, h0.shape[1], L)](
                    h0, h0_relu,
                    h0.stride(0), h0.shape(1), h0.shape(2),
                    h0.stride(0), h0.shape(1), h0.shape(2),
                    BLOCK_L=128,
                    num_warps=1,
                )
                h0 = h0_relu

                # conv1: h0 -> h1
                h1 = triton_conv1d(h0, conv1_w, conv1_b, (N, conv1_w.shape[0], L))
                # ReLU in Triton
                h1_relu = torch.empty_like(h1)
                relu_kernel[(N, h1.shape[1], L)](
                    h1, h1_relu,
                    h1.stride(0), h1.shape(1), h1.shape(2),
                    h1.stride(0), h1.shape(1), h1.shape(2),
                    BLOCK_L=128,
                    num_warps=1,
                )
                h1 = h1_relu

                # conv2: h1 -> h2 (no ReLU)
                h2 = triton_conv1d(h1, conv2_w, conv2_b, (N, conv2_w.shape[0], L))

                # Affine coupling: x1 = x1 + h2
                x1 = x1 + h2

                # Concatenate back
                x_full = torch.empty((N, 2 * half_channels, L), device=x.device, dtype=torch.float32)
                concat_halves_forward_launch(x0, x1, x_full)

                # Update x
                x = x_full

                # Apply mask (broadcast over channel)
                out = torch.empty_like(x, dtype=torch.float32)
                mul_mask_kernel[(N, x.shape[1], L)](
                    x, x_mask, out,
                    N, x.shape[1], L,
                    x.stride(0), x.stride(1), x.stride(2),
                    x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                    BLOCK_L=128,
                    num_warps=1,
                )
                x = out
        else:
            # Reverse pass: apply transformations in reverse order (no ReLU in conv2; ReLU after conv0 and conv1)
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
                # Split into halves
                x0 = x[:, :half_channels, :].contiguous()
                x1 = x[:, half_channels:, :].contiguous()

                # conv0: x0 -> h0
                h0 = triton_conv1d(x0, conv0_w, conv0_b, (N, conv0_w.shape[0], L))
                # ReLU in Triton
                h0_relu = torch.empty_like(h0)
                relu_kernel[(N, h0.shape[1], L)](
                    h0, h0_relu,
                    h0.stride(0), h0.shape(1), h0.shape(2),
                    h0.stride(0), h0.shape(1), h0.shape(2),
                    BLOCK_L=128,
                    num_warps=1,
                )
                h0 = h0_relu

                # conv1: h0 -> h1
                h1 = triton_conv1d(h0, conv1_w, conv1_b, (N, conv1_w.shape[0], L))
                # ReLU in Triton
                h1_relu = torch.empty_like(h1)
                relu_kernel[(N, h1.shape[1], L)](
                    h1, h1_relu,
                    h1.stride(0), h1.shape(1), h1.shape(2),
                    h1.stride(0), h1.shape(1), h1.shape(2),
                    BLOCK_L=128,
                    num_warps=1,
                )
                h1 = h1_relu

                # conv2: h1 -> h2 (no ReLU)
                h2 = triton_conv1d(h1, conv2_w, conv2_b, (N, conv2_w.shape[0], L))

                # Inverse affine coupling: x1 = x1 - h2
                x1 = x1 - h2

                # Concatenate back
                x_full = torch.empty((N, 2 * half_channels, L), device=x.device, dtype=torch.float32)
                concat_halves_forward_launch(x0, x1, x_full)

                # Update x
                x = x_full

                # Apply mask (broadcast over channel)
                out = torch.empty_like(x, dtype=torch.float32)
                mul_mask_kernel[(N, x.shape[1], L)](
                    x, x_mask, out,
                    N, x.shape[1], L,
                    x.stride(0), x.stride(1), x.stride(2),
                    x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                    BLOCK_L=128,
                    num_warps=1,
                )
                x = out

        return x


def run(*args):
    return ModelNew()(*args)
