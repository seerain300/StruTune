import math
import torch
import torch.nn.functional as F

# Triton is required. We import triton and define kernels.
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels: all numerical computation must be here.

if TRITON_AVAILABLE:
    @triton.jit
    def conv1d_forward_kernel(
        x_ptr,         # *float32, [N, C_IN, T_IN]
        w_ptr,         # *float32, [C_OUT, C_IN, K]
        b_ptr,         # *float32, [C_OUT]
        y_ptr,         # *float32, [N, C_OUT, T_OUT]
        N, T_IN, T_OUT, C_IN: tl.constexpr, C_OUT: tl.constexpr, K: tl.constexpr, PAD: tl.constexpr,
        x_stride_n, x_stride_c, x_stride_t,
        w_stride_co, w_stride_ci, w_stride_k,
        y_stride_n, y_stride_c, y_stride_t,
        t_block_start: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        # program ids
        pid_n = tl.program_id(0)
        pid_co = tl.program_id(1)
        pid_tb = tl.program_id(2)

        # time offsets this program computes
        t_offsets = t_block_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T_OUT

        # accumulator for this (n, co, t_block)
        acc = tl.zeros([BLOCK_T], dtype=tl.float32)

        # loop over input channels and kernel taps
        for ci in range(0, C_IN):
            for k in range(0, K):
                t_in = t_offsets + k - PAD
                valid = (t_in >= 0) & (t_in < T_IN) & t_mask
                # load x[n, ci, t_in] with mask
                x_ptrs = x_ptr + pid_n * x_stride_n + ci * x_stride_c + t_in * x_stride_t
                x_vals = tl.load(x_ptrs, mask=valid, other=0.0)
                # load weight w[co, ci, k]
                w_ptrs = w_ptr + pid_co * w_stride_co + ci * w_stride_ci + k * w_stride_k
                w_val = tl.load(w_ptrs)
                acc += x_vals * w_val

        # add bias for this output channel
        b_val = tl.load(b_ptr + pid_co)
        acc += b_val

        # store y[n, co, t_offsets]
        y_ptrs = y_ptr + pid_n * y_stride_n + pid_co * y_stride_c + t_offsets * y_stride_t
        tl.store(y_ptrs, acc, mask=t_mask)

    @triton.jit
    def relu_forward_kernel(
        x_ptr, y_ptr, N, C, T,
        x_stride_n, x_stride_c, x_stride_t,
        y_stride_n, y_stride_c, y_stride_t,
        BLOCK_T: tl.constexpr = 128,
    ):
        pid_nc = tl.program_id(0)  # over N*C
        pid_t  = tl.program_id(1)
        n = pid_nc // C
        c = pid_nc % C

        t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T

        x_ptrs = x_ptr + n * x_stride_n + c * x_stride_c + t_offsets * x_stride_t
        y_ptrs = y_ptr + n * y_stride_n + c * y_stride_c + t_offsets * y_stride_t

        x_vals = tl.load(x_ptrs, mask=t_mask, other=0.0)
        y_vals = tl.maximum(x_vals, 0.0)
        tl.store(y_ptrs, y_vals, mask=t_mask)

    @triton.jit
    def concat_half_channels_kernel_fixed(
        x0_ptr, x1_ptr, y_ptr,
        N, C_HALF, T,
        x0_stride_n, x0_stride_c, x0_stride_t,
        x1_stride_n, x1_stride_c, x1_stride_t,
        y_stride_n, y_stride_c, y_stride_t,
        c_block_start: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        # grid: (N, C_HALF, ceil(T / BLOCK_T))
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        c = c_block_start + pid_c
        if c >= C_HALF:
            return

        t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T

        x0_ptrs = x0_ptr + pid_n * x0_stride_n + c * x0_stride_c + t_offsets * x0_stride_t
        x1_ptrs = x1_ptr + pid_n * x1_stride_n + c * x1_stride_c + t_offsets * x1_stride_t

        y0_ptrs = y_ptr + pid_n * y_stride_n + c * y_stride_c + t_offsets * y_stride_t
        y1_ptrs = y_ptr + pid_n * y_stride_n + (c + C_HALF) * y_stride_c + t_offsets * y_stride_t

        x0_vals = tl.load(x0_ptrs, mask=t_mask, other=0.0)
        x1_vals = tl.load(x1_ptrs, mask=t_mask, other=0.0)
        tl.store(y0_ptrs, x0_vals, mask=t_mask)
        tl.store(y1_ptrs, x1_vals, mask=t_mask)

    @triton.jit
    def affine_add_kernel(
        x1_ptr, h_ptr, y_ptr,
        N, C, T,
        x1_stride_n, x1_stride_c, x1_stride_t,
        h_stride_n, h_stride_c, h_stride_t,
        y_stride_n, y_stride_c, y_stride_t,
        BLOCK_T: tl.constexpr,
    ):
        pid_nc = tl.program_id(0)  # over N*C
        pid_t  = tl.program_id(1)
        n = pid_nc // C
        c = pid_nc % C

        t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T

        x1_ptrs = x1_ptr + n * x1_stride_n + c * x1_stride_c + t_offsets * x1_stride_t
        h_ptrs  = h_ptr  + n * h_stride_n  + c * h_stride_c  + t_offsets * h_stride_t
        y_ptrs  = y_ptr  + n * y_stride_n  + c * y_stride_c  + t_offsets * y_stride_t

        x1_vals = tl.load(x1_ptrs, mask=t_mask, other=0.0)
        h_vals  = tl.load(h_ptrs,  mask=t_mask, other=0.0)
        y_vals  = x1_vals + h_vals
        tl.store(y_ptrs,  y_vals, mask=t_mask)

    @triton.jit
    def affine_sub_kernel(
        x1_ptr, h_ptr, y_ptr,
        N, C, T,
        x1_stride_n, x1_stride_c, x1_stride_t,
        h_stride_n, h_stride_c, h_stride_t,
        y_stride_n, y_stride_c, y_stride_t,
        BLOCK_T: tl.constexpr,
    ):
        pid_nc = tl.program_id(0)  # over N*C
        pid_t  = tl.program_id(1)
        n = pid_nc // C
        c = pid_nc % C

        t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T

        x1_ptrs = x1_ptr + n * x1_stride_n + c * x1_stride_c + t_offsets * x1_stride_t
        h_ptrs  = h_ptr  + n * h_stride_n  + c * h_stride_c  + t_offsets * h_stride_t
        y_ptrs  = y_ptr  + n * y_stride_n  + c * y_stride_c  + t_offsets * y_stride_t

        x1_vals = tl.load(x1_ptrs, mask=t_mask, other=0.0)
        h_vals  = tl.load(h_ptrs,  mask=t_mask, other=0.0)
        y_vals  = x1_vals - h_vals
        tl.store(y_ptrs,  y_vals, mask=t_mask)

    @triton.jit
    def mask_mul_kernel(
        x_ptr, mask_ptr, y_ptr,
        N, C, T,
        x_stride_n, x_stride_c, x_stride_t,
        mask_stride_n, mask_stride_c, mask_stride_t,
        y_stride_n, y_stride_c, y_stride_t,
        BLOCK_T: tl.constexpr,
    ):
        pid_nc = tl.program_id(0)  # over N*C
        pid_t  = tl.program_id(1)
        n = pid_nc // C
        c = pid_nc % C

        t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T

        x_ptrs  = x_ptr    + n * x_stride_n    + c * x_stride_c    + t_offsets * x_stride_t
        mask_ptrs = mask_ptr + n * mask_stride_n + c * mask_stride_c + t_offsets * mask_stride_t
        y_ptrs  = y_ptr    + n * y_stride_n    + c * y_stride_c    + t_offsets * y_stride_t

        x_vals  = tl.load(x_ptrs,    mask=t_mask, other=0.0)
        mask_vals = tl.load(mask_ptrs, mask=t_mask, other=0.0)
        y_vals = x_vals * mask_vals
        tl.store(y_ptrs, y_vals, mask=t_mask)


# We will implement the Triton-only run as ModelNew.forward. It mirrors the original signature.
def run_triton(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    reverse: bool,
    # first transform
    transform_0_conv0_weight: torch.Tensor,
    transform_0_conv0_bias: torch.Tensor,
    transform_0_conv1_weight: torch.Tensor,
    transform_0_conv1_bias: torch.Tensor,
    transform_0_conv2_weight: torch.Tensor,
    transform_0_conv2_bias: torch.Tensor,
    # second transform
    transform_1_conv0_weight: torch.Tensor,
    transform_1_conv0_bias: torch.Tensor,
    transform_1_conv1_weight: torch.Tensor,
    transform_1_conv1_bias: torch.Tensor,
    transform_1_conv2_weight: torch.Tensor,
    transform_1_conv2_bias: torch.Tensor,
    # third transform
    transform_2_conv0_weight: torch.Tensor,
    transform_2_conv0_bias: torch.Tensor,
    transform_2_conv1_weight: torch.Tensor,
    transform_2_conv1_bias: torch.Tensor,
    transform_2_conv2_weight: torch.Tensor,
    transform_2_conv2_bias: torch.Tensor,
    # fourth transform
    transform_3_conv0_weight: torch.Tensor,
    transform_3_conv0_bias: torch.Tensor,
    transform_3_conv1_weight: torch.Tensor,
    transform_3_conv1_bias: torch.Tensor,
    transform_3_conv2_weight: torch.Tensor,
    transform_3_conv2_bias: torch.Tensor,
):
    """
    Triton-only implementation of the original run function:
    - Forward: x1 = x1 + transform(x0) for each layer
    - Reverse: x1 = x1 - transform(x0) for each layer in reverse order
    """
    assert TRITON_AVAILABLE, "Triton is not available. Please run on a CUDA device with Triton installed."

    batch_size = x.shape[0]
    channels = x.shape[1]
    time = x.shape[2]

    half_channels = channels // 2

    # Utility: launch a conv1d Triton kernel and return the result (no torch ops).
    def triton_conv1d(x_nchw, w_cico, b_co, T_out):
        # x_nchw: [N, C_IN, T_IN], w_cico: [C_OUT, C_IN, K], b_co: [C_OUT]
        N, C_IN, T_IN = x_nchw.shape
        C_OUT = w_cico.shape[0]
        K = w_cico.shape[2]
        PAD = K // 2

        # Allocate output
        y = torch.empty((N, C_OUT, T_out), dtype=x_nchw.dtype, device=x_nchw.device)

        # Strides
        x_stride_n, x_stride_c, x_stride_t = x_nchw.stride()
        w_stride_co, w_stride_ci, w_stride_k = w_cico.stride()
        y_stride_n, y_stride_c, y_stride_t = y.stride()

        # Launch grid: (N, C_OUT, ceil(T_out / BLOCK_T))
        BLOCK_T = 128
        grid = (N, C_OUT, triton.cdiv(T_out, BLOCK_T))
        conv1d_forward_kernel[grid](
            x_nchw, w_cico, b_co, y,
            N, T_IN, T_out, C_IN, C_OUT, K, PAD,
            x_stride_n, x_stride_c, x_stride_t,
            w_stride_co, w_stride_ci, w_stride_k,
            y_stride_n, y_stride_c, y_stride_t,
            0, BLOCK_T,
            num_warps=4, num_stages=2,
        )
        return y

    # Utility: elementwise ReLU via Triton
    def triton_relu(inp):
        out = torch.empty_like(inp)
        N, C, T = inp.shape
        x_stride_n, x_stride_c, x_stride_t = inp.stride()
        y_stride_n, y_stride_c, y_stride_t = out.stride()
        BLOCK_T = 128
        grid = (N * C, triton.cdiv(T, BLOCK_T))
        relu_forward_kernel[grid](
            inp, out, N, C, T,
            x_stride_n, x_stride_c, x_stride_t,
            y_stride_n, y_stride_c, y_stride_t,
            BLOCK_T,
            num_warps=4, num_stages=2,
        )
        return out

    # Utility: concat two halves along channel dimension via Triton
    def triton_concat_half_channels(x0, x1):
        N, C_HALF, T = x0.shape
        out = torch.empty((N, 2 * C_HALF, T), dtype=x0.dtype, device=x0.device)
        x0_stride_n, x0_stride_c, x0_stride_t = x0.stride()
        x1_stride_n, x1_stride_c, x1_stride_t = x1.stride()
        y_stride_n, y_stride_c, y_stride_t = out.stride()
        BLOCK_T = 128
        grid = (N, C_HALF, triton.cdiv(T, BLOCK_T))
        concat_half_channels_kernel_fixed[grid](
            x0, x1, out,
            N, C_HALF, T,
            x0_stride_n, x0_stride_c, x0_stride_t,
            x1_stride_n, x1_stride_c, x1_stride_t,
            y_stride_n, y_stride_c, y_stride_t,
            0, BLOCK_T,
            num_warps=4, num_stages=2,
        )
        return out

    # Utility: elementwise affine add/sub via Triton
    def triton_affine_add(x1, h):
        out = torch.empty_like(x1)
        N, C, T = x1.shape
        x1_stride_n, x1_stride_c, x1_stride_t = x1.stride()
        h_stride_n, h_stride_c, h_stride_t = h.stride()
        y_stride_n, y_stride_c, y_stride_t = out.stride()
        BLOCK_T = 128
        grid = (N * C, triton.cdiv(T, BLOCK_T))
        affine_add_kernel[grid](
            x1, h, out,
            N, C, T,
            x1_stride_n, x1_stride_c, x1_stride_t,
            h_stride_n, h_stride_c, h_stride_t,
            y_stride_n, y_stride_c, y_stride_t,
            BLOCK_T,
            num_warps=4, num_stages=2,
        )
        return out

    def triton_affine_sub(x1, h):
        out = torch.empty_like(x1)
        N, C, T = x1.shape
        x1_stride_n, x1_stride_c, x1_stride_t = x1.stride()
        h_stride_n, h_stride_c, h_stride_t = h.stride()
        y_stride_n, y_stride_c, y_stride_t = out.stride()
        BLOCK_T = 128
        grid = (N * C, triton.cdiv(T, BLOCK_T))
        affine_sub_kernel[grid](
            x1, h, out,
            N, C, T,
            x1_stride_n, x1_stride_c, x1_stride_t,
            h_stride_n, h_stride_c, h_stride_t,
            y_stride_n, y_stride_c, y_stride_t,
            BLOCK_T,
            num_warps=4, num_stages=2,
        )
        return out

    # Utility: elementwise mask multiply via Triton
    def triton_mask_mul(x, mask):
        out = torch.empty_like(x)
        N, C, T = x.shape
        x_stride_n, x_stride_c, x_stride_t = x.stride()
        mask_stride_n, mask_stride_c, mask_stride_t = mask.stride()
        y_stride_n, y_stride_c, y_stride_t = out.stride()
        BLOCK_T = 128
        grid = (N * C, triton.cdiv(T, BLOCK_T))
        mask_mul_kernel[grid](
            x, mask, out,
            N, C, T,
            x_stride_n, x_stride_c, x_stride_t,
            mask_stride_n, mask_stride_c, mask_stride_t,
            y_stride_n, y_stride_c, y_stride_t,
            BLOCK_T,
            num_warps=4, num_stages=2,
        )
        return out

    # List of transforms (each is a tuple of weights/biases for conv0, conv1, conv2)
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

    x = x.contiguous()
    x_mask = x_mask.contiguous()

    if not reverse:
        # Forward pass: apply transformations sequentially
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            # Split into two halves along channel dimension
            x0 = x[:, :half_channels, :]
            x1 = x[:, half_channels:, :]

            # Compute transformation conditioned on x0 using Triton convs
            # Each conv uses padding = K//2
            h = triton_conv1d(x0, conv0_w, conv0_b, time)
            h = triton_relu(h)
            h = triton_conv1d(h, conv1_w, conv1_b, time)
            h = triton_relu(h)
            h = triton_conv1d(h, conv2_w, conv2_b, time)

            # Apply mask (mask is ones in given setup, but keep generic)
            h = triton_mask_mul(h, x_mask)

            # Affine coupling: x1 = x1 + h
            x1 = triton_affine_add(x1, h)

            # Concatenate back along channel dimension
            x = triton_concat_half_channels(x0, x1)

            # Apply mask to output
            x = triton_mask_mul(x, x_mask)
    else:
        # Reverse pass: apply transformations in reverse order
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
            # Split into two halves along channel dimension
            x0 = x[:, :half_channels, :]
            x1 = x[:, half_channels:, :]

            # Compute transformation conditioned on x0 using Triton convs
            h = triton_conv1d(x0, conv0_w, conv0_b, time)
            h = triton_relu(h)
            h = triton_conv1d(h, conv1_w, conv1_b, time)
            h = triton_relu(h)
            h = triton_conv1d(h, conv2_w, conv2_b, time)

            # Apply mask
            h = triton_mask_mul(h, x_mask)

            # Inverse affine coupling: x1 = x1 - h
            x1 = triton_affine_sub(x1, h)

            # Concatenate back
            x = triton_concat_half_channels(x0, x1)

            # Apply mask to output
            x = triton_mask_mul(x, x_mask)

    return x


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The harness will pass all required tensors. We mirror the original run signature.
        return run_triton(*args)


def run(*args):
    return ModelNew()(*args)
