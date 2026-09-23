import math
import torch

# Triton import
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Triton kernels: all computations must be in Triton
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
        x_ptr, y_ptr,
        N, C, T,
        x_stride_n, x_stride_c, x_stride_t,
        y_stride_n, y_stride_c, y_stride_t,
        BLOCK_T: tl.constexpr = 128,
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        # compute time offsets
        t_block_start = pid_t * BLOCK_T
        t_offsets = t_block_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T

        # load x
        x_ptrs = x_ptr + pid_n * x_stride_n + pid_c * x_stride_c + t_offsets * x_stride_t
        x_vals = tl.load(x_ptrs, mask=t_mask, other=0.0)
        # ReLU
        y_vals = tl.maximum(x_vals, 0.0)
        # store
        y_ptrs = y_ptr + pid_n * y_stride_n + pid_c * y_stride_c + t_offsets * y_stride_t
        tl.store(y_ptrs, y_vals, mask=t_mask)

    @triton.jit
    def concat_half_channels_kernel_fixed(
        x0_ptr, x1_ptr, y_ptr,
        N, C_HALF, T,
        x0_stride_n, x0_stride_c, x0_stride_t,
        x1_stride_n, x1_stride_c, x1_stride_t,
        y_stride_n, y_stride_c, y_stride_t,
        half_first: tl.constexpr,  # True: y[:, :C_HALF, :] = x0; False: y[:, C_HALF:, :] = x1
        BLOCK_T: tl.constexpr = 128,
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        # compute time offsets
        t_block_start = pid_t * BLOCK_T
        t_offsets = t_block_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T

        if half_first:
            src_ptrs = x0_ptr + pid_n * x0_stride_n + pid_c * x0_stride_c + t_offsets * x0_stride_t
            dst_ptrs = y_ptr + pid_n * y_stride_n + pid_c * y_stride_c + t_offsets * y_stride_t
        else:
            src_ptrs = x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + t_offsets * x1_stride_t
            dst_ptrs = y_ptr + pid_n * y_stride_n + (pid_c + C_HALF) * y_stride_c + t_offsets * y_stride_t

        vals = tl.load(src_ptrs, mask=t_mask, other=0.0)
        tl.store(dst_ptrs, vals, mask=t_mask)

    @triton.jit
    def affine_add_kernel(
        x1_ptr, h_ptr, y_ptr,
        N, C_HALF, T,
        x1_stride_n, x1_stride_c, x1_stride_t,
        h_stride_n, h_stride_c, h_stride_t,
        y_stride_n, y_stride_c, y_stride_t,
        BLOCK_T: tl.constexpr = 128,
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        t_block_start = pid_t * BLOCK_T
        t_offsets = t_block_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T

        x1_ptrs = x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + t_offsets * x1_stride_t
        h_ptrs = h_ptr + pid_n * h_stride_n + pid_c * h_stride_c + t_offsets * h_stride_t
        y_ptrs = y_ptr + pid_n * y_stride_n + pid_c * y_stride_c + t_offsets * y_stride_t

        x1_vals = tl.load(x1_ptrs, mask=t_mask, other=0.0)
        h_vals = tl.load(h_ptrs, mask=t_mask, other=0.0)
        y_vals = x1_vals + h_vals
        tl.store(y_ptrs, y_vals, mask=t_mask)

    @triton.jit
    def affine_sub_kernel(
        x1_ptr, h_ptr, y_ptr,
        N, C_HALF, T,
        x1_stride_n, x1_stride_c, x1_stride_t,
        h_stride_n, h_stride_c, h_stride_t,
        y_stride_n, y_stride_c, y_stride_t,
        BLOCK_T: tl.constexpr = 128,
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        t_block_start = pid_t * BLOCK_T
        t_offsets = t_block_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T

        x1_ptrs = x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + t_offsets * x1_stride_t
        h_ptrs = h_ptr + pid_n * h_stride_n + pid_c * h_stride_c + t_offsets * h_stride_t
        y_ptrs = y_ptr + pid_n * y_stride_n + pid_c * y_stride_c + t_offsets * y_stride_t

        x1_vals = tl.load(x1_ptrs, mask=t_mask, other=0.0)
        h_vals = tl.load(h_ptrs, mask=t_mask, other=0.0)
        y_vals = x1_vals - h_vals
        tl.store(y_ptrs, y_vals, mask=t_mask)

    @triton.jit
    def mask_mul_kernel(
        x_ptr, mask_ptr, y_ptr,
        N, C_HALF, T,
        x_stride_n, x_stride_c, x_stride_t,
        mask_stride_n, mask_stride_c, mask_stride_t,
        y_stride_n, y_stride_c, y_stride_t,
        BLOCK_T: tl.constexpr = 128,
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        t_block_start = pid_t * BLOCK_T
        t_offsets = t_block_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T

        x_ptrs = x_ptr + pid_n * x_stride_n + pid_c * x_stride_c + t_offsets * x_stride_t
        mask_ptrs = mask_ptr + pid_n * mask_stride_n + 0 * mask_stride_c + t_offsets * mask_stride_t  # mask has C=1
        y_ptrs = y_ptr + pid_n * y_stride_n + pid_c * y_stride_c + t_offsets * y_stride_t

        x_vals = tl.load(x_ptrs, mask=t_mask, other=0.0)
        mask_vals = tl.load(mask_ptrs, mask=t_mask, other=1.0)
        y_vals = x_vals * mask_vals
        tl.store(y_ptrs, y_vals, mask=t_mask)

def conv1d_triton(x, w, b, padding):
    # x: [N, C_in, T_in], w: [C_out, C_in, K], b: [C_out]
    assert x.is_cuda and TRITON_AVAILABLE, "x must be on CUDA for Triton conv1d"
    N, C_in, T_in = x.shape
    C_out, C_in_w, K = w.shape
    assert C_in == C_in_w, "Input channels mismatch for conv1d"
    T_out = T_in
    y = torch.empty((N, C_out, T_out), dtype=x.dtype, device=x.device)

    # Strides
    x_stride_n, x_stride_c, x_stride_t = x.stride()
    w_stride_co, w_stride_ci, w_stride_k = w.stride()
    y_stride_n, y_stride_c, y_stride_t = y.stride()

    # Grid: (N, C_out, ceil_div(T_out, BLOCK_T))
    BLOCK_T = 128
    grid = (N, C_out, triton.cdiv(T_out, BLOCK_T))

    conv1d_forward_kernel[grid](
        x, w, b, y,
        N, T_in, T_out, C_in, C_out, K, padding,
        x_stride_n, x_stride_c, x_stride_t,
        w_stride_co, w_stride_ci, w_stride_k,
        y_stride_n, y_stride_c, y_stride_t,
        t_block_start=0,
        BLOCK_T=BLOCK_T,
        num_warps=4,
    )
    return y

def relu_triton(inp):
    assert inp.is_cuda and TRITON_AVAILABLE, "Input must be on CUDA for Triton ReLU"
    N, C, T = inp.shape
    out = torch.empty_like(inp)
    x_stride_n, x_stride_c, x_stride_t = inp.stride()
    y_stride_n, y_stride_c, y_stride_t = out.stride()
    BLOCK_T = 128
    grid = (N, C, triton.cdiv(T, BLOCK_T))
    relu_forward_kernel[grid](
        inp, out,
        N, C, T,
        x_stride_n, x_stride_c, x_stride_t,
        y_stride_n, y_stride_c, y_stride_t,
        BLOCK_T=BLOCK_T,
        num_warps=4,
    )
    return out

def concat_half_channels_triton(x0, x1, half_first=True):
    # x0: [N, C_half, T], x1: [N, C_half, T] -> y: [N, 2*C_half, T]
    assert x0.is_cuda and x1.is_cuda and TRITON_AVAILABLE, "Tensors must be on CUDA for Triton concat"
    N, C_half, T = x0.shape
    y = torch.empty((N, 2 * C_half, T), dtype=x0.dtype, device=x0.device)
    x0_stride_n, x0_stride_c, x0_stride_t = x0.stride()
    x1_stride_n, x1_stride_c, x1_stride_t = x1.stride()
    y_stride_n, y_stride_c, y_stride_t = y.stride()
    BLOCK_T = 128
    grid = (N, C_half, triton.cdiv(T, BLOCK_T))
    concat_half_channels_kernel_fixed[grid](
        x0, x1, y,
        N, C_half, T,
        x0_stride_n, x0_stride_c, x0_stride_t,
        x1_stride_n, x1_stride_c, x1_stride_t,
        y_stride_n, y_stride_c, y_stride_t,
        half_first=half_first,
        BLOCK_T=BLOCK_T,
        num_warps=4,
    )
    return y

def affine_add_triton(x1, h):
    # x1, h: [N, C_half, T]
    assert x1.is_cuda and h.is_cuda and TRITON_AVAILABLE, "Tensors must be on CUDA for Triton affine add"
    N, C_half, T = x1.shape
    y = torch.empty_like(x1)
    x1_stride_n, x1_stride_c, x1_stride_t = x1.stride()
    h_stride_n, h_stride_c, h_stride_t = h.stride()
    y_stride_n, y_stride_c, y_stride_t = y.stride()
    BLOCK_T = 128
    grid = (N, C_half, triton.cdiv(T, BLOCK_T))
    affine_add_kernel[grid](
        x1, h, y,
        N, C_half, T,
        x1_stride_n, x1_stride_c, x1_stride_t,
        h_stride_n, h_stride_c, h_stride_t,
        y_stride_n, y_stride_c, y_stride_t,
        BLOCK_T=BLOCK_T,
        num_warps=4,
    )
    return y

def affine_sub_triton(x1, h):
    # x1, h: [N, C_half, T]
    assert x1.is_cuda and h.is_cuda and TRITON_AVAILABLE, "Tensors must be on CUDA for Triton affine sub"
    N, C_half, T = x1.shape
    y = torch.empty_like(x1)
    x1_stride_n, x1_stride_c, x1_stride_t = x1.stride()
    h_stride_n, h_stride_c, h_stride_t = h.stride()
    y_stride_n, y_stride_c, y_stride_t = y.stride()
    BLOCK_T = 128
    grid = (N, C_half, triton.cdiv(T, BLOCK_T))
    affine_sub_kernel[grid](
        x1, h, y,
        N, C_half, T,
        x1_stride_n, x1_stride_c, x1_stride_t,
        h_stride_n, h_stride_c, h_stride_t,
        y_stride_n, y_stride_c, y_stride_t,
        BLOCK_T=BLOCK_T,
        num_warps=4,
    )
    return y

def mask_mul_triton(x, mask):
    # x: [N, C_half, T], mask: [N, 1, T] -> y = x * mask
    assert x.is_cuda and mask.is_cuda and TRITON_AVAILABLE, "Tensors must be on CUDA for Triton mask mul"
    N, C_half, T = x.shape
    y = torch.empty_like(x)
    x_stride_n, x_stride_c, x_stride_t = x.stride()
    mask_stride_n, mask_stride_c, mask_stride_t = mask.stride()
    y_stride_n, y_stride_c, y_stride_t = y.stride()
    BLOCK_T = 128
    grid = (N, C_half, triton.cdiv(T, BLOCK_T))
    mask_mul_kernel[grid](
        x, mask, y,
        N, C_half, T,
        x_stride_n, x_stride_c, x_stride_t,
        mask_stride_n, mask_stride_c, mask_stride_t,
        y_stride_n, y_stride_c, y_stride_t,
        BLOCK_T=BLOCK_T,
        num_warps=4,
    )
    return y

class ModelNew(torch.nn.Module):
    def forward(self, x, x_mask, reverse, transform_0_conv0_weight, transform_0_conv0_bias,
                transform_0_conv1_weight, transform_0_conv1_bias, transform_0_conv2_weight, transform_0_conv2_bias,
                transform_1_conv0_weight, transform_1_conv0_bias, transform_1_conv1_weight, transform_1_conv1_bias,
                transform_1_conv2_weight, transform_1_conv2_bias, transform_2_conv0_weight, transform_2_conv0_bias,
                transform_2_conv1_weight, transform_2_conv1_bias, transform_2_conv2_weight, transform_2_conv2_bias,
                transform_3_conv0_weight, transform_3_conv0_bias, transform_3_conv1_weight, transform_3_conv1_bias,
                transform_3_conv2_weight, transform_3_conv2_bias):
        # Ensure CUDA for Triton
        assert x.is_cuda and TRITON_AVAILABLE, "Input x must be on CUDA device for Triton"

        N = x.shape[0]
        C = x.shape[1]
        T_in = x.shape[2]
        half_channels = C // 2
        assert C == 192, "Expected channels=192"
        assert half_channels == 96, "Expected half_channels=96"

        # List of transform parameter tuples
        transforms = [
            (transform_0_conv0_weight, transform_0_conv0_bias, transform_0_conv1_weight, transform_0_conv1_bias,
             transform_0_conv2_weight, transform_0_conv2_bias),
            (transform_1_conv0_weight, transform_1_conv0_bias, transform_1_conv1_weight, transform_1_conv1_bias,
             transform_1_conv2_weight, transform_1_conv2_bias),
            (transform_2_conv0_weight, transform_2_conv0_bias, transform_2_conv1_weight, transform_2_conv1_bias,
             transform_2_conv2_weight, transform_2_conv2_bias),
            (transform_3_conv0_weight, transform_3_conv0_bias, transform_3_conv1_weight, transform_3_conv1_bias,
             transform_3_conv2_weight, transform_3_conv2_bias),
        ]
        K = 5
        PAD = K // 2

        if not reverse:
            # Forward: x1 = x1 + transform(x0) per layer
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
                # Split input along channels
                x0 = x[:, :half_channels, :]
                x1 = x[:, half_channels:, :]

                # Conv1d -> ReLU -> Conv1d -> ReLU -> Conv1d
                # conv0
                h = conv1d_triton(x0, conv0_w, conv0_b, PAD)
                h = relu_triton(h)
                # conv1
                h = conv1d_triton(h, conv1_w, conv1_b, PAD)
                h = relu_triton(h)
                # conv2
                h = conv1d_triton(h, conv2_w, conv2_b, PAD)

                # Affine coupling: x1 = x1 + h
                x1 = affine_add_triton(x1, h)

                # Concatenate halves along channels
                x = concat_half_channels_triton(x0, x1, half_first=True)

                # Apply mask (mask is ones in the provided setup, but keep generic behavior)
                x = mask_mul_triton(x, x_mask)
        else:
            # Reverse: x1 = x1 - transform(x0) per layer (in reverse order)
            for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
                x0 = x[:, :half_channels, :]
                x1 = x[:, half_channels:, :]

                # conv0
                h = conv1d_triton(x0, conv0_w, conv0_b, PAD)
                h = relu_triton(h)
                # conv1
                h = conv1d_triton(h, conv1_w, conv1_b, PAD)
                h = relu_triton(h)
                # conv2
                h = conv1d_triton(h, conv2_w, conv2_b, PAD)

                # Inverse affine coupling: x1 = x1 - h
                x1 = affine_sub_triton(x1, h)

                # Concatenate halves along channels
                x = concat_half_channels_triton(x0, x1, half_first=True)

                # Apply mask
                x = mask_mul_triton(x, x_mask)

        return x


def run(*args):
    return ModelNew()(*args)
