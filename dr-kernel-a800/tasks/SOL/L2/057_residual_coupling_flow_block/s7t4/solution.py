import math
import torch
import torch.nn.functional as F

# Try to import Triton; we will use only Triton in forward.
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels: all numerical computations are here.
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
        # Program ids
        pid_n = tl.program_id(0)  # batch index
        pid_co = tl.program_id(1) # output channel index
        pid_tb = tl.program_id(2) # time block index

        # Compute time offsets this program handles
        t_offsets = t_block_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T_OUT

        # Accumulator for this (n, co, time_block)
        acc = tl.zeros([BLOCK_T], dtype=tl.float32)

        # Loop over input channels and kernel taps
        for ci in range(0, C_IN):
            for k in range(0, K):
                t_in = t_offsets + k - PAD
                valid = (t_in >= 0) & (t_in < T_IN) & t_mask
                # Load x[n, ci, t_in]
                x_ptrs = x_ptr + pid_n * x_stride_n + ci * x_stride_c + t_in * x_stride_t
                x_vals = tl.load(x_ptrs, mask=valid, other=0.0)
                # Load weight w[co, ci, k]
                w_ptrs = w_ptr + pid_co * w_stride_co + ci * w_stride_ci + k * w_stride_k
                w_val = tl.load(w_ptrs)
                acc += x_vals * w_val

        # Add bias for this output channel
        b_val = tl.load(b_ptr + pid_co)
        acc += b_val

        # Store y[n, co, t_offsets]
        y_ptrs = y_ptr + pid_n * y_stride_n + pid_co * y_stride_c + t_offsets * y_stride_t
        tl.store(y_ptrs, acc, mask=t_mask)

    @triton.jit
    def relu_forward_kernel(
        inp_ptr,        # *float32, input [N, C, T]
        out_ptr,        # *float32, output [N, C, T]
        N, C, T,
        in_stride_n, in_stride_c, in_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK_TC: tl.constexpr,  # block over T*C
    ):
        grid_nc = N * C
        pid_nc = tl.program_id(0)
        pid_t = tl.program_id(1)
        # Each program handles one (n,c) and a block of T
        # Compute t offsets
        t_start = pid_t * BLOCK_TC
        t_offsets = t_start + tl.arange(0, BLOCK_TC)
        t_mask = t_offsets < T

        # Map pid_nc to n,c
        n = pid_nc // C
        c = pid_nc % C

        # Compute pointers
        inp_ptrs = inp_ptr + n * in_stride_n + c * in_stride_c + t_offsets * in_stride_t
        out_ptrs = out_ptr + n * out_stride_n + c * out_stride_c + t_offsets * out_stride_t

        # Load, ReLU, store
        vals = tl.load(inp_ptrs, mask=t_mask, other=0.0)
        vals = tl.maximum(vals, 0.0)
        tl.store(out_ptrs, vals, mask=t_mask)

    @triton.jit
    def concat_half_channels_kernel_fixed(
        x0_ptr, x1_ptr, out_ptr,
        N, C_HALF, T,
        x0_stride_n, x0_stride_c, x0_stride_t,
        x1_stride_n, x1_stride_c, x1_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK_TC: tl.constexpr,  # block over T*C_HALF
    ):
        # Grid over (N, C_HALF, time blocks)
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        t_start = pid_t * BLOCK_TC
        t_offsets = t_start + tl.arange(0, BLOCK_TC)
        t_mask = t_offsets < T

        # From x0 (first half channels)
        x0_ptrs = x0_ptr + pid_n * x0_stride_n + pid_c * x0_stride_c + t_offsets * x0_stride_t
        vals0 = tl.load(x0_ptrs, mask=t_mask, other=0.0)
        out0_ptrs = out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + t_offsets * out_stride_t
        tl.store(out0_ptrs, vals0, mask=t_mask)

        # From x1 (second half channels), write at channel index C_HALF + pid_c
        x1_ptrs = x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + t_offsets * x1_stride_t
        vals1 = tl.load(x1_ptrs, mask=t_mask, other=0.0)
        out1_ptrs = out_ptr + pid_n * out_stride_n + (pid_c + C_HALF) * out_stride_c + t_offsets * out_stride_t
        tl.store(out1_ptrs, vals1, mask=t_mask)

    @triton.jit
    def add_affine_kernel(
        x1_ptr, h_ptr, out_ptr,
        N, C_HALF, T,
        x1_stride_n, x1_stride_c, x1_stride_t,
        h_stride_n, h_stride_c, h_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK_TC: tl.constexpr,
    ):
        # Grid over (N, C_HALF, time blocks)
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        t_start = pid_t * BLOCK_TC
        t_offsets = t_start + tl.arange(0, BLOCK_TC)
        t_mask = t_offsets < T

        x1_ptrs = x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + t_offsets * x1_stride_t
        h_ptrs = h_ptr + pid_n * h_stride_n + pid_c * h_stride_c + t_offsets * h_stride_t
        out_ptrs = out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + t_offsets * out_stride_t

        x1_vals = tl.load(x1_ptrs, mask=t_mask, other=0.0)
        h_vals = tl.load(h_ptrs, mask=t_mask, other=0.0)
        out_vals = x1_vals + h_vals
        tl.store(out_ptrs, out_vals, mask=t_mask)

    @triton.jit
    def sub_affine_kernel(
        x1_ptr, h_ptr, out_ptr,
        N, C_HALF, T,
        x1_stride_n, x1_stride_c, x1_stride_t,
        h_stride_n, h_stride_c, h_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK_TC: tl.constexpr,
    ):
        # Grid over (N, C_HALF, time blocks)
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        t_start = pid_t * BLOCK_TC
        t_offsets = t_start + tl.arange(0, BLOCK_TC)
        t_mask = t_offsets < T

        x1_ptrs = x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + t_offsets * x1_stride_t
        h_ptrs = h_ptr + pid_n * h_stride_n + pid_c * h_stride_c + t_offsets * h_stride_t
        out_ptrs = out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + t_offsets * out_stride_t

        x1_vals = tl.load(x1_ptrs, mask=t_mask, other=0.0)
        h_vals = tl.load(h_ptrs, mask=t_mask, other=0.0)
        out_vals = x1_vals - h_vals
        tl.store(out_ptrs, out_vals, mask=t_mask)

    @triton.jit
    def mask_mul_kernel(
        x_ptr, mask_ptr, out_ptr,
        N, C, T,
        x_stride_n, x_stride_c, x_stride_t,
        mask_stride_n, mask_stride_c, mask_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK_TC: tl.constexpr,
    ):
        # Grid over (N, C, time blocks). mask has shape [N, 1, T], so we load mask[n, 0, t].
        grid_nc = N * C
        pid_nc = tl.program_id(0)
        pid_t = tl.program_id(1)

        n = pid_nc // C
        c = pid_nc % C

        t_start = pid_t * BLOCK_TC
        t_offsets = t_start + tl.arange(0, BLOCK_TC)
        t_mask = t_offsets < T

        x_ptrs = x_ptr + n * x_stride_n + c * x_stride_c + t_offsets * x_stride_t
        mask_ptrs = mask_ptr + n * mask_stride_n + 0 * mask_stride_c + t_offsets * mask_stride_t
        out_ptrs = out_ptr + n * out_stride_n + c * out_stride_c + t_offsets * out_stride_t

        x_vals = tl.load(x_ptrs, mask=t_mask, other=0.0)
        mask_vals = tl.load(mask_ptrs, mask=t_mask, other=1.0)
        out_vals = x_vals * mask_vals
        tl.store(out_ptrs, out_vals, mask=t_mask)


def conv1d_triton(x, w, b, padding=2):
    """
    x: [N, C_in, T_in], w: [C_out, C_in, K], b: [C_out]
    Returns y: [N, C_out, T_out] with T_out = T_in (padding symmetric).
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert x.is_cuda and w.is_cuda and b.is_cuda, "Triton kernels require CUDA tensors"
    assert x.dtype == torch.float32 and w.dtype == torch.float32 and b.dtype == torch.float32

    N, C_in, T_in = x.shape
    C_out, C_in_w, K = w.shape
    assert C_in == C_in_w, "Weight in_channels must match input channels"
    assert K == w.shape[2], "Invalid weight shape for kernel"
    T_out = T_in

    y = torch.empty((N, C_out, T_out), device=x.device, dtype=x.dtype)

    # Strides
    x_stride_n, x_stride_c, x_stride_t = x.stride()
    w_stride_co, w_stride_ci, w_stride_k = w.stride()
    y_stride_n, y_stride_c, y_stride_t = y.stride()

    # Launch conv1d kernel over (N, C_out, time blocks)
    BLOCK_T = 128
    grid = (N, C_out, triton.cdiv(T_out, BLOCK_T))
    conv1d_forward_kernel[grid](
        x, w, b, y,
        N, T_in, T_out, C_in, C_out, K, padding,
        x_stride_n, x_stride_c, x_stride_t,
        w_stride_co, w_stride_ci, w_stride_k,
        y_stride_n, y_stride_c, y_stride_t,
        0,  # t_block_start will be handled by grid's third dim; not needed as constexpr
        BLOCK_T,
        num_warps=4, num_stages=2,
    )
    return y


def relu_triton(inp):
    """
    Elementwise ReLU: out = max(inp, 0)
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert inp.is_cuda, "Triton kernels require CUDA tensors"
    assert inp.dtype == torch.float32

    N, C, T = inp.shape
    out = torch.empty_like(inp)

    in_stride_n, in_stride_c, in_stride_t = inp.stride()
    out_stride_n, out_stride_c, out_stride_t = out.stride()

    # Use a 2D grid over (N*C, time blocks)
    BLOCK_TC = 128
    grid = (N * C, triton.cdiv(T, BLOCK_TC))
    relu_forward_kernel[grid](
        inp, out,
        N, C, T,
        in_stride_n, in_stride_c, in_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK_TC,
        num_warps=4, num_stages=2,
    )
    return out


def concat_half_channels_triton(x0, x1):
    """
    x0: [N, C_HALF, T], x1: [N, C_HALF, T]
    Returns out: [N, 2*C_HALF, T] with first C_HALF channels from x0, second C_HALF from x1.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert x0.is_cuda and x1.is_cuda, "Triton kernels require CUDA tensors"
    assert x0.dtype == torch.float32 and x1.dtype == torch.float32

    N, C_HALF, T = x0.shape
    out = torch.empty((N, 2 * C_HALF, T), device=x0.device, dtype=x0.dtype)

    x0_stride_n, x0_stride_c, x0_stride_t = x0.stride()
    x1_stride_n, x1_stride_c, x1_stride_t = x1.stride()
    out_stride_n, out_stride_c, out_stride_t = out.stride()

    BLOCK_TC = 128
    grid = (N, C_HALF, triton.cdiv(T, BLOCK_TC))
    concat_half_channels_kernel_fixed[grid](
        x0, x1, out,
        N, C_HALF, T,
        x0_stride_n, x0_stride_c, x0_stride_t,
        x1_stride_n, x1_stride_c, x1_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK_TC,
        num_warps=4, num_stages=2,
    )
    return out


def add_affine_triton(x1, h):
    """
    Elementwise add: out = x1 + h
    x1: [N, C_HALF, T], h: [N, C_HALF, T]
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert x1.is_cuda and h.is_cuda, "Triton kernels require CUDA tensors"
    assert x1.dtype == torch.float32 and h.dtype == torch.float32

    N, C_HALF, T = x1.shape
    out = torch.empty_like(x1)

    x1_stride_n, x1_stride_c, x1_stride_t = x1.stride()
    h_stride_n, h_stride_c, h_stride_t = h.stride()
    out_stride_n, out_stride_c, out_stride_t = out.stride()

    BLOCK_TC = 128
    grid = (N, C_HALF, triton.cdiv(T, BLOCK_TC))
    add_affine_kernel[grid](
        x1, h, out,
        N, C_HALF, T,
        x1_stride_n, x1_stride_c, x1_stride_t,
        h_stride_n, h_stride_c, h_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK_TC,
        num_warps=4, num_stages=2,
    )
    return out


def sub_affine_triton(x1, h):
    """
    Elementwise subtract: out = x1 - h
    x1: [N, C_HALF, T], h: [N, C_HALF, T]
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert x1.is_cuda and h.is_cuda, "Triton kernels require CUDA tensors"
    assert x1.dtype == torch.float32 and h.dtype == torch.float32

    N, C_HALF, T = x1.shape
    out = torch.empty_like(x1)

    x1_stride_n, x1_stride_c, x1_stride_t = x1.stride()
    h_stride_n, h_stride_c, h_stride_t = h.stride()
    out_stride_n, out_stride_c, out_stride_t = out.stride()

    BLOCK_TC = 128
    grid = (N, C_HALF, triton.cdiv(T, BLOCK_TC))
    sub_affine_kernel[grid](
        x1, h, out,
        N, C_HALF, T,
        x1_stride_n, x1_stride_c, x1_stride_t,
        h_stride_n, h_stride_c, h_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK_TC,
        num_warps=4, num_stages=2,
    )
    return out


def mask_mul_triton(x, mask):
    """
    Elementwise multiply: out = x * mask
    x: [N, C, T], mask: [N, 1, T]
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert x.is_cuda and mask.is_cuda, "Triton kernels require CUDA tensors"
    assert x.dtype == torch.float32 and mask.dtype == torch.float32

    N, C, T = x.shape
    out = torch.empty_like(x)

    x_stride_n, x_stride_c, x_stride_t = x.stride()
    mask_stride_n, mask_stride_c, mask_stride_t = mask.stride()
    out_stride_n, out_stride_c, out_stride_t = out.stride()

    # Use a 2D grid over (N*C, time blocks)
    BLOCK_TC = 128
    grid = (N * C, triton.cdiv(T, BLOCK_TC))
    mask_mul_kernel[grid](
        x, mask, out,
        N, C, T,
        x_stride_n, x_stride_c, x_stride_t,
        mask_stride_n, mask_stride_c, mask_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK_TC,
        num_warps=4, num_stages=2,
    )
    return out


# Triton-based apply_transform (no torch ops)
def apply_transform_triton(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b):
    """
    Single transform: Conv1d -> ReLU -> Conv1d -> ReLU -> Conv1d
    x0: [N, C_half, T], weights: [C_out, C_in, K]
    Returns h: [N, C_out, T]
    Padding = K//2 (for K=5 => PAD=2)
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert x0.is_cuda and conv0_w.is_cuda and conv1_w.is_cuda and conv2_w.is_cuda and \
           conv0_b.is_cuda and conv1_b.is_cuda and conv2_b.is_cuda, "Triton kernels require CUDA tensors"

    # conv0
    h0 = conv1d_triton(x0, conv0_w, conv0_b, padding=conv0_w.shape[2] // 2)
    # ReLU
    h0 = relu_triton(h0)
    # conv1
    h1 = conv1d_triton(h0, conv1_w, conv1_b, padding=conv1_w.shape[2] // 2)
    # ReLU
    h1 = relu_triton(h1)
    # conv2
    h2 = conv1d_triton(h1, conv2_w, conv2_b, padding=conv2_w.shape[2] // 2)
    return h2


# Triton-based run (no torch ops)
def run_triton(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    reverse: bool,
    # 4 transforms:
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
    Residual coupling flow block.
    Forward: x1 = x1 + transform(x0) for each layer
    Reverse: x1 = x1 - transform(x0) for each layer (in reverse order)
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert x.is_cuda, "Triton kernels require CUDA tensors"
    assert x.dtype == torch.float32

    N = x.shape[0]
    C = x.shape[1]
    T = x.shape[2]
    half_channels = C // 2
    assert half_channels == 96, "half_channels must be 96 per provided get_inputs"

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
        # Forward: apply transforms sequentially
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            x0 = x[:, :half_channels, :]  # [N, 96, T]
            x1 = x[:, half_channels:, :]  # [N, 96, T]

            # Compute transformation conditioned on x0
            h = apply_transform_triton(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b)  # [N, 192, T]

            # Apply mask (generic, although in provided setup mask is ones)
            h = mask_mul_triton(h, x_mask)

            # Affine coupling: x1 = x1 + h
            x1 = add_affine_triton(x1, h)

            # Concatenate back along channel dimension
            x = concat_half_channels_triton(x0, x1)  # [N, 192, T]

            # Apply mask to output
            x = mask_mul_triton(x, x_mask)
    else:
        # Reverse pass: apply transformations in reverse order
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
            x0 = x[:, :half_channels, :]  # [N, 96, T]
            x1 = x[:, half_channels:, :]  # [N, 96, T]

            h = apply_transform_triton(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b)  # [N, 192, T]
            h = mask_mul_triton(h, x_mask)

            # Inverse affine coupling: x1 = x1 - h
            x1 = sub_affine_triton(x1, h)

            x = concat_half_channels_triton(x0, x1)  # [N, 192, T]
            x = mask_mul_triton(x, x_mask)

    return x


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # ModelNew uses Triton-only kernels in forward
        # The signature matches the original run function:
        # (x, x_mask, reverse, transform_* weights/biases)
        return run_triton(*args)


def run(*args):
    return ModelNew()(*args)
