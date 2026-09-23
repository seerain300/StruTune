import math
import torch
import torch.nn as nn

# Try to import Triton; if unavailable, we won't run Triton kernels (but in evaluation, Triton should be available).
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Define Triton kernels (all computation will be performed here when TRITON_AVAILABLE is True).

if TRITON_AVAILABLE:
    @triton.jit
    def conv1d_forward_kernel(
        x_ptr,         # *float32, input tensor [N, C_IN, T_IN]
        w_ptr,         # *float32, weights tensor [C_OUT, C_IN, K]
        b_ptr,         # *float32, biases tensor [C_OUT]
        y_ptr,         # *float32, output tensor [N, C_OUT, T_OUT]
        N, T_IN, T_OUT,  # int32
        x_stride_n, x_stride_c, x_stride_t,      # int32 strides
        w_stride_co, w_stride_ci, w_stride_k,    # int32 strides
        y_stride_n, y_stride_c, y_stride_t,      # int32 strides
        BLOCK_T: tl.constexpr,                   # int32 compile-time
        C_IN: tl.constexpr,                      # int32 compile-time
        C_OUT: tl.constexpr,                     # int32 compile-time
        K: tl.constexpr,                         # int32 compile-time
    ):
        # program ids: over (n, co, time block)
        pid_n = tl.program_id(0)
        pid_co = tl.program_id(1)
        pid_tb = tl.program_id(2)

        # time offsets computed by this program
        t_offsets = pid_tb * BLOCK_T + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T_OUT

        # accumulator for this (n, co, time block)
        acc = tl.zeros([BLOCK_T], dtype=tl.float32)

        # loop over input channels and kernel taps
        for ci in range(0, C_IN):
            for k in range(0, K):
                t_in = t_offsets + k  # pad implicitly via mask
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
        inp_ptr,        # *float32, input tensor
        out_ptr,        # *float32, output tensor (can alias inp_ptr)
        N, C, T,        # int32
        in_stride_n, in_stride_c, in_stride_t,  # int32
        out_stride_n, out_stride_c, out_stride_t,  # int32
        BLOCK: tl.constexpr = 1,
    ):
        # simple elementwise ReLU kernel over the whole tensor
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        offs = pid_t * BLOCK + tl.arange(0, BLOCK)
        mask = offs < T

        base = pid_n * in_stride_n + pid_c * in_stride_c
        ptrs = inp_ptr + base + offs * in_stride_t
        vals = tl.load(ptrs, mask=mask, other=0.0)
        vals = tl.maximum(vals, 0.0)
        out_ptrs = out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + offs * out_stride_t
        tl.store(out_ptrs, vals, mask=mask)

    @triton.jit
    def concat_half_channels_kernel(
        src0_ptr,       # *float32, [N, C_HALF, T]
        src1_ptr,       # *float32, [N, C_HALF, T]
        out_ptr,        # *float32, [N, 2*C_HALF, T]
        N, C_HALF, T,   # int32
        s0_stride_n, s0_stride_c, s0_stride_t,    # int32
        s1_stride_n, s1_stride_c, s1_stride_t,    # int32
        out_stride_n, out_stride_c, out_stride_t, # int32
        BLOCK: tl.constexpr = 1,
    ):
        # copy src0 into out[:, :C_HALF, :]
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        offs = pid_t * BLOCK + tl.arange(0, BLOCK)
        mask = offs < T

        s0_ptrs = src0_ptr + pid_n * s0_stride_n + pid_c * s0_stride_c + offs * s0_stride_t
        vals0 = tl.load(s0_ptrs, mask=mask, other=0.0)
        out_ptrs0 = out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + offs * out_stride_t
        tl.store(out_ptrs0, vals0, mask=mask)

        # copy src1 into out[:, C_HALF:, :]
        # src1 channel index is same pid_c (0..C_HALF-1), but out channel index is pid_c + C_HALF
        s1_ptrs = src1_ptr + pid_n * s1_stride_n + pid_c * s1_stride_c + offs * s1_stride_t
        vals1 = tl.load(s1_ptrs, mask=mask, other=0.0)
        out_ptrs1 = out_ptr + pid_n * out_stride_n + (pid_c + C_HALF) * out_stride_c + offs * out_stride_t
        tl.store(out_ptrs1, vals1, mask=mask)

    @triton.jit
    def affine_add_kernel(
        a_ptr, b_ptr, out_ptr,
        N, C, T,
        a_stride_n, a_stride_c, a_stride_t,
        b_stride_n, b_stride_c, b_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK: tl.constexpr = 1,
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        offs = pid_t * BLOCK + tl.arange(0, BLOCK)
        mask = offs < T

        a_ptrs = a_ptr + pid_n * a_stride_n + pid_c * a_stride_c + offs * a_stride_t
        b_ptrs = b_ptr + pid_n * b_stride_n + pid_c * b_stride_c + offs * b_stride_t
        a_vals = tl.load(a_ptrs, mask=mask, other=0.0)
        b_vals = tl.load(b_ptrs, mask=mask, other=0.0)
        out_vals = a_vals + b_vals

        out_ptrs = out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + offs * out_stride_t
        tl.store(out_ptrs, out_vals, mask=mask)

    @triton.jit
    def affine_sub_kernel(
        a_ptr, b_ptr, out_ptr,
        N, C, T,
        a_stride_n, a_stride_c, a_stride_t,
        b_stride_n, b_stride_c, b_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK: tl.constexpr = 1,
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        offs = pid_t * BLOCK + tl.arange(0, BLOCK)
        mask = offs < T

        a_ptrs = a_ptr + pid_n * a_stride_n + pid_c * a_stride_c + offs * a_stride_t
        b_ptrs = b_ptr + pid_n * b_stride_n + pid_c * b_stride_c + offs * b_stride_t
        a_vals = tl.load(a_ptrs, mask=mask, other=0.0)
        b_vals = tl.load(b_ptrs, mask=mask, other=0.0)
        out_vals = a_vals - b_vals

        out_ptrs = out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + offs * out_stride_t
        tl.store(out_ptrs, out_vals, mask=mask)

    @triton.jit
    def mask_mul_kernel(
        inp_ptr, mask_ptr, out_ptr,
        N, C, T,
        in_stride_n, in_stride_c, in_stride_t,
        mask_stride_n, mask_stride_c, mask_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK: tl.constexpr = 1,
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        offs = pid_t * BLOCK + tl.arange(0, BLOCK)
        mask = offs < T

        inp_ptrs = inp_ptr + pid_n * in_stride_n + pid_c * in_stride_c + offs * in_stride_t
        mask_ptrs = mask_ptr + pid_n * mask_stride_n + pid_c * mask_stride_c + offs * mask_stride_t
        inp_vals = tl.load(inp_ptrs, mask=mask, other=0.0)
        mask_vals = tl.load(mask_ptrs, mask=mask, other=1.0)  # mask is float32
        out_vals = inp_vals * mask_vals

        out_ptrs = out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + offs * out_stride_t
        tl.store(out_ptrs, out_vals, mask=mask)


def _conv1d_triton(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, padding: int) -> torch.Tensor:
    """
    Triton conv1d: x [N, C_in, T_in], w [C_out, C_in, K], b [C_out]
    Output y [N, C_out, T_out], T_out = T_in - K + 1 + 2*padding (for odd K and symmetric padding, equals T_in).
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert x.is_cuda and w.is_cuda and b.is_cuda, "Tensors must be on CUDA device for Triton"

    N, C_in, T_in = x.shape
    C_out, C_in_w, K = w.shape
    assert C_in == C_in_w, "Weight in_channels must match input channels"
    T_out = T_in - K + 1 + 2 * padding

    # Ensure contiguous
    x = x.contiguous()
    w = w.contiguous()
    b = b.contiguous()

    # Allocate output
    y = torch.empty((N, C_out, T_out), dtype=x.dtype, device=x.device)

    # Strides
    x_stride_n, x_stride_c, x_stride_t = x.stride()
    w_stride_co, w_stride_ci, w_stride_k = w.stride()
    y_stride_n, y_stride_c, y_stride_t = y.stride()

    # Launch Triton kernel
    BLOCK_T = 128  # tuneable
    grid = (N, C_out, triton.cdiv(T_out, BLOCK_T))
    conv1d_forward_kernel[grid](
        x, w, b, y,
        N, T_in, T_out,
        x_stride_n, x_stride_c, x_stride_t,
        w_stride_co, w_stride_ci, w_stride_k,
        y_stride_n, y_stride_c, y_stride_t,
        BLOCK_T=BLOCK_T,
        C_IN=C_in, C_OUT=C_out, K=K,
        num_warps=4,  # tuneable
        num_stages=2, # tuneable
    )
    return y


def _relu_triton(inp: torch.Tensor) -> torch.Tensor:
    """
    Elementwise ReLU via Triton.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert inp.is_cuda, "Input must be on CUDA device for Triton"
    inp = inp.contiguous()
    out = torch.empty_like(inp)
    N, C, T = inp.shape
    in_stride_n, in_stride_c, in_stride_t = inp.stride()
    out_stride_n, out_stride_c, out_stride_t = out.stride()
    # Simple grid (N, C, T) with BLOCK=1; Triton handles vectorization internally.
    grid = (N, C, T)
    relu_forward_kernel[grid](
        inp, out,
        N, C, T,
        in_stride_n, in_stride_c, in_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK=1,
        num_warps=4,
        num_stages=2,
    )
    return out


def _concat_half_channels_triton(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """
    Concatenate along channel dimension: out[:, :C_half, :] = x0, out[:, C_half:, :] = x1.
    x0: [N, C_half, T], x1: [N, C_half, T], out: [N, 2*C_half, T].
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert x0.is_cuda and x1.is_cuda, "Tensors must be on CUDA device for Triton"
    N, C_half, T = x0.shape
    assert x1.shape == (N, C_half, T), "x1 must match x0 shape"
    out = torch.empty((N, 2 * C_half, T), dtype=x0.dtype, device=x0.device)
    s0_stride_n, s0_stride_c, s0_stride_t = x0.stride()
    s1_stride_n, s1_stride_c, s1_stride_t = x1.stride()
    out_stride_n, out_stride_c, out_stride_t = out.stride()
    grid = (N, C_half, T)
    concat_half_channels_kernel[grid](
        x0, x1, out,
        N, C_half, T,
        s0_stride_n, s0_stride_c, s0_stride_t,
        s1_stride_n, s1_stride_c, s1_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK=1,
        num_warps=4,
        num_stages=2,
    )
    return out


def _affine_add_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Elementwise add via Triton: out = a + b
    Assumes a, b have same shape and are CUDA tensors.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA device for Triton"
    a = a.contiguous()
    b = b.contiguous()
    out = torch.empty_like(a)
    N, C, T = a.shape
    a_stride_n, a_stride_c, a_stride_t = a.stride()
    b_stride_n, b_stride_c, b_stride_t = b.stride()
    out_stride_n, out_stride_c, out_stride_t = out.stride()
    grid = (N, C, T)
    affine_add_kernel[grid](
        a, b, out,
        N, C, T,
        a_stride_n, a_stride_c, a_stride_t,
        b_stride_n, b_stride_c, b_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK=1,
        num_warps=4,
        num_stages=2,
    )
    return out


def _affine_sub_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Elementwise subtract via Triton: out = a - b
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA device for Triton"
    a = a.contiguous()
    b = b.contiguous()
    out = torch.empty_like(a)
    N, C, T = a.shape
    a_stride_n, a_stride_c, a_stride_t = a.stride()
    b_stride_n, b_stride_c, b_stride_t = b.stride()
    out_stride_n, out_stride_c, out_stride_t = out.stride()
    grid = (N, C, T)
    affine_sub_kernel[grid](
        a, b, out,
        N, C, T,
        a_stride_n, a_stride_c, a_stride_t,
        b_stride_n, b_stride_c, b_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK=1,
        num_warps=4,
        num_stages=2,
    )
    return out


def _mask_mul_triton(inp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Elementwise multiply by mask via Triton.
    inp: [N, C, T], mask: [N, C, T]
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert inp.is_cuda and mask.is_cuda, "Tensors must be on CUDA device for Triton"
    inp = inp.contiguous()
    mask = mask.contiguous()
    out = torch.empty_like(inp)
    N, C, T = inp.shape
    in_stride_n, in_stride_c, in_stride_t = inp.stride()
    mask_stride_n, mask_stride_c, mask_stride_t = mask.stride()
    out_stride_n, out_stride_c, out_stride_t = out.stride()
    grid = (N, C, T)
    mask_mul_kernel[grid](
        inp, mask, out,
        N, C, T,
        in_stride_n, in_stride_c, in_stride_t,
        mask_stride_n, mask_stride_c, mask_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK=1,
        num_warps=4,
        num_stages=2,
    )
    return out


def apply_transform_triton(x0: torch.Tensor, conv0_w: torch.Tensor, conv0_b: torch.Tensor,
                           conv1_w: torch.Tensor, conv1_b: torch.Tensor,
                           conv2_w: torch.Tensor, conv2_b: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of a single transform: Conv1d -> ReLU -> Conv1d -> ReLU -> Conv1d.
    x0: [N, C_in_half, T], conv* shapes: [C_out, C_in, K]
    Returns h: [N, C_out, T] where C_out is the last layer's out_channels (typically 96 or 192).
    """
    # conv0
    h = _conv1d_triton(x0, conv0_w, conv0_b, padding=conv0_w.shape[2] // 2)
    # ReLU
    h = _relu_triton(h)
    # conv1
    h = _conv1d_triton(h, conv1_w, conv1_b, padding=conv1_w.shape[2] // 2)
    # ReLU
    h = _relu_triton(h)
    # conv2
    h = _conv1d_triton(h, conv2_w, conv2_b, padding=conv2_w.shape[2] // 2)
    return h


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
    Residual coupling flow block.
    Forward: x1 = x1 + transform(x0) for each layer
    Reverse: x1 = x1 - transform(x0) for each layer (in reverse order)
    Triton-only implementation; no torch ops used in forward path.
    """
    assert TRITON_AVAILABLE, "Triton is not available"
    assert x.is_cuda and x_mask.is_cuda, "All tensors must be on CUDA device for Triton"
    # x: [N, 192, T]
    N, C, T = x.shape
    half_channels = C // 2
    assert half_channels == 96, "This implementation expects half_channels=96"

    # Prepare transforms (each is a 3-conv chain). We'll apply them in the loop.
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
        # Forward pass: apply transformations sequentially
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            # Split into two halves along channel dimension
            x0 = x[:, :half_channels, :]           # [N, 96, T]
            x1 = x[:, half_channels:, :]           # [N, 96, T]

            # Compute transformation conditioned on x0
            h = apply_transform_triton(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b)  # [N, C_out, T]
            # Apply mask (mask is [N,1,T], treat as [N,*,T])
            # Broadcast h to [N,1,T] then elementwise multiply; but h already is [N,C_out,T].
            # To apply mask, we can expand x_mask to [N,1,T] and multiply with h expanded to [N,1,T] by summing channels (not needed for mask, since mask is ones).
            # However, mask here is [N,1,T]. We'll ensure h has same [N,1,T] by masking using broadcasting:
            # Create a copy of h and multiply elementwise by x_mask.
            # Since mask is [N,1,T], we can multiply h with broadcasted mask: h * x_mask (PyTorch broadcast). But we need Triton.
            # We implement mask multiply via Triton kernel.
            h_masked = _mask_mul_triton(h, x_mask)

            # Affine coupling: x1 = x1 + h_masked, but h_masked has [N, C_out, T]; we need to align channel dimensions.
            # Note: In the original code, h has the same time length T, and we add it to x1 which also has [N, 96, T].
            # The original code doesn't perform ReLU on x1 before adding. We should not do extra ReLU here.
            # However, the original apply_transform returns a tensor of shape [N, conv2_out_channels, T], not [N, 96, T].
            # In our setup, conv2_out_channels for each transform is 96 (half_channels). So h has shape [N, 96, T].
            # We can add h_masked to x1 directly, elementwise, since both are [N, 96, T].
            # But we must ensure h_masked has the same channel size as x1. The code above assumes conv2_out_channels == half_channels.
            # Given the provided get_inputs, this is true. So we proceed with elementwise add via Triton.

            # Concatenate back along channel dimension: x0 (96) + updated x1 (96) -> x with 192 channels
            # First, we need to construct out tensor [N, 2*96=192, T] using Triton concat kernel.
            # However, concatenation here is just torch.cat; we cannot avoid torch.cat in host. But we can avoid torch in computation. To adhere to Triton-only, we implement concat via Triton:
            # We have x0 and updated x1 after add. Let's define updated_x1 = x1 + h_masked. Then concatenate:
            updated_x1 = _affine_add_triton(x1, h_masked)
            x = _concat_half_channels_triton(x0, updated_x1)
            # Apply mask to output: multiply by x_mask. But x_mask is [N,1,T], and x is [N,192,T].
            # Elementwise multiply of [N,192,T] with [N,1,T] is broadcast along channels. Implement via Triton:
            x = _mask_mul_triton(x, x_mask)

    else:
        # Reverse pass: apply transformations in reverse order
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
            x0 = x[:, :half_channels, :]
            x1 = x[:, half_channels:, :]

            h = apply_transform_triton(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b)  # [N, 96, T]
            h_masked = _mask_mul_triton(h, x_mask)

            # Inverse affine coupling: x1 = x1 - h_masked
            x1 = _affine_sub_triton(x1, h_masked)

            # Concatenate back
            x = _concat_half_channels_triton(x0, x1)
            # Apply mask to output
            x = _mask_mul_triton(x, x_mask)

    return x


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The harness will pass all required tensors. We mirror the original run signature and use only Triton kernels.
        return run_triton(*args)


def run(*args):
    return ModelNew()(*args)
