try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels: Conv1d, ReLU, concat, affine coupling, mask multiply

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
        inp_ptr,        # *float32, input tensor [N, C, T]
        out_ptr,        # *float32, output tensor [N, C, T] (can alias inp_ptr if in-place)
        N, C, T,
        in_stride_n, in_stride_c, in_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        total_programs: tl.constexpr,
        # grid sizes: total_programs = N*C*T
    ):
        # Each program handles one element (n, c, t) by linear indexing.
        pid = tl.program_id(0)
        # Compute n, c, t from pid (note: tl.constexpr grid is fixed)
        T_factor = C * T
        C_factor = T
        n = pid // T_factor
        rem = pid % T_factor
        c = rem // C_factor
        t = rem % C_factor

        in_ptrs = inp_ptr + n * in_stride_n + c * in_stride_c + t * in_stride_t
        val = tl.load(in_ptrs)
        val = tl.maximum(val, 0.0)
        out_ptrs = out_ptr + n * out_stride_n + c * out_stride_c + t * out_stride_t
        tl.store(out_ptrs, val)

    @triton.jit
    def concat_half_channels_kernel(
        x0_ptr, x1_ptr, out_ptr,
        N, C_HALF, T,
        x0_stride_n, x0_stride_c, x0_stride_t,
        x1_stride_n, x1_stride_c, x1_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        mode: tl.constexpr,  # 0: copy x0 -> out[:, :C_HALF, :], 1: copy x1 -> out[:, C_HALF:, :]
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        # load from source
        if mode == 0:
            src_ptr = x0_ptr + pid_n * x0_stride_n + pid_c * x0_stride_c + pid_t * x0_stride_t
            val = tl.load(src_ptr)
            # write to out[:, :C_HALF, :]
            out_ptr_target = out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t
            tl.store(out_ptr_target, val)
        else:
            src_ptr = x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + pid_t * x1_stride_t
            val = tl.load(src_ptr)
            # write to out[:, C_HALF + pid_c, :]
            out_ptr_target = out_ptr + pid_n * out_stride_n + (pid_c + C_HALF) * out_stride_c + pid_t * out_stride_t
            tl.store(out_ptr_target, val)

    @triton.jit
    def add_affine_kernel(
        x1_ptr, h_ptr, out_ptr,
        N, C, T,
        x1_stride_n, x1_stride_c, x1_stride_t,
        h_stride_n, h_stride_c, h_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        x1_ptr_el = x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + pid_t * x1_stride_t
        h_ptr_el = h_ptr + pid_n * h_stride_n + pid_c * h_stride_c + pid_t * h_stride_t
        out_ptr_el = out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t

        x1_val = tl.load(x1_ptr_el)
        h_val = tl.load(h_ptr_el)
        out_val = x1_val + h_val
        tl.store(out_ptr_el, out_val)

    @triton.jit
    def sub_affine_kernel(
        x1_ptr, h_ptr, out_ptr,
        N, C, T,
        x1_stride_n, x1_stride_c, x1_stride_t,
        h_stride_n, h_stride_c, h_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        x1_ptr_el = x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + pid_t * x1_stride_t
        h_ptr_el = h_ptr + pid_n * h_stride_n + pid_c * h_stride_c + pid_t * h_stride_t
        out_ptr_el = out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t

        x1_val = tl.load(x1_ptr_el)
        h_val = tl.load(h_ptr_el)
        out_val = x1_val - h_val
        tl.store(out_ptr_el, out_val)

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

        x_ptr_el = x_ptr + pid_n * x_stride_n + pid_c * x_stride_c + pid_t * x_stride_t
        mask_ptr_el = mask_ptr + pid_n * mask_stride_n + pid_c * mask_stride_c + pid_t * mask_stride_t
        out_ptr_el = out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t

        x_val = tl.load(x_ptr_el)
        m_val = tl.load(mask_ptr_el)
        out_val = x_val * m_val
        tl.store(out_ptr_el, out_val)


def _ceil_div(a, b):
    return (a + b - 1) // b


# Triton-only run implementation
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
    Triton-only forward/reverse pass for the Residual Coupling Flow block.
    Assumes tensors are on CUDA and Triton is available.
    """
    assert x.is_cuda and TRITON_AVAILABLE, "Triton version requires CUDA tensors"
    N, C, T = x.shape
    C_HALF = C // 2
    device = x.device
    dtype = x.dtype

    # Prepare weight/bias pointers and shapes
    # Note: All convs share same padding = K//2 = 2 for K=5. We keep it generic.
    # We'll compute time output consistently (no change for padding here).

    # Helper: perform a single transform on x0 -> h using conv1d -> ReLU -> conv1d -> ReLU -> conv1d
    def apply_transform_triton(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b):
        # conv0: [C_OUT0=192, C_IN0=96, K=5]
        C_OUT0 = conv0_w.shape[0]
        C_IN0 = conv0_w.shape[1]
        K0 = conv0_w.shape[2]
        PAD0 = K0 // 2

        # Allocate h0 for conv0 output [N, C_OUT0, T]
        h0 = torch.empty((N, C_OUT0, T), device=device, dtype=dtype)

        # Launch conv1d kernel for conv0
        grid0 = (N, C_OUT0, _ceil_div(T, 64))  # T is large, 64 is a reasonable block for time
        conv1d_forward_kernel[grid0](
            x0, conv0_w, conv0_b, h0,
            N, T, T, C_IN0, C_OUT0, K0, PAD0,
            x0.stride(0), x0.stride(1), x0.stride(2),
            conv0_w.stride(0), conv0_w.stride(1), conv0_w.stride(2),
            h0.stride(0), h0.stride(1), h0.stride(2),
            0, 64,
            num_warps=4, num_stages=2,
        )

        # ReLU for h0
        h0_relu = torch.empty_like(h0)
        grid_relu = (N * C_OUT0 * T,)
        relu_forward_kernel[grid_relu](
            h0, h0_relu,
            N, C_OUT0, T,
            h0.stride(0), h0.stride(1), h0.stride(2),
            h0_relu.stride(0), h0_relu.stride(1), h0_relu.stride(2),
            grid_relu,
            num_warps=4, num_stages=2,
        )

        # conv1: [C_OUT1=192, C_IN1=192, K=5]
        C_OUT1 = conv1_w.shape[0]
        C_IN1 = conv1_w.shape[1]
        K1 = conv1_w.shape[2]
        PAD1 = K1 // 2

        h1 = torch.empty((N, C_OUT1, T), device=device, dtype=dtype)

        grid1 = (N, C_OUT1, _ceil_div(T, 64))
        conv1d_forward_kernel[grid1](
            h0_relu, conv1_w, conv1_b, h1,
            N, C_OUT0, C_OUT1, K1, PAD1,
            h0_relu.stride(0), h0_relu.stride(1), h0_relu.stride(2),
            conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2),
            h1.stride(0), h1.stride(1), h1.stride(2),
            0, 64,
            num_warps=4, num_stages=2,
        )

        # ReLU for h1
        h1_relu = torch.empty_like(h1)
        grid_relu2 = (N * C_OUT1 * T,)
        relu_forward_kernel[grid_relu2](
            h1, h1_relu,
            N, C_OUT1, T,
            h1.stride(0), h1.stride(1), h1.stride(2),
            h1_relu.stride(0), h1_relu.stride(1), h1_relu.stride(2),
            grid_relu2,
            num_warps=4, num_stages=2,
        )

        # conv2: [C_OUT2=96, C_IN2=192, K=5]
        C_OUT2 = conv2_w.shape[0]
        C_IN2 = conv2_w.shape[1]
        K2 = conv2_w.shape[2]
        PAD2 = K2 // 2

        h = torch.empty((N, C_OUT2, T), device=device, dtype=dtype)

        grid2 = (N, C_OUT2, _ceil_div(T, 64))
        conv1d_forward_kernel[grid2](
            h1_relu, conv2_w, conv2_b, h,
            N, C_OUT1, C_OUT2, K2, PAD2,
            h1_relu.stride(0), h1_relu.stride(1), h1_relu.stride(2),
            conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2),
            h.stride(0), h.stride(1), h.stride(2),
            0, 64,
            num_warps=4, num_stages=2,
        )
        return h

    # Main loop over transforms
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

    # Split input into halves along channel
    x0 = x[:, :C_HALF, :]
    x1 = x[:, C_HALF:, :]

    if not reverse:
        # Forward: x1 += transform(x0) for each transform
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            # Compute h for this transform
            h = apply_transform_triton(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b)

            # Mask multiply (generic, though mask is ones here)
            h_masked = torch.empty_like(h)
            grid_mask = (N * h.shape[1] * T,)
            mask_mul_kernel[grid_mask](
                h, x_mask, h_masked,
                N, h.shape[1], T,
                h.stride(0), h.stride(1), h.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                num_warps=4, num_stages=2,
            )

            # Affine coupling: x1 = x1 + h
            out_x1 = torch.empty_like(x1)
            grid_add = (N * x1.shape[1] * T,)
            add_affine_kernel[grid_add](
                x1, h_masked, out_x1,
                N, x1.shape[1], T,
                x1.stride(0), x1.stride(1), x1.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                out_x1.stride(0), out_x1.stride(1), out_x1.stride(2),
                num_warps=4, num_stages=2,
            )

            # Concatenate halves: out has shape [N, 2*C_HALF, T]
            out = torch.empty((N, 2 * C_HALF, T), device=device, dtype=dtype)
            # Copy x0 into first half
            grid_copy0 = (N, C_HALF, T)
            concat_half_channels_kernel[grid_copy0](
                x0, x1, out,
                N, C_HALF, T,
                x0.stride(0), x0.stride(1), x0.stride(2),
                x1.stride(0), x1.stride(1), x1.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                mode=0,
                num_warps=4, num_stages=2,
            )
            # Copy out_x1 into second half
            grid_copy1 = (N, C_HALF, T)
            concat_half_channels_kernel[grid_copy1](
                x0, out_x1, out,
                N, C_HALF, T,
                x0.stride(0), x0.stride(1), x0.stride(2),
                out_x1.stride(0), out_x1.stride(1), out_x1.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                mode=1,
                num_warps=4, num_stages=2,
            )

            # Update x to the new concatenated tensor for next transform
            x = out

    else:
        # Reverse: x1 -= transform(x0) for each transform in reversed order
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
            h = apply_transform_triton(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b)

            h_masked = torch.empty_like(h)
            grid_mask = (N * h.shape[1] * T,)
            mask_mul_kernel[grid_mask](
                h, x_mask, h_masked,
                N, h.shape[1], T,
                h.stride(0), h.stride(1), h.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                num_warps=4, num_stages=2,
            )

            out_x1 = torch.empty_like(x1)
            grid_sub = (N * x1.shape[1] * T,)
            sub_affine_kernel[grid_sub](
                x1, h_masked, out_x1,
                N, x1.shape[1], T,
                x1.stride(0), x1.stride(1), x1.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                out_x1.stride(0), out_x1.stride(1), out_x1.stride(2),
                num_warps=4, num_stages=2,
            )

            # Concatenate: first half from x0, second half from out_x1
            out = torch.empty((N, 2 * C_HALF, T), device=device, dtype=dtype)
            grid_copy0 = (N, C_HALF, T)
            concat_half_channels_kernel[grid_copy0](
                x0, x1, out,
                N, C_HALF, T,
                x0.stride(0), x0.stride(1), x0.stride(2),
                x1.stride(0), x1.stride(1), x1.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                mode=0,
                num_warps=4, num_stages=2,
            )
            grid_copy1 = (N, C_HALF, T)
            concat_half_channels_kernel[grid_copy1](
                x0, out_x1, out,
                N, C_HALF, T,
                x0.stride(0), x0.stride(1), x0.stride(2),
                out_x1.stride(0), out_x1.stride(1), out_x1.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                mode=1,
                num_warps=4, num_stages=2,
            )

            x = out

    # After loop, apply x_mask to output (though mask is ones, keep generic)
    x_masked = torch.empty_like(x)
    grid_mask_final = (N * x.shape[1] * T,)
    mask_mul_kernel[grid_mask_final](
        x, x_mask, x_masked,
        N, x.shape[1], T,
        x.stride(0), x.stride(1), x.stride(2),
        x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
        x_masked.stride(0), x_masked.stride(1), x_masked.stride(2),
        num_warps=4, num_stages=2,
    )
    return x_masked


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Mirror the original signature and run the Triton-only implementation.
        return run_triton(*args)


def run(*args):
    return ModelNew()(*args)
