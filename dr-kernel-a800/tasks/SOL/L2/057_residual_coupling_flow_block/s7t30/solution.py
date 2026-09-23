# Triton-based implementation with TRITON-ONLY computation

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# 1) Triton Conv1d forward: direct accumulation with padding
if TRITON_AVAILABLE:
    @triton.jit
    def conv1d_forward_kernel(
        x_ptr,         # *fp32, [N, C_IN, T_IN]
        w_ptr,         # *fp32, [C_OUT, C_IN, K]
        b_ptr,         # *fp32, [C_OUT]
        y_ptr,         # *fp32, [N, C_OUT, T_OUT]
        N, T_IN, T_OUT, C_IN: tl.constexpr, C_OUT: tl.constexpr, K: tl.constexpr, PAD: tl.constexpr,
        x_stride_n, x_stride_c, x_stride_t,
        w_stride_co, w_stride_ci, w_stride_k,
        y_stride_n, y_stride_c, y_stride_t,
        t_block_start: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        # Program ids
        pid_n = tl.program_id(0)
        pid_co = tl.program_id(1)
        pid_tb = tl.program_id(2)

        # Time offsets computed by this program
        t_offsets = t_block_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T_OUT

        # Accumulator
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

        # Add bias
        b_val = tl.load(b_ptr + pid_co)
        acc += b_val

        # Store y[n, co, t_offsets]
        y_ptrs = y_ptr + pid_n * y_stride_n + pid_co * y_stride_c + t_offsets * y_stride_t
        tl.store(y_ptrs, acc, mask=t_mask)

    # 2) Triton ReLU forward
    @triton.jit
    def relu_forward_kernel(
        inp_ptr,        # *fp32, input [NC, T]
        out_ptr,        # *fp32, output [NC, T]
        NC, T,
        in_stride_n, in_stride_c, in_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        grid0: tl.constexpr,  # grid[0] = NC
        grid1: tl.constexpr,  # grid[1] = number of tiles along T
        BLOCK_T: tl.constexpr,
    ):
        pid_n = tl.program_id(0)  # 0..NC-1
        pid_tb = tl.program_id(1)
        t_start = pid_tb * BLOCK_T
        t_offsets = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T

        inp_ptrs = inp_ptr + pid_n * in_stride_n + t_offsets * in_stride_t
        out_ptrs = out_ptr + pid_n * out_stride_n + t_offsets * out_stride_t

        x = tl.load(inp_ptrs, mask=t_mask, other=0.0)
        y = tl.maximum(x, 0.0)
        tl.store(out_ptrs, y, mask=t_mask)

    # 3) Concatenate two channel halves along channels: out = [x0, x1]
    # Assumes x0 has shape [N, C_half, T], x1 has shape [N, C_half, T], out has shape [N, 2*C_half, T].
    @triton.jit
    def concat_half_channels_kernel(
        x0_ptr,         # *fp32, [N, C_half, T]
        x1_ptr,         # *fp32, [N, C_half, T]
        out_ptr,        # *fp32, [N, 2*C_half, T]
        N, C_half, T,
        x0_stride_n, x0_stride_c, x0_stride_t,
        x1_stride_n, x1_stride_c, x1_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        c_block_start: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_tb = tl.program_id(2)

        c_offsets = c_block_start + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C_half

        # For first half channels from x0
        x0_ptrs = x0_ptr + pid_n * x0_stride_n + c_offsets * x0_stride_c + pid_tb * BLOCK_T * x0_stride_t
        x0_vals = tl.load(x0_ptrs, mask=c_mask, other=0.0)

        # For second half channels from x1, shifted by C_half
        x1_ptrs = x1_ptr + pid_n * x1_stride_n + c_offsets * x1_stride_c + pid_tb * BLOCK_T * x1_stride_t
        x1_vals = tl.load(x1_ptrs, mask=c_mask, other=0.0)

        # Store into out: first half
        out_ptrs_first = out_ptr + pid_n * out_stride_n + (c_offsets) * out_stride_c + pid_tb * BLOCK_T * out_stride_t
        tl.store(out_ptrs_first, x0_vals, mask=c_mask)

        # Store into out: second half
        out_ptrs_second = out_ptr + pid_n * out_stride_n + (c_offsets + C_half) * out_stride_c + pid_tb * BLOCK_T * out_stride_t
        tl.store(out_ptrs_second, x1_vals, mask=c_mask)

    # 4) Elementwise add and subtract (affine coupling)
    @triton.jit
    def add_affine_kernel(
        a_ptr, b_ptr, out_ptr,
        NC, T,
        a_stride_n, a_stride_c, a_stride_t,
        b_stride_n, b_stride_c, b_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK_T: tl.constexpr,
    ):
        pid_n = tl.program_id(0)  # 0..NC-1
        pid_tb = tl.program_id(1)
        t_start = pid_tb * BLOCK_T
        t_offsets = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T

        a_ptrs = a_ptr + pid_n * a_stride_n + t_offsets * a_stride_t
        b_ptrs = b_ptr + pid_n * b_stride_n + t_offsets * b_stride_t
        out_ptrs = out_ptr + pid_n * out_stride_n + t_offsets * out_stride_t

        a = tl.load(a_ptrs, mask=t_mask, other=0.0)
        b = tl.load(b_ptrs, mask=t_mask, other=0.0)
        out = a + b
        tl.store(out_ptrs, out, mask=t_mask)

    @triton.jit
    def sub_affine_kernel(
        a_ptr, b_ptr, out_ptr,
        NC, T,
        a_stride_n, a_stride_c, a_stride_t,
        b_stride_n, b_stride_c, b_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK_T: tl.constexpr,
    ):
        pid_n = tl.program_id(0)  # 0..NC-1
        pid_tb = tl.program_id(1)
        t_start = pid_tb * BLOCK_T
        t_offsets = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T

        a_ptrs = a_ptr + pid_n * a_stride_n + t_offsets * a_stride_t
        b_ptrs = b_ptr + pid_n * b_stride_n + t_offsets * b_stride_t
        out_ptrs = out_ptr + pid_n * out_stride_n + t_offsets * out_stride_t

        a = tl.load(a_ptrs, mask=t_mask, other=0.0)
        b = tl.load(b_ptrs, mask=t_mask, other=0.0)
        out = a - b
        tl.store(out_ptrs, out, mask=t_mask)

    # 5) Elementwise mask multiply (generic, though mask is ones in given setup)
    @triton.jit
    def mask_mul_kernel(
        inp_ptr, mask_ptr, out_ptr,
        N, C, T,
        inp_stride_n, inp_stride_c, inp_stride_t,
        mask_stride_n, mask_stride_c, mask_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK_T: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_tb = tl.program_id(2)

        t_start = pid_tb * BLOCK_T
        t_offsets = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T

        inp_ptrs = inp_ptr + pid_n * inp_stride_n + pid_c * inp_stride_c + t_offsets * inp_stride_t
        mask_ptrs = mask_ptr + pid_n * mask_stride_n + pid_c * mask_stride_c + t_offsets * mask_stride_t
        out_ptrs = out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + t_offsets * out_stride_t

        x = tl.load(inp_ptrs, mask=t_mask, other=0.0)
        m = tl.load(mask_ptrs, mask=t_mask, other=1.0)  # mask is ones in given setup
        y = x * m
        tl.store(out_ptrs, y, mask=t_mask)


# Host-side ModelNew with Triton kernels only (no torch ops)
class ModelNew(torch.nn.Module):
    def forward(self, *args):
        """
        This mirrors the original 'run' signature. The original run takes:
        - x: [N, C, T]
        - x_mask: [N, 1, T]
        - reverse: bool
        - 4 sets of weights/biases for 3-conv transforms
        """
        # Extract inputs
        # Note: In the original run, the caller passes all tensors. Here we assume the same signature.
        x = args[0]  # [N, C, T]
        x_mask = args[1]  # [N, 1, T]
        reverse = bool(args[2])
        # The next 16 are weights/biases for 4 transforms:
        # transform_0_conv0_weight, transform_0_conv0_bias,
        # transform_0_conv1_weight, transform_0_conv1_bias,
        # transform_0_conv2_weight, transform_0_conv2_bias,
        # transform_1_conv0_weight, transform_1_conv0_bias,
        # transform_1_conv1_weight, transform_1_conv1_bias,
        # transform_1_conv2_weight, transform_1_conv2_bias,
        # transform_2_conv0_weight, transform_2_conv0_bias,
        # transform_2_conv1_weight, transform_2_conv1_bias,
        # transform_2_conv2_weight, transform_2_conv2_bias,
        # transform_3_conv0_weight, transform_3_conv0_bias,
        # transform_3_conv1_weight, transform_3_conv1_bias,
        # transform_3_conv2_weight, transform_3_conv2_bias,
        # We'll iterate them 4 times (same as original).

        # Sanity: require CUDA for Triton
        if not TRITON_AVAILABLE:
            # Fallback to original torch ops if Triton unavailable (but evaluation should have Triton)
            raise RuntimeError("Triton is not available. Please ensure Triton is installed and tensors are on CUDA.")

        N, C, T = x.shape
        C_half = C // 2
        K = 5
        PAD = K // 2  # 2
        T_OUT = T  # padding keeps time length

        # We'll operate in float32 (inputs from get_inputs are float32). If not, cast.
        if x.dtype != torch.float32:
            x = x.float()
        if x_mask.dtype != torch.float32:
            x_mask = x_mask.float()

        # Process 4 transforms as in original run
        for _ in range(4):
            # Extract the current set of 3 convs weights/biases for a single transform (original loop)
            # The caller passes 24 tensors after x, x_mask, reverse.
            # Mapping:
            # i = 0 -> conv0_w, conv0_b; i = 1 -> conv1_w, conv1_b; i = 2 -> conv2_w, conv2_b
            # There are 4 such sets. We'll read them sequentially from args.
            conv0_w = args[3 + 0]
            conv0_b = args[3 + 1]
            conv1_w = args[3 + 2]
            conv1_b = args[3 + 3]
            conv2_w = args[3 + 4]
            conv2_b = args[3 + 5]

            # Split into halves
            x0 = x[:, :C_half, :]
            x1 = x[:, C_half:, :]

            # Compute h = apply_transform(x0) using Triton convs+ReLU
            # h0 = conv0
            h = x0  # placeholder, will overwrite

            # conv0
            h0 = self.conv1d_forward(x0, conv0_w, conv0_b, N, C_half, conv0_w.shape[0], T, conv0_w.shape[2], PAD)
            h0 = self.relu_forward(h0)

            # conv1
            h1 = self.conv1d_forward(h0, conv1_w, conv1_b, N, conv1_w.shape[1], conv1_w.shape[0], T, conv1_w.shape[2], PAD)
            h1 = self.relu_forward(h1)

            # conv2
            h = self.conv1d_forward(h1, conv2_w, conv2_b, N, conv2_w.shape[1], conv2_w.shape[0], T, conv2_w.shape[2], PAD)

            # Affine coupling: x1 = x1 + h (forward) or x1 = x1 - h (reverse)
            if not reverse:
                # Elementwise add
                x1 = self.add_affine(x1, h)
            else:
                # Elementwise subtract (reverse order)
                x1 = self.sub_affine(x1, h)

            # Concatenate back along channel dimension: out has 2*C_half channels
            out = self.concat_half_channels(x0, x1, N, C_half, T)

            # Apply mask (mask is ones in given setup; keep generic)
            # Note: x_mask shape is [N, 1, T]; broadcast over channels
            out = self.mask_mul(out, x_mask, N, out.shape[1], T)

            # Update x for next iteration
            x = out

        return out

    # Helper methods to launch Triton kernels
    def conv1d_forward(self, x, w, b, N, C_IN, C_OUT, T_IN, K, PAD):
        # x: [N, C_IN, T_IN], w: [C_OUT, C_IN, K], b: [C_OUT]
        assert x.is_cuda and w.is_cuda and b.is_cuda
        # Allocate output [N, C_OUT, T_OUT]
        T_OUT = T_IN
        y = torch.empty((N, C_OUT, T_OUT), device=x.device, dtype=torch.float32)

        # Strides
        x_stride_n, x_stride_c, x_stride_t = x.stride()
        w_stride_co, w_stride_ci, w_stride_k = w.stride()
        y_stride_n, y_stride_c, y_stride_t = y.stride()

        # Grid over (N, C_OUT, time blocks)
        BLOCK_T = 128
        grid = (N, C_OUT, triton.cdiv(T_OUT, BLOCK_T))

        # Launch Triton conv1d kernel
        conv1d_forward_kernel[grid](
            x, w, b, y,
            N, T_IN, T_OUT, C_IN, C_OUT, K, PAD,
            x_stride_n, x_stride_c, x_stride_t,
            w_stride_co, w_stride_ci, w_stride_k,
            y_stride_n, y_stride_c, y_stride_t,
            0, BLOCK_T, num_warps=4, num_stages=2
        )
        return y

    def relu_forward(self, inp):
        # inp: [N, C, T], output same shape
        N, C, T = inp.shape
        out = torch.empty_like(inp)
        # Strides
        in_stride_n, in_stride_c, in_stride_t = inp.stride()
        out_stride_n, out_stride_c, out_stride_t = out.stride()
        # Grid over (N*C, time tiles)
        BLOCK_T = 128
        grid = (N * C, triton.cdiv(T, BLOCK_T))
        # Launch Triton ReLU kernel (elementwise over [NC, T] logical view)
        relu_forward_kernel[grid](
            inp, out,
            N * C, T,
            in_stride_n, in_stride_c, in_stride_t,
            out_stride_n, out_stride_c, out_stride_t,
            N * C, triton.cdiv(T, BLOCK_T), BLOCK_T,
            num_warps=4, num_stages=2
        )
        return out

    def concat_half_channels(self, x0, x1, N, C_half, T):
        # x0: [N, C_half, T], x1: [N, C_half, T], out: [N, 2*C_half, T]
        out = torch.empty((N, 2 * C_half, T), device=x0.device, dtype=torch.float32)
        # Strides
        x0_stride_n, x0_stride_c, x0_stride_t = x0.stride()
        x1_stride_n, x1_stride_c, x1_stride_t = x1.stride()
        out_stride_n, out_stride_c, out_stride_t = out.stride()

        BLOCK_C = 64  # channels per tile
        BLOCK_T = 128 # time per tile

        grid = (N, triton.cdiv(C_half, BLOCK_C), triton.cdiv(T, BLOCK_T))

        concat_half_channels_kernel[grid](
            x0, x1, out,
            N, C_half, T,
            x0_stride_n, x0_stride_c, x0_stride_t,
            x1_stride_n, x1_stride_c, x1_stride_t,
            out_stride_n, out_stride_c, out_stride_t,
            0, BLOCK_C, BLOCK_T,
            num_warps=4, num_stages=2
        )
        return out

    def add_affine(self, a, b):
        # a, b: [N, C, T], elementwise a + b
        N, C, T = a.shape
        out = torch.empty_like(a)
        # Strides
        a_stride_n, a_stride_c, a_stride_t = a.stride()
        b_stride_n, b_stride_c, b_stride_t = b.stride()
        out_stride_n, out_stride_c, out_stride_t = out.stride()

        grid = (N * C, triton.cdiv(T, 128))
        add_affine_kernel[grid](
            a, b, out,
            N * C, T,
            a_stride_n, a_stride_c, a_stride_t,
            b_stride_n, b_stride_c, b_stride_t,
            out_stride_n, out_stride_c, out_stride_t,
            128, num_warps=4, num_stages=2
        )
        return out

    def sub_affine(self, a, b):
        # a, b: [N, C, T], elementwise a - b
        N, C, T = a.shape
        out = torch.empty_like(a)
        # Strides
        a_stride_n, a_stride_c, a_stride_t = a.stride()
        b_stride_n, b_stride_c, b_stride_t = b.stride()
        out_stride_n, out_stride_c, out_stride_t = out.stride()

        grid = (N * C, triton.cdiv(T, 128))
        sub_affine_kernel[grid](
            a, b, out,
            N * C, T,
            a_stride_n, a_stride_c, a_stride_t,
            b_stride_n, b_stride_c, b_stride_t,
            out_stride_n, out_stride_c, out_stride_t,
            128, num_warps=4, num_stages=2
        )
        return out

    def mask_mul(self, inp, mask, N, C, T):
        # inp: [N, C, T], mask: [N, 1, T] -> broadcast over channels
        out = torch.empty_like(inp)
        # Ensure mask broadcastable: [N, 1, T] and [N, C, 1] would be ideal, but here we rely on broadcasting
        # We load mask per (N, channel, T) by indexing mask[n, 0, t] and using channel stride in out as 1.
        inp_stride_n, inp_stride_c, inp_stride_t = inp.stride()
        mask_stride_n, mask_stride_c, mask_stride_t = mask.stride()  # mask is [N, 1, T]
        out_stride_n, out_stride_c, out_stride_t = out.stride()

        BLOCK_T = 128
        grid = (N, C, triton.cdiv(T, BLOCK_T))

        mask_mul_kernel[grid](
            inp, mask, out,
            N, C, T,
            inp_stride_n, inp_stride_c, inp_stride_t,
            mask_stride_n, mask_stride_c, mask_stride_t,
            out_stride_n, out_stride_c, out_stride_t,
            BLOCK_T, num_warps=4, num_stages=2
        )
        return out


def run(*args):
    return ModelNew()(*args)
