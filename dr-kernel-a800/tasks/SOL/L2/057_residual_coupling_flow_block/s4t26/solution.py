import math
import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    # Kernel: Conv1d forward with ReLU, fixed K=5, padding=PAD=2
    # Computes y[n, co, t] = ReLU( sum_{ci,k} x[n, ci, t + k - PAD] * w[co, ci, k] + b[co] )
    @triton.jit
    def conv1d_relu_triton(
        x_ptr,          # *const float, shape [N, C_in, T_in], contiguous
        w_ptr,          # *const float, shape [C_out, C_in, K], contiguous
        b_ptr,          # *const float, shape [C_out], contiguous
        y_ptr,          # *float,       shape [N, C_out, T_out], contiguous
        N: tl.int32,
        C_in: tl.int32,
        T_in: tl.int32,
        C_out: tl.int32,
        T_out: tl.int32,
        K: tl.constexpr,                  # kernel size (5)
        PAD: tl.constexpr,               # padding (2)
        BLOCK_CO: tl.constexpr,          # tile along output channels
        BLOCK_T: tl.constexpr            # tile along time
    ):
        pid_n = tl.program_id(0)         # batch index
        pid_co_block = tl.program_id(1)  # block along output channels
        pid_t_block = tl.program_id(2)   # block along time

        co_start = pid_co_block * BLOCK_CO
        t_start = pid_t_block * BLOCK_T

        co_offsets = co_start + tl.arange(0, BLOCK_CO)   # [BLOCK_CO]
        t_offsets = t_start + tl.arange(0, BLOCK_T)      # [BLOCK_T]

        co_mask = co_offsets < C_out
        t_mask = t_offsets < T_out
        out_mask = co_mask[:, None] & t_mask[None, :]

        # accumulator [BLOCK_CO, BLOCK_T]
        acc = tl.zeros((BLOCK_CO, BLOCK_T), dtype=tl.float32)

        # loop over input channels and kernel taps
        for ci in range(0, C_in):
            for k in range(0, K):
                t_in = t_offsets + (k - PAD)               # [BLOCK_T]
                in_bounds = (t_in >= 0) & (t_in < T_in) & t_mask  # [BLOCK_T]

                # load x[n, ci, t_in]
                x_offs = ((pid_n * C_in + ci) * T_in) + t_in  # [BLOCK_T]
                x_vals = tl.load(x_ptr + x_offs, mask=in_bounds, other=0.0).to(tl.float32)

                # load weights w[co, ci, k]
                w_offs = co_offsets * (C_in * K) + ci * K + k  # [BLOCK_CO]
                w_vals = tl.load(w_ptr + w_offs, mask=co_mask, other=0.0).to(tl.float32)

                # outer product accumulate
                acc += w_vals[:, None] * x_vals[None, :]

        # add bias and ReLU
        b_vals = tl.load(b_ptr + co_offsets, mask=co_mask, other=0.0).to(tl.float32)
        acc = acc + b_vals[:, None]
        acc = tl.maximum(acc, 0.0)

        # store to y
        y_offs = ((pid_n * C_out + co_offsets[:, None]) * T_out) + t_offsets[None, :]
        tl.store(y_ptr + y_offs, acc, mask=out_mask)


    # Kernel: Split input x into two halves along channels: out0 = x[:, :C_half, :], out1 = x[:, C_half:, :]
    @triton.jit
    def split_halves_triton(
        x_ptr,       # *const float, shape [N, C, T], contiguous
        out0_ptr,    # *float,       shape [N, C_half, T], contiguous
        out1_ptr,    # *float,       shape [N, C_half, T], contiguous
        N: tl.int32,
        C_half: tl.int32,
        C: tl.int32,
        T: tl.int32,
        BLOCK_C: tl.constexpr,
        BLOCK_T: tl.constexpr
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        c_start = pid_c * BLOCK_C
        t_start = pid_t * BLOCK_T

        c_offsets = c_start + tl.arange(0, BLOCK_C)  # [BLOCK_C]
        t_offsets = t_start + tl.arange(0, BLOCK_T)  # [BLOCK_T]

        c_mask0 = c_offsets < C_half
        c_mask1 = c_offsets < C_half  # second half also uses C_half channels (after original C_half)
        t_mask = t_offsets < T

        # x0: channels 0..C_half-1
        x_offs = (pid_n * C * T) + c_offsets[:, None] * T + t_offsets[None, :]
        mask0 = c_mask0[:, None] & t_mask[None, :]
        vals0 = tl.load(x_ptr + x_offs, mask=mask0, other=0.0)
        out_offs0 = (pid_n * C_half * T) + c_offsets[:, None] * T + t_offsets[None, :]
        tl.store(out0_ptr + out_offs0, vals0, mask=mask0)

        # x1: channels C_half..C-1, but we take only first C_half
        # Note: out1 corresponds to the second half channels in the original x; here we write the same C_half channels.
        x_offs1 = (pid_n * C * T) + (c_offsets[:, None] + C_half) * T + t_offsets[None, :]
        mask1 = c_mask1[:, None] & t_mask[None, :]
        vals1 = tl.load(x_ptr + x_offs1, mask=mask1, other=0.0)
        out_offs1 = (pid_n * C_half * T) + c_offsets[:, None] * T + t_offsets[None, :]
        tl.store(out1_ptr + out_offs1, vals1, mask=mask1)


    # Kernel: Concatenate two tensors along channel dimension:
    # out has channels C_out = C_half_out1 + C_half_out2
    # out[:, :C_half_out1, :] = in0, out[:, C_half_out1:, :] = in1
    @triton.jit
    def concat_triton(
        in0_ptr,       # *const float, shape [N, C_half, T], contiguous
        in1_ptr,       # *const float, shape [N, C_half, T], contiguous
        out_ptr,       # *float,       shape [N, C_out, T], contiguous
        N: tl.int32,
        C_half: tl.int32,
        T: tl.int32,
        C_out: tl.int32,
        BLOCK_C: tl.constexpr,
        BLOCK_T: tl.constexpr
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        c_start = pid_c * BLOCK_C
        t_start = pid_t * BLOCK_T

        c_offsets = c_start + tl.arange(0, BLOCK_C)  # [BLOCK_C]
        t_offsets = t_start + tl.arange(0, BLOCK_T)  # [BLOCK_T]

        c_mask_in0 = c_offsets < C_half
        c_mask_in1 = c_offsets < C_half
        t_mask = t_offsets < T

        # first half from in0
        in0_offs = (pid_n * C_half * T) + c_offsets[:, None] * T + t_offsets[None, :]
        mask_in0 = c_mask_in0[:, None] & t_mask[None, :]
        vals0 = tl.load(in0_ptr + in0_offs, mask=mask_in0, other=0.0)
        out_offs0 = (pid_n * C_out * T) + c_offsets[:, None] * T + t_offsets[None, :]
        tl.store(out_ptr + out_offs0, vals0, mask=mask_in0)

        # second half from in1, starting at channel offset C_half
        in1_offs = (pid_n * C_half * T) + c_offsets[:, None] * T + t_offsets[None, :]
        mask_in1 = c_mask_in1[:, None] & t_mask[None, :]
        vals1 = tl.load(in1_ptr + in1_offs, mask=mask_in1, other=0.0)
        out_offs1 = (pid_n * C_out * T) + (c_offsets[:, None] + C_half) * T + t_offsets[None, :]
        tl.store(out_ptr + out_offs1, vals1, mask=mask_in1)


    # Kernel: Elementwise multiply tensor by mask (mask has shape [N, 1, T], broadcast across channels)
    @triton.jit
    def mask_mul_triton(
        x_ptr,       # *const float, shape [N, C, T], contiguous
        mask_ptr,    # *const float, shape [N, 1, T], contiguous (we can pass [N, T] too, but here it's [N, 1, T] with stride ignoring channel)
        out_ptr,     # *float,       shape [N, C, T], contiguous
        N: tl.int32,
        C: tl.int32,
        T: tl.int32,
        BLOCK_C: tl.constexpr,
        BLOCK_T: tl.constexpr
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        c_start = pid_c * BLOCK_C
        t_start = pid_t * BLOCK_T

        c_offsets = c_start + tl.arange(0, BLOCK_C)  # [BLOCK_C]
        t_offsets = t_start + tl.arange(0, BLOCK_T)  # [BLOCK_T]

        c_mask = c_offsets < C
        t_mask = t_offsets < T
        mask = c_mask[:, None] & t_mask[None, :]

        # load x
        x_offs = (pid_n * C * T) + c_offsets[:, None] * T + t_offsets[None, :]
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0)

        # load mask (assuming [N, 1, T] layout, we ignore channel in mask indexing by using base pointer + t)
        # mask_ptr has strides: (N, 1, T). We index as (n, 0, t)
        m_offs = (pid_n * 1 * T) + t_offsets  # base for channel 0
        m_vals = tl.load(mask_ptr + m_offs, mask=t_mask, other=1.0).to(tl.float32)  # [BLOCK_T]
        m_vals = m_vals[None, :]  # broadcast across channels

        out_vals = x_vals * m_vals
        out_offs = (pid_n * C * T) + c_offsets[:, None] * T + t_offsets[None, :]
        tl.store(out_ptr + out_offs, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # We expect the same inputs as the original: x, x_mask, reverse, and 16 conv weights/biases.
        # Extract them. We will not use torch ops; only Triton kernels.
        # Note: args order must match original Model.run signature:
        # x: [N, C, T], x_mask: [N, 1, T], reverse: bool, then 4*3 = 12 conv weights/bias tuples.
        # The original run function applies 4 transforms sequentially, each with 3 convs. We will emulate that with Triton.
        # However, since the evaluation requires Triton-only, we implement the full forward using Triton kernels.

        # Extract first argument 'x' which is the input tensor, shape [N, C, T]
        # args[0] should be x
        x = args[0]
        N, C, T = x.shape
        device = x.device
        dtype = x.dtype

        # x_mask: [N, 1, T] (binary mask)
        x_mask = args[1]  # [N, 1, T]

        # Constants for kernels
        half = C // 2  # 96

        # Prepare output buffer for the final result; we will build it step-by-step using Triton kernels.
        # Since Triton kernels cannot modify caller's tensors, we maintain an internal output tensor.
        out = torch.empty((N, C, T), device=device, dtype=dtype)

        # We need to apply 4 transforms sequentially. Each transform:
        # 1) Split x into x0 and x1.
        # 2) Do conv0 -> ReLU -> conv1 -> ReLU -> conv2.
        # 3) h is the result of conv2; update x1 = x1 + h; concatenate to produce next x.
        # We implement this in Triton.

        # To avoid dynamic tuple unpacking complexity, we iterate over the args in 3-conv chunks manually.
        # Original signature expects exactly 16 conv weights/biases (4 transforms, each 3): w0, b0, w1, b1, w2, b2, ... w15, b15.
        # We will read them via args slicing.

        # Number of conv weights provided: len(args) - 4 (since first 4 are x, x_mask, reverse, remaining are weights/biases)
        # We'll process 4 transforms in a loop: each has 3 convs.
        transforms = 4
        per_transform = 3

        for t in range(transforms):
            # Prepare x0 and x1: x0 = x[:, :half, :], x1 = x[:, half:, :]
            x0 = torch.empty((N, half, T), device=device, dtype=dtype)
            x1 = torch.empty((N, half, T), device=device, dtype=dtype)

            # Launch split kernel
            BLOCK_C_SPLIT = 64
            BLOCK_T_SPLIT = 128
            grid_split0 = (N, triton.cdiv(half, BLOCK_C_SPLIT), triton.cdiv(T, BLOCK_T_SPLIT))
            split_halves_triton[grid_split0](
                x, x0, x1,
                N, half, C, T,
                BLOCK_C=BLOCK_C_SPLIT, BLOCK_T=BLOCK_T_SPLIT
            )

            # Now apply 3 convs: conv0 -> ReLU -> conv1 -> ReLU -> conv2
            # Read conv parameters for this transform.
            # Indexing: conv0 = t*per_transform, conv1 = t*per_transform + 1, conv2 = t*per_transform + 2
            conv0_w = args[2 + t * per_transform]  # shape [C_out, C_in, K] where C_out=C, C_in=half, K=5
            conv0_b = args[3 + t * per_transform]  # shape [C_out]
            conv1_w = args[2 + t * per_transform + 1]
            conv1_b = args[3 + t * per_transform + 1]
            conv2_w = args[2 + t * per_transform + 2]
            conv2_b = args[3 + t * per_transform + 2]

            C_out0, C_in0, K0 = conv0_w.shape  # C_out0 = C (192), C_in0 = half (96), K0 = 5
            C_out1, C_in1, K1 = conv1_w.shape  # C_out1 = C (192), C_in1 = C_out0 (192), K1 = 5
            C_out2, C_in2, K2 = conv2_w.shape  # C_out2 = half (96), C_in2 = C_out1 (192), K2 = 5

            # Allocate h0, h1, h2 outputs for convs
            h0 = torch.empty((N, C_out0, T), device=device, dtype=dtype)   # [N, 192, T]
            h1 = torch.empty((N, C_out1, T), device=device, dtype=dtype)   # [N, 192, T]
            h2 = torch.empty((N, C_out2, T), device=device, dtype=dtype)   # [N, 96, T]

            # Launch conv0 + ReLU
            BLOCK_CO0 = 64
            BLOCK_T0 = 128
            grid_conv0 = (N, triton.cdiv(C_out0, BLOCK_CO0), triton.cdiv(T, BLOCK_T0))
            conv1d_relu_triton[grid_conv0](
                x0, conv0_w, conv0_b, h0,
                N, C_in0, T, C_out0, T, K0, 2,
                BLOCK_CO=BLOCK_CO0, BLOCK_T=BLOCK_T0
            )

            # conv1 + ReLU on h0
            BLOCK_CO1 = 64
            BLOCK_T1 = 128
            grid_conv1 = (N, triton.cdiv(C_out1, BLOCK_CO1), triton.cdiv(T, BLOCK_T1))
            conv1d_relu_triton[grid_conv1](
                h0, conv1_w, conv1_b, h1,
                N, C_in1, T, C_out1, T, K1, 2,
                BLOCK_CO=BLOCK_CO1, BLOCK_T=BLOCK_T1
            )

            # conv2 on h1 (final h = conv2 output, shape [N, 96, T])
            BLOCK_CO2 = 64
            BLOCK_T2 = 128
            grid_conv2 = (N, triton.cdiv(C_out2, BLOCK_CO2), triton.cdiv(T, BLOCK_T2))
            conv1d_relu_triton[grid_conv2](
                h1, conv2_w, conv2_b, h2,
                N, C_in2, T, C_out2, T, K2, 2,
                BLOCK_CO=BLOCK_CO2, BLOCK_T=BLOCK_T2
            )

            # Update x1: x1 = x1 + h2
            # We need to add h2 (N, 96, T) to x1 (N, 96, T)
            # Launch elementwise add kernel (we can reuse split kernel tiling or just simple 3D grid)
            # Allocate x1_plus_h2
            x1_plus_h2 = torch.empty_like(x1)

            BLOCK_C_ADD = 64
            BLOCK_T_ADD = 128
            grid_add = (N, triton.cdiv(half, BLOCK_C_ADD), triton.cdiv(T, BLOCK_T_ADD))
            add_triton[grid_add](
                x1, h2,
                x1_plus_h2,
                N, half, T,
                BLOCK_C=BLOCK_C_ADD, BLOCK_T=BLOCK_T_ADD
            )

            # Now concatenate [x0, x1_plus_h2] along channel dimension to produce next x (channels = 192)
            next_out = torch.empty((N, C, T), device=device, dtype=dtype)

            # x0 has channels half=96, x1_plus_h2 also has channels half=96, total 192
            BLOCK_C_CONCAT = 64
            BLOCK_T_CONCAT = 128
            grid_concat = (N, triton.cdiv(C, BLOCK_C_CONCAT), triton.cdiv(T, BLOCK_T_CONCAT))
            # We need in0=x0, in1=x1_plus_h2, out=next_out
            concat_triton[grid_concat](
                x0, x1_plus_h2, next_out,
                N, half, T, C,
                BLOCK_C=BLOCK_C_CONCAT, BLOCK_T=BLOCK_T_CONCAT
            )

            # Apply mask: out = out * x_mask (broadcast along channel)
            # Launch mask_mul_triton
            # x_mask shape is [N, 1, T]; Triton kernel expects [N, C, T] for out, we will broadcast along channels by multiplying with mask[:, 0, :].
            # Create a mask tensor with channel dimension for simplicity: we broadcast by using the same mask across channels.
            # But mask_mul_triton assumes mask is [N, 1, T]; we pass it as-is and ignore channel index in mask (we multiply across all channels).
            BLOCK_C_MASK = 128
            BLOCK_T_MASK = 128
            grid_mask = (N, triton.cdiv(C, BLOCK_C_MASK), triton.cdiv(T, BLOCK_T_MASK))
            mask_mul_triton[grid_mask](
                next_out, x_mask,
                next_out,
                N, C, T,
                BLOCK_C=BLOCK_C_MASK, BLOCK_T=BLOCK_T_MASK
            )

            # Update x for next iteration
            x = next_out

        # Return final x
        return x


def run(*args):
    return ModelNew()(*args)
