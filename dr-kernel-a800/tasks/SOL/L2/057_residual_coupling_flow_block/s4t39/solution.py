import math
import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: Conv1d with fixed K=5, padding=2, and ReLU
# Computes y[n, co, t_out] = ReLU( sum_{ci=0..C_in-1, k=0..4} x[n, ci, t_out + k - 2] * w[co, ci, k] + b[co] )
if TRITON_AVAILABLE:
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
        K: tl.constexpr,            # kernel size (5)
        PAD: tl.constexpr,          # padding (2)
        BLOCK_CO: tl.constexpr,     # tile along output channels
        BLOCK_T: tl.constexpr       # tile along time
    ):
        pid_n = tl.program_id(0)    # batch index
        pid_co = tl.program_id(1)   # block index along output channels
        pid_t = tl.program_id(2)    # block index along time

        co_start = pid_co * BLOCK_CO
        t_start = pid_t * BLOCK_T

        co_offsets = co_start + tl.arange(0, BLOCK_CO)   # [BLOCK_CO]
        t_offsets = t_start + tl.arange(0, BLOCK_T)      # [BLOCK_T]

        co_mask = co_offsets < C_out
        t_mask = t_offsets < T_out
        mask_out = co_mask[:, None] & t_mask[None, :]

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
        tl.store(y_ptr + y_offs, acc, mask=mask_out)


    @triton.jit
    def split_copy_triton(
        in_ptr,         # *const float, shape [N, 2*C_in, T], input x
        out0_ptr,       # *float,       shape [N, C_in, T], output for first half
        out1_ptr,       # *float,       shape [N, C_in, T], output for second half
        N: tl.int32,
        C_in: tl.int32,
        T: tl.int32,
        BLOCK_C: tl.constexpr,
        BLOCK_T: tl.constexpr
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        c_start = pid_c * BLOCK_C
        t_start = pid_t * BLOCK_T

        c_offsets = c_start + tl.arange(0, BLOCK_C)   # [BLOCK_C]
        t_offsets = t_start + tl.arange(0, BLOCK_T)   # [BLOCK_T]

        c_mask = c_offsets < C_in
        t_mask = t_offsets < T
        mask = c_mask[:, None] & t_mask[None, :]

        # First half channels [0..C_in-1]
        in_offs0 = ((pid_n * (2 * C_in) + c_offsets[:, None]) * T) + t_offsets[None, :]
        out_offs0 = ((pid_n * C_in + c_offsets[:, None]) * T) + t_offsets[None, :]
        vals0 = tl.load(in_ptr + in_offs0, mask=mask, other=0.0).to(tl.float32)
        tl.store(out0_ptr + out_offs0, vals0, mask=mask)

        # Second half channels [C_in..2*C_in-1]
        in_offs1 = ((pid_n * (2 * C_in) + (c_offsets[:, None] + C_in)) * T) + t_offsets[None, :]
        out_offs1 = ((pid_n * C_in + c_offsets[:, None]) * T) + t_offsets[None, :]
        vals1 = tl.load(in_ptr + in_offs1, mask=mask, other=0.0).to(tl.float32)
        tl.store(out1_ptr + out_offs1, vals1, mask=mask)


    @triton.jit
    def add_sub_second_triton(
        x1_ptr,         # *float, shape [N, C_in, T], second half input (current x1)
        h_ptr,          # *const float, shape [N, C_in, T], h from conv2 of current transform
        out_ptr,        # *float, shape [N, C_in, T], output x1 updated (x1 + h or x1 - h)
        N: tl.int32,
        C_in: tl.int32,
        T: tl.int32,
        ADD: tl.constexpr,           # True for forward (add), False for reverse (subtract)
        BLOCK_C: tl.constexpr,
        BLOCK_T: tl.constexpr
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        c_start = pid_c * BLOCK_C
        t_start = pid_t * BLOCK_T

        c_offsets = c_start + tl.arange(0, BLOCK_C)   # [BLOCK_C]
        t_offsets = t_start + tl.arange(0, BLOCK_T)   # [BLOCK_T]

        c_mask = c_offsets < C_in
        t_mask = t_offsets < T
        mask = c_mask[:, None] & t_mask[None, :]

        x1_offs = ((pid_n * C_in + c_offsets[:, None]) * T) + t_offsets[None, :]
        x1_vals = tl.load(x1_ptr + x1_offs, mask=mask, other=0.0).to(tl.float32)

        h_offs = ((pid_n * C_in + c_offsets[:, None]) * T) + t_offsets[None, :]
        h_vals = tl.load(h_ptr + h_offs, mask=mask, other=0.0).to(tl.float32)

        delta = h_vals if ADD else (-h_vals)
        new_vals = x1_vals + delta

        tl.store(out_ptr + x1_offs, new_vals, mask=mask)


    @triton.jit
    def concat_halves_triton(
        x0_ptr,         # *const float, shape [N, C_in, T], first half
        x1_ptr,         # *const float, shape [N, C_in, T], second half (updated)
        out_ptr,        # *float,       shape [N, 2*C_in, T], output concatenated
        N: tl.int32,
        C_in: tl.int32,
        T: tl.int32,
        BLOCK_C: tl.constexpr,
        BLOCK_T: tl.constexpr
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        c_start = pid_c * BLOCK_C
        t_start = pid_t * BLOCK_T

        c_offsets = c_start + tl.arange(0, BLOCK_C)   # [BLOCK_C]
        t_offsets = t_start + tl.arange(0, BLOCK_T)   # [BLOCK_T]

        c_mask = c_offsets < C_in
        t_mask = t_offsets < T
        mask = c_mask[:, None] & t_mask[None, :]

        # x0 (first half): channels 0..C_in-1, output channels 0..C_in-1
        x0_offs = ((pid_n * C_in + c_offsets[:, None]) * T) + t_offsets[None, :]
        x0_vals = tl.load(x0_ptr + x0_offs, mask=mask, other=0.0).to(tl.float32)

        # x1 (second half): channels 0..C_in-1, output channels C_in..2*C_in-1
        x1_offs = ((pid_n * C_in + c_offsets[:, None]) * T) + t_offsets[None, :]
        x1_vals = tl.load(x1_ptr + x1_offs, mask=mask, other=0.0).to(tl.float32)

        # store x0
        out_offs0 = ((pid_n * (2 * C_in) + c_offsets[:, None]) * T) + t_offsets[None, :]
        tl.store(out_ptr + out_offs0, x0_vals, mask=mask)

        # store x1 at offset C_in
        out_offs1 = ((pid_n * (2 * C_in) + (c_offsets[:, None] + C_in)) * T) + t_offsets[None, :]
        tl.store(out_ptr + out_offs1, x1_vals, mask=mask)


    @triton.jit
    def mask_mul_triton(
        x_ptr,         # *const float, shape [N, C, T]
        mask_ptr,      # *const float, shape [N, 1, T], mask[:, 0, :]
        out_ptr,       # *float,       shape [N, C, T]
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

        c_offsets = c_start + tl.arange(0, BLOCK_C)   # [BLOCK_C]
        t_offsets = t_start + tl.arange(0, BLOCK_T)   # [BLOCK_T]

        c_mask = c_offsets < C
        t_mask = t_offsets < T
        mask = c_mask[:, None] & t_mask[None, :]

        # load x
        x_offs = ((pid_n * C + c_offsets[:, None]) * T) + t_offsets[None, :]
        x = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

        # load mask per batch, per time (channel is broadcast)
        mask_offs = ((pid_n * 1 + 0) * T) + t_offsets  # mask has Cdim=1
        m = tl.load(mask_ptr + mask_offs, mask=t_mask, other=1.0).to(tl.float32)

        y = x * m[None, :]  # broadcast along channel dimension
        tl.store(out_ptr + x_offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                x_mask: torch.Tensor,
                reverse: bool,
                # weights for transform 0
                transform_0_conv0_weight: torch.Tensor,
                transform_0_conv0_bias: torch.Tensor,
                transform_0_conv1_weight: torch.Tensor,
                transform_0_conv1_bias: torch.Tensor,
                transform_0_conv2_weight: torch.Tensor,
                transform_0_conv2_bias: torch.Tensor,
                # weights for transform 1
                transform_1_conv0_weight: torch.Tensor,
                transform_1_conv0_bias: torch.Tensor,
                transform_1_conv1_weight: torch.Tensor,
                transform_1_conv1_bias: torch.Tensor,
                transform_1_conv2_weight: torch.Tensor,
                transform_1_conv2_bias: torch.Tensor,
                # weights for transform 2
                transform_2_conv0_weight: torch.Tensor,
                transform_2_conv0_bias: torch.Tensor,
                transform_2_conv1_weight: torch.Tensor,
                transform_2_conv1_bias: torch.Tensor,
                transform_2_conv2_weight: torch.Tensor,
                transform_2_conv2_bias: torch.Tensor,
                # weights for transform 3
                transform_3_conv0_weight: torch.Tensor,
                transform_3_conv0_bias: torch.Tensor,
                transform_3_conv1_weight: torch.Tensor,
                transform_3_conv1_bias: torch.Tensor,
                transform_3_conv2_weight: torch.Tensor,
                transform_3_conv2_bias: torch.Tensor):
        # x: [N, 192, T]
        N, C, T = x.shape
        half = C // 2  # 96
        device = x.device

        # Precompute blocks for Triton
        BLOCK_CO = 64
        BLOCK_T = 128
        BLOCK_C = 64

        # We will apply transforms sequentially; update x in place via Triton kernels
        # Note: x is modified in Triton by constructing new tensors (we return the final x).
        # However, Triton cannot modify caller's tensor, so we keep a running x and return it at the end.

        # For each transform, we need to:
        # 1) Split x into x0 and x1 (slices). We implement split_copy_triton using x itself.
        # 2) conv0(x0) -> y0, ReLU
        # 3) conv1(y0) -> y1, ReLU
        # 4) conv2(y1) -> h
        # 5) Update x1: x1 = x1 + h (forward) or x1 = x1 - h (reverse)
        # 6) Concatenate [x0, x1] into x
        # 7) Multiply by mask
        # We repeat steps for 4 transforms.

        # Launch a temp output x_out to hold the final result
        x_out = torch.empty_like(x)

        for i in range(4):
            # Determine which weights to use
            if i == 0:
                w0, b0 = transform_0_conv0_weight, transform_0_conv0_bias
                w1, b1 = transform_0_conv1_weight, transform_0_conv1_bias
                w2, b2 = transform_0_conv2_weight, transform_0_conv2_bias
            elif i == 1:
                w0, b0 = transform_1_conv0_weight, transform_1_conv0_bias
                w1, b1 = transform_1_conv1_weight, transform_1_conv1_bias
                w2, b2 = transform_1_conv2_weight, transform_1_conv2_bias
            elif i == 2:
                w0, b0 = transform_2_conv0_weight, transform_2_conv0_bias
                w1, b1 = transform_2_conv1_weight, transform_2_conv1_bias
                w2, b2 = transform_2_conv2_weight, transform_2_conv2_bias
            else:
                w0, b0 = transform_3_conv0_weight, transform_3_conv0_bias
                w1, b1 = transform_3_conv1_weight, transform_3_conv1_bias
                w2, b2 = transform_3_conv2_weight, transform_3_conv2_bias

            # Launch split_copy_triton: x0, x1 from x_out[:, :half, :], x_out[:, half:, :]
            x0_tmp = torch.empty((N, half, T), device=device, dtype=x.dtype)
            x1_tmp = torch.empty((N, half, T), device=device, dtype=x.dtype)
            grid_split = (N, triton.cdiv(half, BLOCK_C), triton.cdiv(T, BLOCK_T))
            split_copy_triton[grid_split](
                x_out, x0_tmp, x1_tmp,
                N, half, T,
                BLOCK_C, BLOCK_T
            )

            # conv0: y0 = conv1d_relu_triton(x0_tmp, w0, b0)
            y0 = torch.empty((N, half, T), device=device, dtype=x.dtype)
            T_out0 = T - 1  # padding=2, K=5 -> T_out = T - 1
            grid_conv0 = (N, triton.cdiv(half, BLOCK_CO), triton.cdiv(T_out0, BLOCK_T))
            conv1d_relu_triton[grid_conv0](
                x0_tmp, w0, b0, y0,
                N, half, T, half, T_out0,
                5, 2,
                BLOCK_CO, BLOCK_T
            )

            # conv1: y1 = conv1d_relu_triton(y0, w1, b1)
            y1 = torch.empty((N, half, T), device=device, dtype=x.dtype)
            grid_conv1 = (N, triton.cdiv(half, BLOCK_CO), triton.cdiv(T, BLOCK_T))
            conv1d_relu_triton[grid_conv1](
                y0, w1, b1, y1,
                N, half, T_out0, half, T,
                5, 2,
                BLOCK_CO, BLOCK_T
            )

            # conv2: h = conv1d_relu_triton(y1, w2, b2)  [this should be just conv+bias, no ReLU?]
            # In original code, the sequence is conv -> ReLU -> conv -> ReLU -> conv, so we apply ReLU after conv2 as well.
            h = torch.empty((N, half, T), device=device, dtype=x.dtype)
            grid_conv2 = (N, triton.cdiv(half, BLOCK_CO), triton.cdiv(T, BLOCK_T))
            conv1d_relu_triton[grid_conv2](
                y1, w2, b2, h,
                N, half, T, half, T,
                5, 2,
                BLOCK_CO, BLOCK_T
            )

            # Update second half: x1_tmp = x1_tmp + h (forward) or x1_tmp = x1_tmp - h (reverse)
            x1_new = torch.empty_like(x1_tmp)
            grid_addsub = (N, triton.cdiv(half, BLOCK_C), triton.cdiv(T, BLOCK_T))
            add_sub_second_triton[grid_addsub](
                x1_tmp, h, x1_new,
                N, half, T,
                reverse,  # False => add, True => subtract
                BLOCK_C, BLOCK_T
            )

            # Concatenate x0_tmp and x1_new into x_out
            grid_concat = (N, triton.cdiv(C, BLOCK_C), triton.cdiv(T, BLOCK_T))
            concat_halves_triton[grid_concat](
                x0_tmp, x1_new, x_out,
                N, C,
                T,
                BLOCK_C, BLOCK_T
            )

            # Apply mask: x_out = x_out * x_mask (broadcast over channels)
            # x_mask is [N, 1, T] -> broadcast along C
            x_mask_flat = x_mask  # already shape [N,1,T]
            x_out_masked = torch.empty_like(x_out)
            grid_mask = (N, triton.cdiv(C, BLOCK_C), triton.cdiv(T, BLOCK_T))
            mask_mul_triton[grid_mask](
                x_out, x_mask_flat, x_out_masked,
                N, C, T,
                BLOCK_C, BLOCK_T
            )
            x_out = x_out_masked

        return x_out


def run(*args):
    return ModelNew()(*args)
