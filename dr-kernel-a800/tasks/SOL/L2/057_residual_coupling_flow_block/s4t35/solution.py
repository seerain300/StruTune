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
        pid_co = tl.program_id(1)        # output channel block id
        pid_t = tl.program_id(2)         # time block id

        co_start = pid_co * BLOCK_CO
        t_start = pid_t * BLOCK_T

        co_offsets = co_start + tl.arange(0, BLOCK_CO)   # [BLOCK_CO]
        t_offsets = t_start + tl.arange(0, BLOCK_T)      # [BLOCK_T]

        co_mask = co_offsets < C_out
        t_mask = t_offsets < T_out
        mask_out = co_mask[:, None] & t_mask[None, :]

        acc = tl.zeros((BLOCK_CO, BLOCK_T), dtype=tl.float32)

        # Accumulate over input channels and kernel taps
        for ci in range(0, C_in):
            for k in range(0, K):
                t_in = t_offsets + (k - PAD)               # [BLOCK_T]
                in_bounds = (t_in >= 0) & (t_in < T_in) & t_mask  # [BLOCK_T]

                # load x[n, ci, t_in]
                x_offs = ((pid_n * C_in + ci) * T_in) + t_in  # [BLOCK_T]
                x_vals = tl.load(x_ptr + x_offs, mask=in_bounds, other=0.0).to(tl.float32)

                # load w[co, ci, k]
                w_offs = co_offsets * (C_in * K) + ci * K + k  # [BLOCK_CO]
                w_vals = tl.load(w_ptr + w_offs, mask=co_mask, other=0.0).to(tl.float32)

                acc += w_vals[:, None] * x_vals[None, :]

        # add bias and ReLU
        b_vals = tl.load(b_ptr + co_offsets, mask=co_mask, other=0.0).to(tl.float32)
        acc = acc + b_vals[:, None]
        acc = tl.maximum(acc, 0.0)

        # store y[n, co, t]
        y_offs = ((pid_n * C_out + co_offsets[:, None]) * T_out) + t_offsets[None, :]
        tl.store(y_ptr + y_offs, acc, mask=mask_out)


if TRITON_AVAILABLE:
    @triton.jit
    def add_triton(
        a_ptr, b_ptr, out_ptr,
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

        a_offs = ((pid_n * C + c_offsets[:, None]) * T) + t_offsets[None, :]
        b_offs = ((pid_n * C + c_offsets[:, None]) * T) + t_offsets[None, :]
        a = tl.load(a_ptr + a_offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + b_offs, mask=mask, other=0.0).to(tl.float32)
        out = a + b

        out_offs = ((pid_n * C + c_offsets[:, None]) * T) + t_offsets[None, :]
        tl.store(out_ptr + out_offs, out, mask=mask)


if TRITON_AVAILABLE:
    @triton.jit
    def concat_halves_triton(
        y0_ptr,         # *const float, shape [N, C_in, T]
        y1_ptr,         # *const float, shape [N, C_in, T]
        out_ptr,        # *float,       shape [N, 2*C_in, T]
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

        # copy first half
        y0_offs = ((pid_n * C_in + c_offsets[:, None]) * T) + t_offsets[None, :]
        y1_offs = ((pid_n * C_in + c_offsets[:, None]) * T) + t_offsets[None, :]
        x0 = tl.load(y0_ptr + y0_offs, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(y1_ptr + y1_offs, mask=mask, other=0.0).to(tl.float32)

        # store to out: first half at channels [0..C_in-1], second half at [C_in..2*C_in-1]
        out_offs0 = ((pid_n * (2 * C_in) + c_offsets[:, None]) * T) + t_offsets[None, :]
        out_offs1 = ((pid_n * (2 * C_in) + (c_offsets[:, None] + C_in)) * T) + t_offsets[None, :]
        tl.store(out_ptr + out_offs0, x0, mask=mask)
        tl.store(out_ptr + out_offs1, x1, mask=mask)


if TRITON_AVAILABLE:
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
        super().__


def run(*args):
    return ModelNew()(*args)
