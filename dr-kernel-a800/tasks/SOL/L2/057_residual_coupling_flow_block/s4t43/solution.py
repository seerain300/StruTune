import math
import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel: Conv1d with fixed K=5, padding=2, ReLU
# Computes y[n, co, t_out] = ReLU( sum_{ci=0..C_in-1, k=0..4} x[n, ci, t_out + k - 2] * w[co, ci, k] + b[co] )
# y shape: [N, C_out, T_out], T_out = T_in - 1
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

# Kernel: copy a slice from input x into output y with channel offset
# x: [N, C, T], y: [N, C_dest, T], offset channels written starting at offset
if TRITON_AVAILABLE:
    @triton.jit
    def slice_copy_triton(
        x_ptr,      # *const float, [N, C_src, T]
        y_ptr,      # *float,       [N, C_dest, T]
        N: tl.int32,
        C_src: tl.int32,
        T: tl.int32,
        C_dest: tl.int32,
        offset: tl.int32,            # starting channel index in y
        BLOCK_C: tl.constexpr,       # tile along channel
        BLOCK_T: tl.constexpr        # tile along time
    ):
        pid_n = tl.program_id(0)
        pid_c = tl.program_id(1)
        pid_t = tl.program_id(2)

        c_src_start = pid_c * BLOCK_C
        t_start = pid_t * BLOCK_T

        c_src_offsets = c_src_start + tl.arange(0, BLOCK_C)   # [BLOCK_C]
        t_offsets = t_start + tl.arange(0, BLOCK_T)           # [BLOCK_T]

        c_src_mask = c_src_offsets < C_src
        t_mask = t_offsets < T
        mask_out = c_src_mask[:, None] & t_mask[None, :]

        # input offsets
        x_offs = (pid_n * C_src + c_src_offsets[:, None]) * T + t_offsets[None, :]
        x_vals = tl.load(x_ptr + x_offs, mask=mask_out, other=0.0).to(tl.float32)

        # output offsets: write into y starting at 'offset' channels
        c_dest_offsets = c_src_offsets + offset
        c_dest_mask = c_dest_offsets < C_dest
        out_mask = mask_out & (c_dest_mask[:, None])

        y_offs = (pid_n * C_dest + c_dest_offsets[:, None]) * T + t_offsets[None, :]
        tl.store(y_ptr + y_offs, x_vals, mask=out_mask)

# Kernel: concatenate x0 [N, C0, T] and x1_plus_h [N, C1, T] into y_out [N, C0+C1, T]
# x0: channels 0..C0-1, x1_plus_h: channels C0..C0+C1-1
if TRITON_AVAILABLE:
    @triton.jit
    def concat_add_triton(
        x0_ptr,        # *const float, [N, C0, T]
        x1_ptr,        # *const float, [N, C1, T]
        h_ptr,         # *const float, [N, C1, T] (conv2 output)
        y_ptr,         # *float,       [N, C0+C1, T]
        N: tl.int32,
        C0: tl.int32,
        C1: tl.int32,
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

        c_mask = c_offsets < (C0 + C1)
        t_mask = t_offsets < T
        mask_out = c_mask[:, None] & t_mask[None, :]

        # Copy first half: channels 0..C0-1
        c0_mask = c_offsets < C0
        c1_mask = (c_offsets >= C0) & c_mask
        c0_offsets = c_offsets - (c1_mask * C1)  # [BLOCK_C]: 0..C0-1 or -1 when c_offsets >= C0

        # For c_offsets < C0: write to y[:, c_offsets, :]
        # For c_offsets >= C0: write to y[:, c_offsets - C1, :]
        # Implement via two masked stores:
        # Store to x0 for c_offsets < C0
        if tl.any(c0_mask):
            x0_offs = (pid_n * C0 + c_offsets[c0_mask][:, None]) * T + t_offsets[None, :]
            vals0 = tl.load(x0_ptr + x0_offs, mask=(c0_mask[:, None] & t_mask[None, :]), other=0.0).to(tl.float32)
            y_offs0 = (pid_n * (C0 + C1) + c_offsets[c0_mask][:, None]) * T + t_offsets[None, :]
            tl.store(y_ptr + y_offs0, vals0, mask=(c0_mask[:, None] & t_mask[None, :]))

        # Store to x1_plus_h for c_offsets >= C0
        if tl.any(c1_mask):
            c1_offsets = c_offsets[c1_mask] - C0  # [BLOCK_C] positions in x1
            x1_offs = (pid_n * C1 + c1_offsets[:, None]) * T + t_offsets[None, :]
            h_offs = x1_offs  # same offsets, h_ptr is [N, C1, T]
            x1_vals = tl.load(x1_ptr + x1_offs, mask=(c1_mask[:, None] & t_mask[None, :]), other=0.0).to(tl.float32)
            h_vals = tl.load(h_ptr + h_offs, mask=(c1_mask[:, None] & t_mask[None, :]), other=0.0).to(tl.float32)
            out_vals = x1_vals + h_vals
            y_offs1 = (pid_n * (C0 + C1) + c_offsets[c1_mask][:, None]) * T + t_offsets[None, :]
            tl.store(y_ptr + y_offs1, out_vals, mask=(c1_mask[:, None] & t_mask[None, :]))

# Kernel: elementwise multiply y [N, C, T] by mask [N, 1, T], broadcasting across channels
if TRITON_AVAILABLE:
    @triton.jit
    def mask_mul_triton(
        y_ptr,      # *float, [N, C, T]
        mask_ptr,   # *const float, [N, 1, T]
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
        mask_out = c_mask[:, None] & t_mask[None, :]

        # load y
        y_offs = (pid_n * C + c_offsets[:, None]) * T + t_offsets[None, :]
        y_vals = tl.load(y_ptr + y_offs, mask=mask_out, other=0.0).to(tl.float32)

        # load mask along n and t (mask has shape [N,1,T])
        mask_offs = (pid_n * T) + t_offsets
        mask_vals = tl.load(mask_ptr + mask_offs, mask=t_mask, other=1.0).to(tl.float32)  # [BLOCK_T]
        mask_vals = mask_vals[None, :]  # broadcast along channel dimension

        out_vals = y_vals * mask_vals
        tl.store(y_ptr + y_offs, out_vals, mask=mask_out)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,                # [N, C=192, T]
        x_mask: torch.Tensor,           # [N, 1, T]
        reverse: bool,                  # not used; kept for signature
        # weights and biases for 4 transforms
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
        # We will perform all computation via Triton kernels, no torch ops in host code.
        N, C, T = x.shape
        C0 = 96  # first half
        C1 = 96  # second half
        K = 5
        PAD = 2
        T_out = T - 1  # per conv1d with padding=2 and K=5, output length is T_in - 1
        # Launch Triton kernels for 4 transforms in sequence
        # For each transform: conv0->ReLU->conv1->ReLU->conv2, then update x1 via concat_add_triton
        # We will not return per-transform state; we return the final x after all transforms.

        # Helper to run one transform and return updated x:
        def run_one_transform(x_in, mask_in,
                              conv0_w, conv0_b,
                              conv1_w, conv1_b,
                              conv2_w, conv2_b):
            # 1) conv0
            y0 = torch.empty((N, 192, T_out), dtype=x_in.dtype, device=x_in.device)
            grid0 = (N, triton.cdiv(192, 32), triton.cdiv(T_out, 64))
            conv1d_relu_triton[grid0](
                x_in, conv0_w, conv0_b, y0,
                N, C, T, 192, T_out, K, PAD,
                BLOCK_CO=32, BLOCK_T=64
            )
            # 2) conv1 on y0
            y1 = torch.empty((N, 192, T_out), dtype=x_in.dtype, device=x_in.device)
            grid1 = (N, triton.cdiv(192, 32), triton.cdiv(T_out, 64))
            conv1d_relu_triton[grid1](
                y0, conv1_w, conv1_b, y1,
                N, 192, T_out, 192, T_out, K, PAD,
                BLOCK_CO=32, BLOCK_T=64
            )
            # 3) conv2
            h = torch.empty((N, 96, T_out), dtype=x_in.dtype, device=x_in.device)
            grid2 = (N, triton.cdiv(96, 32), triton.cdiv(T_out, 64))
            conv1d_relu_triton[grid2](
                y1, conv2_w, conv2_b, h,
                N, 192, T_out, 96, T_out, K, PAD,
                BLOCK_CO=32, BLOCK_T=64
            )
            # 4) split x_in into x0 and x1
            x0 = torch.empty((N, C0, T), dtype=x_in.dtype, device=x_in.device)
            x1 = torch.empty((N, C1, T), dtype=x_in.dtype, device=x_in.device)
            grid_slice = (N, triton.cdiv(C0, 32), triton.cdiv(T, 64))
            slice_copy_triton[grid_slice](
                x_in, x0, N, C, T, C0, 0, BLOCK_C=32, BLOCK_T=64
            )
            grid_slice1 = (N, triton.cdiv(C1, 32), triton.cdiv(T, 64))
            slice_copy_triton[grid_slice1](
                x_in, x1, N, C, T, C1, 96, BLOCK_C=32, BLOCK_T=64
            )
            # 5) concat x0 and x1 + h into y_out [N, 192, T]
            y_out = torch.empty((N, C0 + C1, T), dtype=x_in.dtype, device=x_in.device)
            grid_concat = (N, triton.cdiv(C0 + C1, 32), triton.cdiv(T, 64))
            concat_add_triton[grid_concat](
                x0, x1, h, y_out,
                N, C0, C1, T,
                BLOCK_C=32, BLOCK_T=64
            )
            # 6) mask multiply
            # If mask_in is all ones, this is a no-op; we still launch kernel for correctness.
            grid_mask = (N, triton.cdiv(C0 + C1, 32), triton.cdiv(T, 64))
            mask_mul_triton[grid_mask](
                y_out, mask_in,
                N, C0 + C1, T,
                BLOCK_C=32, BLOCK_T=64
            )
            return y_out

        # Run 4 transforms sequentially; x_in becomes the output of the previous transform.
        x_cur = x
        # transform 0
        x_cur = run_one_transform(x_cur, x_mask,
                                  transform_0_conv0_weight, transform_0_conv0_bias,
                                  transform_0_conv1_weight, transform_0_conv1_bias,
                                  transform_0_conv2_weight, transform_0_conv2_bias)
        # transform 1
        x_cur = run_one_transform(x_cur, x_mask,
                                  transform_1_conv0_weight, transform_1_conv0_bias,
                                  transform_1_conv1_weight, transform_1_conv1_bias,
                                  transform_1_conv2_weight, transform_1_conv2_bias)
        # transform 2
        x_cur = run_one_transform(x_cur, x_mask,
                                  transform_2_conv0_weight, transform_2_conv0_bias,
                                  transform_2_conv1_weight, transform_2_conv1_bias,
                                  transform_2_conv2_weight, transform_2_conv2_bias)
        # transform 3
        x_cur = run_one_transform(x_cur, x_mask,
                                  transform_3_conv0_weight, transform_3_conv0_bias,
                                  transform_3_conv1_weight, transform_3_conv1_bias,
                                  transform_3_conv2_weight, transform_3_conv2_bias)

        return x_cur


def run(*args):
    return ModelNew()(*args)
