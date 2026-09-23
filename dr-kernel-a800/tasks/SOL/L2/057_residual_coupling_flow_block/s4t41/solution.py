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

    # Kernels for slicing/copy (copy x to y with channel offset)
    @triton.jit
    def slice_copy_triton(
        x_ptr,          # *const float, shape [N, Cin, T], contiguous
        y_ptr,          # *float,       shape [N, Cout, T], contiguous
        N: tl.int32,
        Cin: tl.int32,
        T: tl.int32,
        Cout: tl.int32,
        offset: tl.int32,                 # channel offset in y
        BLOCK_C: tl.constexpr,
        BLOCK_T: tl.constexpr
    ):
        pid_n = tl.program_id(0)          # batch index
        pid_c = tl.program_id(1)          # channel block id
        pid_t = tl.program_id(2)          # time block id

        c_start = pid_c * BLOCK_C
        t_start = pid_t * BLOCK_T

        c_offsets = c_start + tl.arange(0, BLOCK_C)      # [BLOCK_C], in input space
        t_offsets = t_start + tl.arange(0, BLOCK_T)      # [BLOCK_T]

        c_mask = c_offsets < Cin
        t_mask = t_offsets < T
        mask_out = c_mask[:, None] & t_mask[None, :]

        # load from x
        x_offs = (pid_n * Cin + c_offsets[:, None]) * T + t_offsets[None, :]
        x_vals = tl.load(x_ptr + x_offs, mask=mask_out, other=0.0).to(tl.float32)

        # store to y at offset
        y_c = c_offsets + offset                        # [BLOCK_C]
        y_c_mask = y_c < Cout                          # [BLOCK_C]
        y_offs = (pid_n * Cout + y_c[:, None]) * T + t_offsets[None, :]
        tl.store(y_ptr + y_offs, x_vals, mask=(mask_out & y_c_mask[:, None]))

    # Kernel: concatenate two halves and add h to second half
    @triton.jit
    def concat_add_triton(
        x0_ptr,         # *const float, shape [N, C_half, T], contiguous (first half)
        x1_ptr,         # *const float, shape [N, C_half, T], contiguous (second half)
        h_ptr,          # *const float, shape [N, C_half, T], contiguous
        y_ptr,          # *float,       shape [N, C_half*2, T], contiguous
        N: tl.int32,
        C_half: tl.int32,
        T: tl.int32,
        BLOCK_C: tl.constexpr,
        BLOCK_T: tl.constexpr
    ):
        pid_n = tl.program_id(0)          # batch index
        pid_c = tl.program_id(1)          # channel block id
        pid_t = tl.program_id(2)          # time block id

        t_start = pid_t * BLOCK_T
        c_start = pid_c * BLOCK_C

        t_offsets = t_start + tl.arange(0, BLOCK_T)      # [BLOCK_T]
        c_offsets = c_start + tl.arange(0, BLOCK_T)      # [BLOCK_C] -> should be T? Actually for channels
        # We need two regions: first half channels 0..C_half-1 and second half channels C_half..2*C_half-1
        # So we have two stores:
        # 1) y[n, c, t] = x0[n, c, t] for c in 0..C_half-1
        # 2) y[n, c + C_half, t] = x1[n, c, t] + h[n, c, t]

        # First half copy
        c1_offsets = c_start + tl.arange(0, BLOCK_C)     # [BLOCK_C], channels in x0
        c1_mask = c1_offsets < C_half
        t_mask = t_offsets < T
        mask1 = c1_mask[:, None] & t_mask[None, :]

        x0_offs = (pid_n * C_half + c1_offsets[:, None]) * T + t_offsets[None, :]
        vals1 = tl.load(x0_ptr + x0_offs, mask=mask1, other=0.0).to(tl.float32)

        y_c1 = c1_offsets                                         # [BLOCK_C], channels in y
        y_c1_mask = y_c1 < C_half
        y_offs1 = (pid_n * (2 * C_half) + y_c1[:, None]) * T + t_offsets[None, :]
        tl.store(y_ptr + y_offs1, vals1, mask=(mask1 & y_c1_mask[:, None]))

        # Second half: add h to x1
        c2_offsets = c_start + tl.arange(0, BLOCK_C)          # [BLOCK_C], channels in x1
        c2_mask = c2_offsets < C_half
        t_mask2 = t_offsets < T
        mask2 = c2_mask[:, None] & t_mask2[None, :]

        x1_offs = (pid_n * C_half + c2_offsets[:, None]) * T + t_offsets[None, :]
        x1_vals = tl.load(x1_ptr + x1_offs, mask=mask2, other=0.0).to(tl.float32)

        h_offs = (pid_n * C_half + c2_offsets[:, None]) * T + t_offsets[None, :]
        h_vals = tl.load(h_ptr + h_offs, mask=mask2, other=0.0).to(tl.float32)
        vals2 = x1_vals + h_vals

        y_c2 = c2_offsets + C_half                             # [BLOCK_C], channels in y second half
        y_c2_mask = y_c2 < (2 * C_half)
        y_offs2 = (pid_n * (2 * C_half) + y_c2[:, None]) * T + t_offsets[None, :]
        tl.store(y_ptr + y_offs2, vals2, mask=(mask2 & y_c2_mask[:, None]))

    # Kernel: elementwise multiply by mask [N, 1, T] (broadcast over channels)
    @triton.jit
    def mask_mul_triton(
        inp_ptr,        # *const float, shape [N, C, T], contiguous
        mask_ptr,       # *const float, shape [N, 1, T], contiguous (or [N, T])
        out_ptr,        # *float,       shape [N, C, T], contiguous
        N: tl.int32,
        C: tl.int32,
        T: tl.int32,
        BLOCK_C: tl.constexpr,
        BLOCK_T: tl.constexpr
    ):
        pid_n = tl.program_id(0)          # batch index
        pid_c = tl.program_id(1)          # channel block id
        pid_t = tl.program_id(2)          # time block id

        c_start = pid_c * BLOCK_C
        t_start = pid_t * BLOCK_T

        c_offsets = c_start + tl.arange(0, BLOCK_C)      # [BLOCK_C]
        t_offsets = t_start + tl.arange(0, BLOCK_T)      # [BLOCK_T]

        c_mask = c_offsets < C
        t_mask = t_offsets < T
        mask_out = c_mask[:, None] & t_mask[None, :]

        inp_offs = (pid_n * C + c_offsets[:, None]) * T + t_offsets[None, :]
        inp_vals = tl.load(inp_ptr + inp_offs, mask=mask_out, other=0.0).to(tl.float32)

        # load mask: shape [N, 1, T], index over n and t, ignore c (broadcast)
        mask_offs = (pid_n * T) + t_offsets
        mask_vals = tl.load(mask_ptr + mask_offs, mask=t_mask, other=1.0).to(tl.float32)  # [BLOCK_T]
        mask_vals = mask_vals[None, :]                                                       # broadcast along channel dimension

        out_vals = inp_vals * mask_vals
        out_offs = (pid_n * C + c_offsets[:, None]) * T + t_offsets[None, :]
        tl.store(out_ptr + out_offs, out_vals, mask=mask_out)

# ModelNew: Triton-only forward
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,                # [N, C, T]
        x_mask: torch.Tensor,           # [N, 1, T]
        reverse: bool,                  # not used in this forward (kept for signature symmetry)
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
        # We implement the forward path entirely in Triton kernels.
        # Each iteration: compute conv0->relu->conv1->relu->conv2 (h), slice x into x0 and x1,
        # concatenate x0 and (x1 + h), multiply by mask, and use the result as input for next iteration.
        # Finally, we return the last result.

        N, C, T = x.shape
        half = C // 2
        assert C == 192 and half == 96, "This Triton implementation assumes C=192, half=96"

        # Helper to launch conv1d + ReLU
        def conv_relu(x_in, w, b):
            C_in = x_in.shape[1]
            K = 5
            PAD = 2
            T_out = T - K + 1 + 2 * PAD  # with PAD=2 => T_out = T - 1
            y = torch.empty((N, w.shape[0], T_out), device=x_in.device, dtype=x_in.dtype)
            # Choose tile sizes. We cover full channel and time via blocks.
            BLOCK_CO = 64  # good for C_out up to 192
            BLOCK_T = 128  # good for T_out up to large values; mask handles edges
            grid = (N, triton.cdiv(w.shape[0], BLOCK_CO), triton.cdiv(T_out, BLOCK_T))
            conv1d_relu_triton[grid](
                x_in, w, b, y,
                N, C_in, T, w.shape[0], T_out, K, PAD,
                BLOCK_CO=BLOCK_CO, BLOCK_T=BLOCK_T
            )
            return y

        # Run 4 transforms sequentially
        for _ in range(4):
            # conv0
            y0 = conv_relu(x, transform_0_conv0_weight, transform_0_conv0_bias)
            # conv1
            y1 = conv_relu(y0, transform_0_conv1_weight, transform_0_conv1_bias)
            # conv2
            h = conv_relu(y1, transform_0_conv2_weight, transform_0_conv2_bias)

            # Slice x into x0 and x1
            x0 = torch.empty((N, half, T), device=x.device, dtype=x.dtype)
            x1 = torch.empty((N, half, T), device=x.device, dtype=x.dtype)
            # copy first half
            slice_copy_triton[(N, triton.cdiv(half, 64), triton.cdiv(T, 128))](
                x, x0, N, half, T, 0, BLOCK_C=64, BLOCK_T=128
            )
            # copy second half (offset half channels)
            slice_copy_triton[(N, triton.cdiv(half, 64), triton.cdiv(T, 128))](
                x, x1, N, half, T, half, BLOCK_C=64, BLOCK_T=128
            )

            # Concatenate x0 and (x1 + h) into y_out [N, 192, T]
            y_out = torch.empty((N, C, T), device=x.device, dtype=x.dtype)
            concat_add_triton[(N, triton.cdiv(C, 64), triton.cdiv(T, 128))](
                x0, x1, h, y_out, N, half, T, BLOCK_C=64, BLOCK_T=128
            )

            # Multiply by mask
            # Ensure mask is [N, 1, T] contiguous
            out = torch.empty_like(y_out)
            mask = x_mask  # [N, 1, T]
            # Triton expects mask_ptr to be contiguous over N*T. It is [N, 1, T], so flatten N*T
            # Note: Triton cannot index 2D like [N, 1, T] directly; we flatten to [N*T] since C dimension is broadcasted
            mask_flat = mask.reshape(N * T).contiguous()  # [N*T]
            mask_mul_triton[(N, triton.cdiv(C, 64), triton.cdiv(T, 128))](
                y_out, mask_flat, out, N, C, T, BLOCK_C=64, BLOCK_T=128
            )

            # For the next iteration, input is y_out (after mask)
            x = out

        # Return final transformed output
        return x


def run(*args):
    return ModelNew()(*args)
