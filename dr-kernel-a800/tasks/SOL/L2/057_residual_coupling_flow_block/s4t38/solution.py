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
                x_vals = tl.load(x_ptr + x_offs, mask=in_bounds, other=0.0).to(tl.float32)  # [BLOCK_T]

                # load weights w[co, ci, k]
                w_offs = co_offsets * (C_in * K) + ci * K + k  # [BLOCK_CO]
                w_vals = tl.load(w_ptr + w_offs, mask=co_mask, other=0.0).to(tl.float32)    # [BLOCK_CO]

                # outer product accumulate: [BLOCK_CO, 1] * [1, BLOCK_T]
                acc += w_vals[:, None] * x_vals[None, :]

        # add bias and ReLU
        b_vals = tl.load(b_ptr + co_offsets, mask=co_mask, other=0.0).to(tl.float32)   # [BLOCK_CO]
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
    def concat_halves_triton(
        y0_ptr,         # *const float, shape [N, C_in, T], first half
        y1_ptr,         # *const float, shape [N, C_in, T], second half
        h_ptr,          # *const float, shape [N, C_in, T], add/subtract from y1
        out_ptr,        # *float,       shape [N, 2*C_in, T], output concatenated
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

        # y0 (first half): channels 0..C_in-1
        y0_offs = ((pid_n * C_in + c_offsets[:, None]) * T) + t_offsets[None, :]
        y0_vals = tl.load(y0_ptr + y0_offs, mask=mask, other=0.0).to(tl.float32)

        # y1 (second half): channels 0..C_in-1 but stored as second half
        y1_offs = ((pid_n * C_in + c_offsets[:, None]) * T) + t_offsets[None, :]
        y1_vals = tl.load(y1_ptr + y1_offs, mask=mask, other=0.0).to(tl.float32)

        # h (same shape as y1)
        h_offs = ((pid_n * C_in + c_offsets[:, None]) * T) + t_offsets[None, :]
        h_vals = tl.load(h_ptr + h_offs, mask=mask, other=0.0).to(tl.float32)

        # update: forward add, reverse subtract
        delta = h_vals if ADD else (-h_vals)
        y1_new = y1_vals + delta

        # write out: first half at channels [0..C_in-1], second half at [C_in..2*C_in-1]
        out_offs0 = ((pid_n * (2 * C_in) + c_offsets[:, None]) * T) + t_offsets[None, :]
        out_offs1 = ((pid_n * (2 * C_in) + (c_offsets[:, None] + C_in)) * T) + t_offsets[None, :]
        tl.store(out_ptr + out_offs0, y0_vals, mask=mask)
        tl.store(out_ptr + out_offs1, y1_new, mask=mask)


    @triton.jit
    def mask_mul_triton(
        x_ptr,         # *const float, shape [N, 2*C_in, T]
        mask_ptr,      # *const float, shape [N, 1, T], mask[:, 0, :]
        out_ptr,       # *float,       shape [N, 2*C_in, T]
        N: tl.int32,
        C_in: tl.int32,  # half channels (96)
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

        c_mask = c_offsets < (2 * C_in)
        t_mask = t_offsets < T
        mask = c_mask[:, None] & t_mask[None, :]

        # load x
        x_offs = ((pid_n * (2 * C_in) + c_offsets[:, None]) * T) + t_offsets[None, :]
        x_vals = tl.load(x_ptr + x_offs, mask=mask, other=0.0).to(tl.float32)

        # load mask per batch, per time (channel is broadcast)
        mask_offs = ((pid_n * 1 + 0) * T) + t_offsets  # mask has Cdim=1
        m = tl.load(mask_ptr + mask_offs, mask=t_mask, other=1.0).to(tl.float32)

        y = x_vals * m[None, :]  # broadcast along channel dimension
        tl.store(out_ptr + x_offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
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
    ) -> torch.Tensor:
        # Allocate output for final result
        N, C, T = x.shape
        C_in = C // 2
        C_out = C_in  # since half_channels is used in each transform

        # Prepare constants
        K = 5
        PAD = 2
        T_out = T - 1  # for conv1d with padding 2 and K=5, T_out = T - K + 1 + 2*PAD == T - 1

        # Launch transforms sequentially
        x_in = x  # initial input
        for i in range(4):
            # Determine which set of weights to use for this transform; the original passes all transforms' weights
            # We will use the next set in list order to emulate sequential transforms.
            # To keep it generic, we construct weight/bias pointers for the current transform using the given args.
            # Here, i is the transform index; since all args are provided, we can pick weights accordingly.
            # For clarity, we use the i-th set of weights passed in the argument list:
            # transforms[i] = (conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b)
            # But the arguments are all packed. We can index as follows:
            # transform_?_conv*_weight/bias are the variables; we need to select each transform's weights.
            # Since the caller passes 4 transforms in order, we can compute base index per transform.
            # For Triton invocation, we need to slice the inputs. We'll extract conv weights/bias using indexing logic below.

            # We'll emulate conv0, conv1, conv2 by selecting weights from the provided args. The args are ordered:
            # transform_0 conv0/conv1/conv2, transform_1 conv0/conv1/conv2, ..., transform_3 conv0/conv1/conv2
            # Each transform has 3 weights: conv0_w, conv1_w, conv2_w, and biases. There are 4 transforms * 3 = 12 weight/bias tensors per kind.
            # To extract the i-th transform's weights, we need to pick weight_i0, weight_i1, weight_i2 from the provided 12.

            # Helper function to get weight/bias for step j in transform i:
            # The ordering is: args[0..2] for transform 0, [3..5] for transform 1, ..., [21..23] for transform 3.
            # Each transform occupies 3 args. So for transform i (0..3), conv0 is args[3*i], conv1 is args[3*i+1], conv2 is args[3*i+2].
            def get_wb(step_j, i):
                idx = 3 * i + step_j
                return (eval(f'transform_{i}_conv{step_j}_weight'), eval(f'transform_{i}_conv{step_j}_bias'))

            # conv0 -> ReLU
            (w0, b0) = get_wb(0, i)
            x0 = torch.empty( (N, C_in, T), device=x.device, dtype=torch.float32 )
            y0 = torch.empty( (N, C_out, T_out), device=x.device, dtype=torch.float32 )
            # Launch conv1d_relu_triton
            BLOCK_CO = 32
            BLOCK_T = 128
            grid0 = (N, triton.cdiv(C_out, BLOCK_CO), triton.cdiv(T_out, BLOCK_T))
            conv1d_relu_triton[grid0](
                x_in[:, :C_in, :].contiguous(), w0.contiguous(), b0.contiguous(), y0, N, C_in, T, C_out, T_out, K, PAD, BLOCK_CO, BLOCK_T
            )
            # conv1 -> ReLU
            (w1, b1) = get_wb(1, i)
            x1 = torch.empty( (N, C_out, T_out), device=x.device, dtype=torch.float32 )
            y1 = torch.empty( (N, C_out, T_out), device=x.device, dtype=torch.float32 )
            grid1 = (N, triton.cdiv(C_out, BLOCK_CO), triton.cdiv(T_out, BLOCK_T))
            conv1d_relu_triton[grid1](
                y0.contiguous(), w1.contiguous(), b1.contiguous(), y1, N, C_out, T_out, C_out, T_out, K, PAD, BLOCK_CO, BLOCK_T
            )
            # conv2
            (w2, b2) = get_wb(2, i)
            y2 = torch.empty( (N, C_out, T_out), device=x.device, dtype=torch.float32 )
            grid2 = (N, triton.cdiv(C_out, BLOCK_CO), triton.cdiv(T_out, BLOCK_T))
            conv1d_relu_triton[grid2](
                y1.contiguous(), w2.contiguous(), b2.contiguous(), y2, N, C_out, T_out, C_out, T_out, K, PAD, BLOCK_CO, BLOCK_T
            )
            # h = y2 has shape [N, 96, T_out]
            # Split x_in into x0 and x1 halves
            x0_half = torch.empty( (N, C_in, T), device=x.device, dtype=torch.float32 )
            x1_half = torch.empty( (N, C_in, T), device=x.device, dtype=torch.float32 )
            split_copy_triton[(N, triton.cdiv(C_in, 32), triton.cdiv(T, 128))](
                x_in.contiguous(), x0_half, x1_half, N, C_in, T, 32, 128
            )
            # Concatenate updated x1 = x1_half + y2 (forward) or -y2 (reverse)
            y_in = torch.empty( (N, 2 * C_in, T), device=x.device, dtype=torch.float32 )
            ADD = True if not reverse else False
            concat_halves_triton[(N, triton.cdiv(C_in, 32), triton.cdiv(T, 128))](
                x0_half, x1_half, y2.contiguous(), y_in, N, C_in, T, ADD, 32, 128
            )
            # Apply mask
            out = torch.empty_like(y_in, dtype=torch.float32)
            mask_mul_triton[(N, triton.cdiv(2 * C_in, 32), triton.cdiv(T, 128))](
                y_in.contiguous(), x_mask.contiguous(), out, N, C_in, T, 32, 128
            )
            # Update x_in for next transform
            x_in = out

        # Return final transformed x (after 4 transforms)
        return x_in


def run(*args):
    return ModelNew()(*args)
