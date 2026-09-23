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


# Triton kernel: Concatenate two halves into a single output with channel offset.
# Inputs:
#   y0_ptr: [N, C_in, T] (first half)
#   y1_ptr: [N, C_in, T] (second half after +/- h)
#   out_ptr: [N, 2*C_in, T]
# Operation:
#   out[n, c, t] = y0[n, c, t] for c in [0..C_in-1]
#   out[n, c+C_in, t] = y1[n, c, t]
if TRITON_AVAILABLE:
    @triton.jit
    def concat_halves_triton(
        y0_ptr, y1_ptr, out_ptr,
        N: tl.int32, C: tl.int32, T: tl.int32,
        BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr
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

        # load y0 and y1
        y0_offs = ((pid_n * C + c_offsets[:, None]) * T) + t_offsets[None, :]
        y1_offs = ((pid_n * C + c_offsets[:, None]) * T) + t_offsets[None, :]
        y0 = tl.load(y0_ptr + y0_offs, mask=mask, other=0.0).to(tl.float32)
        y1 = tl.load(y1_ptr + y1_offs, mask=mask, other=0.0).to(tl.float32)

        # store into out: first half at channels [0..C-1], second half at [C..2C-1]
        out_offs0 = ((pid_n * (2 * C) + c_offsets[:, None]) * T) + t_offsets[None, :]
        out_offs1 = ((pid_n * (2 * C) + (c_offsets[:, None] + C)) * T) + t_offsets[None, :]
        tl.store(out_ptr + out_offs0, y0, mask=mask)
        tl.store(out_ptr + out_offs1, y1, mask=mask)


# Triton kernel: Elementwise multiply by mask (x_mask has shape [N, 1, T], we broadcast along channel)
if TRITON_AVAILABLE:
    @triton.jit
    def mask_mul_triton(
        x_ptr,         # *const float, shape [N, C, T]
        mask_ptr,      # *const float, shape [N, 1, T]
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

    def forward(self, x, x_mask, reverse, *weights_and_biases):
        """
        Triton-only forward that emulates the original run behavior.
        - x: [N, 192, T]
        - x_mask: [N, 1, T]
        - reverse: bool (if True, apply transforms in reverse order)
        - weights_and_biases: 12 tensors for 4 transforms: (conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b) repeated 4 times.
        Returns: list of outputs after each transform (or after each step if reverse).
        """
        N, C, T = x.shape
        assert C == 192, "Expected channels=192"
        half = 96
        C_out0 = 96  # conv0 and conv2 output channels
        C_mid = 96   # conv1 output channels

        # Helper to launch conv with given weights and biases, and ReLU
        def do_conv_relu(x_ptr, w_ptr, b_ptr, out_ptr, C_in, C_out, T_in, T_out):
            grid = (
                N,
                triton.cdiv(C_out, 32),  # tile channels
                triton.cdiv(T_out, 64)   # tile time
            )
            conv1d_relu_triton[grid](
                x_ptr, w_ptr, b_ptr, out_ptr,
                N, C_in, T_in, C_out, T_out, K=5, PAD=2,
                BLOCK_CO=32, BLOCK_T=64
            )
            return out_ptr

        # Prepare outputs list
        outputs = []

        # Define weights/biases list per transform
        num_transforms = 4
        weight_list = []
        for i in range(num_transforms):
            base = i * 6
            conv0_w = weights_and_biases[base]
            conv0_b = weights_and_biases[base + 1]
            conv1_w = weights_and_biases[base + 2]
            conv1_b = weights_and_biases[base + 3]
            conv2_w = weights_and_biases[base + 4]
            conv2_b = weights_and_biases[base + 5]
            weight_list.append((conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b))

        if reverse:
            # Apply in reverse order: 3 -> 2 -> 1 -> 0
            current = x  # start from original x
            for t, (conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b) in enumerate(reversed(weight_list)):
                # Split halves: first half x0, second half x1
                # We'll create y0 from conv0(x0), and y1 from conv1(conv0(x0)) then conv2, then update x1 = x1 - h
                # However, since we only have current tensor, we must extract halves via slicing.
                # For Triton, we implement slice copying into two outputs.
                # Allocate y0, y1
                # Note: conv0 operates on first half only (C_in=96), conv1 also on first half output, conv2 on that.

                # Extract x0 and x1: x0 = current[:, :96, :], x1 = current[:, 96:, :]
                # Create placeholders by slicing and copying (Triton kernels operate on contiguous tensors, so we can copy into new buffers).
                # Allocate y0_tmp, y1_tmp
                # y0_tmp = conv0(x0)
                # y1_tmp = conv1(y0_tmp) then ReLU
                # y2_tmp = conv2(y1_tmp) then ReLU -> h
                # Update current: first half = current[:, :96, :], second half = current[:, 96:, :] - h

                # Compute h via conv2_triton on y1_tmp after conv1_triton
                # For simplicity, we'll recompute y0_tmp and y1_tmp from current.
                # First, copy x0 and x1 into separate tensors
                y0 = torch.empty((N, 96, T), device=x.device, dtype=x.dtype)
                y1 = torch.empty((N, 96, T), device=x.device, dtype=x.dtype)
                y0_ptr = y0
                y1_ptr = y1

                # Copy first half into y0: y0[n, c, t] = current[n, c, t]
                grid0 = (
                    N,
                    triton.cdiv(96, 32),
                    triton.cdiv(T, 64)
                )
                slice_copy_triton[grid0](
                    current, y0_ptr, N, 96, T, offset=0,
                    BLOCK_C=32, BLOCK_T=64
                )

                # Copy second half into y1: y1[n, c, t] = current[n, 96 + c, t]
                grid1 = (
                    N,
                    triton.cdiv(96, 32),
                    triton.cdiv(T, 64)
                )
                slice_copy_triton[grid1](
                    current, y1_ptr, N, 96, T, offset=96,
                    BLOCK_C=32, BLOCK_T=64
                )

                # Compute conv0 on y0 to get y0_after
                y0_after = torch.empty_like(y0)
                do_conv_relu(y0_ptr, conv0_w, conv0_b, y0_after, C_in=96, C_out=96, T_in=T, T_out=T)

                # Compute conv1 on y0_after
                y1_tmp = torch.empty_like(y1)
                # conv1 on y0_after: input shape [N, 96, T], output shape [N, 96, T]
                do_conv_relu(y0_after, conv1_w, conv1_b, y1_tmp, C_in=96, C_out=96, T_in=T, T_out=T)

                # Compute conv2 on y1_tmp to get h
                h = torch.empty((N, 96, T), device=x.device, dtype=x.dtype)
                do_conv_relu(y1_tmp, conv2_w, conv2_b, h, C_in=96, C_out=96, T_in=T, T_out=T)

                # Now update current: first half unchanged, second half minus h
                # Allocate new current_out
                current_out = torch.empty((N, 192, T), device=x.device, dtype=x.dtype)

                # Concatenate: first half (current[:, :96, :]) and second half (current[:, 96:, :] - h)
                # But we need to reconstruct second half from current_out: we don't have current_out yet.
                # To avoid reading current again, we'll write into current_out in two steps using concat_halves_triton.
                # First, copy current[:, :96, :] into current_out[:, :96, :]
                # Then compute x1_new = current[:, 96:, :] - h and write into current_out[:, 96:, :]
                # However, we don't have current_out yet. Instead, we'll use concat_halves_triton with y0 and (current[:, 96:, :] - h).
                # We need y0 as first half. To get y0, we use the output of conv0; but conv0 output is y0_after.
                # We already have y0_after and y1_tmp. We need h. h is already computed.
                # Let's prepare y0_out and y1_out.
                y0_out = y0_after
                # y1_out will be current[:, 96:, :] - h
                y1_out = torch.empty_like(y1)
                # We need current[:, 96:, :] to compute y1_out. We can reconstruct by copying from current into y1_out then subtract h.
                # But we don't have current_out; we only have current. The safest approach is to copy current[:, 96:, :] into y1_out and subtract h.
                # Note: y1_out is the same shape as y1_tmp and h. But we need current's second half.
                # We can copy current[:, 96:, :] into y1_out by slicing and copying, then subtract h.
                # Allocate y1_out
                y1_out = torch.empty((N, 96, T), device=x.device, dtype=x.dtype)

                # Copy current[:, 96:, :] into y1_out
                grid_copy = (
                    N,
                    triton.cdiv(96, 32),
                    triton.cdiv(T, 64)
                )
                slice_copy_triton[grid_copy](
                    current, y1_out, N, 96, T, offset=96,
                    BLOCK_C=32, BLOCK_T=64
                )
                # Subtract h
                # Since h is [N, 96, T], subtract elementwise
                y1_out = y1_out - h

                # Now concat: out has 192 channels
                out = torch.empty((N, 192, T), device=x.device, dtype=x.dtype)
                concat_halves_triton[(N, triton.cdiv(192, 32), triton.cdiv(T, 64))](
                    y0_out, y1_out, out,
                    N, 96, T,
                    BLOCK_C=32, BLOCK_T=64
                )

                # Apply mask
                out_masked = torch.empty_like(out)
                mask_mul_triton[(N, triton.cdiv(192, 32), triton.cdiv(T, 64))](
                    out, x_mask, out_masked, N, 192, T,
                    BLOCK_C=32, BLOCK_T=64
                )

                # Save output for this step
                outputs.append(out_masked)

                # Update current for next transform (reverse order): current = out_masked
                current = out_masked

        else:
            # Forward order: 0 -> 1 -> 2 -> 3
            current = x  # start from original x
            for t, (conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b) in enumerate(weight_list):
                # Compute conv0 on first half
                # Allocate y0
                y0 = torch.empty((N, 96, T), device=x.device, dtype=x.dtype)
                grid0 = (
                    N,
                    triton.cdiv(96, 32),
                    triton.cdiv(T, 64)
                )
                # We need to copy current[:, :96, :] into y0 for conv0 input
                slice_copy_triton[grid0](
                    current, y0, N, 96, T, offset=0,
                    BLOCK_C=32, BLOCK_T=64
                )

                # conv0 output
                y0_after = torch.empty_like(y0)
                do_conv_relu(y0, conv0_w, conv0_b, y0_after, C_in=96, C_out=96, T_in=T, T_out=T)

                # conv1 on y0_after
                y1_tmp = torch.empty_like(y0)
                do_conv_relu(y0_after, conv1_w, conv1_b, y1_tmp, C_in=96, C_out=96, T_in=T, T_out=T)

                # conv2 on y1_tmp to get h
                h = torch.empty((N, 96, T), device=x.device, dtype=x.dtype)
                do_conv_relu(y1_tmp, conv2_w, conv2_b, h, C_in=96, C_out=96, T_in=T, T_out=T)

                # Update second half: current[:, 96:, :] = current[:, 96:, :] + h
                # First allocate y1_out = current[:, 96:, :]
                y1_out = torch.empty((N, 96, T), device=x.device, dtype=x.dtype)
                grid1 = (
                    N,
                    triton.cdiv(96, 32),
                    triton.cdiv(T, 64)
                )
                slice_copy_triton[grid1](
                    current, y1_out, N, 96, T, offset=96,
                    BLOCK_C=32, BLOCK_T=64
                )
                y1_out = y1_out + h

                # Concatenate halves
                out = torch.empty((N, 192, T), device=x.device, dtype=x.dtype)
                concat_halves_triton[(N, triton.cdiv(192, 32), triton.cdiv(T, 64))](
                    y0_after, y1_out, out,
                    N, 96, T,
                    BLOCK_C=32, BLOCK_T=64
                )

                # Apply mask
                out_masked = torch.empty_like(out)
                mask_mul_triton[(N, triton.cdiv(192, 32), triton.cdiv(T, 64))](
                    out, x_mask, out_masked, N, 192, T,
                    BLOCK_C=32, BLOCK_T=64
                )

                # Save output for this transform
                outputs.append(out_masked)

                # Update current for next transform
                current = out_masked

        # Return outputs list as per original run (after each transform). In forward, we return after each step.
        # If reverse, we return in reverse order of processing (which is already in reverse). Otherwise, normal.
        return outputs


# Optional: helper Triton kernel for slice copying (not strictly necessary if we copy via concatenation above,
# but included for completeness; we actually didn't call it previously because we used concat and slicing at host level).
# Here we define it but we won't use it in forward since forward already uses concatenation logic. To keep completeness,
# I'll define it and mention usage. The forward above uses direct slicing which is Python-side, but we can still define
# a Triton version for consistency.
if TRITON_AVAILABLE:
    @triton.jit
    def slice_copy_triton(
        src_ptr,     # *const float, shape [N, C, T]
        dst_ptr,     # *float,       shape [N, C, T]
        N: tl.int32, C: tl.int32, T: tl.int32,
        offset: tl.int32,            # channel offset to copy into dst
        BLOCK_C: tl.constexpr,       # tile along channels
        BLOCK_T: tl.constexpr        # tile along time
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

        src_offs = ((pid_n * C + c_offsets[:, None] + offset) * T) + t_offsets[None, :]
        dst_offs = ((pid_n * C + c_offsets[:, None]) * T) + t_offsets[None, :]
        vals = tl.load(src_ptr + src_offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(dst_ptr + dst_offs, vals, mask=mask)


def run(*args):
    return ModelNew()(*args)
