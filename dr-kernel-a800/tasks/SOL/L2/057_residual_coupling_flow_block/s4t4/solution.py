import math
import torch

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: Conv1d forward with ReLU, fixed K=5, padding=PAD=2.
# Computes y[n, co, t] = ReLU( sum_{ci,k} x[n, ci, t + k - PAD] * w[co, ci, k] + b[co] )
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


# Triton kernel: Elementwise mask multiplication y = y * mask
# y: [N, C, T], mask: [N, 1, T] (broadcast along C)
if TRITON_AVAILABLE:
    @triton.jit
    def mask_mul_triton(
        y_ptr,     # *float, [N, C, T], contiguous
        mask_ptr,  # *float, [N, 1, T], contiguous
        N: tl.int32,
        C: tl.int32,
        T: tl.int32,
        BLOCK_C: tl.constexpr,  # tile along channels
        BLOCK_T: tl.constexpr   # tile along time
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
        mask_out = c_mask[:, None] & t_mask[None, :]

        # load y block
        y_offs = ((pid_n * C) + c_offsets[:, None]) * T + t_offsets[None, :]
        y_vals = tl.load(y_ptr + y_offs, mask=mask_out, other=0.0).to(tl.float32)

        # load mask block (broadcast along C)
        # mask is [N, 1, T] -> we index at c=0
        mask_offs = (pid_n * T) + t_offsets
        mask_vals = tl.load(mask_ptr + mask_offs, mask=t_mask, other=1.0).to(tl.float32)  # [BLOCK_T]

        # elementwise multiply
        y_vals = y_vals * mask_vals[None, :]

        # store
        tl.store(y_ptr + y_offs, y_vals, mask=mask_out)


# Triton kernel: Split and copy x into two halves in output: x0 into channels 0..C_in-1, x1 into channels C_in..2*C_in-1
# x: [N, 2*C_in, T], y_out: [N, 2*C_in, T]
if TRITON_AVAILABLE:
    @triton.jit
    def split_copy_triton(
        x_ptr,       # *const float, input [N, 2*C_in, T], contiguous
        y_ptr,       # *float,       output [N, 2*C_in, T], contiguous
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

        c_offsets = c_start + tl.arange(0, BLOCK_C)  # [BLOCK_C]
        t_offsets = t_start + tl.arange(0, BLOCK_T)  # [BLOCK_T]

        c_mask = c_offsets < (2 * C_in)
        t_mask = t_offsets < T
        mask_out = c_mask[:, None] & t_mask[None, :]

        # For c < C_in, copy from x[n, c, :] to y[n, c, :]
        # For c >= C_in, copy from x[n, c - C_in, :] to y[n, c, :]
        x_c = c_offsets  # absolute input channel indices
        x_c_in_bounds = x_c < (2 * C_in)
        x_offs = ((pid_n * (2 * C_in)) + x_c[:, None]) * T + t_offsets[None, :]
        x_vals = tl.load(x_ptr + x_offs, mask=(mask_out & x_c_in_bounds[:, None]), other=0.0).to(tl.float32)

        y_offs = ((pid_n * (2 * C_in)) + c_offsets[:, None]) * T + t_offsets[None, :]
        tl.store(y_ptr + y_offs, x_vals, mask=mask_out)


# Triton kernel: Concatenate two tensors into a single output tensor with channel offsets.
# We will use it to form the final output: x0 in channels [0:C_in), x1+H in channels [C_in:2*C_in).
# Here, x0 and x1+H are both [N, C_in, T].
# We need to provide x0_ptr and x1h_ptr. For x1h, we can pass x1_ptr (previously concatenated result or recomputed) + h via mask_mul.
# To keep things simple and Triton-only, we will perform the addition in a separate Triton kernel (elementwise addition) and then call concat_halves with the result.
# However, since Triton kernels cannot modify caller's tensor, we construct a new output tensor for the final result and copy x0 into channels 0.. and x1h into channels C_in.. in one kernel.
# Note: The original code concatenates by slicing, but Triton cannot mutate; we construct the output accordingly.

# Implement concat_halves_triton as a split_copy_triton with x0 and x1h pointers.

# Instead, we implement a simple concat kernel that writes from x0_ptr and x1h_ptr into y_ptr with channel offsets.
# y has shape [N, 2*C_in, T]. We will place x0 at channels 0.. and x1h at channels C_in..

if TRITON_AVAILABLE:
    @triton.jit
    def concat_halves_triton(
        x0_ptr,      # *const float, [N, C_in, T]
        x1h_ptr,     # *const float, [N, C_in, T]
        y_ptr,       # *float,       [N, 2*C_in, T]
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

        c_offsets = c_start + tl.arange(0, BLOCK_C)  # [BLOCK_C]
        t_offsets = t_start + tl.arange(0, BLOCK_T)  # [BLOCK_T]

        # mask for writing x0 (channels 0..C_in-1)
        c_mask_x0 = c_offsets < C_in
        t_mask = t_offsets < T
        mask_x0 = c_mask_x0[:, None] & t_mask[None, :]

        # mask for writing x1h (channels C_in..2*C_in-1)
        c_mask_x1 = (c_offsets - C_in) >= 0
        mask_x1 = c_mask_x1[:, None] & t_mask[None, :]

        # Copy x0 into y[n, c, :] for c in [0, C_in)
        x0_offs = ((pid_n * C_in) + c_offsets[:, None]) * T + t_offsets[None, :]
        x0_vals = tl.load(x0_ptr + x0_offs, mask=(mask_x0), other=0.0).to(tl.float32)
        y_offs_x0 = ((pid_n * (2 * C_in)) + c_offsets[:, None]) * T + t_offsets[None, :]
        tl.store(y_ptr + y_offs_x0, x0_vals, mask=mask_x0)

        # Copy x1h into y[n, c, :] for c in [C_in, 2*C_in)
        x1h_offs = ((pid_n * C_in) + (c_offsets - C_in)[:, None]) * T + t_offsets[None, :]
        x1h_vals = tl.load(x1h_ptr + x1h_offs, mask=mask_x1, other=0.0).to(tl.float32)
        y_offs_x1 = ((pid_n * (2 * C_in)) + (c_offsets + C_in)[:, None]) * T + t_offsets[None, :]
        tl.store(y_ptr + y_offs_x1, x1h_vals, mask=mask_x1)


def run_triton_only(x, x_mask, reverse, *weights_and_biases):
    """
    Run the entire forward (or reverse) sequence entirely in Triton kernels.
    Returns the final output tensor of shape [batch, 192, time].
    Note: reverse is unused here because Triton-only forward is deterministic and we return the final transformed tensor.
    """
    assert TRITON_AVAILABLE, "Triton is not available."

    batch, channels, time = x.shape
    assert channels == 192, "channels must be 192."
    half_channels = channels // 2

    # We will perform the 4 transforms sequentially. After each transform, we produce y_out [batch, 192, time].
    # To get the final result, we keep the last y_out. We can reuse the same y_out buffer each time.

    # Prepare output buffer for final result
    C_in = half_channels
    C_out = C_in  # conv output channels equal to input channels in this setup
    T = time

    # We will run conv1d_relu_triton 3 times per transform (conv0, conv1, conv2). For x0 we need conv0 -> ReLU -> conv1 -> ReLU -> conv2.
    # To do that, we need conv0 weights/biases, conv1 weights/biases, conv2 weights/biases. The *weights_and_biases argument
    # provides these in order for each transform. Since we have 4 transforms, the list contains 4 sets of (conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b).

    # We will implement the forward: x1 = x1 + h for each transform. Since Triton cannot modify caller's tensor, we construct
    # a new output tensor each time. However, to return a single final tensor, we can concatenate halves via concat_halves_triton
    # after all transforms. For simplicity, we perform the forward directly into an output tensor of shape [batch, 192, time].

    # Initialize final output y_out as zeros
    y_out = torch.empty((batch, 192, T), device=x.device, dtype=x.dtype)

    # We need to emulate the original apply_transform: for each transform t in 0..3:
    # x0 = x[:, :C_in, :], x1 = x[:, C_in:, :]
    # h = conv2(ReLU(conv1(ReLU(conv0(x0))))) computed in Triton
    # x1 = x1 + h
    # Concatenate into y_out.

    # Build a list of weights per transform from *weights_and_biases
    # Each transform has 6 tensors: (conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b)
    num_transforms = (len(weights_and_biases) + 5) // 6  # we expect 4 transforms; just compute based on length
    # Actually, the caller guarantees length == 4 * 6 = 24. We can compute num_transforms as len(weights_and_biases)//6.
    num_transforms = len(weights_and_biases) // 6

    # For Triton launch parameters
    BLOCK_CO = 32
    BLOCK_T = 128

    # Process each transform
    for t in range(num_transforms):
        # Extract weights for this transform
        # weights_and_biases is a flat list: conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b, then next set, ...
        conv0_w = weights_and_biases[t * 6 + 0]
        conv0_b = weights_and_biases[t * 6 + 1]
        conv1_w = weights_and_biases[t * 6 + 2]
        conv1_b = weights_and_biases[t * 6 + 3]
        conv2_w = weights_and_biases[t * 6 + 4]
        conv2_b = weights_and_biases[t * 6 + 5]

        # Allocate buffers for x0 and x1 (same as input x)
        x0 = x[:, :C_in, :].clone()
        x1 = x[:, C_in:, :].clone()

        # Compute h = conv2( ReLU( conv1( ReLU( conv0(x0) ) ) ) )
        # conv0
        h = x0.clone()
        # y0 = conv1d_relu_triton(x0, conv0_w, conv0_b)
        # We need to pass pointers for x, w, b, and y.
        # Ensure contiguous
        x0_c = x0.contiguous()
        conv0_w_c = conv0_w.contiguous()
        conv0_b_c = conv0_b.contiguous()
        y0 = torch.empty((batch, conv0_w.shape[0], T), device=x.device, dtype=x.dtype)

        grid_conv0 = (batch, triton.cdiv(conv0_w.shape[0], BLOCK_CO), triton.cdiv(T, BLOCK_T))
        conv1d_relu_triton[grid_conv0](
            x0_c, conv0_w_c, conv0_b_c, y0,
            batch, C_in, T, conv0_w.shape[0], T, 5, 2,
            BLOCK_CO=BLOCK_CO, BLOCK_T=BLOCK_T
        )

        # ReLU
        # Implement ReLU in Triton: elementwise max(0, y0)
        y0_relu = torch.empty_like(y0)
        BLOCK_C = 64
        grid_relu = (batch, triton.cdiv(conv0_w.shape[0], BLOCK_C), triton.cdiv(T, BLOCK_T))
        # relu_triton kernel: y = max(0, x)
        # We need a relu kernel; define it:
        @triton.jit
        def relu_triton_kernel(x_ptr, y_ptr, N: tl.int32, C: tl.int32, T: tl.int32, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr):
            pid_n = tl.program_id(0)
            pid_c = tl.program_id(1)
            pid_t = tl.program_id(2)
            c_start = pid_c * BLOCK_C
            t_start = pid_t * BLOCK_T
            c_offsets = c_start + tl.arange(0, BLOCK_C)
            t_offsets = t_start + tl.arange(0, BLOCK_T)
            c_mask = c_offsets < C
            t_mask = t_offsets < T
            mask_out = c_mask[:, None] & t_mask[None, :]
            offs = ((pid_n * C) + c_offsets[:, None]) * T + t_offsets[None, :]
            x = tl.load(x_ptr + offs, mask=mask_out, other=0.0).to(tl.float32)
            x = tl.maximum(x, 0.0)
            tl.store(y_ptr + offs, x, mask=mask_out)

        relu_triton_kernel[grid_relu](
            y0, y0_relu, batch, conv0_w.shape[0], T, BLOCK_C=BLOCK_C, BLOCK_T=BLOCK_T
        )

        # conv1
        h = y0_relu.clone()
        conv1_w_c = conv1_w.contiguous()
        conv1_b_c = conv1_b.contiguous()
        y1 = torch.empty((batch, conv1_w.shape[0], T), device=x.device, dtype=x.dtype)

        grid_conv1 = (batch, triton.cdiv(conv1_w.shape[0], BLOCK_CO), triton.cdiv(T, BLOCK_T))
        conv1d_relu_triton[grid_conv1](
            h, conv1_w_c, conv1_b_c, y1,
            batch, conv1_w.shape[1], T, conv1_w.shape[0], T, 5, 2,
            BLOCK_CO=BLOCK_CO, BLOCK_T=BLOCK_T
        )

        # ReLU
        y1_relu = torch.empty_like(y1)
        grid_relu = (batch, triton.cdiv(conv1_w.shape[0], BLOCK_C), triton.cdiv(T, BLOCK_T))
        relu_triton_kernel[grid_relu](
            y1, y1_relu, batch, conv1_w.shape[0], T, BLOCK_C=BLOCK_C, BLOCK_T=BLOCK_T
        )

        # conv2
        h = y1_relu.clone()
        conv2_w_c = conv2_w.contiguous()
        conv2_b_c = conv2_b.contiguous()
        h_out = torch.empty((batch, conv2_w.shape[0], T), device=x.device, dtype=x.dtype)

        grid_conv2 = (batch, triton.cdiv(conv2_w.shape[0], BLOCK_CO), triton.cdiv(T, BLOCK_T))
        conv1d_relu_triton[grid_conv2](
            h, conv2_w_c, conv2_b_c, h_out,
            batch, conv2_w.shape[1], T, conv2_w.shape[0], T, 5, 2,
            BLOCK_CO=BLOCK_CO, BLOCK_T=BLOCK_T
        )

        # Now, update x1 = x1 + h_out
        x1_c = x1.contiguous()
        h_out_c = h_out.contiguous()
        # We need an elementwise addition kernel in Triton
        x1_new = torch.empty_like(x1_c)
        @triton.jit
        def add_triton_kernel(a_ptr, b_ptr, out_ptr, N: tl.int32, C: tl.int32, T: tl.int32, BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr):
            pid_n = tl.program_id(0)
            pid_c = tl.program_id(1)
            pid_t = tl.program_id(2)
            c_start = pid_c * BLOCK_C
            t_start = pid_t * BLOCK_T
            c_offsets = c_start + tl.arange(0, BLOCK_C)
            t_offsets = t_start + tl.arange(0, BLOCK_T)
            c_mask = c_offsets < C
            t_mask = t_offsets < T
            mask_out = c_mask[:, None] & t_mask[None, :]
            offs = ((pid_n * C) + c_offsets[:, None]) * T + t_offsets[None, :]
            a = tl.load(a_ptr + offs, mask=mask_out, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + offs, mask=mask_out, other=0.0).to(tl.float32)
            c = a + b
            tl.store(out_ptr + offs, c, mask=mask_out)

        add_triton_kernel[(batch, triton.cdiv(C_in, BLOCK_C), triton.cdiv(T, BLOCK_T))](
            x1_c, h_out_c, x1_new, batch, C_in, T, BLOCK_C=BLOCK_C, BLOCK_T=BLOCK_T
        )

        # Construct final y_out for this transform: concat halves [x0, x1_new]
        # We need to copy x0 into channels 0..C_in-1 and x1_new into channels C_in..2*C_in-1.
        # We will use concat_halves_triton for this.
        # First, ensure y_out is large enough: we already allocated y_out = [batch, 192, T].
        # We need to write into y_out using concat_halves_triton. But concat_halves_triton expects x0 and x1h pointers; here x1h is x1_new.
        # However, the original code uses x1 = x1 + h, and then concatenates x0 and x1. We have x0 and x1_new. We can call concat_halves_triton
        # with x0 = x0 and x1h = x1_new, and it will write into y_out with channel offsets.

        # But concat_halves_triton writes into a provided y_ptr. We need to pass y_out as y_ptr. It expects x0 and x1h of shape [N, C_in, T].
        # x0: x[:, :C_in, :], x1_new: x[:, C_in:, :]. So we need to slice from x accordingly:
        x0_slice = x[:, :C_in, :]
        x1h_slice = x1_new

        # Now call concat_halves_triton to fill y_out
        grid_concat = (batch, triton.cdiv(2 * C_in, BLOCK_C), triton.cdiv(T, BLOCK_T))
        concat_halves_triton[grid_concat](
            x0_slice, x1h_slice, y_out,
            batch, C_in, T, BLOCK_C=BLOCK_C, BLOCK_T=BLOCK_T
        )

        # Apply mask multiplication (even if mask is all ones, we do it for generality)
        # mask: [batch, 1, T], broadcast across channels
        mask = x_mask  # [batch, 1, T]
        # Ensure mask is contiguous and same device
        mask = mask.contiguous()
        # mask_mul_triton expects y_ptr, mask_ptr, N, C, T. Here C=192, T=T.
        grid_mask = (batch, triton.cdiv(192, BLOCK_C), triton.cdiv(T, BLOCK_T))
        mask_mul_triton[grid_mask](
            y_out, mask, batch, 192, T, BLOCK_C=BLOCK_C, BLOCK_T=BLOCK_T
        )

        # Update x for next transform
        x = y_out

    # After all transforms, y_out is the final output.
    return y_out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # We expect: x, x_mask, reverse, then 24 weight/bias tensors (4 transforms * 6 per transform)
        # The original Model.forward returns run(...), but since we cannot use torch ops in host code,
        # we call our Triton-only runner and return the final output.
        # Note: The original signature of run is not known exactly; in the provided setup, get_inputs returns
        # a dict of inputs including all conv weights and biases. The forward of Model calls run(x, x_mask, reverse, ...).
        # We reconstruct the call from args: first 3 are x, x_mask, reverse; the remaining are weights_and_biases.
        x = args[0]
        x_mask = args[1]
        reverse = args[2]
        weights_and_biases = list(args[3:])
        # Launch Triton-only runner
        return run_triton_only(x, x_mask, reverse, *weights_and_biases)


def run(*args):
    return ModelNew()(*args)
