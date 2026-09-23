import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels: all computation is done here; no torch ops in forward.

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
        # program ids: over (N, C_OUT, time blocks)
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
        inp_ptr,        # *float32, input tensor
        out_ptr,        # *float32, output tensor (can alias inp_ptr)
        N, C, T,
        in_stride_n, in_stride_c, in_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        grid0: tl.constexpr,  # grid[0] = N*C
        grid1: tl.constexpr,  # grid[1] = T blocks
        BLOCK_T: tl.constexpr,
    ):
        pid_nc = tl.program_id(0)
        pid_tb = tl.program_id(1)
        n = pid_nc // C
        c = pid_nc % C

        t_start = pid_tb * BLOCK_T
        t_offsets = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T

        inp_ptrs = inp_ptr + n * in_stride_n + c * in_stride_c + t_offsets * in_stride_t
        out_ptrs = out_ptr + n * out_stride_n + c * out_stride_c + t_offsets * out_stride_t

        x = tl.load(inp_ptrs, mask=t_mask, other=0.0)
        y = tl.maximum(x, 0.0)
        tl.store(out_ptrs, y, mask=t_mask)

    @triton.jit
    def concat_half_channels_kernel(
        x0_ptr,         # *float32, [N, C_half, T]
        x1_ptr,         # *float32, [N, C_half, T]
        out_ptr,        # *float32, [N, 2*C_half, T]
        N, C_half, T,
        x0_stride_n, x0_stride_c, x0_stride_t,
        x1_stride_n, x1_stride_c, x1_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        c_block_start: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        # grid over (N, C_blocks, T_blocks)
        pid_n = tl.program_id(0)
        pid_cb = tl.program_id(1)
        pid_tb = tl.program_id(2)

        c_offsets = c_block_start + tl.arange(0, BLOCK_C)
        t_offsets = pid_tb * BLOCK_T + tl.arange(0, BLOCK_T)

        c_mask = c_offsets < C_half
        t_mask = t_offsets < T

        # Copy x0 into out[:, :C_half, :]
        for ci in range(0, BLOCK_C):
            c_i = c_offsets[ci]
            mask_c = c_mask[ci]
            if mask_c:
                x0_ptrs = x0_ptr + pid_n * x0_stride_n + c_i * x0_stride_c + t_offsets * x0_stride_t
                out_ptrs = out_ptr + pid_n * out_stride_n + c_i * out_stride_c + t_offsets * out_stride_t
                vals = tl.load(x0_ptrs, mask=t_mask, other=0.0)
                tl.store(out_ptrs, vals, mask=t_mask)

        # Copy x1 into out[:, C_half:, :]
        for ci in range(0, BLOCK_C):
            c_i = c_offsets[ci]
            mask_c = c_mask[ci]
            if mask_c:
                x1_ptrs = x1_ptr + pid_n * x1_stride_n + c_i * x1_stride_c + t_offsets * x1_stride_t
                out_ptrs = out_ptr + pid_n * out_stride_n + (C_half + c_i) * out_stride_c + t_offsets * out_stride_t
                vals = tl.load(x1_ptrs, mask=t_mask, other=0.0)
                tl.store(out_ptrs, vals, mask=t_mask)

    @triton.jit
    def add_affine_kernel(
        x_ptr,          # *float32, [N, C, T] (x1)
        h_ptr,          # *float32, [N, C_half, T] (h from transform on x0)
        out_ptr,        # *float32, [N, C, T]
        N, C, T,
        x_stride_n, x_stride_c, x_stride_t,
        h_stride_n, h_stride_c, h_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        c_block_start: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        # grid over (N, C_blocks, T_blocks)
        pid_n = tl.program_id(0)
        pid_cb = tl.program_id(1)
        pid_tb = tl.program_id(2)

        c_offsets = c_block_start + tl.arange(0, BLOCK_C)
        t_offsets = pid_tb * BLOCK_T + tl.arange(0, BLOCK_T)

        c_mask = c_offsets < C
        t_mask = t_offsets < T

        for ci in range(0, BLOCK_C):
            c_i = c_offsets[ci]
            mask_c = c_mask[ci]
            if mask_c:
                x_ptrs = x_ptr + pid_n * x_stride_n + c_i * x_stride_c + t_offsets * x_stride_t
                h_ptrs = h_ptr + pid_n * h_stride_n + c_i * h_stride_c + t_offsets * h_stride_t
                out_ptrs = out_ptr + pid_n * out_stride_n + c_i * out_stride_c + t_offsets * out_stride_t
                x_vals = tl.load(x_ptrs, mask=t_mask, other=0.0)
                h_vals = tl.load(h_ptrs, mask=t_mask, other=0.0)
                y = x_vals + h_vals
                tl.store(out_ptrs, y, mask=t_mask)

    @triton.jit
    def sub_affine_kernel(
        x_ptr,          # *float32, [N, C, T] (x1)
        h_ptr,          # *float32, [N, C_half, T] (h from transform on x0)
        out_ptr,        # *float32, [N, C, T]
        N, C, T,
        x_stride_n, x_stride_c, x_stride_t,
        h_stride_n, h_stride_c, h_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        c_block_start: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        # grid over (N, C_blocks, T_blocks)
        pid_n = tl.program_id(0)
        pid_cb = tl.program_id(1)
        pid_tb = tl.program_id(2)

        c_offsets = c_block_start + tl.arange(0, BLOCK_C)
        t_offsets = pid_tb * BLOCK_T + tl.arange(0, BLOCK_T)

        c_mask = c_offsets < C
        t_mask = t_offsets < T

        for ci in range(0, BLOCK_C):
            c_i = c_offsets[ci]
            mask_c = c_mask[ci]
            if mask_c:
                x_ptrs = x_ptr + pid_n * x_stride_n + c_i * x_stride_c + t_offsets * x_stride_t
                h_ptrs = h_ptr + pid_n * h_stride_n + c_i * h_stride_c + t_offsets * h_stride_t
                out_ptrs = out_ptr + pid_n * out_stride_n + c_i * out_stride_c + t_offsets * out_stride_t
                x_vals = tl.load(x_ptrs, mask=t_mask, other=0.0)
                h_vals = tl.load(h_ptrs, mask=t_mask, other=0.0)
                y = x_vals - h_vals
                tl.store(out_ptrs, y, mask=t_mask)

    @triton.jit
    def mask_mul_kernel(
        inp_ptr,        # *float32, [N, C, T] input (x or h)
        mask_ptr,       # *float32, [N, 1, T] mask (broadcast along channel)
        out_ptr,        # *float32, [N, C, T]
        N, C, T,
        in_stride_n, in_stride_c, in_stride_t,
        mask_stride_n, mask_stride_c, mask_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        grid0: tl.constexpr,  # grid[0] = N*C
        grid1: tl.constexpr,  # grid[1] = T blocks
        BLOCK_T: tl.constexpr,
    ):
        pid_nc = tl.program_id(0)
        pid_tb = tl.program_id(1)
        n = pid_nc // C
        c = pid_nc % C

        t_start = pid_tb * BLOCK_T
        t_offsets = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T

        inp_ptrs = inp_ptr + n * in_stride_n + c * in_stride_c + t_offsets * in_stride_t
        mask_ptrs = mask_ptr + n * mask_stride_n + 0 * mask_stride_c + t_offsets * mask_stride_t
        out_ptrs = out_ptr + n * out_stride_n + c * out_stride_c + t_offsets * out_stride_t

        x = tl.load(inp_ptrs, mask=t_mask, other=0.0)
        m = tl.load(mask_ptrs, mask=t_mask, other=0.0)
        y = x * m
        tl.store(out_ptrs, y, mask=t_mask)


def _run_triton(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    reverse: bool,
    # weights for 4 transforms
    t0_w0, t0_b0, t0_w1, t0_b1, t0_w2, t0_b2,
    t1_w0, t1_b0, t1_w1, t1_b1, t1_w2, t1_b2,
    t2_w0, t2_b0, t2_w1, t2_b1, t2_w2, t2_b2,
    t3_w0, t3_b0, t3_w1, t3_b1, t3_w2, t3_b2,
):
    assert TRITON_AVAILABLE, "Triton is not available"
    assert x.is_cuda, "Input must be on CUDA for Triton kernels"

    N, C, T = x.shape
    C_half = C // 2
    K = 5
    PAD = K // 2  # 2

    # Launch parameters
    BLOCK_T = 128
    BLOCK_C = 64

    # We'll apply the transforms sequentially. For each, we compute:
    # x0 = x[:, :C_half, :], x1 = x[:, C_half:, :], h = apply_transform(x0), then update x1, concatenate.

    # Helper to run one transform: return h (shape [N, C_half, T]) and also update x1 if out is provided.
    def one_transform(x0, x1, w0, b0, w1, b1, w2, b2, out_x1=None):
        # Compute h = apply_transform(x0) using Triton convs and ReLUs
        # conv0
        y0 = torch.empty((N, w0.shape[0], T), dtype=x.dtype, device=x.device)
        grid0 = (N, w0.shape[0], triton.cdiv(T, BLOCK_T))
        conv1d_forward_kernel[grid0](
            x0, w0, b0, y0,
            N, x0.shape[1], T, w0.shape[1], w0.shape[0], K, PAD,
            x0.stride(0), x0.stride(1), x0.stride(2),
            w0.stride(0), w0.stride(1), w0.stride(2),
            y0.stride(0), y0.stride(1), y0.stride(2),
            0, BLOCK_T,
            num_warps=4,
        )
        y0 = y0  # y0 is conv0 output [N, C_out0, T]
        # ReLU
        y0 = torch.empty_like(y0)
        grid_relu = (N * w0.shape[0], triton.cdiv(T, BLOCK_T))
        relu_forward_kernel[grid_relu](
            y0, y0,
            N, w0.shape[0], T,
            y0.stride(0), y0.stride(1), y0.stride(2),
            y0.stride(0), y0.stride(1), y0.stride(2),
            N * w0.shape[0], triton.cdiv(T, BLOCK_T), BLOCK_T,
            num_warps=4,
        )
        # conv1
        y1 = torch.empty((N, w1.shape[0], T), dtype=x.dtype, device=x.device)
        grid1 = (N, w1.shape[0], triton.cdiv(T, BLOCK_T))
        conv1d_forward_kernel[grid1](
            y0, w1, b1, y1,
            N, w0.shape[0], T, w1.shape[1], w1.shape[0], K, PAD,
            y0.stride(0), y0.stride(1), y0.stride(2),
            w1.stride(0), w1.stride(1), w1.stride(2),
            y1.stride(0), y1.stride(1), y1.stride(2),
            0, BLOCK_T,
            num_warps=4,
        )
        y1 = y1  # conv1 output [N, C_out1, T]
        # ReLU
        y1 = torch.empty_like(y1)
        grid_relu1 = (N * w1.shape[0], triton.cdiv(T, BLOCK_T))
        relu_forward_kernel[grid_relu1](
            y1, y1,
            N, w1.shape[0], T,
            y1.stride(0), y1.stride(1), y1.stride(2),
            y1.stride(0), y1.stride(1), y1.stride(2),
            N * w1.shape[0], triton.cdiv(T, BLOCK_T), BLOCK_T,
            num_warps=4,
        )
        # conv2
        h = torch.empty((N, w2.shape[0], T), dtype=x.dtype, device=x.device)
        grid2 = (N, w2.shape[0], triton.cdiv(T, BLOCK_T))
        conv1d_forward_kernel[grid2](
            y1, w2, b2, h,
            N, w1.shape[0], T, w2.shape[1], w2.shape[0], K, PAD,
            y1.stride(0), y1.stride(1), y1.stride(2),
            w2.stride(0), w2.stride(1), w2.stride(2),
            h.stride(0), h.stride(1), h.stride(2),
            0, BLOCK_T,
            num_warps=4,
        )
        # h is [N, C_half, T]
        # Optionally update x1
        if out_x1 is not None:
            if reverse:
                sub_affine_kernel[(N, triton.cdiv(C_half, BLOCK_C), triton.cdiv(T, BLOCK_T))](
                    x1, h, out_x1,
                    N, C_half, T,
                    x1.stride(0), x1.stride(1), x1.stride(2),
                    h.stride(0), h.stride(1), h.stride(2),
                    out_x1.stride(0), out_x1.stride(1), out_x1.stride(2),
                    0, BLOCK_C, BLOCK_T,
                    num_warps=4,
                )
            else:
                add_affine_kernel[(N, triton.cdiv(C_half, BLOCK_C), triton.cdiv(T, BLOCK_T))](
                    x1, h, out_x1,
                    N, C_half, T,
                    x1.stride(0), x1.stride(1), x1.stride(2),
                    h.stride(0), h.stride(1), h.stride(2),
                    out_x1.stride(0), out_x1.stride(1), out_x1.stride(2),
                    0, BLOCK_C, BLOCK_T,
                    num_warps=4,
                )
        return h

    # Main loop: apply 4 transforms
    x0 = x[:, :C_half, :]
    x1 = x[:, C_half:, :]
    # We'll build output x step-by-step by concatenating halves after each transform
    # However, concatenation in Triton is done per step; to keep output tensor, we do it after each.

    # Allocate final output buffer as [N, 2*C_half, T] (two halves)
    out_channels = 2 * C_half
    final_out = torch.empty((N, out_channels, T), dtype=x.dtype, device=x.device)

    # Apply each transform in sequence
    # Forward: update x1 += h
    # Reverse: update x1 -= h (we pass h and out_x1=x1 to sub_affine_kernel)
    for _ in range(4):
        # Choose weights by their order; in the original, they are identical sets, but we keep generic
        # Here, we use t0 weights for the first transform, t1 for second, etc. Since the original sets are identical,
        # each transform uses the same weights, but we support passing different sets.
        if _ == 0:
            w0 = t0_w0; b0 = t0_b0; w1 = t0_w1; b1 = t0_b1; w2 = t0_w2; b2 = t0_b2
        elif _ == 1:
            w0 = t1_w0; b0 = t1_b0; w1 = t1_w1; b1 = t1_b1; w2 = t1_w2; b2 = t1_b2
        elif _ == 2:
            w0 = t2_w0; b0 = t2_b0; w1 = t2_w1; b1 = t2_b1; w2 = t2_w2; b2 = t2_b2
        else:
            w0 = t3_w0; b0 = t3_b0; w1 = t3_w1; b1 = t3_b1; w2 = t3_w2; b2 = t3_b2

        # Compute h on x0
        h = one_transform(x0, x1, w0, b0, w1, b1, w2, b2, out_x1=x1)[0]  # out_x1 is updated inside

        # Apply mask to h (broadcast along channels)
        # x_mask is [N, 1, T], we need [N, C_half, T]
        masked_h = torch.empty((N, C_half, T), dtype=x.dtype, device=x.device)
        grid_mask = (N * C_half, triton.cdiv(T, BLOCK_T))
        mask_mul_kernel[grid_mask](
            h, x_mask, masked_h,
            N, C_half, T,
            h.stride(0), h.stride(1), h.stride(2),
            x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
            masked_h.stride(0), masked_h.stride(1), masked_h.stride(2),
            N * C_half, triton.cdiv(T, BLOCK_T), BLOCK_T,
            num_warps=4,
        )

        # Concatenate halves into final_out: first half channels from x0, second half from x1
        # Initialize final_out to zeros and fill
        final_out.zero_()  # safe to zero-initialize for overwrite
        # Copy x0 into final_out[:, :C_half, :]
        grid_concat = (N, triton.cdiv(C_half, BLOCK_C), triton.cdiv(T, BLOCK_T))
        concat_half_channels_kernel[grid_concat](
            x0, x1, final_out,
            N, C_half, T,
            x0.stride(0), x0.stride(1), x0.stride(2),
            x1.stride(0), x1.stride(1), x1.stride(2),
            final_out.stride(0), final_out.stride(1), final_out.stride(2),
            0, BLOCK_C, BLOCK_T,
            num_warps=4,
        )

        # Update x: set x to final_out
        x = final_out

    return x


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # The harness will pass all required tensors: x, x_mask, reverse flag, then the 24 weight/bias tensors.
        # We mirror the original run signature.
        return _run_triton(*args)


def run(*args):
    return ModelNew()(*args)
