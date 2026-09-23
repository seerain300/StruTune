import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


# Conv1d forward: no padding, stride=1
@triton.jit
def conv1d_forward_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, T_in, C_out, T_out, K,
    x_stride_n, x_stride_c, x_stride_t,
    w_stride_co, w_stride_ci, w_stride_k,
    out_stride_n, out_stride_c, out_stride_t,
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # Grid dims: (N, ceil(T_out/BLOCK_T), ceil(C_out/BLOCK_C))
    pid_n = tl.program_id(0)
    pid_tblk = tl.program_id(1)
    pid_cblk = tl.program_id(2)

    t_offsets = pid_tblk * BLOCK_T + tl.arange(0, BLOCK_T)
    co_offsets = pid_cblk * BLOCK_C + tl.arange(0, BLOCK_C)
    t_mask = t_offsets < T_out
    co_mask = co_offsets < C_out

    acc = tl.zeros([BLOCK_C, BLOCK_T], dtype=tl.float32)

    ci = 0
    while ci < C_in:
        k = 0
        while k < K:
            t_in = t_offsets - k  # for padding=0, only valid when 0 <= t_in < T_in
            in_bounds = (t_in >= 0) & (t_in < T_in)

            # Load x[n, ci, t_in] for all t_offsets (vectorized)
            x_offsets = pid_n * x_stride_n + ci * x_stride_c + t_in * x_stride_t
            # Broadcast over co_offsets
            x_ptrs = x_ptr + x_offsets[None, :] + co_offsets[:, None] * x_stride_c
            x_vals = tl.load(x_ptrs, mask=co_mask[:, None] & in_bounds[None, :], other=0.0)

            # Load w[co, ci, k]
            w_ptrs = w_ptr + co_offsets[:, None] * w_stride_co + ci * w_stride_ci + k * w_stride_k
            w_vals = tl.load(w_ptrs, mask=co_mask[:, None], other=0.0)

            # Accumulate
            acc += w_vals[:, None] * x_vals
            k += 1
        ci += 1

    # Add bias
    b_vals = tl.load(b_ptr + co_offsets, mask=co_mask, other=0.0)
    acc += b_vals[:, None]

    # Store results
    out_ptrs = out_ptr + pid_n * out_stride_n + co_offsets[:, None] * out_stride_c + t_offsets[None, :] * out_stride_t
    tl.store(out_ptrs, acc, mask=co_mask[:, None] & t_mask[None, :])


# Conv1d + ReLU (apply ReLU on conv result)
@triton.jit
def conv1d_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, T_in, C_out, T_out, K,
    x_stride_n, x_stride_c, x_stride_t,
    w_stride_co, w_stride_ci, w_stride_k,
    out_stride_n, out_stride_c, out_stride_t,
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_tblk = tl.program_id(1)
    pid_cblk = tl.program_id(2)

    t_offsets = pid_tblk * BLOCK_T + tl.arange(0, BLOCK_T)
    co_offsets = pid_cblk * BLOCK_C + tl.arange(0, BLOCK_C)
    t_mask = t_offsets < T_out
    co_mask = co_offsets < C_out

    acc = tl.zeros([BLOCK_C, BLOCK_T], dtype=tl.float32)

    ci = 0
    while ci < C_in:
        k = 0
        while k < K:
            t_in = t_offsets - k
            in_bounds = (t_in >= 0) & (t_in < T_in)

            x_offsets = pid_n * x_stride_n + ci * x_stride_c + t_in * x_stride_t
            x_ptrs = x_ptr + x_offsets[None, :] + co_offsets[:, None] * x_stride_c
            x_vals = tl.load(x_ptrs, mask=co_mask[:, None] & in_bounds[None, :], other=0.0)

            w_ptrs = w_ptr + co_offsets[:, None] * w_stride_co + ci * w_stride_ci + k * w_stride_k
            w_vals = tl.load(w_ptrs, mask=co_mask[:, None], other=0.0)

            acc += w_vals[:, None] * x_vals
            k += 1
        ci += 1

    b_vals = tl.load(b_ptr + co_offsets, mask=co_mask, other=0.0)
    acc += b_vals[:, None]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    out_ptrs = out_ptr + pid_n * out_stride_n + co_offsets[:, None] * out_stride_c + t_offsets[None, :] * out_stride_t
    tl.store(out_ptrs, acc, mask=co_mask[:, None] & t_mask[None, :])


# Split halves along channel dimension: x [N, 2*C_half, T] -> x0 [N, C_half, T], x1 [N, C_half, T]
@triton.jit
def split_halves_kernel(
    x_ptr, x0_ptr, x1_ptr,
    N, C_half, T,
    x_stride_n, x_stride_c, x_stride_t,
    x0_stride_n, x0_stride_c, x0_stride_t,
    x1_stride_n, x1_stride_c, x1_stride_t,
):
    # Grid: (N, C_half, T)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    x0_offset = pid_n * x_stride_n + pid_c * x_stride_c + pid_t * x_stride_t
    x1_offset = pid_n * x_stride_n + (pid_c + C_half) * x_stride_c + pid_t * x_stride_t

    val0 = tl.load(x_ptr + x0_offset)
    val1 = tl.load(x_ptr + x1_offset)

    tl.store(x0_ptr + pid_n * x0_stride_n + pid_c * x0_stride_c + pid_t * x0_stride_t, val0)
    tl.store(x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + pid_t * x1_stride_t, val1)


# Add coupling: out[n, c, t] = x1[n, c, t] + h[n, c, t] (forward) or - (reverse)
@triton.jit
def add_halves_kernel(
    x1_ptr, h_ptr, out_ptr,
    N, C, T,
    x1_stride_n, x1_stride_c, x1_stride_t,
    h_stride_n, h_stride_c, h_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
    ADD: tl.constexpr,
):
    # Grid: (N, C, T)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    v1 = tl.load(x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + pid_t * x1_stride_t)
    v2 = tl.load(h_ptr + pid_n * h_stride_n + pid_c * h_stride_c + pid_t * h_stride_t)
    res = v1 + v2 if ADD else v1 - v2
    tl.store(out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t, res)


# Concatenate x0 and x1 along channel dimension: out [N, 2*C_half, T]
@triton.jit
def cat_halves_kernel(
    x0_ptr, x1_ptr, out_ptr,
    N, C_half, T,
    x0_stride_n, x0_stride_c, x0_stride_t,
    x1_stride_n, x1_stride_c, x1_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
):
    # Grid: (N, 2*C_half, T)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    # First half
    val0 = tl.load(x0_ptr + pid_n * x0_stride_n + pid_c * x0_stride_c + pid_t * x0_stride_t)
    out_offset = pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t
    tl.store(out_ptr + out_offset, val0)

    # Second half (original channel index c' = c + C_half)
    val1 = tl.load(x1_ptr + pid_n * x1_stride_n + (pid_c + C_half) * x1_stride_c + pid_t * x1_stride_t)
    out_offset = pid_n * out_stride_n + (pid_c + C_half) * out_stride_c + pid_t * out_stride_t
    tl.store(out_ptr + out_offset, val1)


# Multiply by mask (mask is [N, 1, T], keep generality). In provided inputs, mask is all ones.
@triton.jit
def mask_mul_kernel(
    h_ptr, mask_ptr, out_ptr,
    N, C, T,
    h_stride_n, h_stride_c, h_stride_t,
    mask_stride_n, mask_stride_c, mask_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
):
    # Grid: (N, C, T)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    h_val = tl.load(h_ptr + pid_n * h_stride_n + pid_c * h_stride_c + pid_t * h_stride_t)
    mask_val = tl.load(mask_ptr + pid_n * mask_stride_n + 0 * mask_stride_c + pid_t * mask_stride_t)  # channel dim is 1
    res = h_val * mask_val
    tl.store(out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t, res)


# ===== Host-side helpers (pure PyTorch for setup) =====
def conv1d_triton_forward(x, w, b, N, C_in, T_in, C_out, T_out, K,
                           BLOCK_C=128, BLOCK_T=128):
    # Allocate output
    out = torch.empty((N, C_out, T_out), device=x.device, dtype=torch.float32)
    # Strides
    x_stride_n, x_stride_c, x_stride_t = x.stride()
    w_stride_co, w_stride_ci, w_stride_k = w.stride()
    out_stride_n, out_stride_c, out_stride_t = out.stride()
    # Launch grid
    grid = (N, triton.cdiv(T_out, BLOCK_T), triton.cdiv(C_out, BLOCK_C))
    conv1d_forward_kernel[grid](
        x, w, b, out,
        N, C_in, T_in, C_out, T_out, K,
        x_stride_n, x_stride_c, x_stride_t,
        w_stride_co, w_stride_ci, w_stride_k,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK_C=BLOCK_C, BLOCK_T=BLOCK_T,
        num_warps=4, num_stages=2,
    )
    return out


def conv1d_triton_relu(x, w, b, N, C_in, T_in, C_out, T_out, K,
                        BLOCK_C=128, BLOCK_T=128):
    out = torch.empty((N, C_out, T_out), device=x.device, dtype=torch.float32)
    x_stride_n, x_stride_c, x_stride_t = x.stride()
    w_stride_co, w_stride_ci, w_stride_k = w.stride()
    out_stride_n, out_stride_c, out_stride_t = out.stride()
    grid = (N, triton.cdiv(T_out, BLOCK_T), triton.cdiv(C_out, BLOCK_C))
    conv1d_relu_kernel[grid](
        x, w, b, out,
        N, C_in, T_in, C_out, T_out, K,
        x_stride_n, x_stride_c, x_stride_t,
        w_stride_co, w_stride_ci, w_stride_k,
        out_stride_n, out_stride_c, out_stride_t,
        BLOCK_C=BLOCK_C, BLOCK_T=BLOCK_T,
        num_warps=4, num_stages=2,
    )
    return out


def split_halves_triton(x, x0, x1, N, C_half, T):
    # x: [N, 2*C_half, T], x0, x1: [N, C_half, T]
    x_stride_n, x_stride_c, x_stride_t = x.stride()
    x0_stride_n, x0_stride_c, x0_stride_t = x0.stride()
    x1_stride_n, x1_stride_c, x1_stride_t = x1.stride()
    grid = (N, C_half, T)
    split_halves_kernel[grid](
        x, x0, x1,
        N, C_half, T,
        x_stride_n, x_stride_c, x_stride_t,
        x0_stride_n, x0_stride_c, x0_stride_t,
        x1_stride_n, x1_stride_c, x1_stride_t,
        num_warps=2, num_stages=1,
    )


def add_halves_triton(x1, h, out, N, C, T, ADD=True):
    x1_stride_n, x1_stride_c, x1_stride_t = x1.stride()
    h_stride_n, h_stride_c, h_stride_t = h.stride()
    out_stride_n, out_stride_c, out_stride_t = out.stride()
    grid = (N, C, T)
    add_halves_kernel[grid](
        x1, h, out,
        N, C, T,
        x1_stride_n, x1_stride_c, x1_stride_t,
        h_stride_n, h_stride_c, h_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        ADD=ADD,
        num_warps=4, num_stages=2,
    )


def cat_halves_triton(x0, x1, out, N, C_half, T):
    x0_stride_n, x0_stride_c, x0_stride_t = x0.stride()
    x1_stride_n, x1_stride_c, x1_stride_t = x1.stride()
    out_stride_n, out_stride_c, out_stride_t = out.stride()
    grid = (N, 2 * C_half, T)
    cat_halves_kernel[grid](
        x0, x1, out,
        N, C_half, T,
        x0_stride_n, x0_stride_c, x0_stride_t,
        x1_stride_n, x1_stride_c, x1_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        num_warps=4, num_stages=2,
    )


def mask_mul_triton(h, mask, out, N, C, T):
    h_stride_n, h_stride_c, h_stride_t = h.stride()
    mask_stride_n, mask_stride_c, mask_stride_t = mask.stride()
    out_stride_n, out_stride_c, out_stride_t = out.stride()
    grid = (N, C, T)
    mask_mul_kernel[grid](
        h, mask, out,
        N, C, T,
        h_stride_n, h_stride_c, h_stride_t,
        mask_stride_n, mask_stride_c, mask_stride_t,
        out_stride_n, out_stride_c, out_stride_t,
        num_warps=4, num_stages=2,
    )


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor,
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
                transform_3_conv2_bias: torch.Tensor):
        # Ensure dtype/device
        x = x.contiguous().to(torch.float32)
        x_mask = x_mask.contiguous().to(torch.float32)
        N, C, T = x.shape
        C_half = C // 2

        # Allocate intermediate tensors
        x0 = torch.empty((N, C_half, T), device=x.device, dtype=torch.float32)
        x1 = torch.empty((N, C_half, T), device=x.device, dtype=torch.float32)

        # We need to apply 4 transforms sequentially:
        # Each transform: conv0 -> ReLU -> conv1 -> ReLU -> conv2
        # Then split x along channels: first C_half to x0, second C_half to x1
        # Update x1 = x1 + h (forward) or x1 = x1 - h (reverse), then concatenate back.
        for t_idx in range(4):
            # Prepare for current transform
            # 1) conv0: in=96, out=192, k=5
            conv0_w = eval(f"transform_{t_idx}_conv0_weight")
            conv0_b = eval(f"transform_{t_idx}_conv0_bias")
            conv1_w = eval(f"transform_{t_idx}_conv1_weight")
            conv1_b = eval(f"transform_{t_idx}_conv1_bias")
            conv2_w = eval(f"transform_{t_idx}_conv2_weight")
            conv2_b = eval(f"transform_{t_idx}_conv2_bias")

            # Split halves
            split_halves_triton(x, x0, x1, N, C_half, T)

            # conv0 (forward) -> ReLU -> conv1 (forward) -> ReLU -> conv2 (forward)
            # Note: In original apply_transform, ReLU is between conv0->conv1 and conv1->conv2.
            # We'll compute conv0, ReLU, conv1, ReLU, conv2 entirely in Triton.

            # conv0: x0 -> h0
            h0 = conv1d_triton_forward(x0, conv0_w, conv0_b, N, C_half, T, conv0_w.shape[0], T - 4, 5)
            # ReLU on h0
            h0 = conv1d_triton_relu(h0, conv0_w, conv0_b, N, conv0_w.shape[0], T - 4, conv1_w.shape[0], T - 4, 5)
            # conv1: h0 -> h1
            h1 = conv1d_triton_forward(h0, conv1_w, conv1_b, N, conv1_w.shape[1], T - 4, conv1_w.shape[0], T - 4, 5)
            # ReLU on h1
            h1 = conv1d_triton_relu(h1, conv1_w, conv1_b, N, conv1_w.shape[0], T - 4, conv1_w.shape[0], T - 4, 5)
            # conv2: h1 -> h (final transform output)
            h = conv1d_triton_forward(h1, conv2_w, conv2_b, N, conv2_w.shape[1], T - 4, conv2_w.shape[0], T - 4, 5)

            # Apply mask
            # mask is [N, 1, T]; mask multiplication in Triton
            h_masked = torch.empty_like(h)
            mask_mul_triton(h, x_mask, h_masked, N, h.shape[1], T)

            # Update x1
            x1_out = torch.empty_like(x1)
            add_halves_triton(x1, h_masked, x1_out, N, C_half, T, ADD=(not reverse))

            # Concatenate back to form new x: [x0, x1_out]
            new_x = torch.empty((N, C, T), device=x.device, dtype=torch.float32)
            cat_halves_triton(x0, x1_out, new_x, N, C_half, T)

            # Update x for next transform
            x = new_x

        return x


# The original get_inputs and run can remain as-is; ModelNew.forward is the entry point.
# Example of how to instantiate and run (not part of ModelNew itself):
# m = ModelNew().to('cuda')
# inputs = get_inputs({'batch_size': 16, 'time': 2447}, torch.device('cuda'))
# out = m(*inputs)  # This will launch all Triton kernels


def run(*args):
    return ModelNew()(*args)
