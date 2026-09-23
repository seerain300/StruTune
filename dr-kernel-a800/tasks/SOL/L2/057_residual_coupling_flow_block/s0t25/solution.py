import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv1d_forward_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, T_in, C_out, T_out, K,
    x_stride_n, x_stride_c, x_stride_t,
    w_stride_co, w_stride_ci, w_stride_k,
    out_stride_n, out_stride_c, out_stride_t,
    BLOCK_C: tl.constexpr,
):
    # Grid: (N, T_out, C_out_blocks)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_cblk = tl.program_id(2)

    co_start = pid_cblk * BLOCK_C
    co_offsets = co_start + tl.arange(0, BLOCK_C)
    co_mask = co_offsets < C_out

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    # Sum over input channels and kernel taps
    ci = 0
    while ci < C_in:
        k = 0
        while k < K:
            # For padding=0, only valid t_in gives non-zero; we guard loads with in_bounds.
            t_in = pid_t - k  # output index t, kernel at k: input index = t - k
            in_bounds = (t_in >= 0) & (t_in < T_in)

            # Load x for all co in block
            x_offsets = pid_n * x_stride_n + ci * x_stride_c + t_in * x_stride_t
            co_vec_offsets = co_offsets * x_stride_c
            x_ptrs = x_ptr + x_offsets + co_vec_offsets
            # Masked load; out-of-bounds gives 0.0 (zero padding)
            x_vals = tl.load(x_ptrs, mask=co_mask & in_bounds, other=0.0)

            # Load w for all co in block for this (ci, k)
            w_ptrs = w_ptr + co_offsets * w_stride_co + ci * w_stride_ci + k * w_stride_k
            w_vals = tl.load(w_ptrs, mask=co_mask, other=0.0)

            acc += x_vals * w_vals
            k += 1
        ci += 1

    # Add bias
    b_vals = tl.load(b_ptr + co_offsets, mask=co_mask, other=0.0)
    acc += b_vals

    # Store output
    out_offsets = pid_n * out_stride_n + co_offsets * out_stride_c + pid_t * out_stride_t
    tl.store(out_ptr + out_offsets, acc, mask=co_mask)


@triton.jit
def conv1d_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, T_in, C_out, T_out, K,
    x_stride_n, x_stride_c, x_stride_t,
    w_stride_co, w_stride_ci, w_stride_k,
    out_stride_n, out_stride_c, out_stride_t,
    BLOCK_C: tl.constexpr,
):
    # Same as conv1d_forward, then apply ReLU
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_cblk = tl.program_id(2)

    co_start = pid_cblk * BLOCK_C
    co_offsets = co_start + tl.arange(0, BLOCK_C)
    co_mask = co_offsets < C_out

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    ci = 0
    while ci < C_in:
        k = 0
        while k < K:
            t_in = pid_t - k
            in_bounds = (t_in >= 0) & (t_in < T_in)

            x_offsets = pid_n * x_stride_n + ci * x_stride_c + t_in * x_stride_t
            co_vec_offsets = co_offsets * x_stride_c
            x_ptrs = x_ptr + x_offsets + co_vec_offsets
            x_vals = tl.load(x_ptrs, mask=co_mask & in_bounds, other=0.0)

            w_ptrs = w_ptr + co_offsets * w_stride_co + ci * w_stride_ci + k * w_stride_k
            w_vals = tl.load(w_ptrs, mask=co_mask, other=0.0)

            acc += x_vals * w_vals
            k += 1
        ci += 1

    b_vals = tl.load(b_ptr + co_offsets, mask=co_mask, other=0.0)
    acc += b_vals

    # Apply ReLU
    acc = tl.maximum(acc, 0.0)

    out_offsets = pid_n * out_stride_n + co_offsets * out_stride_c + pid_t * out_stride_t
    tl.store(out_ptr + out_offsets, acc, mask=co_mask)


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

    # First half channels
    x_offsets0 = pid_n * x_stride_n + pid_c * x_stride_c + pid_t * x_stride_t
    val0 = tl.load(x_ptr + x_offsets0)
    tl.store(x0_ptr + pid_n * x0_stride_n + pid_c * x0_stride_c + pid_t * x0_stride_t, val0)

    # Second half channels (original index c' = c + C_half)
    x_offsets1 = pid_n * x_stride_n + (pid_c + C_half) * x_stride_c + pid_t * x_stride_t
    val1 = tl.load(x_ptr + x_offsets1)
    tl.store(x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + pid_t * x1_stride_t, val1)


@triton.jit
def add_halves_kernel(
    x1_ptr, h_ptr, out_ptr,
    N, C, T,
    x1_stride_n, x1_stride_c, x1_stride_t,
    h_stride_n, h_stride_c, h_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
    ADD: tl.constexpr,  # True for forward (add), False for reverse (subtract)
):
    # Grid: (N, C, T)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    x1_val = tl.load(x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + pid_t * x1_stride_t)
    h_val = tl.load(h_ptr + pid_n * h_stride_n + pid_c * h_stride_c + pid_t * h_stride_t)
    res = x1_val + h_val if ADD else x1_val - h_val
    tl.store(out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t, res)


@triton.jit
def cat_halves_kernel(
    x0_ptr, x1_ptr, out_ptr,
    N, C_half, T,
    x0_stride_n, x0_stride_c, x0_stride_t,
    x1_stride_n, x1_stride_c, x1_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
):
    # Grid: (N, C_half, T) for both halves; write out as (N, 2*C_half, T)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    # First half
    val0 = tl.load(x0_ptr + pid_n * x0_stride_n + pid_c * x0_stride_c + pid_t * x0_stride_t)
    out_idx_c = pid_c  # first half remains pid_c
    tl.store(out_ptr + pid_n * out_stride_n + out_idx_c * out_stride_c + pid_t * out_stride_t, val0)

    # Second half
    val1 = tl.load(x1_ptr + pid_n * x1_stride_n + pid_c * x1_stride_c + pid_t * x1_stride_t)
    out_idx_c = pid_c + C_half  # second half starts at C_half
    tl.store(out_ptr + pid_n * out_stride_n + out_idx_c * out_stride_c + pid_t * out_stride_t, val1)


@triton.jit
def mask_mul_kernel(
    x_ptr, mask_ptr, out_ptr,
    N, C, T,
    x_stride_n, x_stride_c, x_stride_t,
    mask_stride_n, mask_stride_c, mask_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
):
    # Grid: (N, C, T)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    x_val = tl.load(x_ptr + pid_n * x_stride_n + pid_c * x_stride_c + pid_t * x_stride_t)
    # mask has shape [N, 1, T]; for c dimension, it's 1, so we just load mask at c=0
    mask_val = tl.load(mask_ptr + pid_n * mask_stride_n + 0 * mask_stride_c + pid_t * mask_stride_t)
    res = x_val * mask_val
    tl.store(out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t, res)


@triton.jit
def relu_inplace_kernel(
    x_ptr, out_ptr,
    N, C, T,
    x_stride_n, x_stride_c, x_stride_t,
    out_stride_n, out_stride_c, out_stride_t,
):
    # In-place ReLU; write result to out_ptr (can be same as x_ptr if out_ptr=x_ptr)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    val = tl.load(x_ptr + pid_n * x_stride_n + pid_c * x_stride_c + pid_t * x_stride_t)
    val = tl.maximum(val, 0.0)
    tl.store(out_ptr + pid_n * out_stride_n + pid_c * out_stride_c + pid_t * out_stride_t, val)


def _compute_T_out(T_in, K):
    # PyTorch conv1d default padding=0 => T_out = T_in - K + 1
    return T_in - K + 1


class ModelNew(torch.nn.Module):
    def __init__(self, BLOCK_C: int = 64):
        super().__init__()
        self.BLOCK_C = BLOCK_C

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        reverse: bool,
        # 4 transforms' weights and biases
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
        """
        Triton-optimized forward that performs the same sequence as 'run' but using Triton kernels:
        - split halves, conv1d (forward+ReLU twice), coupling (add/sub), concatenate, mask multiply.
        """
        N, C, T = x.shape
        assert C == 192, "This implementation expects channels=192."
        C_half = C // 2  # 96
        K = 5

        # Prepare mask (broadcast to [N, C, T] for elementwise multiply)
        # x_mask is [N, 1, T]; we broadcast to [N, C, T] by repeating along C dimension
        # We will use Triton kernel to multiply.
        mask_expanded = x_mask.expand(N, C, T).contiguous()  # [N, C, T]

        # We will use Triton for all computations; no torch.conv1d or torch.relu in host code.
        device = x.device
        dtype = x.dtype  # assume float32

        # For forward, we process 4 transforms in order
        # For reverse, we process in reversed order but the coupling uses subtraction.
        transforms = [
            (transform_0_conv0_weight, transform_0_conv0_bias,
             transform_0_conv1_weight, transform_0_conv1_bias,
             transform_0_conv2_weight, transform_0_conv2_bias),
            (transform_1_conv0_weight, transform_1_conv0_bias,
             transform_1_conv1_weight, transform_1_conv1_bias,
             transform_1_conv2_weight, transform_1_conv2_bias),
            (transform_2_conv0_weight, transform_2_conv0_bias,
             transform_2_conv1_weight, transform_2_conv1_bias,
             transform_2_conv2_weight, transform_2_conv2_bias),
            (transform_3_conv0_weight, transform_3_conv0_bias,
             transform_3_conv1_weight, transform_3_conv1_bias,
             transform_3_conv2_weight, transform_3_conv2_bias),
        ]

        # We need to compute x1 updated each iteration; start with original x for both halves
        # However, original x is [N, C_half, T] split. For the first iteration, we need x0, x1 from x.
        # We'll initialize x0 and x1 tensors as we go. But we need to split current x into x0 and x1.

        # We'll process sequentially:
        for i in range(4):
            # Get current x (we'll overwrite it after each coupling)
            # First, split halves
            x0 = torch.empty((N, C_half, T), device=device, dtype=dtype)
            x1 = torch.empty((N, C_half, T), device=device, dtype=dtype)
            # Triton split kernel (We need to pass pointers; Triton expects torch tensors with .data_ptr(),
            # but Triton can read torch tensors directly. So we launch kernel with these tensors.)
            grid_split = (N, C_half, T)
            split_halves_kernel[grid_split](
                x, x0, x1,
                N, C_half, T,
                x.stride(0), x.stride(1), x.stride(2),
                x0.stride(0), x0.stride(1), x0.stride(2),
                x1.stride(0), x1.stride(1), x1.stride(2),
            )

            # We will compute h = apply_transform(x0) using Triton conv+ReLU steps
            # conv0: in=96, out=192
            C_in0 = C_half
            C_out0 = 192
            K0 = 5
            T_in0 = T
            T_out0 = _compute_T_out(T_in0, K0)
            h0 = torch.empty((N, C_out0, T_out0), device=device, dtype=dtype)

            # Launch conv1d_forward_kernel for conv0
            grid_conv0 = (N, T_out0, triton.cdiv(C_out0, self.BLOCK_C))
            conv1d_forward_kernel[grid_conv0](
                x0,  # input tensor x0 with shape [N, 96, T]
                transforms[i][0],  # conv0 weight [192, 96, 5]
                transforms[i][1],  # conv0 bias [192]
                h0,
                N, C_in0, T_in0, C_out0, T_out0, K0,
                x0.stride(0), x0.stride(1), x0.stride(2),
                transforms[i][0].stride(0), transforms[i][0].stride(1), transforms[i][0].stride(2),
                h0.stride(0), h0.stride(1), h0.stride(2),
                BLOCK_C=self.BLOCK_C,
            )
            # ReLU after conv0
            h0_relu = torch.empty_like(h0)
            grid_relu0 = (N, T_out0, triton.cdiv(C_out0, self.BLOCK_C))
            relu_inplace_kernel[grid_relu0](
                h0, h0_relu,
                N, C_out0, T_out0,
                h0.stride(0), h0.stride(1), h0.stride(2),
                h0_relu.stride(0), h0_relu.stride(1), h0_relu.stride(2),
            )
            # conv1: in=192, out=192
            C_in1 = 192
            C_out1 = 192
            K1 = 5
            T_in1 = T_out0
            T_out1 = _compute_T_out(T_in1, K1)
            h1 = torch.empty((N, C_out1, T_out1), device=device, dtype=dtype)

            grid_conv1 = (N, T_out1, triton.cdiv(C_out1, self.BLOCK_C))
            conv1d_forward_kernel[grid_conv1](
                h0_relu,  # input after conv0 + ReLU
                transforms[i][2],  # conv1 weight [192, 192, 5]
                transforms[i][3],  # conv1 bias [192]
                h1,
                N, C_in1, T_in1, C_out1, T_out1, K1,
                h0_relu.stride(0), h0_relu.stride(1), h0_relu.stride(2),
                transforms[i][2].stride(0), transforms[i][2].stride(1), transforms[i][2].stride(2),
                h1.stride(0), h1.stride(1), h1.stride(2),
                BLOCK_C=self.BLOCK_C,
            )
            # ReLU after conv1
            h1_relu = torch.empty_like(h1)
            grid_relu1 = (N, T_out1, triton.cdiv(C_out1, self.BLOCK_C))
            relu_inplace_kernel[grid_relu1](
                h1, h1_relu,
                N, C_out1, T_out1,
                h1.stride(0), h1.stride(1), h1.stride(2),
                h1_relu.stride(0), h1_relu.stride(1), h1_relu.stride(2),
            )
            # conv2: in=192, out=96
            C_in2 = 192
            C_out2 = C_half
            K2 = 5
            T_in2 = T_out1
            T_out2 = _compute_T_out(T_in2, K2)
            h = torch.empty((N, C_out2, T_out2), device=device, dtype=dtype)

            grid_conv2 = (N, T_out2, triton.cdiv(C_out2, self.BLOCK_C))
            conv1d_forward_kernel[grid_conv2](
                h1_relu,  # input after conv1 + ReLU
                transforms[i][4],  # conv2 weight [96, 192, 5]
                transforms[i][5],  # conv2 bias [96]
                h,
                N, C_in2, T_in2, C_out2, T_out2, K2,
                h1_relu.stride(0), h1_relu.stride(1), h1_relu.stride(2),
                transforms[i][4].stride(0), transforms[i][4].stride(1), transforms[i][4].stride(2),
                h.stride(0), h.stride(1), h.stride(2),
                BLOCK_C=self.BLOCK_C,
            )

            # Apply mask (broadcasted along C)
            h_masked = torch.empty_like(h)
            grid_mask = (N, C_out2, T_out2)
            mask_mul_kernel[grid_mask](
                h, mask_expanded, h_masked,
                N, C_out2, T_out2,
                h.stride(0), h.stride(1), h.stride(2),
                mask_expanded.stride(0), mask_expanded.stride(1), mask_expanded.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
            )

            # Now update x1: if reverse, subtract; else add
            # We need current x1 (the second half of current x) and h_masked
            # We'll launch add_halves_kernel. To do that, we need to have the current x1 from the previous iteration.
            # But we haven't updated x yet. We will maintain x1 buffer across iterations. To keep it simple, we
            # can reconstruct x1 from the original x by splitting, and then update it in each iteration.
            # However, since we don't have a running x1 tensor across iterations, we'll recompute the split each iteration.
            # This is fine; each iteration operates on the original x at that moment, then updates in-place conceptualy via coupling.
            # But Triton kernels expect pointers; we need to have x1 tensor. So we keep x1 as torch.empty initialized per iteration.
            # We'll implement coupling by reading x1 from x (split) and writing updated x1.

            # Note: For the very first iteration, x0 and x1 are derived from x as above. For subsequent iterations, we use
            # the updated x? The original code updates x1 after each transform; to mimic, we need to keep track of x1 across iterations.
            # Since we don't have a running x in the signature, we'll re-split x each iteration and update. This is correct for forward.

            # Update x1 via Triton add/subtract
            # We need to allocate an output tensor for the updated x1
            x1_out = torch.empty_like(x1)
            grid_add = (N, C_half, T)
            if reverse:
                add_halves_kernel[grid_add](
                    x1, h_masked, x1_out,
                    N, C_half, T,
                    x1.stride(0), x1.stride(1), x1.stride(2),
                    h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                    x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                    ADD=False  # subtract
                )
            else:
                add_halves_kernel[grid_add](
                    x1, h_masked, x1_out,
                    N, C_half, T,
                    x1.stride(0), x1.stride(1), x1.stride(2),
                    h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                    x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                    ADD=True  # add
                )
            # Replace x1 with updated x1_out
            x1 = x1_out

            # Concatenate x0 and updated x1 to form new x for next iteration
            new_x = torch.empty((N, C, T), device=device, dtype=dtype)
            grid_cat = (N, C_half, T)
            cat_halves_kernel[grid_cat](
                x0, x1, new_x,
                N, C_half, T,
                x0.stride(0), x0.stride(1), x0.stride(2),
                x1.stride(0), x1.stride(1), x1.stride(2),
                new_x.stride(0), new_x.stride(1), new_x.stride(2),
            )
            # Update x for next iteration
            x = new_x

            # Finally, apply mask to x (x_mask is all ones; this is a no-op here, but we keep it for structure)
            x = x * mask_expanded

        # Return final x
        return x


def run(*args):
    return ModelNew()(*args)
