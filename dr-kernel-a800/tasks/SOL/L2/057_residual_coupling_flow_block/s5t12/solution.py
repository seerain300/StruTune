import math
import torch
import triton
import triton.language as tl


@triton.jit
def conv1d_triton_kernel(
    x_ptr,         # *f32, [B, Cin, T]
    w_ptr,         # *f32, [Cout, Cin*K]
    b_ptr,         # *f32, [Cout]
    y_ptr,         # *f32, [B, Cout, T]
    B, Cin, T, Cout, K,
    PAD,
    x_stride0, x_stride1, x_stride2,
    w_stride0, w_stride1,
    y_stride0, y_stride1, y_stride2,
    BLOCK_CO: tl.constexpr, BLOCK_T: tl.constexpr
):
    # Grid: (B, ceil(Cout/BLOCK_CO), ceil(T/BLOCK_T))
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_pos = tl.program_id(2)

    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    pos_offsets = pid_pos * BLOCK_T + tl.arange(0, BLOCK_T)

    co_mask = co_offsets < Cout
    pos_mask = pos_offsets < T

    # Accumulator for output channels
    acc = tl.zeros((BLOCK_CO, BLOCK_T), dtype=tl.float32)

    # Flatten weights per output channel: w[co, k*Cin + c]
    # Loop over input channels and kernel positions
    for c in range(0, Cin):
        for k in range(0, K):
            in_pos = pos_offsets - (k - PAD)
            in_pos = tl.maximum(in_pos, 0)
            in_pos = tl.minimum(in_pos, T - 1)

            # Load x[b, c, in_pos]
            x_base = pid_b * x_stride0 + c * x_stride1
            x_ptrs = x_ptr + x_base + in_pos * x_stride2
            x_vals = tl.load(x_ptrs, mask=pos_mask, other=0.0)  # [BLOCK_T]
            x_vals = x_vals[:, None]  # broadcast over co_offsets

            # Load weights for this (co, c, k): w[co, k*Cin + c]
            w_base = co_offsets * w_stride0
            w_ptrs = w_ptr + w_base + (k * Cin + c) * w_stride1
            w_vals = tl.load(w_ptrs, mask=co_mask, other=0.0)  # [BLOCK_CO]

            acc += w_vals[:, None] * x_vals  # [BLOCK_CO, BLOCK_T]

    # Add bias and apply ReLU
    b_vals = tl.load(b_ptr + co_offsets, mask=co_mask, other=0.0)  # [BLOCK_CO]
    acc += b_vals[:, None]
    acc = tl.maximum(acc, 0.0)  # ReLU

    # Store output y[b, co, pos]
    y_base = pid_b * y_stride0
    y_ptrs = y_ptr + y_base + co_offsets[:, None] * y_stride1 + pos_offsets[None, :] * y_stride2
    tl.store(y_ptrs, acc, mask=co_mask[:, None] & pos_mask[None, :])


@triton.jit
def apply_mask_to_h_triton(
    h_ptr,         # *f32, [B, C_half, T]
    mask_ptr,      # *f32, [B, 1, T]
    h_out_ptr,     # *f32, [B, C_half, T]
    B, C_half, T,
    h_stride0, h_stride1, h_stride2,
    mask_stride0, mask_stride1, mask_stride2,
    h_out_stride0, h_out_stride1, h_out_stride2,
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)

    c_mask = c_offsets < C_half
    t_mask = t_offsets < T

    h_ptrs = h_ptr + pid_b * h_stride0 + c_offsets[:, None] * h_stride1 + t_offsets[None, :] * h_stride2
    h_vals = tl.load(h_ptrs, mask=c_mask[:, None] & t_mask[None, :], other=0.0)

    # mask is [B, 1, T]; load scalar per time
    mask_ptrs = mask_ptr + pid_b * mask_stride0 + 0 * mask_stride1 + t_offsets * mask_stride2
    mask_vals = tl.load(mask_ptrs, mask=t_mask, other=1.0)  # [BLOCK_T], assume mask is 1s (default)

    h_vals = h_vals * mask_vals[None, :]
    h_out_ptrs = h_out_ptr + pid_b * h_out_stride0 + c_offsets[:, None] * h_out_stride1 + t_offsets[None, :] * h_out_stride2
    tl.store(h_out_ptrs, h_vals, mask=c_mask[:, None] & t_mask[None, :])


@triton.jit
def add_h_to_x1_triton(
    x1_ptr,        # *f32, [B, C_half, T]
    h_ptr,         # *f32, [B, C_half, T]
    x1_out_ptr,    # *f32, [B, C_half, T]
    B, C_half, T,
    x1_stride0, x1_stride1, x1_stride2,
    h_stride0, h_stride1, h_stride2,
    x1_out_stride0, x1_out_stride1, x1_out_stride2,
    ADD: tl.constexpr,              # bool: True for add, False for subtract
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)

    c_mask = c_offsets < C_half
    t_mask = t_offsets < T

    x1_ptrs = x1_ptr + pid_b * x1_stride0 + c_offsets[:, None] * x1_stride1 + t_offsets[None, :] * x1_stride2
    h_ptrs = h_ptr + pid_b * h_stride0 + c_offsets[:, None] * h_stride1 + t_offsets[None, :] * h_stride2

    x1_vals = tl.load(x1_ptrs, mask=c_mask[:, None] & t_mask[None, :], other=0.0)
    h_vals = tl.load(h_ptrs, mask=c_mask[:, None] & t_mask[None, :], other=0.0)

    if ADD:
        x1_new = x1_vals + h_vals
    else:
        x1_new = x1_vals - h_vals

    x1_out_ptrs = x1_out_ptr + pid_b * x1_out_stride0 + c_offsets[:, None] * x1_out_stride1 + t_offsets[None, :] * x1_out_stride2
    tl.store(x1_out_ptrs, x1_new, mask=c_mask[:, None] & t_mask[None, :])


@triton.jit
def concat_add_v1_triton(
    x0_ptr,        # *f32, [B, C_half, T]
    x1_plus_ptr,   # *f32, [B, C_half, T]
    out_ptr,       # *f32, [B, C, T]
    B, C_half, T,
    x0_stride0, x0_stride1, x0_stride2,
    x1_plus_stride0, x1_plus_stride1, x1_plus_stride2,
    out_stride0, out_stride1, out_stride2,
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr
):
    # Grid: (B, ceil(C_half/BLOCK_C), ceil(T/BLOCK_T))
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)

    c_mask = c_offsets < C_half
    t_mask = t_offsets < T

    # First half channels: write x0 into out[:, :C_half, :]
    x0_ptrs = x0_ptr + pid_b * x0_stride0 + c_offsets[:, None] * x0_stride1 + t_offsets[None, :] * x0_stride2
    x0_vals = tl.load(x0_ptrs, mask=c_mask[:, None] & t_mask[None, :], other=0.0)

    out_base = pid_b * out_stride0
    out_ptrs = out_ptr + out_base + c_offsets[:, None] * out_stride1 + t_offsets[None, :] * out_stride2
    tl.store(out_ptrs, x0_vals, mask=c_mask[:, None] & t_mask[None, :])


@triton.jit
def concat_add_v2_triton(
    x0_ptr,        # *f32, [B, C_half, T]
    x1_minus_ptr,  # *f32, [B, C_half, T]
    out_ptr,       # *f32, [B, C, T]
    B, C_half, T,
    x0_stride0, x0_stride1, x0_stride2,
    x1_minus_stride0, x1_minus_stride1, x1_minus_stride2,
    out_stride0, out_stride1, out_stride2,
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)

    c_mask = c_offsets < C_half
    t_mask = t_offsets < T

    # First half channels: write x0 into out[:, :C_half, :]
    x0_ptrs = x0_ptr + pid_b * x0_stride0 + c_offsets[:, None] * x0_stride1 + t_offsets[None, :] * x0_stride2
    x0_vals = tl.load(x0_ptrs, mask=c_mask[:, None] & t_mask[None, :], other=0.0)

    out_base = pid_b * out_stride0
    out_ptrs = out_ptr + out_base + c_offsets[:, None] * out_stride1 + t_offsets[None, :] * out_stride2
    tl.store(out_ptrs, x0_vals, mask=c_mask[:, None] & t_mask[None, :])


@triton.jit
def copy_x0_to_out_triton(
    x0_ptr,        # *f32, [B, C_half, T]
    out_ptr,       # *f32, [B, C, T]
    B, C_half, T,
    x0_stride0, x0_stride1, x0_stride2,
    out_stride0, out_stride1, out_stride2,
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr
):
    # Grid: (B, ceil(C_half/BLOCK_C), ceil(T/BLOCK_T))
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)

    c_mask = c_offsets < C_half
    t_mask = t_offsets < T

    x0_ptrs = x0_ptr + pid_b * x0_stride0 + c_offsets[:, None] * x0_stride1 + t_offsets[None, :] * x0_stride2
    x0_vals = tl.load(x0_ptrs, mask=c_mask[:, None] & t_mask[None, :], other=0.0)

    out_base = pid_b * out_stride0
    out_ptrs = out_ptr + out_base + c_offsets[:, None] * out_stride1 + t_offsets[None, :] * out_stride2
    tl.store(out_ptrs, x0_vals, mask=c_mask[:, None] & t_mask[None, :])


@triton.jit
def copy_x1_to_out_triton(
    x1_ptr,        # *f32, [B, C_half, T]
    out_ptr,       # *f32, [B, C, T]
    B, C_half, T,
    x1_stride0, x1_stride1, x1_stride2,
    out_stride0, out_stride1, out_stride2,
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr
):
    # Grid: (B, ceil(C_half/BLOCK_C), ceil(T/BLOCK_T))
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)

    c_mask = c_offsets < C_half
    t_mask = t_offsets < T

    x1_ptrs = x1_ptr + pid_b * x1_stride0 + c_offsets[:, None] * x1_stride1 + t_offsets[None, :] * x1_stride2
    x1_vals = tl.load(x1_ptrs, mask=c_mask[:, None] & t_mask[None, :], other=0.0)

    out_base = pid_b * out_stride0
    # write to second half channels: start at C_half
    out_ptrs = out_ptr + out_base + (c_offsets + C_half)[:, None] * out_stride1 + t_offsets[None, :] * out_stride2
    tl.store(out_ptrs, x1_vals, mask=c_mask[:, None] & t_mask[None, :])


def run(
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
):
    """
    Triton-Only Residual Coupling Flow.
    - forward: x1 = x1 + transform(x0)
    - reverse: x1 = x1 - transform(x0) in original order
    """
    assert x.is_cuda, "Inputs must be on CUDA device for Triton kernels."
    B, C, T = x.shape
    half_c = C // 2
    device = x.device

    # Helper to apply a single transform: conv0 -> ReLU -> conv1 -> ReLU -> conv2
    # Split x into x0 and x1
    x0 = x[:, :half_c, :].contiguous()
    x1 = x[:, half_c:, :].contiguous()

    # Prepare masks
    x_mask = x_mask.to(x.dtype).to(device).contiguous()  # [B, 1, T]

    def apply_single_transform(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b, ADD=True):
        """
        Compute: conv0(x0) -> ReLU -> conv1(...) -> ReLU -> conv2(...)
        Returns h of shape [B, half_c, T]
        """
        # conv0: [B, C_half, T] -> [B, hidden, T] (hidden=192)
        y0 = torch.empty((B, conv0_w.shape[0], T), dtype=torch.float32, device=device)
        conv1d_triton_kernel[(B, triton.cdiv(conv0_w.shape[0], 64), triton.cdiv(T, 128))](
            x0, conv0_w, conv0_b, y0,
            B, x0.shape[1], T, conv0_w.shape[0], conv0_w.shape[1], 2,
            x0.stride(0), x0.stride(1), x0.stride(2),
            conv0_w.stride(0), conv0_w.stride(1),
            y0.stride(0), y0.stride(1), y0.stride(2),
            BLOCK_CO=64, BLOCK_T=128
        )
        y0 = tl.maximum(y0, 0.0)

        # conv1: [B, hidden, T] -> [B, hidden, T]
        y1 = torch.empty((B, conv1_w.shape[0], T), dtype=torch.float32, device=device)
        conv1d_triton_kernel[(B, triton.cdiv(conv1_w.shape[0], 64), triton.cdiv(T, 128))](
            y0, conv1_w, conv1_b, y1,
            B, y0.shape[1], T, conv1_w.shape[0], conv1_w.shape[1], 2,
            y0.stride(0), y0.stride(1), y0.stride(2),
            conv1_w.stride(0), conv1_w.stride(1),
            y1.stride(0), y1.stride(1), y1.stride(2),
            BLOCK_CO=64, BLOCK_T=128
        )
        y1 = tl.maximum(y1, 0.0)

        # conv2: [B, hidden, T] -> [B, half_c, T]
        h = torch.empty((B, conv2_w.shape[0], T), dtype=torch.float32, device=device)
        conv1d_triton_kernel[(B, triton.cdiv(conv2_w.shape[0], 64), triton.cdiv(T, 128))](
            y1, conv2_w, conv2_b, h,
            B, y1.shape[1], T, conv2_w.shape[0], conv2_w.shape[1], 2,
            y1.stride(0), y1.stride(1), y1.stride(2),
            conv2_w.stride(0), conv2_w.stride(1),
            h.stride(0), h.stride(1), h.stride(2),
            BLOCK_CO=64, BLOCK_T=128
        )
        return h

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

    # Forward or reverse
    if not reverse:
        # Initialize out as x
        out = x.clone()
        # Process transforms sequentially
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            h = apply_single_transform(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b, ADD=True)
            # Mask h
            h_masked = torch.empty_like(h)
            apply_mask_to_h_triton[(B, triton.cdiv(half_c, 64), triton.cdiv(T, 128))](
                h, x_mask, h_masked,
                B, half_c, T,
                h.stride(0), h.stride(1), h.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                BLOCK_C=64, BLOCK_T=128
            )
            # Update x1
            x1_out = torch.empty_like(x1)
            add_h_to_x1_triton[(B, triton.cdiv(half_c, 64), triton.cdiv(T, 128))](
                x1, h_masked,
                x1_out,
                B, half_c, T,
                x1.stride(0), x1.stride(1), x1.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                ADD=True,
                BLOCK_C=64, BLOCK_T=128
            )
            # Concatenate
            out = torch.empty((B, C, T), dtype=torch.float32, device=device)
            # First half: x0
            copy_x0_to_out_triton[(B, triton.cdiv(half_c, 64), triton.cdiv(T, 128))](
                x0,
                out,
                B, half_c, T,
                x0.stride(0), x0.stride(1), x0.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_C=64, BLOCK_T=128
            )
            # Second half: x1_out
            copy_x1_to_out_triton[(B, triton.cdiv(half_c, 64), triton.cdiv(T, 128))](
                x1_out,
                out,
                B, half_c, T,
                x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_C=64, BLOCK_T=128
            )
            # Update x0 and x1 for next loop
            x0 = x0  # unchanged
            x1 = x1_out
            # Finally, apply x_mask to out (broadcast along channels)
            out_masked = torch.empty_like(out)
            # This mask is [B, 1, T]; broadcast along channels
            out_masked_ptrs = out_masked  # Triton will handle broadcast in the kernel below
            # We'll launch a simple elementwise Triton multiply over C,T:
            out_masked = out * x_mask
        return out
    else:
        # Reverse mode: subtract in original order
        out = x.clone()
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            h = apply_single_transform(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b, ADD=True)
            h_masked = torch.empty_like(h)
            apply_mask_to_h_triton[(B, triton.cdiv(half_c, 64), triton.cdiv(T, 128))](
                h, x_mask, h_masked,
                B, half_c, T,
                h.stride(0), h.stride(1), h.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                BLOCK_C=64, BLOCK_T=128
            )
            x1_out = torch.empty_like(x1)
            add_h_to_x1_triton[(B, triton.cdiv(half_c, 64), triton.cdiv(T, 128))](
                x1, h_masked,
                x1_out,
                B, half_c, T,
                x1.stride(0), x1.stride(1), x1.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                ADD=False,
                BLOCK_C=64, BLOCK_T=128
            )
            # Concatenate
            out = torch.empty((B, C, T), dtype=torch.float32, device=device)
            copy_x0_to_out_triton[(B, triton.cdiv(half_c, 64), triton.cdiv(T, 128))](
                x0,
                out,
                B, half_c, T,
                x0.stride(0), x0.stride(1), x0.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_C=64, BLOCK_T=128
            )
            copy_x1_to_out_triton[(B, triton.cdiv(half_c, 64), triton.cdiv(T, 128))](
                x1_out,
                out,
                B, half_c, T,
                x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_C=64, BLOCK_T=128
            )
            # Update x0 and x1 for next iteration (stay on reversed transform order)
            x0 = x0  # unchanged
            x1 = x1_out
            # Apply x_mask to out (broadcast along channels)
            out = out * x_mask
        return out

# Entry point for the evaluator
class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


def run(*args):
    return ModelNew()(*args)
