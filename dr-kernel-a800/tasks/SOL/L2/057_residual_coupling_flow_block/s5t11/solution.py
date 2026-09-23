import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv1d_triton(x_ptr, w_ptr, b_ptr, out_ptr,
                   B, Cin, Cout, T, K,
                   x_stride0, x_stride1, x_stride2,
                   w_stride0, w_stride1, w_stride2,
                   out_stride0, out_stride1, out_stride2,
                   BLOCK_CO: tl.constexpr, BLOCK_POS: tl.constexpr):
    # Grid: (B, ceil(Cout/BLOCK_CO), ceil(T/BLOCK_POS))
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_pos = tl.program_id(2)

    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    pos_offsets = pid_pos * BLOCK_POS + tl.arange(0, BLOCK_POS)

    co_mask = co_offsets < Cout
    pos_mask = pos_offsets < T

    # Initialize accumulator
    acc = tl.zeros((BLOCK_CO, BLOCK_POS), dtype=tl.float32)

    # Padding for K=5 is 2
    pad = (K - 1) // 2

    # Loop over input channels and kernel positions
    for ci in range(0, Cin):
        for k in range(0, K):
            in_pos = pos_offsets - k - pad  # vector
            in_pos_valid = (in_pos >= 0) & (in_pos < T)

            # Build pointers for x[b, ci, in_pos]
            x_ptrs = x_ptr + pid_b * x_stride0 + ci * x_stride1 + in_pos * x_stride2
            x_vals = tl.load(x_ptrs, mask=in_pos_valid, other=0.0)  # shape [BLOCK_POS]

            # Build pointers for w[co, ci, k]
            w_ptrs = w_ptr + co_offsets * w_stride0 + ci * w_stride1 + k * w_stride2  # shape [BLOCK_CO]
            w_vals = tl.load(w_ptrs, mask=co_mask, other=0.0)

            # Outer product accumulate: [BLOCK_CO, 1] * [1, BLOCK_POS]
            acc += w_vals[:, None] * x_vals[None, :]

    # Add bias
    b_ptrs = b_ptr + co_offsets
    bias = tl.load(b_ptrs, mask=co_mask, other=0.0)  # [BLOCK_CO]
    acc += bias[:, None]

    # Apply ReLU
    acc = tl.maximum(acc, 0.0)

    # Store to out[b, co, pos]
    out_ptrs = out_ptr + pid_b * out_stride0 + co_offsets[:, None] * out_stride1 + pos_offsets[None, :] * out_stride2
    tl.store(out_ptrs, acc, mask=co_mask[:, None] & pos_mask[None, :])


@triton.jit
def apply_mask_to_h_triton(h_ptr, x_mask_ptr, h_out_ptr,
                            B, Cout, T,
                            h_stride0, h_stride1, h_stride2,
                            mask_stride0, mask_stride1, mask_stride2,
                            h_out_stride0, h_out_stride1, h_out_stride2,
                            BLOCK_CO: tl.constexpr, BLOCK_POS: tl.constexpr):
    # Grid: (B, ceil(Cout/BLOCK_CO), ceil(T/BLOCK_POS))
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_pos = tl.program_id(2)

    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    pos_offsets = pid_pos * BLOCK_POS + tl.arange(0, BLOCK_POS)

    co_mask = co_offsets < Cout
    pos_mask = pos_offsets < T

    h_in_ptrs = h_ptr + pid_b * h_stride0 + co_offsets[:, None] * h_stride1 + pos_offsets[None, :] * h_stride2
    h_vals = tl.load(h_in_ptrs, mask=co_mask[:, None] & pos_mask[None, :], other=0.0)

    # x_mask has shape [B, 1, T] -> use stride1 = 1
    mask_ptrs = x_mask_ptr + pid_b * mask_stride0 + 0 * mask_stride1 + pos_offsets[None, :] * mask_stride2
    mask_vals = tl.load(mask_ptrs, mask=pos_mask[None, :], other=1.0)  # [1, BLOCK_POS]

    h_out_vals = h_vals * mask_vals  # broadcast along channels

    h_out_ptrs = h_out_ptr + pid_b * h_out_stride0 + co_offsets[:, None] * h_out_stride1 + pos_offsets[None, :] * h_out_stride2
    tl.store(h_out_ptrs, h_out_vals, mask=co_mask[:, None] & pos_mask[None, :])


@triton.jit
def add_h_to_x1_triton(x1_ptr, h_masked_ptr, x1_out_ptr,
                        B, Cout, T,
                        x1_stride0, x1_stride1, x1_stride2,
                        h_stride0, h_stride1, h_stride2,
                        x1_out_stride0, x1_out_stride1, x1_out_stride2,
                        ADD: tl.constexpr, BLOCK_CO: tl.constexpr, BLOCK_POS: tl.constexpr):
    # Grid: (B, ceil(Cout/BLOCK_CO), ceil(T/BLOCK_POS))
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_pos = tl.program_id(2)

    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    pos_offsets = pid_pos * BLOCK_POS + tl.arange(0, BLOCK_POS)

    co_mask = co_offsets < Cout
    pos_mask = pos_offsets < T

    x1_in_ptrs = x1_ptr + pid_b * x1_stride0 + co_offsets[:, None] * x1_stride1 + pos_offsets[None, :] * x1_stride2
    x1_vals = tl.load(x1_in_ptrs, mask=co_mask[:, None] & pos_mask[None, :], other=0.0)

    h_ptrs = h_masked_ptr + pid_b * h_stride0 + co_offsets[:, None] * h_stride1 + pos_offsets[None, :] * h_stride2
    h_vals = tl.load(h_ptrs, mask=co_mask[:, None] & pos_mask[None, :], other=0.0)

    if ADD:
        x1_out_vals = x1_vals + h_vals
    else:
        x1_out_vals = x1_vals - h_vals

    x1_out_ptrs = x1_out_ptr + pid_b * x1_out_stride0 + co_offsets[:, None] * x1_out_stride1 + pos_offsets[None, :] * x1_out_stride2
    tl.store(x1_out_ptrs, x1_out_vals, mask=co_mask[:, None] & pos_mask[None, :])


@triton.jit
def concat_add_v1_triton(x0_ptr, x1_ptr, out_ptr,
                          B, half_c, T,
                          x0_stride0, x0_stride1, x0_stride2,
                          x1_stride0, x1_stride1, x1_stride2,
                          out_stride0, out_stride1, out_stride2,
                          BLOCK_CO: tl.constexpr, BLOCK_POS: tl.constexpr):
    # Grid: (B, ceil(half_c/BLOCK_CO), ceil(T/BLOCK_POS))
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_pos = tl.program_id(2)

    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    pos_offsets = pid_pos * BLOCK_POS + tl.arange(0, BLOCK_POS)

    co_mask = co_offsets < half_c
    pos_mask = pos_offsets < T

    # Copy x0 into out for first half channels
    base0 = pid_b * x0_stride0
    x0_ptrs = x0_ptr + base0 + co_offsets[:, None] * x0_stride1 + pos_offsets[None, :] * x0_stride2
    x0_vals = tl.load(x0_ptrs, mask=co_mask[:, None] & pos_mask[None, :], other=0.0)

    out_base = pid_b * out_stride0
    out_ptrs = out_ptr + out_base + co_offsets[:, None] * out_stride1 + pos_offsets[None, :] * out_stride2
    tl.store(out_ptrs, x0_vals, mask=co_mask[:, None] & pos_mask[None, :])


@triton.jit
def concat_add_v2_triton(x0_ptr, x1_ptr, out_ptr,
                          B, half_c, T,
                          x0_stride0, x0_stride1, x0_stride2,
                          x1_stride0, x1_stride1, x1_stride2,
                          out_stride0, out_stride1, out_stride2,
                          BLOCK_CO: tl.constexpr, BLOCK_POS: tl.constexpr):
    # Grid: (B, ceil(half_c/BLOCK_CO), ceil(T/BLOCK_POS))
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_pos = tl.program_id(2)

    co_offsets = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    pos_offsets = pid_pos * BLOCK_POS + tl.arange(0, BLOCK_POS)

    co_mask = co_offsets < half_c
    pos_mask = pos_offsets < T

    # Copy x1 into out for second half channels
    base0 = pid_b * x0_stride0  # placeholder
    # We need to copy from x1_ptr
    x1_ptrs = x1_ptr + pid_b * x1_stride0 + co_offsets[:, None] * x1_stride1 + pos_offsets[None, :] * x1_stride2
    x1_vals = tl.load(x1_ptrs, mask=co_mask[:, None] & pos_mask[None, :], other=0.0)

    out_base = pid_b * out_stride0
    out_ptrs = out_ptr + out_base + (co_offsets + half_c)[:, None] * out_stride1 + pos_offsets[None, :] * out_stride2
    tl.store(out_ptrs, x1_vals, mask=co_mask[:, None] & pos_mask[None, :])


def conv1d_relu_triton(x, w, b, T_out, grid):
    # x: [B, Cin, T], w: [Cout, Cin, K], b: [Cout]
    B, Cin, T = x.shape
    Cout = w.shape[0]
    K = w.shape[2]
    pad = (K - 1) // 2
    assert T_out == T - K + 1 + 2 * pad  # T_out should equal T for K=5, pad=2
    y = torch.empty((B, Cout, T_out), dtype=torch.float32, device=x.device)
    x_ = x.contiguous()
    w_ = w.contiguous()
    b_ = b.contiguous()
    conv1d_triton[grid](
        x_, w_, b_, y,
        B, Cin, Cout, T, K,
        x_.stride(0), x_.stride(1), x_.stride(2),
        w_.stride(0), w_.stride(1), w_.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        BLOCK_CO=64, BLOCK_POS=128
    )
    return y


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
    Residual coupling flow block.
    Forward: x1 = x1 + transform(x0) for each layer
    Reverse: x1 = x1 - transform(x0) for each layer (in reverse order)
    """
    B, C, T = x.shape
    half_c = C // 2

    # Ensure tensors are on CUDA
    assert x.is_cuda, "x must be on CUDA"
    # Prepare tensors contiguous
    x0 = x[:, :half_c, :].contiguous()
    x1 = x[:, half_c:, :].contiguous()

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

    if not reverse:
        # Forward pass: apply transformations sequentially
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            # Conv0
            h0 = conv1d_relu_triton(x0, conv0_w, conv0_b, T, (B, triton.cdiv(half_c, 64), triton.cdiv(T, 128)))
            # Conv1 + ReLU
            h1 = conv1d_relu_triton(h0, conv1_w, conv1_b, T, (B, triton.cdiv(half_c, 64), triton.cdiv(T, 128)))
            # Conv2
            h2 = conv1d_relu_triton(h1, conv2_w, conv2_b, T, (B, triton.cdiv(half_c, 64), triton.cdiv(T, 128)))

            # Apply time mask: broadcast [B, 1, T] along channels
            h_masked = torch.empty_like(h2)
            apply_mask_to_h_triton[(B, triton.cdiv(half_c, 64), triton.cdiv(T, 128))](
                h2, x_mask, h_masked,
                B, half_c, T,
                h2.stride(0), h2.stride(1), h2.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                BLOCK_CO=64, BLOCK_POS=128
            )

            # Update x1: x1 = x1 + h_masked
            x1_out = torch.empty_like(x1)
            add_h_to_x1_triton[(B, triton.cdiv(half_c, 64), triton.cdiv(T, 128))](
                x1, h_masked,
                x1_out,
                B, half_c, T,
                x1.stride(0), x1.stride(1), x1.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                ADD=True,
                BLOCK_CO=64, BLOCK_POS=128
            )

            # Concatenate back: out = [x0, x1_out]
            out = torch.empty((B, C, T), dtype=torch.float32, device=x.device)
            concat_add_v1_triton[(B, triton.cdiv(half_c, 64), triton.cdiv(T, 128))](
                x0, x1_out,
                out,
                B, half_c, T,
                x0.stride(0), x0.stride(1), x0.stride(2),
                x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_CO=64, BLOCK_POS=128
            )
            x = out
    else:
        # Reverse pass: apply transformations in reverse order
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
            # Conv0
            h0 = conv1d_relu_triton(x0, conv0_w, conv0_b, T, (B, triton.cdiv(half_c, 64), triton.cdiv(T, 128)))
            # Conv1 + ReLU
            h1 = conv1d_relu_triton(h0, conv1_w, conv1_b, T, (B, triton.cdiv(half_c, 64), triton.cdiv(T, 128)))
            # Conv2
            h2 = conv1d_relu_triton(h1, conv2_w, conv2_b, T, (B, triton.cdiv(half_c, 64), triton.cdiv(T, 128)))

            # Apply time mask: broadcast [B, 1, T] along channels
            h_masked = torch.empty_like(h2)
            apply_mask_to_h_triton[(B, triton.cdiv(half_c, 64), triton.cdiv(T, 128))](
                h2, x_mask, h_masked,
                B, half_c, T,
                h2.stride(0), h2.stride(1), h2.stride(2),
                x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                BLOCK_CO=64, BLOCK_POS=128
            )

            # Update x1: x1 = x1 - h_masked
            x1_out = torch.empty_like(x1)
            add_h_to_x1_triton[(B, triton.cdiv(half_c, 64), triton.cdiv(T, 128))](
                x1, h_masked,
                x1_out,
                B, half_c, T,
                x1.stride(0), x1.stride(1), x1.stride(2),
                h_masked.stride(0), h_masked.stride(1), h_masked.stride(2),
                x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                ADD=False,
                BLOCK_CO=64, BLOCK_POS=128
            )

            # Concatenate back: out = [x0, x1_out]
            out = torch.empty((B, C, T), dtype=torch.float32, device=x.device)
            concat_add_v2_triton[(B, triton.cdiv(half_c, 64), triton.cdiv(T, 128))](
                x0, x1_out,
                out,
                B, half_c, T,
                x0.stride(0), x0.stride(1), x0.stride(2),
                x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_CO=64, BLOCK_POS=128
            )
            x = out

    return x


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect: x, x_mask, reverse, then weights
        x = args[0].to(device="cuda")
        x_mask = args[1].to(device="cuda")
        reverse = bool(args[2] if len(args) > 2 else False)

        # Run with Triton-only computation
        return run(*args)


def run(*args):
    return ModelNew()(*args)
