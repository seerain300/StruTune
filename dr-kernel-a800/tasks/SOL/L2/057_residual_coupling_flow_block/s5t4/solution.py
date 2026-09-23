import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def concat_and_add_triton(
    x0_ptr,        # *const float, [B, C0, T]
    x1_ptr,        # *const float, [B, C1, T]
    h_ptr,         # *const float, [B, C1, T]
    out_ptr,       # *float,       [B, C0+C1, T]
    B: tl.int32, C0: tl.int32, C1: tl.int32, T: tl.int32,
    x0_stride_b: tl.int32, x0_stride_c: tl.int32, x0_stride_t: tl.int32,
    x1_stride_b: tl.int32, x1_stride_c: tl.int32, x1_stride_t: tl.int32,
    h_stride_b: tl.int32, h_stride_c: tl.int32, h_stride_t: tl.int32,
    out_stride_b: tl.int32, out_stride_c: tl.int32, out_stride_t: tl.int32,
    ADD: tl.int32,   # 1 for add, 0 for subtract
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    co = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)        # output channel indices 0..C0+C1-1
    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)         # time indices

    co_mask = co < (C0 + C1)
    t_mask = t < T

    out_base = out_ptr + pid_b * out_stride_b

    # Write x0 to out[:, :C0, :]
    for i in range(0, BLOCK_C):
        c = co[i]
        if (c < C0) & co_mask[i]:
            x0_val = tl.load(
                x0_ptr + pid_b * x0_stride_b + c * x0_stride_c + t * x0_stride_t,
                mask=t_mask,
                other=0.0
            ).to(tl.float32)
            tl.store(
                out_base + c * out_stride_c + t * out_stride_t,
                x0_val,
                mask=t_mask
            )

    # Write x1 (possibly with h) to out[:, C0:, :]
    for i in range(0, BLOCK_C):
        c = co[i]
        if (c >= C0) & co_mask[i]:
            c_rel = c - C0
            x1_val = tl.load(
                x1_ptr + pid_b * x1_stride_b + c_rel * x1_stride_c + t * x1_stride_t,
                mask=t_mask,
                other=0.0
            ).to(tl.float32)
            h_val = tl.load(
                h_ptr + pid_b * h_stride_b + c_rel * h_stride_c + t * h_stride_t,
                mask=t_mask,
                other=0.0
            ).to(tl.float32)
            if ADD:
                new_val = x1_val + h_val
            else:
                new_val = x1_val - h_val
            tl.store(
                out_base + c * out_stride_c + t * out_stride_t,
                new_val,
                mask=t_mask
            )


@triton.jit
def add_mask_to_h_triton(
    h_ptr,         # *const float, [B, Cout, T]
    mask_ptr,      # *const float, [B, 1, T]
    out_h_ptr,     # *float,       [B, Cout, T]
    B: tl.int32, Cout: tl.int32, T: tl.int32,
    h_stride_b: tl.int32, h_stride_c: tl.int32, h_stride_t: tl.int32,
    mask_stride_b: tl.int32, mask_stride_c: tl.int32, mask_stride_t: tl.int32,   # mask has C=1
    out_h_stride_b: tl.int32, out_h_stride_c: tl.int32, out_h_stride_t: tl.int32,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    co = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)  # output channels 0..Cout-1
    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)   # time indices

    co_mask = co < Cout
    t_mask = t < T

    # Load mask for this batch and time; mask has C=1
    mask_val = tl.load(
        mask_ptr + pid_b * mask_stride_b + 0 * mask_stride_c + t * mask_stride_t,
        mask=t_mask,
        other=0.0
    ).to(tl.float32)  # [BLOCK_T]

    # Multiply h by mask (broadcast over channels)
    for i in range(0, BLOCK_C):
        c = co[i]
        if co_mask[i]:
            h_val = tl.load(
                h_ptr + pid_b * h_stride_b + c * h_stride_c + t * h_stride_t,
                mask=t_mask,
                other=0.0
            ).to(tl.float32)
            new_val = h_val * mask_val
            tl.store(
                out_h_ptr + pid_b * out_h_stride_b + c * out_h_stride_c + t * out_h_stride_t,
                new_val,
                mask=t_mask
            )


@triton.jit
def add_h_to_x1_triton(
    x1_ptr,        # *const float, [B, C1, T]
    h_ptr,         # *const float, [B, C1, T]
    out_x1_ptr,    # *float,       [B, C1, T]
    B: tl.int32, C1: tl.int32, T: tl.int32,
    x1_stride_b: tl.int32, x1_stride_c: tl.int32, x1_stride_t: tl.int32,
    h_stride_b: tl.int32, h_stride_c: tl.int32, h_stride_t: tl.int32,
    out_x1_stride_b: tl.int32, out_x1_stride_c: tl.int32, out_x1_stride_t: tl.int32,
    ADD: tl.int32,   # 1 for add, 0 for subtract
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)  # channels 0..C1-1
    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # time indices

    c_mask = c < C1
    t_mask = t < T

    for i in range(0, BLOCK_C):
        ci = c[i]
        if c_mask[i]:
            x1_val = tl.load(
                x1_ptr + pid_b * x1_stride_b + ci * x1_stride_c + t * x1_stride_t,
                mask=t_mask,
                other=0.0
            ).to(tl.float32)
            h_val = tl.load(
                h_ptr + pid_b * h_stride_b + ci * h_stride_c + t * h_stride_t,
                mask=t_mask,
                other=0.0
            ).to(tl.float32)
            if ADD:
                new_val = x1_val + h_val
            else:
                new_val = x1_val - h_val
            tl.store(
                out_x1_ptr + pid_b * out_x1_stride_b + ci * out_x1_stride_c + t * out_x1_stride_t,
                new_val,
                mask=t_mask
            )


@triton.jit
def copy_first_half_kernel(
    out_ptr, x0_ptr,
    B: tl.int32, C0: tl.int32, T: tl.int32,
    out_stride_b: tl.int32, out_stride_c: tl.int32, out_stride_t: tl.int32,
    x0_stride_b: tl.int32, x0_stride_c: tl.int32, x0_stride_t: tl.int32,
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)  # 0..C0-1
    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # 0..T-1

    c_mask = c < C0
    t_mask = t < T

    out_base = out_ptr + pid_b * out_stride_b

    for i in range(0, BLOCK_C):
        ci = c[i]
        if c_mask[i]:
            val = tl.load(
                x0_ptr + pid_b * x0_stride_b + ci * x0_stride_c + t * x0_stride_t,
                mask=t_mask,
                other=0.0
            ).to(tl.float32)
            tl.store(
                out_base + ci * out_stride_c + t * out_stride_t,
                val,
                mask=t_mask
            )


@triton.jit
def copy_second_half_kernel(
    out_ptr, x1_ptr,
    B: tl.int32, C0: tl.int32, C1: tl.int32, T: tl.int32,
    out_stride_b: tl.int32, out_stride_c: tl.int32, out_stride_t: tl.int32,
    x1_stride_b: tl.int32, x1_stride_c: tl.int32, x1_stride_t: tl.int32,
    BLOCK_C: tl.constexpr, BLOCK_T: tl.constexpr,
):
    # This kernel writes out[:, C0:, :] from x1[:, :, :]
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_t = tl.program_id(2)

    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)  # 0..C1-1
    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # 0..T-1

    c_mask = c < C1
    t_mask = t < T

    out_base = out_ptr + pid_b * out_stride_b  # we will store at ci + C0

    for i in range(0, BLOCK_C):
        ci = c[i]
        if c_mask[i]:
            val = tl.load(
                x1_ptr + pid_b * x1_stride_b + ci * x1_stride_c + t * x1_stride_t,
                mask=t_mask,
                other=0.0
            ).to(tl.float32)
            tl.store(
                out_base + (ci + C0) * out_stride_c + t * out_stride_t,
                val,
                mask=t_mask
            )


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
    Triton-optimized forward/reverse with conv computed by PyTorch F.conv1d
    to ensure correctness, and concatenation, mask application, and coupling
    updates handled by Triton kernels.
    """
    assert x.dim() == 3 and x_mask.dim() == 3, "x and x_mask must be [B, C, T]"
    B, C, T = x.shape
    half_channels = C // 2

    # Ensure CUDA and contiguous
    device = x.device
    # We keep convs in PyTorch for correctness
    # Helper: run one transform and update x in-place style via Triton for coupling
    def apply_one_transform(x0, conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b, ADD):
        # F.conv1d expects [N, C_in, L], here x0 is [B, Cin, T]
        # conv0: Cout=C1, Cin=Cin0=half_channels, K=5
        h0 = F.conv1d(x0, conv0_w, conv0_b, padding=conv0_w.shape[2] // 2)
        # ReLU via PyTorch
        h0 = F.relu(h0)
        # conv1: Cout=C2, Cin=C1, K=5
        h1 = F.conv1d(h0, conv1_w, conv1_b, padding=conv1_w.shape[2] // 2)
        h1 = F.relu(h1)
        # conv2: Cout=C3, Cin=C2, K=5
        h2 = F.conv1d(h1, conv2_w, conv2_b, padding=conv2_w.shape[2] // 2)

        # Mask multiply
        h2_masked = torch.empty_like(h2)
        add_mask_to_h_triton[(B, triton.cdiv(half_channels, 64), triton.cdiv(T, 128))](
            h2, x_mask,
            h2_masked,
            B, half_channels, T,
            h2.stride(0), h2.stride(1), h2.stride(2),
            x_mask.stride(0), x_mask.stride(1), x_mask.stride(2),
            h2_masked.stride(0), h2_masked.stride(1), h2_masked.stride(2),
            BLOCK_C=64, BLOCK_T=128
        )

        # Update x1 = x1 + h2_masked or -h2_masked
        x1 = x[:, half_channels:, :]
        x1_out = torch.empty_like(x1)
        add_h_to_x1_triton[(B, triton.cdiv(half_channels, 64), triton.cdiv(T, 128))](
            x1, h2_masked,
            x1_out,
            B, half_channels, T,
            x1.stride(0), x1.stride(1), x1.stride(2),
            h2_masked.stride(0), h2_masked.stride(1), h2_masked.stride(2),
            x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
            ADD=1 if not reverse else 0,
            BLOCK_C=64, BLOCK_T=128
        )

        # Concatenate back: out = [x0, x1_out]
        out = torch.empty((B, C, T), dtype=torch.float32, device=device)
        copy_first_half_kernel[(B, triton.cdiv(half_channels, 64), triton.cdiv(T, 128))](
            out, x0,
            B, half_channels, T,
            out.stride(0), out.stride(1), out.stride(2),
            x0.stride(0), x0.stride(1), x0.stride(2),
            BLOCK_C=64, BLOCK_T=128
        )
        copy_second_half_kernel[(B, triton.cdiv(half_channels, 64), triton.cdiv(T, 128))](
            out, x1_out,
            B, half_channels, T,
            out.stride(0), out.stride(1), out.stride(2),
            x1_out.stride(0), x1_out.stride(1), x1_out.stride(2),
            BLOCK_C=64, BLOCK_T=128
        )
        return out

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
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in transforms:
            x = apply_one_transform(x[:, :half_channels, :], conv0_w, conv0_b,
                                    conv1_w, conv1_b, conv2_w, conv2_b, ADD=1)
    else:
        # Reverse: apply transforms in reverse order
        for conv0_w, conv0_b, conv1_w, conv1_b, conv2_w, conv2_b in reversed(transforms):
            x = apply_one_transform(x[:, :half_channels, :], conv0_w, conv0_b,
                                    conv1_w, conv1_b, conv2_w, conv2_b, ADD=0)

    return x


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args: x, x_mask, reverse, ... weight tensors
        x = args[0]
        x_mask = args[1]
        reverse = args[2] if len(args) > 2 else False

        # Prepare weights: they are provided by get_inputs; we assume they are on the same device as x
        # In the evaluation, they are already on the right device, so we can directly pass them.
        return run(*args)


def run(*args):
    return ModelNew()(*args)
