import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 conv forward (no bias), stride=1, padding=1
# x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), y: (B, C_out, H, W)
# Grid: (B, C_out). Each program handles one (b, oc), iterating over input channels and 3x3 neighborhood.
@triton.jit
def conv3x3_triton(
    x_ptr, w_ptr, y_ptr,
    B, C_in, H, W, C_out,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
):
    b = tl.program_id(0)
    oc = tl.program_id(1)

    # Accumulator for output channel oc
    # We will write the full output plane H*W. Use a simple loop over spatial positions.
    H_out = H
    W_out = W

    # For each output spatial position
    for h_out in range(0, H_out):
        for w_out in range(0, W_out):
            acc = 0.0
            # Loop over input channels and 3x3 neighborhood
            for ic in range(0, C_in):
                for kh in range(3):
                    for kw in range(3):
                        h_in = h_out + (kh - 1)
                        w_in = w_out + (kw - 1)
                        # Guard for padding (always in bounds with padding=1, but keep for safety)
                        if (h_in >= 0) and (h_in < H) and (w_in >= 0) and (w_in < W):
                            x_ptrs = x_ptr \
                                     + b * x_stride_b \
                                     + ic * x_stride_c \
                                     + h_in * x_stride_h \
                                     + w_in * x_stride_w
                            # Load x value as scalar (float32)
                            x_val = tl.load(x_ptrs)
                            w_ptrs = w_ptr \
                                     + ic * w_stride_cin \
                                     + oc * w_stride_cout \
                                     + kh * w_stride_kh \
                                     + kw * w_stride_kw
                            w_val = tl.load(w_ptrs)
                            acc += x_val * w_val

            # Store result
            y_ptrs = y_ptr \
                     + b * y_stride_b \
                     + oc * y_stride_c \
                     + h_out * y_stride_h \
                     + w_out * y_stride_w
            tl.store(y_ptrs, acc)


# Triton kernel: GroupNorm forward + affine (scale, bias) + SiLU
# x: (B, C, H, W), weight/bias: (C,), y: (B, C, H, W)
# Assumes num_groups=32, and C % 32 == 0.
@triton.jit
def group_norm_affine_silu(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W,
    num_groups: tl.constexpr,  # 32
    eps: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,  # reduction tile
):
    b = tl.program_id(0)
    g = tl.program_id(1)
    group_channels = C // num_groups
    group_elements = group_channels * H * W

    # First pass: compute sum and sum of squares over the group
    sum_val = 0.0
    sum_sq = 0.0
    for start in range(0, group_elements, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < group_elements

        ch = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr \
                 + b * x_stride_b \
                 + ch * x_stride_c \
                 + h * x_stride_h \
                 + w * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        sum_val += tl.sum(x_vals, axis=0)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    mean = sum_val / group_elements
    var = sum_sq / group_elements - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize, affine, SiLU
    for start in range(0, group_elements, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < group_elements

        ch = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr \
                 + b * x_stride_b \
                 + ch * x_stride_c \
                 + h * x_stride_h \
                 + w * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        norm_vals = (x_vals - mean) * inv_std

        scale = tl.load(weight_ptr + ch, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + ch, mask=mask, other=0.0)
        z = norm_vals * scale + bias

        s = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * s

        y_ptrs = y_ptr \
                 + b * y_stride_b \
                 + ch * y_stride_c \
                 + h * y_stride_h \
                 + w * y_stride_w
        tl.store(y_ptrs, y_vals, mask=mask)


# Triton kernel: elementwise residual add y = y + x
@triton.jit
def add_residual_kernel(out_ptr, y_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = y + x
    tl.store(out_ptr + offsets, out, mask=mask)


def _run_conv3x3_triton(x, w):
    """
    x: (B, C, H, W), w: (C, C, 3, 3)
    returns y: (B, C, H, W)
    """
    assert x.is_cuda and w.is_cuda
    B, C_in, H, W = x.shape
    C_out = w.shape[0]
    y = torch.empty((B, C_out, H, W), device=x.device, dtype=torch.float32)

    # Launch grid over (B, C_out)
    grid = (B, C_out)
    conv3x3_triton[grid](
        x, w, y,
        B, C_in, H, W, C_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        num_warps=4,
    )
    return y


def _group_norm_affine_silu_triton(x, weight, bias, eps):
    """
    x: (B, C, H, W), weight: (C,), bias: (C,)
    returns y: (B, C, H, W)
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    B, C, H, W = x.shape
    num_groups = 32
    assert C % num_groups == 0, "C must be divisible by num_groups=32"
    y = torch.empty_like(x, dtype=torch.float32, device=x.device)

    grid = (B, num_groups)
    group_norm_affine_silu[grid](
        x, weight, bias, y,
        B, C, H, W,
        num_groups, eps,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK=1024,
        num_warps=4,
    )
    return y


class ModelNew(torch.nn.Module):
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        # Ensure tensors are on CUDA and float32
        assert x.is_cuda and conv1_weight.is_cuda and norm1_weight.is_cuda and norm1_bias.is_cuda \
               and conv2_weight.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda

        # Cast to float32 for numerics
        x = x.to(torch.float32)
        conv1_weight = conv1_weight.to(torch.float32)
        norm1_weight = norm1_weight.to(torch.float32)
        norm1_bias = norm1_bias.to(torch.float32)
        conv2_weight = conv2_weight.to(torch.float32)
        norm2_weight = norm2_weight.to(torch.float32)
        norm2_bias = norm2_bias.to(torch.float32)

        # Stage 1: Conv3x3 -> GroupNorm -> SiLU
        out = _run_conv3x3_triton(x, conv1_weight)  # (B, C, H, W)
        out = _group_norm_affine_silu_triton(out, norm1_weight, norm1_bias, eps)

        # Stage 2: Conv3x3 -> GroupNorm -> SiLU
        out = _run_conv3x3_triton(out, conv2_weight)  # (B, C, H, W)
        out = _group_norm_affine_silu_triton(out, norm2_weight, norm2_bias, eps)

        # Final residual: out = out + x
        N = out.numel()
        y_flat = torch.empty(N, device=out.device, dtype=torch.float32)
        out_flat = out.view(-1)
        x_flat = x.view(-1)
        grid_add = (triton.cdiv(N, 1024),)
        add_residual_kernel[grid_add](y_flat, out_flat, x_flat, N, BLOCK=1024, num_warps=4)
        out = y_flat.view_as(out)

        return out


def run(*args):
    return ModelNew()(*args)
