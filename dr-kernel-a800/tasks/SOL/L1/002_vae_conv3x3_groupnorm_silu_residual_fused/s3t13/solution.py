import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 conv forward (no bias), stride=1, padding=1
# x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), y: (B, C_out, H, W)
# Grid: (B, C_out, H, W). Each program computes exactly one output element y[b, oc, h, w].
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
    h_out = tl.program_id(2)
    w_out = tl.program_id(3)

    # Accumulator for the single output element
    acc = 0.0

    # Loop over input channels
    for ic in range(0, C_in):
        # Loop over 3x3 neighborhood (padding=1, stride=1)
        for kh in range(-1, 2):
            for kw in range(-1, 2):
                h_in = h_out + kh
                w_in = w_out + kw
                # Guard for boundaries (padding=1 maps out-of-bound to zero)
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                if in_bounds:
                    x_ptr_val = x_ptr \
                                + b * x_stride_b \
                                + ic * x_stride_c \
                                + h_in * x_stride_h \
                                + w_in * x_stride_w
                    x_val = tl.load(x_ptr_val)  # scalar
                    w_ptr_val = w_ptr \
                                + ic * w_stride_cin \
                                + oc * w_stride_cout \
                                + kh * w_stride_kh \
                                + kw * w_stride_kw
                    w_val = tl.load(w_ptr_val)  # scalar
                    acc += x_val * w_val

    y_ptr_val = y_ptr \
                + b * y_stride_b \
                + oc * y_stride_c \
                + h_out * y_stride_h \
                + w_out * y_stride_w
    # Store result
    tl.store(y_ptr_val, acc)


# Triton kernel: GroupNorm (per (batch, group)) + affine (scale, bias) + SiLU
# Assumes num_groups=32 and C % 32 == 0. y_out will have same shape as input.
@triton.jit
def group_norm_affine_silu(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W, num_groups: tl.constexpr,  # 32
    eps: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,  # reduction block size (e.g., 2048)
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

    # Second pass: normalize, affine, SiLU, store
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

        # SiLU: x * sigmoid(x)
        s = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * s

        y_ptrs = y_ptr \
                 + b * y_stride_b \
                 + ch * y_stride_c \
                 + h * y_stride_h \
                 + w * y_stride_w
        tl.store(y_ptrs, y_vals, mask=mask)


# Triton kernel: elementwise residual add y = y + x over flattened N elements
@triton.jit
def add_residual_kernel(out_ptr, y_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y_vals + x_vals, mask=mask)


@triton.jit
def silu_kernel(in_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def group_norm_only_kernel(x_ptr, y_ptr, B, C, H, W, num_groups: tl.constexpr, eps: tl.constexpr,
                            x_stride_b, x_stride_c, x_stride_h, x_stride_w,
                            y_stride_b, y_stride_c, y_stride_h, y_stride_w,
                            weight_ptr, bias_ptr, BLOCK: tl.constexpr):
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

    # Second pass: normalize
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

        y_ptrs = y_ptr \
                 + b * y_stride_b \
                 + ch * y_stride_c \
                 + h * y_stride_h \
                 + w * y_stride_w
        tl.store(y_ptrs, norm_vals, mask=mask)


@torch.no_grad()
def run(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    eps: float,
):
    """
    Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
    Args:
        x: Input tensor of shape (B, C, H, W)
        conv1_weight: First conv weights (C, C, 3, 3)
        norm1_weight: First GroupNorm scale (C,)
        norm1_bias: First GroupNorm bias (C,)
        conv2_weight: Second conv weights (C, C, 3, 3)
        norm2_weight: Second GroupNorm scale (C,)
        norm2_bias: Second GroupNorm bias (C,)
        eps: Epsilon for GroupNorm
    Returns:
        Output tensor of shape (B, C, H, W)
    """
    B, C, H, W = x.shape
    assert C % 32 == 0, "num_groups=32 requires C % 32 == 0"
    device = x.device

    # 1) Conv1: Triton 3x3
    out1 = torch.empty((B, C, H, W), device=device, dtype=x.dtype)
    grid_conv = (B, C, H, W)
    conv3x3_triton[grid_conv](
        x, conv1_weight, out1,
        B, C, H, W, C,  # C_in == C, C_out == C
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
        out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        num_warps=4, num_stages=2
    )

    # 2) GroupNorm1 + affine + SiLU (Triton)
    out1_gn = torch.empty_like(out1)
    grid_gn = (B, 32)
    group_norm_affine_silu[grid_gn](
        out1, norm1_weight, norm1_bias, out1_gn,
        B, C, H, W, 32, eps,
        out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        out1_gn.stride(0), out1_gn.stride(1), out1_gn.stride(2), out1_gn.stride(3),
        BLOCK=4096, num_warps=4, num_stages=2
    )

    # 3) SiLU1 (Triton)
    out1_silu = torch.empty_like(out1_gn)
    N = B * C * H * W
    grid_silu = (triton.cdiv(N, 4096),)
    silu_kernel[grid_silu](
        out1_gn, out1_silu, N, 4096, num_warps=4, num_stages=2
    )

    # 4) Conv2: Triton 3x3
    out2 = torch.empty((B, C, H, W), device=device, dtype=x.dtype)
    grid_conv2 = (B, C, H, W)
    conv3x3_triton[grid_conv2](
        out1_silu, conv2_weight, out2,
        B, C, H, W, C,
        out1_silu.stride(0), out1_silu.stride(1), out1_silu.stride(2), out1_silu.stride(3),
        conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
        out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
        num_warps=4, num_stages=2
    )

    # 5) GroupNorm2 + affine + SiLU (Triton)
    out2_gn = torch.empty_like(out2)
    grid_gn2 = (B, 32)
    group_norm_affine_silu[grid_gn2](
        out2, norm2_weight, norm2_bias, out2_gn,
        B, C, H, W, 32, eps,
        out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
        out2_gn.stride(0), out2_gn.stride(1), out2_gn.stride(2), out2_gn.stride(3),
        BLOCK=4096, num_warps=4, num_stages=2
    )

    # 6) SiLU2 (Triton)
    out2_silu = torch.empty_like(out2_gn)
    N2 = B * C * H * W
    grid_silu2 = (triton.cdiv(N2, 4096),)
    silu_kernel[grid_silu2](
        out2_gn, out2_silu, N2, 4096, num_warps=4, num_stages=2
    )

    # 7) Residual add (Triton): out = out2_silu + x
    out = torch.empty_like(out2_silu)
    grid_add = (triton.cdiv(N2, 4096),)
    add_residual_kernel[grid_add](
        out, out2_silu, x, N2, 4096
    )

    return out


class ModelNew(torch.nn.Module):
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        return run(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)


def run(*args):
    return ModelNew()(*args)
