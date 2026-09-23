import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 conv forward (no bias), stride=1, padding=1
# x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), y: (B, C_out, H, W)
# Grid: (B, C_out, H*W). Each program computes one output element y[b, oc, h, w].
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
    sp = tl.program_id(2)  # linear index over H*W
    h = sp // W
    w = sp % W

    acc = 0.0

    # loop over input channels
    for ic in range(0, C_in):
        # loop over 3x3 neighborhood
        for kh in range(0, 3):
            ih = h + kh - 1
            for kw in range(0, 3):
                iw = w + kw - 1
                # load x[b, ic, ih, iw]
                x_ptr_el = x_ptr + b * x_stride_b + ic * x_stride_c + ih * x_stride_h + iw * x_stride_w
                x_val = tl.load(x_ptr_el)  # assumes in-bounds due to padding=1
                # load w[ic, oc, kh, kw]
                w_ptr_el = w_ptr + ic * w_stride_cin + oc * w_stride_cout + kh * w_stride_kh + kw * w_stride_kw
                w_val = tl.load(w_ptr_el)
                acc += x_val * w_val

    # store result
    y_ptr_el = y_ptr + b * y_stride_b + oc * y_stride_c + h * y_stride_h + w * y_stride_w
    tl.store(y_ptr_el, acc)


# Triton kernel: GroupNorm + affine (scale, bias) + SiLU
# Assumes num_groups=32 and C % 32 == 0.
# x: (B, C, H, W), weight: (C,), bias: (C,), y: (B, C, H, W)
@triton.jit
def group_norm_affine_silu(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W, num_groups: tl.constexpr, eps: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,  # reduction tile
):
    b = tl.program_id(0)
    g = tl.program_id(1)
    group_channels = C // num_groups
    group_elements = group_channels * H * W

    # first pass: compute sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    for start in range(0, group_elements, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < group_elements

        ch = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr + b * x_stride_b + ch * x_stride_c + h * x_stride_h + w * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        sum_val += tl.sum(x_vals, axis=0)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    mean = sum_val / group_elements
    var = sum_sq / group_elements - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # second pass: normalize, affine, SiLU
    for start in range(0, group_elements, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < group_elements

        ch = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr + b * x_stride_b + ch * x_stride_c + h * x_stride_h + w * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        norm_vals = (x_vals - mean) * inv_std

        scale = tl.load(weight_ptr + ch, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + ch, mask=mask, other=0.0)
        z = norm_vals * scale + bias

        # SiLU: z * sigmoid(z) = z / (1 + exp(-z))
        s = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * s

        y_ptrs = y_ptr + b * y_stride_b + ch * y_stride_c + h * y_stride_h + w * y_stride_w
        tl.store(y_ptrs, y_vals, mask=mask)


# Triton kernel: elementwise residual add y = y + x
@triton.jit
def add_residual_kernel(out_ptr, y_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y_vals + x_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        """
        Triton-only forward that matches the original:
        out = SiLU(GroupNorm(Conv(x, conv1_weight)) + SiLU(GroupNorm(Conv(SiLU(GroupNorm(Conv(x))), conv2_weight)))
        """
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        B, C, H, W = x.shape

        # First conv path: y1 = conv3x3_triton(x, conv1_weight)
        y1 = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)
        # Ensure contiguous for simple stride math
        x_c = x.contiguous()
        w1_c = conv1_weight.contiguous()
        y1_c = y1  # output will be contiguous since we store elementwise

        grid_conv1 = (B, C, H * W)
        conv3x3_triton[grid_conv1](
            x_c, w1_c, y1_c,
            B, C, H, W, C,  # C_in = C, C_out = C
            x_c.stride(0), x_c.stride(1), x_c.stride(2), x_c.stride(3),
            w1_c.stride(0), w1_c.stride(1), w1_c.stride(2), w1_c.stride(3),
            y1_c.stride(0), y1_c.stride(1), y1_c.stride(2), y1_c.stride(3),
        )

        # GroupNorm + affine (norm1) + SiLU
        y1g = torch.empty_like(y1_c)
        grid_gn1 = (B, 32)
        group_norm_affine_silu[grid_gn1](
            y1_c, norm1_weight, norm1_bias, y1g,
            B, C, H, W, 32, eps,
            y1_c.stride(0), y1_c.stride(1), y1_c.stride(2), y1_c.stride(3),
            y1g.stride(0), y1g.stride(1), y1g.stride(2), y1g.stride(3),
            BLOCK=1024,
        )

        # Second conv path: y2 = conv3x3_triton(y1g, conv2_weight)
        y2 = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)
        y1g_c = y1g.contiguous()
        w2_c = conv2_weight.contiguous()
        y2_c = y2

        grid_conv2 = (B, C, H * W)
        conv3x3_triton[grid_conv2](
            y1g_c, w2_c, y2_c,
            B, C, H, W, C,  # C_in = C, C_out = C
            y1g_c.stride(0), y1g_c.stride(1), y1g_c.stride(2), y1g_c.stride(3),
            w2_c.stride(0), w2_c.stride(1), w2_c.stride(2), w2_c.stride(3),
            y2_c.stride(0), y2_c.stride(1), y2_c.stride(2), y2_c.stride(3),
        )

        # GroupNorm + affine (norm2) + SiLU
        y2g = torch.empty_like(y2_c)
        grid_gn2 = (B, 32)
        group_norm_affine_silu[grid_gn2](
            y2_c, norm2_weight, norm2_bias, y2g,
            B, C, H, W, 32, eps,
            y2_c.stride(0), y2_c.stride(1), y2_c.stride(2), y2_c.stride(3),
            y2g.stride(0), y2g.stride(1), y2g.stride(2), y2g.stride(3),
            BLOCK=1024,
        )

        # Final residual add: out = y2g + x
        out = torch.empty_like(x)
        N = B * C * H * W
        grid_add = (triton.cdiv(N, 1024),)
        add_residual_kernel[grid_add](
            out, y2g, x_c,
            N, BLOCK=1024,
        )

        return out


def run(*args):
    return ModelNew()(*args)
