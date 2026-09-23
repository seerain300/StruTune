import torch
import triton
import triton.language as tl


# Triton kernel: GroupNorm forward + affine (scale, bias) + SiLU
# x: (B, C, H, W), weight_ptr: (C,), bias_ptr: (C,), y: (B, C, H, W)
# Grid: (B, num_groups). Each program handles one (batch, group).
@triton.jit
def group_norm_affine_silu_triton(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, eps: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,  # reduction/block size
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

        # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
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
def add_residual_kernel(out_ptr, y_ptr, x_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # Map linear offsets to (b, c, h, w) assuming N = B*C*H*W
    tmp = offsets
    w = tmp % W
    tmp = tmp // W
    h = tmp % H
    tmp = tmp // H
    c = tmp % C
    b = tmp // C

    x_ptrs = x_ptr + b * C * H * W + c * H * W + h * W + w
    y_ptrs = y_ptr + b * C * H * W + c * H * W + h * W + w
    out_ptrs = out_ptr + b * C * H * W + c * H * W + h * W + w

    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
    y_vals = tl.load(y_ptrs, mask=mask, other=0.0)
    out_vals = y_vals + x_vals
    tl.store(out_ptrs, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        # Ensure CUDA and contiguous
        assert x.is_cuda, "Input must be on CUDA"
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        B, C, H, W = x.shape
        num_groups = 32  # match original code
        assert C % num_groups == 0, "C must be divisible by num_groups=32"

        # First conv using PyTorch (robust and fast)
        conv1 = torch.nn.functional.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        conv1 = conv1.contiguous()

        # GroupNorm + affine + SiLU for conv1 output (Triton)
        conv1_norm = torch.empty_like(conv1)
        grid = (B, num_groups)
        group_norm_affine_silu_triton[grid](
            conv1, norm1_weight, norm1_bias, conv1_norm,
            B, C, H, W, num_groups, eps,
            conv1.stride(0), conv1.stride(1), conv1.stride(2), conv1.stride(3),
            conv1_norm.stride(0), conv1_norm.stride(1), conv1_norm.stride(2), conv1_norm.stride(3),
            BLOCK=1024, num_warps=4
        )

        # Second conv
        conv2 = torch.nn.functional.conv2d(conv1_norm, conv2_weight, bias=None, stride=1, padding=1)
        conv2 = conv2.contiguous()

        # GroupNorm + affine + SiLU for conv2 output (Triton)
        conv2_norm = torch.empty_like(conv2)
        grid2 = (B, num_groups)
        group_norm_affine_silu_triton[grid2](
            conv2, norm2_weight, norm2_bias, conv2_norm,
            B, C, H, W, num_groups, eps,
            conv2.stride(0), conv2.stride(1), conv2.stride(2), conv2.stride(3),
            conv2_norm.stride(0), conv2_norm.stride(1), conv2_norm.stride(2), conv2_norm.stride(3),
            BLOCK=1024, num_warps=4
        )

        # Final residual addition via Triton
        out = torch.empty_like(x)
        N = B * C * H * W
        grid_add = (triton.cdiv(N, 1024),)
        add_residual_kernel[grid_add](
            out, conv2_norm, x,
            N, BLOCK=1024, num_warps=4
        )

        return out


def run(*args):
    return ModelNew()(*args)
