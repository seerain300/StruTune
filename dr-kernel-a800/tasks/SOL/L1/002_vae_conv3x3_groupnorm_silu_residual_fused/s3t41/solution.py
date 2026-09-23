import torch
import triton
import triton.language as tl


# Triton kernel: GroupNorm + affine (scale, bias) + SiLU
# x: (B, C, H, W), y: (B, C, H, W), weight: (C,), bias: (C,)
# Grid: (B, num_groups). Each program handles one (batch, group).
@triton.jit
def group_norm_affine_silu_kernel(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W, num_groups: tl.constexpr, eps: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,  # reduction block size
):
    b = tl.program_id(0)
    g = tl.program_id(1)

    group_channels = C // num_groups
    group_elements = group_channels * H * W

    # First pass: compute sum and sum of squares
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

    # Second pass: normalize, affine, SiLU
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

        # SiLU: z * sigmoid(z)
        s = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * s

        y_ptrs = y_ptr + b * y_stride_b + ch * y_stride_c + h * y_stride_h + w * y_stride_w
        tl.store(y_ptrs, y_vals, mask=mask)


# Triton kernel: elementwise residual add y = y + x
@triton.jit
def add_residual_kernel(y_ptr, x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y + x, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        conv1_weight: torch.Tensor,
        norm1_weight: torch.Tensor,
        norm1_bias: torch.Tensor,
        conv2_weight: torch.Tensor,
        norm2_weight: torch.Tensor,
        norm2_bias: torch.Tensor,
        eps: float = 1e-5,
    ):
        # Ensure tensors are on CUDA and contiguous
        assert x.is_cuda, "Input tensor must be on CUDA device for Triton kernels."
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        B, C, H, W = x.shape
        # Sanity check: C must be divisible by num_groups
        assert C % self.num_groups == 0, "C must be divisible by num_groups (32)."

        # conv1: F.conv2d (no bias), stride=1, padding=1
        y1 = torch.nn.functional.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # Triton GroupNorm + affine + SiLU for conv1
        y1_grouped = torch.empty_like(y1)
        grid_gn1 = (B, self.num_groups)
        group_norm_affine_silu_kernel[grid_gn1](
            y1, norm1_weight, norm1_bias, y1_grouped,
            B, C, H, W, self.num_groups, self.eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_grouped.stride(0), y1_grouped.stride(1), y1_grouped.stride(2), y1_grouped.stride(3),
            BLOCK=1024,
        )

        # conv2: F.conv2d (no bias), stride=1, padding=1
        y2_pre = torch.nn.functional.conv2d(y1_grouped, conv2_weight, bias=None, stride=1, padding=1)

        # Triton GroupNorm + affine + SiLU for conv2
        y2_grouped = torch.empty_like(y2_pre)
        grid_gn2 = (B, self.num_groups)
        group_norm_affine_silu_kernel[grid_gn2](
            y2_pre, norm2_weight, norm2_bias, y2_grouped,
            B, C, y2_pre.shape[2], y2_pre.shape[3], self.num_groups, self.eps,
            y2_pre.stride(0), y2_pre.stride(1), y2_pre.stride(2), y2_pre.stride(3),
            y2_grouped.stride(0), y2_grouped.stride(1), y2_grouped.stride(2), y2_grouped.stride(3),
            BLOCK=1024,
        )

        # Final residual add: out = y2_grouped + x
        out = torch.empty_like(x)
        N = B * C * y2_grouped.shape[2] * y2_grouped.shape[3]
        grid_add = (triton.cdiv(N, 1024),)
        add_residual_kernel[grid_add](y2_grouped, x, out, N, BLOCK=1024)
        return out


def run(*args):
    return ModelNew()(*args)
