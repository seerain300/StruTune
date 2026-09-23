import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton kernel: GroupNorm + affine (scale, bias) + SiLU per (batch, group)
# x: (B, C, H, W), weight_ptr: (C,), bias_ptr: (C,), y: (B, C, H, W)
# Grid: (B, num_groups). Each program handles one sample and one group.
@triton.jit
def group_norm_affine_silu(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W, num_groups, eps,
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

        c_vec = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr + b * x_stride_b + c_vec * x_stride_c + h * x_stride_h + w * x_stride_w
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

        c_vec = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr + b * x_stride_b + c_vec * x_stride_c + h * x_stride_h + w * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        norm_vals = (x_vals - mean) * inv_std

        scale = tl.load(weight_ptr + c_vec, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + c_vec, mask=mask, other=0.0)
        z = norm_vals * scale + bias

        # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        s = z / (1.0 + tl.exp(-z))
        y_vals = z * s

        y_ptrs = y_ptr + b * y_stride_b + c_vec * y_stride_c + h * y_stride_h + w * y_stride_w
        tl.store(y_ptrs, y_vals, mask=mask)


# Triton kernel: elementwise residual add y = y + x over all elements
@triton.jit
def add_residual_kernel(y_ptr, out_ptr, x_ptr, B, C, H, W, BLOCK: tl.constexpr):
    # Flatten and process tiles of size BLOCK
    N = B * C * H * W
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # map offsets to (b, c, h, w)
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

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        """
        Fused residual block with Triton: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        Inputs:
          x: (B, C, H, W), float32 CUDA
          conv1_weight, conv2_weight: (C, C, 3, 3), float32 CUDA
          norm1_weight, norm2_weight: (C,), float32 CUDA
          norm1_bias, norm2_bias: (C,), float32 CUDA
          eps: float
        """
        assert x.is_cuda, "Input must be on CUDA for Triton kernels"
        assert x.dtype == torch.float32 and conv1_weight.dtype == torch.float32 and conv2_weight.dtype == torch.float32
        B, C, H, W = x.shape
        num_groups = 32
        assert C % num_groups == 0, "C must be divisible by num_groups (32) for GroupNorm"

        # First path: Conv1 -> GroupNorm -> SiLU
        out1 = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)  # (B, C, H, W)
        y1 = torch.empty_like(out1)
        grid1 = (B, num_groups)
        group_norm_affine_silu[grid1](
            out1, norm1_weight, norm1_bias, y1,
            B, C, H, W, num_groups, eps,
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK=1024,
        )

        # Second path: Conv2 -> GroupNorm -> SiLU
        out2 = F.conv2d(y1, conv2_weight, bias=None, stride=1, padding=1)  # (B, C, H, W)
        y2 = torch.empty_like(out2)
        grid2 = (B, num_groups)
        group_norm_affine_silu[grid2](
            out2, norm2_weight, norm2_bias, y2,
            B, C, H, W, num_groups, eps,
            out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK=1024,
        )

        # Residual connection: out = y2 + x
        final = torch.empty_like(y2)
        N = B * C * H * W
        BLOCK = 1024
        grid_add = (triton.cdiv(N, BLOCK),)
        add_residual_kernel[grid_add](
            y2, final, x,
            B, C, H, W,
            BLOCK=BLOCK,
        )
        return final


def run(*args):
    return ModelNew()(*args)
