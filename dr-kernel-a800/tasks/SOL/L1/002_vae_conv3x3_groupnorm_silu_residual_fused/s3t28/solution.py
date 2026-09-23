import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton kernel: GroupNorm forward + affine (scale, bias) + SiLU
# Assumes num_groups=32 and C % 32 == 0. Operates over (B, groups).
@triton.jit
def group_norm_affine_silu(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    B, C, H, W,
    num_groups: tl.constexpr,  # 32
    eps: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,  # e.g., 1024
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

        x_ptrs = x_ptr \
                 + b * x_stride_b \
                 + c_vec * x_stride_c \
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

        c_vec = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr \
                 + b * x_stride_b \
                 + c_vec * x_stride_c \
                 + h * x_stride_h \
                 + w * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        norm_vals = (x_vals - mean) * inv_std

        scale = tl.load(weight_ptr + c_vec, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + c_vec, mask=mask, other=0.0)
        z = norm_vals * scale + bias

        s = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * s

        y_ptrs = y_ptr \
                 + b * y_stride_b \
                 + c_vec * y_stride_c \
                 + h * y_stride_h \
                 + w * y_stride_w
        tl.store(y_ptrs, y_vals, mask=mask)


# Triton kernel: elementwise residual add y = y + x
@triton.jit
def add_residual_kernel(out_ptr, y_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def _triton_groupnorm_silu_and_residual(x, weight, bias, eps):
    """
    Perform:
      - GroupNorm (num_groups=32, affine) on x -> y_norm
      - SiLU activation on y_norm
      - Residual add with x -> out = y_norm + x
    Using Triton for GroupNorm + affine + SiLU and an elementwise kernel for addition.
    x: (B, C, H, W), weight: (C,), bias: (C,), eps: float
    """
    assert x.is_cuda, "Triton kernels require CUDA tensors."
    assert x.dim() == 4, "Input must be NCHW."
    B, C, H, W = x.shape
    assert C % 32 == 0, "num_groups=32 requires C % 32 == 0."

    # Ensure weight/bias on same device/dtype and contiguous
    weight = weight.to(device=x.device, dtype=x.dtype).contiguous()
    bias = bias.to(device=x.device, dtype=x.dtype).contiguous()

    # GroupNorm + affine + SiLU
    y_norm = torch.empty_like(x)
    grid = (B, 32)
    group_norm_affine_silu[grid](
        x, y_norm, weight, bias,
        B, C, H, W,
        32, eps,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y_norm.stride(0), y_norm.stride(1), y_norm.stride(2), y_norm.stride(3),
        1024,
        num_warps=4, num_stages=2
    )

    # Residual add: out = y_norm + x
    out = torch.empty_like(x)
    N = B * C * H * W
    grid_add = (triton.cdiv(N, 1024),)
    add_residual_kernel[grid_add](out, y_norm, x, N, 1024, num_warps=4, num_stages=2)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps=1e-5):
        """
        Fused residual block using PyTorch convs and Triton for GroupNorm + SiLU + residual add.
        x: Input tensor of shape (B, C, H, W)
        conv1_weight: (C, C, 3, 3), no bias
        norm1_weight: (C,), norm1_bias: (C,)
        conv2_weight: (C, C, 3, 3), no bias
        norm2_weight: (C,), norm2_bias: (C,)
        eps: epsilon for numerical stability
        Returns: (B, C, H, W)
        """
        # 1) Conv1: stride=1, padding=1, no bias
        y1 = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # 2) GroupNorm + affine (norm1) + SiLU and residual add via Triton
        y1_silu = _triton_groupnorm_silu_and_residual(y1, norm1_weight, norm1_bias, eps)

        # 3) Conv2: same as above
        y2 = F.conv2d(y1_silu, conv2_weight, bias=None, stride=1, padding=1)

        # 4) GroupNorm + affine (norm2) + SiLU
        y2_silu = _triton_groupnorm_silu_and_residual(y2, norm2_weight, norm2_bias, eps)

        # 5) Final residual: add input x
        out = y2_silu + x
        return out


def run(*args):
    return ModelNew()(*args)
