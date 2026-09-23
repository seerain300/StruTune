import torch
import triton
import triton.language as tl


# Triton kernel: GroupNorm forward + affine (scale, bias) + SiLU
# Input x: (B, C, H, W), weight/scale: (C,), bias: (C,), output y: (B, C, H, W)
# Grid: (B, num_groups). Assumes num_groups=32 and C % 32 == 0.
@triton.jit
def group_norm_affine_silu(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W, num_groups: tl.constexpr, eps: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,
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

        s = 1.0 / (1.0 + tl.exp(-z))  # sigmoid
        y_vals = z * s

        y_ptrs = y_ptr \
                 + b * y_stride_b \
                 + ch * y_stride_c \
                 + h * y_stride_h \
                 + w * y_stride_w
        tl.store(y_ptrs, y_vals, mask=mask)


# Triton kernel: elementwise residual add y = y + x (flattened)
@triton.jit
def add_residual_kernel(y_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    y_vals = y_vals + x_vals
    tl.store(y_ptr + offsets, y_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor):
        # Conv1 via PyTorch
        out1 = torch.nn.functional.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        # GroupNorm + affine + SiLU via Triton for out1
        out1 = out1.contiguous()
        y1 = torch.empty_like(out1)
        grid = (out1.shape[0], self.num_groups)
        group_norm_affine_silu[grid](
            out1, norm1_weight, norm1_bias, y1,
            out1.shape[0], out1.shape[1], out1.shape[2], out1.shape[3],
            self.num_groups, self.eps,
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK=1024,
        )
        # SiLU activation (PyTorch)
        y1 = torch.nn.functional.silu(y1)

        # Conv2 via PyTorch
        out2 = torch.nn.functional.conv2d(y1, conv2_weight, bias=None, stride=1, padding=1)
        out2 = out2.contiguous()

        # GroupNorm + affine + SiLU via Triton for out2
        y2 = torch.empty_like(out2)
        group_norm_affine_silu[grid](
            out2, norm2_weight, norm2_bias, y2,
            out2.shape[0], out2.shape[1], out2.shape[2], out2.shape[3],
            self.num_groups, self.eps,
            out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK=1024,
        )
        # SiLU activation (PyTorch)
        y2 = torch.nn.functional.silu(y2)

        # Add residual x using Triton
        y2_flat = y2.view(-1)
        x_flat = x.view(-1)
        y3_flat = torch.empty_like(x_flat)
        N = y2_flat.numel()
        add_residual_kernel[(triton.cdiv(N, 1024),)](y2_flat, x_flat, N, BLOCK=1024)
        y3 = y3_flat.view_as(x)

        return y3


def run(*args):
    return ModelNew()(*args)
