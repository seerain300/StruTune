import torch
import triton
import triton.language as tl


@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    # Elementwise SiLU: y = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # compute sigmoid
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def add_residual_kernel(x_ptr, residual_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # Elementwise addition: out = x + residual
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    res = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    out = x + res
    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        # conv1: y1 = conv2d(x, conv1_weight, stride=1, padding=1, no bias)
        y1 = torch.nn.functional.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # GroupNorm1: (num_groups=32) with affine
        # Note: PyTorch expects weight and bias as (C,) matching channel dimension.
        y1_gn = torch.nn.functional.group_norm(y1, num_groups=32, weight=norm1_weight, bias=norm1_bias, eps=eps)

        # SiLU via Triton
        y1_silu = torch.empty_like(y1_gn, device=y1_gn.device, dtype=y1_gn.dtype)
        total1 = y1_silu.numel()
        grid1 = (triton.cdiv(total1, 1024),)
        # For robustness, cast to float32 for SiLU computation
        y1_silu_fp32 = torch.empty_like(y1_gn, device=y1_gn.device, dtype=torch.float32)
        torch.nn.functional.silu(y1_gn.to(torch.float32), out=y1_silu_fp32)
        silu_kernel[grid1](y1_silu_fp32, y1_silu, total1, BLOCK=1024)

        # conv2: y2 = conv2d(y1_silu, conv2_weight, stride=1, padding=1, no bias)
        y2 = torch.nn.functional.conv2d(y1_silu, conv2_weight, bias=None, stride=1, padding=1)

        # GroupNorm2
        y2_gn = torch.nn.functional.group_norm(y2, num_groups=32, weight=norm2_weight, bias=norm2_bias, eps=eps)

        # SiLU via Triton
        y2_silu = torch.empty_like(y2_gn, device=y2_gn.device, dtype=y2_gn.dtype)
        total2 = y2_silu.numel()
        grid2 = (triton.cdiv(total2, 1024),)
        y2_silu_fp32 = torch.empty_like(y2_gn, device=y2_gn.device, dtype=torch.float32)
        torch.nn.functional.silu(y2_gn.to(torch.float32), out=y2_silu_fp32)
        silu_kernel[grid2](y2_silu_fp32, y2_silu, total2, BLOCK=1024)

        # Residual add: out = y2_silu + x  (x is original input, shape (B, C, H, W))
        # We use Triton for elementwise add
        out = torch.empty_like(y2_silu, device=y2_silu.device, dtype=y2_silu.dtype)
        total_out = out.numel()
        grid_add = (triton.cdiv(total_out, 1024),)
        add_residual_kernel[grid_add](y2_silu, x.to(torch.float32), out, total_out, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
