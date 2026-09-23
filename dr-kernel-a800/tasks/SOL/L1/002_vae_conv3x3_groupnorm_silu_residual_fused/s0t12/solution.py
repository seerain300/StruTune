import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# Triton kernels for elementwise ops

# SiLU: y = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
@triton.jit
def silu_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # sigmoid
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)

# Residual add: y = out + residual (elementwise)
@triton.jit
def add_residual_kernel(out_ptr, res_ptr, y_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    out = tl.load(out_ptr + offs, mask=mask, other=0.0)
    res = tl.load(res_ptr + offs, mask=mask, other=0.0)
    y = out + res
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps
        self.num_groups = 32

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        # Ensure device and dtype
        device = x.device
        dtype = torch.float32

        # 1) First conv: stride=1, padding=1, bias=None
        # Using torch conv for correctness
        conv1 = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        # 2) GroupNorm1 (num_groups=32), weight, bias, eps
        group_norm1 = F.group_norm(conv1, self.num_groups, weight=norm1_weight, bias=norm1_bias, eps=self.eps)
        # 3) SiLU via Triton
        group_norm1_flat = group_norm1.contiguous().to(dtype)
        N1 = group_norm1_flat.numel()
        BLOCK = 4096
        y1 = torch.empty_like(group_norm1_flat)
        grid = (triton.cdiv(N1, BLOCK),)
        silu_kernel[grid](group_norm1_flat, y1, N1, BLOCK)
        group_norm1_silu = y1.view_as(group_norm1)

        # 4) Second conv: stride=1, padding=1, bias=None
        conv2 = F.conv2d(group_norm1_silu, conv2_weight, bias=None, stride=1, padding=1)
        # 5) GroupNorm2 (num_groups=32), weight, bias, eps
        group_norm2 = F.group_norm(conv2, self.num_groups, weight=norm2_weight, bias=norm2_bias, eps=self.eps)
        # 6) SiLU via Triton
        group_norm2_flat = group_norm2.contiguous().to(dtype)
        N2 = group_norm2_flat.numel()
        y2 = torch.empty_like(group_norm2_flat)
        grid2 = (triton.cdiv(N2, BLOCK),)
        silu_kernel[grid2](group_norm2_flat, y2, N2, BLOCK)
        group_norm2_silu = y2.view_as(group_norm2)

        # 7) Add residual x
        residual_flat = x.contiguous().to(dtype)
        Nfinal = group_norm2_silu.numel()
        yfinal = torch.empty_like(group_norm2_silu)
        grid_add = (triton.cdiv(Nfinal, BLOCK),)
        add_residual_kernel[grid_add](group_norm2_silu, residual_flat, yfinal, Nfinal, BLOCK)

        # Return result, keeping original shape and device
        return yfinal.view_as(group_norm2).to(x.dtype)


def run(*args):
    return ModelNew()(*args)
