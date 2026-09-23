import torch
import triton
import triton.language as tl


# SiLU elementwise: y = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
@triton.jit
def silu_kernel(x_ptr, out_ptr, total_elems: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offs, y, mask=mask)


# Elementwise residual addition: out = out + x
@triton.jit
def add_residual_kernel(x_ptr, out_ptr, res_ptr, total_elems: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems
    a = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(out_ptr + offs, mask=mask, other=0.0)
    c = a + b
    tl.store(res_ptr + offs, c, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        """
        Fused residual block:
        Conv3x3 -> GroupNorm -> SiLU
        Conv3x3 -> GroupNorm -> SiLU
        Add original input (elementwise in Triton)
        """
        # Ensure float32 for consistency
        x_fp32 = x.to(torch.float32).contiguous()
        device = x_fp32.device

        # Path 1: Conv1 -> GroupNorm1 -> SiLU1
        # Convolution 1 (PyTorch) - stride=1, padding=1, no bias
        y1 = torch.nn.functional.conv2d(x_fp32, conv1_weight.to(torch.float32).contiguous(), bias=None, stride=1, padding=1)
        # GroupNorm 1 (PyTorch), num_groups=32
        y2 = torch.nn.functional.group_norm(y1, 32, weight=norm1_weight.to(torch.float32).contiguous(), bias=norm1_bias.to(torch.float32).contiguous(), eps=eps)
        # SiLU 1 (Triton)
        total1 = y2.numel()
        y3 = torch.empty_like(y2, device=device, dtype=torch.float32)
        grid_silu1 = (triton.cdiv(total1, 1024),)
        silu_kernel[grid_silu1](y2, y3, total1, 1024, num_warps=4)

        # Convolution 2 (PyTorch)
        y4 = torch.nn.functional.conv2d(y3, conv2_weight.to(torch.float32).contiguous(), bias=None, stride=1, padding=1)
        # GroupNorm 2 (PyTorch)
        y5 = torch.nn.functional.group_norm(y4, 32, weight=norm2_weight.to(torch.float32).contiguous(), bias=norm2_bias.to(torch.float32).contiguous(), eps=eps)
        # SiLU 2 (Triton)
        total2 = y5.numel()
        y6 = torch.empty_like(y5, device=device, dtype=torch.float32)
        grid_silu2 = (triton.cdiv(total2, 1024),)
        silu_kernel[grid_silu2](y5, y6, total2, 1024, num_warps=4)

        # Residual addition: out = y6 + x (elementwise in Triton)
        total_final = y6.numel()
        out_final = torch.empty_like(y6, device=device, dtype=torch.float32)
        grid_add = (triton.cdiv(total_final, 1024),)
        add_residual_kernel[grid_add](y6, y6, out_final, total_final, 1024, num_warps=4)

        return out_final


def run(*args):
    return ModelNew()(*args)
