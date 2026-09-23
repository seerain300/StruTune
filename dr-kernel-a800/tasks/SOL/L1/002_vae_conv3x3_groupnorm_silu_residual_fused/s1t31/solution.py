import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    """
    Elementwise SiLU: y = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
    Works on flattened tensors. Assumes float32.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def add_residual_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    """
    Elementwise addition: out = a + b
    Assumes both inputs have the same shape and are float32.
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    out = a + b
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
        """
        Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        PyTorch handles convolutions and GroupNorm for correctness; Triton is used for SiLU and residual addition.
        """
        # First path: Conv1 -> GroupNorm1 -> SiLU
        out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)  # NCHW
        num_groups = 32
        out = F.group_norm(out, num_groups, weight=norm1_weight, bias=norm1_bias, eps=eps)
        # Triton SiLU
        out_silu = torch.empty_like(out, dtype=torch.float32, device=out.device)
        n_elements = out.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n_elements, BLOCK),)
        silu_kernel[grid](out, out_silu, n_elements, BLOCK=BLOCK)

        # Second path: Conv2 -> GroupNorm2 -> SiLU
        out = F.conv2d(out_silu, conv2_weight, bias=None, stride=1, padding=1)
        out = F.group_norm(out, num_groups, weight=norm2_weight, bias=norm2_bias, eps=eps)
        # Triton SiLU
        out_silu2 = torch.empty_like(out, dtype=torch.float32, device=out.device)
        n_elements2 = out.numel()
        grid2 = (triton.cdiv(n_elements2, BLOCK),)
        silu_kernel[grid2](out, out_silu2, n_elements2, BLOCK=BLOCK)

        # Final residual addition using Triton: out = out_silu2 + x (original input)
        # Ensure dtype and contiguity for Triton
        x_fp32 = x.to(dtype=torch.float32).contiguous()
        out_fp32 = out_silu2.to(dtype=torch.float32).contiguous()
        out_final = torch.empty_like(out_fp32, device=out_fp32.device, dtype=out_fp32.dtype)
        total = out_fp32.numel()
        grid_add = (triton.cdiv(total, BLOCK),)
        add_residual_kernel[grid_add](out_fp32, x_fp32, out_final, total, BLOCK=BLOCK)

        return out_final


def run(*args):
    return ModelNew()(*args)
