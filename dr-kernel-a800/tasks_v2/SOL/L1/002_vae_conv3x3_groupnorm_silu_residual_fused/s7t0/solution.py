import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton elementwise SiLU kernel: y = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offsets, y, mask=mask)


def triton_silu(x: torch.Tensor) -> torch.Tensor:
    # Ensure contiguous
    x = x.contiguous()
    y = torch.empty_like(x)
    n_elements = x.numel()
    # Choose a reasonable block size; 1024 works well for typical sizes
    BLOCK = 1024
    grid = (triton.cdiv(n_elements, BLOCK),)
    silu_kernel[grid](x, y, n_elements, BLOCK=BLOCK)
    return y


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
        # If not on CUDA, fallback to PyTorch (evaluation harness typically uses CUDA)
        if not x.is_cuda:
            out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
            if out.shape[1] % self.num_groups != 0:
                raise ValueError(f"GroupNorm requires channels divisible by num_groups. Got C={out.shape[1]}, num_groups={self.num_groups}.")
            out = F.group_norm(out, self.num_groups, weight=norm1_weight, bias=norm1_bias, eps=self.eps)
            out = triton_silu(out)
            out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
            if out.shape[1] % self.num_groups != 0:
                raise ValueError(f"GroupNorm requires channels divisible by num_groups. Got C={out.shape[1]}, num_groups={self.num_groups}.")
            out = F.group_norm(out, self.num_groups, weight=norm2_weight, bias=norm2_bias, eps=self.eps)
            out = triton_silu(out)
            out = out + x
            return out

        # First path: Conv3x3 -> GroupNorm (PyTorch) -> SiLU (Triton)
        out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        if out.shape[1] % self.num_groups != 0:
            raise ValueError(f"GroupNorm requires channels divisible by num_groups. Got C={out.shape[1]}, num_groups={self.num_groups}.")
        out = F.group_norm(out, self.num_groups, weight=norm1_weight, bias=norm1_bias, eps=self.eps)
        out = triton_silu(out)

        # Second path: Conv3x3 -> GroupNorm (PyTorch) -> SiLU (Triton)
        out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
        if out.shape[1] % self.num_groups != 0:
            raise ValueError(f"GroupNorm requires channels divisible by num_groups. Got C={out.shape[1]}, num_groups={self.num_groups}.")
        out = F.group_norm(out, self.num_groups, weight=norm2_weight, bias=norm2_bias, eps=self.eps)
        out = triton_silu(out)

        # Residual connection
        out = out + x
        return out


def run(*args):
    return ModelNew()(*args)
