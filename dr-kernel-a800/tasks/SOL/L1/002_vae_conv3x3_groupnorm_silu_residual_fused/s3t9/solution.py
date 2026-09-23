import torch
import triton
import triton.language as tl


# Triton kernel: SiLU activation elementwise
# y: input (flattened), out: output (flattened), N: total number of elements
@triton.jit
def silu_kernel(y_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    s = 1.0 / (1.0 + tl.exp(-y))
    tl.store(out_ptr + offsets, y * s, mask=mask)


# Triton kernel: elementwise residual add out = y + x
@triton.jit
def add_residual_kernel(out_ptr, y_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y + x, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups=32, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        # Ensure CUDA tensors
        assert x.is_cuda, "Input tensor must be on CUDA for Triton kernels."
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Conv weights must be on CUDA."
        assert norm1_weight.is_cuda and norm1_bias.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "Norm params must be on CUDA."

        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # 1) conv1: F.conv2d (no bias, stride=1, padding=1)
        out1 = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        # 2) GroupNorm with provided per-channel scale/bias (use PyTorch to compute mean/var exactly)
        #    We can't implement full GroupNorm reduction in Triton here reliably; we will apply normalization
        #    using PyTorch GroupNorm which uses our weight/bias and computes per-channel mean/var per sample.
        #    This ensures correctness. Then we apply SiLU in Triton.
        #    Note: F.group_norm requires inputs of shape (N, C, L) where L=H*W; we will use it per sample.
        out1_norm = F.group_norm(out1, num_groups=self.num_groups, weight=norm1_weight, bias=norm1_bias, eps=self.eps)

        # 3) SiLU in Triton (elementwise)
        out1_silu = torch.empty_like(out1_norm)
        N1 = out1_norm.numel()
        silu_kernel[(triton.cdiv(N1, 1024),)](out1_norm, out1_silu, N1, BLOCK=1024, num_warps=4, num_stages=2)

        # 4) conv2: F.conv2d (no bias, stride=1, padding=1)
        out2 = F.conv2d(out1_silu, conv2_weight, bias=None, stride=1, padding=1)

        # 5) GroupNorm for conv2 output using norm2 params
        out2_norm = F.group_norm(out2, num_groups=self.num_groups, weight=norm2_weight, bias=norm2_bias, eps=self.eps)

        # 6) SiLU in Triton (elementwise)
        out2_silu = torch.empty_like(out2_norm)
        N2 = out2_norm.numel()
        silu_kernel[(triton.cdiv(N2, 1024),)](out2_norm, out2_silu, N2, BLOCK=1024, num_warps=4, num_stages=2)

        # 7) Residual addition: out = out2_silu + x (Triton)
        out = torch.empty_like(out2_silu)
        N = out2_silu.numel()
        add_residual_kernel[(triton.cdiv(N, 1024),)](out, out2_silu, x, N, BLOCK=1024, num_warps=4, num_stages=2)

        return out


def run(*args):
    return ModelNew()(*args)
