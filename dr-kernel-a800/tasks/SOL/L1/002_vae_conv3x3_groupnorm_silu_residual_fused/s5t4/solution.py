import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton kernel: apply per-channel affine and SiLU per element.
# Input y: (B, C, H, W) flattened as [B*C, HW], HW=H*W.
# out = SiLU(y) * scale + bias
# Grid: (B, C, HW)
@triton.jit
def affine_silu_kernel(
    y_ptr, out_ptr, scale_ptr, bias_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    hw = tl.program_id(2)  # linear index over spatial elements

    h = hw // W
    w = hw % W

    idx = ((n * C) + c) * (H * W) + hw

    y_val = tl.load(y_ptr + idx)
    scale = tl.load(scale_ptr + c)
    bias = tl.load(bias_ptr + c)

    sig = 1.0 / (1.0 + tl.exp(-y_val))  # sigmoid
    out_val = y_val * sig * scale + bias

    tl.store(out_ptr + idx, out_val)


# Triton kernel: elementwise add residual: out = out + x
# Grid: (B, C, HW)
@triton.jit
def residual_add_kernel(
    out_ptr, x_ptr, res_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    hw = tl.program_id(2)

    h = hw // W
    w = hw % W

    idx = ((n * C) + c) * (H * W) + hw

    out_val = tl.load(out_ptr + idx)
    x_val = tl.load(x_ptr + idx)
    tl.store(res_ptr + idx, out_val + x_val)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        x: (B, C, H, W)
        conv weights: (C_out, C_in, 3, 3)
        norm scales/bias: (C,)
        Returns: (B, C_out2, H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        x = x.contiguous()

        # First conv
        out1 = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # First GroupNorm
        out1_norm = F.group_norm(out1, self.num_groups, weight=norm1_weight, bias=norm1_bias, eps=self.eps)

        # Apply SiLU in Triton using per-channel scale/bias from norm1
        out1_silu = torch.empty_like(out1_norm)
        B1, C1, H1, W1 = out1_silu.shape
        grid1 = (B1, C1, H1 * W1)
        affine_silu_kernel[grid1](
            out1_norm, out1_silu, norm1_weight.to(torch.float32), norm1_bias.to(torch.float32),
            B=B1, C=C1, H=H1, W=W1,
            num_warps=4,
            num_stages=2,
        )

        # Second conv
        out2_pre = F.conv2d(out1_silu, conv2_weight, bias=None, stride=1, padding=1)

        # Second GroupNorm
        out2_norm = F.group_norm(out2_pre, self.num_groups, weight=norm2_weight, bias=norm2_bias, eps=self.eps)

        # Apply SiLU in Triton using per-channel scale/bias from norm2
        out2_silu = torch.empty_like(out2_norm)
        B2, C2, H2, W2 = out2_silu.shape
        grid2 = (B2, C2, H2 * W2)
        affine_silu_kernel[grid2](
            out2_norm, out2_silu, norm2_weight.to(torch.float32), norm2_bias.to(torch.float32),
            B=B2, C=C2, H=H2, W=W2,
            num_warps=4,
            num_stages=2,
        )

        # Final residual add: out2_silu + x
        out = torch.empty_like(out2_silu)
        residual_add_kernel[(B2, C2, H2 * W2)](
            out2_silu, x.to(torch.float32), out,
            B=B2, C=C2, H=H2, W=W2,
            num_warps=4,
            num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
