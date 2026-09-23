import torch
import triton
import triton.language as tl


@triton.jit
def add_residual_kernel(inp_ptr, res_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    y = tl.load(inp_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(res_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    z = y + x
    tl.store(out_ptr + offsets, z, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        conv1_weight: torch.Tensor,
        norm1_weight: torch.Tensor,
        norm1_bias: torch.Tensor,
        conv2_weight: torch.Tensor,
        norm2_weight: torch.Tensor,
        norm2_bias: torch.Tensor,
        eps: float,
    ):
        # First conv: 3x3 stride=1, padding=1, bias=None
        out1 = torch.nn.functional.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # First GroupNorm(num_groups=32), with per-channel scale and bias
        out1_gn = torch.nn.functional.group_norm(out1, num_groups=self.num_groups,
                                                  weight=norm1_weight, bias=norm1_bias, eps=self.eps)

        # SiLU: y = x * sigmoid(x)
        out1_silu = out1_gn * torch.sigmoid(out1_gn)

        # Second conv
        out2_pre = torch.nn.functional.conv2d(out1_silu, conv2_weight, bias=None, stride=1, padding=1)

        # Second GroupNorm
        out2_gn = torch.nn.functional.group_norm(out2_pre, num_groups=self.num_groups,
                                                  weight=norm2_weight, bias=norm2_bias, eps=self.eps)

        # SiLU
        out2_silu = out2_gn * torch.sigmoid(out2_gn)

        # Residual add using Triton
        N = out2_silu.numel()
        out = torch.empty_like(out2_silu)

        # Triton addition over flattened buffer
        out2_silu_c = out2_silu.contiguous()
        x_c = x.contiguous()
        grid = (triton.cdiv(N, 1024),)
        add_residual_kernel[grid](out2_silu_c, x_c, out, N, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
