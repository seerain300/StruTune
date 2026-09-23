import torch
import triton
import triton.language as tl


# GroupNorm + Affine + SiLU in a single Triton kernel, implemented in two passes:
# 1) reduction kernel to compute per-channel mean and rstd across H*W
# 2) apply kernel to normalize + affine + SiLU
@triton.jit
def group_norm_reduce_kernel(
    x_ptr, mean_ptr, rstd_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, eps: tl.constexpr,
):
    # program ids: (n, group, channel_in_group)
    n = tl.program_id(0)
    g = tl.program_id(1)
    c_start = g * (C // num_groups)
    c = tl.program_id(2) + c_start

    total = H * W
    sum_val = 0.0
    sumsq_val = 0.0

    # tile over H*W in chunks of 128
    N_TILES = (total + 127) // 128
    for t in range(0, N_TILES):
        tile_start = t * 128
        offs = tile_start + tl.arange(0, 128)
        mask = offs < total
        base = n * C * H * W
        idx = base + c * H * W + offs
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(vals, axis=0)
        sumsq_val += tl.sum(vals * vals, axis=0)

    mean = sum_val / total
    var = sumsq_val / total - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + c, mean)
    tl.store(rstd_ptr + c, rstd)


@triton.jit
def group_norm_apply_affine_silu_kernel(
    x_ptr, mean_ptr, rstd_ptr, scale_ptr, bias_ptr, y_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
):
    # program ids: (n, group, channel_in_group, tile)
    n = tl.program_id(0)
    g = tl.program_id(1)
    c_start = g * (C // num_groups)
    c = tl.program_id(2) + c_start

    total = H * W
    N_TILES = (total + 127) // 128
    for t in range(0, N_TILES):
        tile_start = t * 128
        offs = tile_start + tl.arange(0, 128)
        mask = offs < total
        base = n * C * H * W
        idx = base + c * H * W + offs
        x_vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
        mean = tl.load(mean_ptr + c)
        rstd = tl.load(rstd_ptr + c)
        scale = tl.load(scale_ptr + c)
        bias = tl.load(bias_ptr + c)
        # normalize and affine
        norm = (x_vals - mean) * rstd
        norm = norm * scale + bias
        # SiLU: x * sigmoid(x)
        silu = norm * (1.0 / (1.0 + tl.exp(-norm)))
        tl.store(y_ptr + idx, silu, mask=mask)


# Elementwise residual add: y = y + x
@triton.jit
def residual_add_kernel(
    a_ptr, b_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    idx = ((n * C) + c) * H * W + h * W + w
    a_val = tl.load(a_ptr + idx)
    b_val = tl.load(b_ptr + idx)
    tl.store(out_ptr + idx, a_val + b_val)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        x: (B, C_in, H, W), conv weights: (C_out, C_in, 3, 3), norm scales/bias: (C_out,)
        Returns: (B, conv2_weight.shape[0], H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"

        # Ensure contiguous and float32 for Triton computations
        x = x.contiguous().to(torch.float32)
        conv1_weight = conv1_weight.contiguous().to(torch.float32)  # (C_out1, C_in, 3, 3)
        conv2_weight = conv2_weight.contiguous().to(torch.float32)  # (C_out2, C_in, 3, 3)
        norm1_weight = norm1_weight.contiguous().to(torch.float32)
        norm1_bias = norm1_bias.contiguous().to(torch.float32)
        norm2_weight = norm2_weight.contiguous().to(torch.float32)
        norm2_bias = norm2_bias.contiguous().to(torch.float32)

        B, C_in, H, W = x.shape
        C_out1 = conv1_weight.shape[0]
        C_out2 = conv2_weight.shape[0]

        # First conv: y1 = conv3x3(x, conv1_weight) using PyTorch (to avoid Triton conv issues)
        out1 = torch.nn.functional.conv2d(
            x, conv1_weight, bias=None, stride=1, padding=1, dilation=1, groups=1
        )

        # GroupNorm + SiLU for first block
        mean1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        rstd1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        grid_reduce1 = (B, self.num_groups, C_out1 // self.num_groups)
        group_norm_reduce_kernel[grid_reduce1](
            out1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups, eps=self.eps,
            num_warps=4, num_stages=2
        )
        out1_norm = torch.empty_like(out1)
        grid_apply1 = (B, self.num_groups, C_out1 // self.num_groups, ((H * W + 127) // 128))
        group_norm_apply_affine_silu_kernel[grid_apply1](
            out1, mean1, rstd1, norm1_weight, norm1_bias, out1_norm,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups,
            num_warps=4, num_stages=2
        )

        # Second conv: y2 = conv3x3(out1_norm, conv2_weight) using PyTorch
        out2 = torch.nn.functional.conv2d(
            out1_norm, conv2_weight, bias=None, stride=1, padding=1, dilation=1, groups=1
        )

        # GroupNorm + SiLU for second block
        mean2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        rstd2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        grid_reduce2 = (B, self.num_groups, C_out2 // self.num_groups)
        group_norm_reduce_kernel[grid_reduce2](
            out2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups, eps=self.eps,
            num_warps=4, num_stages=2
        )
        out2_norm = torch.empty_like(out2)
        grid_apply2 = (B, self.num_groups, C_out2 // self.num_groups, ((H * W + 127) // 128))
        group_norm_apply_affine_silu_kernel[grid_apply2](
            out2, mean2, rstd2, norm2_weight, norm2_bias, out2_norm,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups,
            num_warps=4, num_stages=2
        )

        # Final residual add: out2_norm + x
        out = torch.empty_like(out2_norm)
        grid_add = (B, C_out2, H, W)
        residual_add_kernel[grid_add](
            out2_norm, x, out,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
