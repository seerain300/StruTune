import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def groupnorm_reduce_sums(
    x_ptr, sums_ptr,
    N, C, H, W, num_groups,
    X_sN, X_sC, X_sH, X_sW,
    BLOCK_HW: tl.constexpr
):
    # Each program handles one (n, group)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    C_per_group = C // num_groups
    start_chan = pid_n * C_per_group + pid_g

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Loop over channels in this group
    for i in range(0, C_per_group):
        chan = start_chan + i
        num_hw = H * W
        s = tl.zeros((), dtype=tl.float32)
        ss = tl.zeros((), dtype=tl.float32)
        # Iterate over H*W in chunks
        for off in range(0, num_hw, BLOCK_HW):
            hw_off = off + tl.arange(0, BLOCK_HW)
            mask_hw = hw_off < num_hw
            h = hw_off // W
            w = hw_off % W
            x_off = pid_n * X_sN + chan * X_sC + h * X_sH + w * X_sW
            x_val = tl.load(x_ptr + x_off, mask=mask_hw, other=0.0)
            x_f32 = x_val.to(tl.float32)
            s += tl.sum(x_f32, axis=0)
            ss += tl.sum(x_f32 * x_f32, axis=0)
        sum_val += s
        sumsq_val += ss

    M = C_per_group * H * W
    base = pid_n * (num_groups * 2) + pid_g * 2
    tl.store(sums_ptr + base + 0, sum_val)    # sum
    tl.store(sums_ptr + base + 1, sumsq_val)  # sumsq


@triton.jit
def groupnorm_apply_affine_silu_nc(
    x_ptr, y_ptr, weight_ptr, bias_ptr, sums_ptr,
    N, C, H, W, num_groups, eps,
    X_sN, X_sC, X_sH, X_sW,
    Y_sN, Y_sC, Y_sH, Y_sW,
    GROUP_sN
):
    # One program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    C_per_group = C // num_groups
    g = c // C_per_group

    base = n * (num_groups * 2) + g * 2
    sum_val = tl.load(GROUP_sN + base + 0)
    sumsq_val = tl.load(GROUP_sN + base + 1)

    M = C_per_group * H * W
    mean = sum_val / M
    var = sumsq_val / M - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Affine params for this channel
    weight_c = tl.load(weight_ptr + c)
    bias_c = tl.load(bias_ptr + c)

    # Elementwise apply over H*W
    num_hw = H * W
    for off in range(0, num_hw):
        h = off // W
        w = off % W
        x_off = n * X_sN + c * X_sC + h * X_sH + w * X_sW
        x_val = tl.load(x_ptr + x_off)
        x_f32 = x_val.to(tl.float32)

        y_norm = (x_f32 - mean) * rstd
        y_affine = y_norm * weight_c + bias_c

        # SiLU: y = x * sigmoid(x)
        sig = 1.0 / (1.0 + tl.exp(-y_affine))
        y = y_affine * sig

        y_off = n * Y_sN + c * Y_sC + h * Y_sH + w * Y_sW
        tl.store(y_ptr + y_off, y)


@triton.jit
def add_residual(x_ptr, out_ptr, N, C, H, W):
    total = N * C * H * W
    pid = tl.program_id(0)
    if pid < total:
        n = pid // (C * H * W)
        rem = pid % (C * H * W)
        c = rem // (H * W)
        rem2 = rem % (H * W)
        h = rem2 // W
        w = rem2 % W

        off = n * C * H * W + c * H * W + h * W + w
        a = tl.load(out_ptr + off)
        b = tl.load(x_ptr + off)
        tl.store(out_ptr + off, a + b)


class ModelNew(nn.Module):
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
        """
        Triton-processed residual block:
        conv1 -> GroupNorm -> SiLU
        conv2 -> GroupNorm -> SiLU
        + residual x

        Note: Convolutions are performed by PyTorch/cuDNN for robustness.
              GroupNorm (statistics + apply), SiLU, and residual addition are Triton kernels.
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels"
        N, C, H, W = x.shape
        assert C % self.num_groups == 0, "C must be divisible by num_groups=32 for GroupNorm"

        # 1st conv via PyTorch (cuDNN)
        # Ensure dtype is float32 for stability; F.conv2d handles strides and padding
        y1 = torch.nn.functional.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # Buffers for GroupNorm stats (float32)
        sums1 = torch.empty(N * self.num_groups * 2, device=x.device, dtype=torch.float32)

        # GroupNorm and SiLU for y1 -> y2 using Triton
        X_sN, X_sC, X_sH, X_sW = y1.stride()
        y2 = torch.empty_like(y1, dtype=torch.float32)

        groupnorm_reduce_sums[(N, self.num_groups)](
            y1, sums1,
            N, C, H, W, self.num_groups,
            X_sN, X_sC, X_sH, X_sW,
            BLOCK_HW=1024,
            num_warps=2
        )

        groupnorm_apply_affine_silu_nc[(N * C)](
            y1, y2, norm1_weight, norm1_bias, sums1,
            N, C, H, W, self.num_groups, self.eps,
            X_sN, X_sC, X_sH, X_sW,
            y2.stride()[0], y2.stride()[1], y2.stride()[2], y2.stride()[3],
            sums1,  # GROUP_sN pointer
            num_warps=2
        )

        # 2nd conv via PyTorch (cuDNN)
        y3 = torch.nn.functional.conv2d(y2, conv2_weight, bias=None, stride=1, padding=1)

        # Buffers for GroupNorm stats
        sums2 = torch.empty(N * self.num_groups * 2, device=x.device, dtype=torch.float32)

        # GroupNorm and SiLU for y3 -> y4 using Triton
        X_sN2, X_sC2, X_sH2, X_sW2 = y3.stride()
        y4 = torch.empty_like(y3, dtype=torch.float32)

        groupnorm_reduce_sums[(N, self.num_groups)](
            y3, sums2,
            N, C, H, W, self.num_groups,
            X_sN2, X_sC2, X_sH2, X_sW2,
            BLOCK_HW=1024,
            num_warps=2
        )

        groupnorm_apply_affine_silu_nc[(N * C)](
            y3, y4, norm2_weight, norm2_bias, sums2,
            N, C, H, W, self.num_groups, self.eps,
            X_sN2, X_sC2, X_sH2, X_sW2,
            y4.stride()[0], y4.stride()[1], y4.stride()[2], y4.stride()[3],
            sums2,  # GROUP_sN pointer
            num_warps=2
        )

        # Residual add via Triton (out = y4 + x)
        add_residual[(N * C * H * W,)](
            x, y4, N, C, H, W,
            num_warps=2
        )

        return y4


def run(*args):
    return ModelNew()(*args)
