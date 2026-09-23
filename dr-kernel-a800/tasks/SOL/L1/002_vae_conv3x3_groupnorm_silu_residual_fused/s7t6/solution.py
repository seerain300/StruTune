import torch
import triton
import triton.language as tl


@triton.jit
def groupnorm_affine_kernel(
    x_ptr,            # *float32 input tensor (B, C, H, W)
    gamma_ptr,        # *float32 per-channel scale (C,)
    beta_ptr,         # *float32 per-channel bias (C,)
    y_ptr,            # *float32 output tensor (B, C, H, W)
    B,                # int
    C,                # int
    H,                # int
    W,                # int
    num_groups: tl.constexpr,  # int (32)
    eps,              # float
):
    # Grid: (B, num_groups)
    n = tl.program_id(0)
    g = tl.program_id(1)

    channels_per_group = C // num_groups
    group_start = g * channels_per_group

    # Compute sum and sum of squares over group and spatial
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for cin in range(channels_per_group):
        c = group_start + cin
        for h in range(H):
            for w in range(W):
                x_index = (((n * C + c) * H + h) * W + w)
                x_val = tl.load(x_ptr + x_index)
                sum_val += x_val
                sum_sq += x_val * x_val

    m = channels_per_group * H * W
    mean = sum_val / m
    var = sum_sq / m - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine per channel
    for cin in range(channels_per_group):
        c = group_start + cin
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)
        for h in range(H):
            for w in range(W):
                x_index = (((n * C + c) * H + h) * W + w)
                x_val = tl.load(x_ptr + x_index)
                y_val = (x_val - mean) * inv_std
                y_val = y_val * gamma + beta
                tl.store(y_ptr + x_index, y_val)


@triton.jit
def silu_kernel(
    x_ptr,            # *float32 input tensor (B, C, H, W)
    y_ptr,            # *float32 output tensor (B, C, H, W)
    B,                # int
    C,                # int
    H,                # int
    W,                # int
):
    # Grid: (B, C, H, W)
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    x_index = (((n * C + c) * H + h) * W + w)
    x_val = tl.load(x_ptr + x_index)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig
    tl.store(y_ptr + x_index, y_val)


@triton.jit
def add_residual_kernel(
    y_ptr,            # *float32 first tensor (B, C, H, W)
    x_ptr,            # *float32 second tensor (B, C, H, W)
    out_ptr,          # *float32 output tensor (B, C, H, W)
    B,                # int
    C,                # int
    H,                # int
    W,                # int
):
    # Grid: (B, C, H, W)
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    x_index = (((n * C + c) * H + h) * W + w)
    y_index = x_index
    y_val = tl.load(y_ptr + y_index)
    x_val = tl.load(x_ptr + y_index)
    out_val = y_val + x_val
    tl.store(out_ptr + y_index, out_val)


class ModelNew(torch.nn.Module):
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
        """
        Fused residual block:
        Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add (x)
        Triton kernels are used for GroupNorm, SiLU, and residual add.
        Convolution is performed using torch.nn.functional.conv2d for correctness.
        """
        assert x.is_cuda, "Input tensor must be on CUDA device for Triton kernels."
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Weights must be on CUDA device."

        N, C, H, W = x.shape

        # 1) Conv1: F.conv2d(x, conv1_weight, stride=1, padding=1, bias=None)
        out1 = torch.nn.functional.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # Enforce GroupNorm divisibility requirement
        num_groups = 32
        assert out1.shape[1] % num_groups == 0 and out1.shape[1] % num_groups == 0, "num_groups=32 must divide channels for GroupNorm."

        # 2) GroupNorm1 (num_groups=32, per-channel affine), eps
        y2 = torch.empty_like(out1)
        groupnorm_affine_kernel[(N, num_groups)](
            out1, norm1_weight, norm1_bias, y2,
            N, out1.shape[1], out1.shape[2], out1.shape[3],
            num_groups, eps,
            num_warps=4, num_stages=2,
        )

        # 3) SiLU1
        y3 = torch.empty_like(y2)
        silu_kernel[(N, out1.shape[1], out1.shape[2], out1.shape[3])](
            y2, y3, N, out1.shape[1], out1.shape[2], out1.shape[3],
            num_warps=4, num_stages=2,
        )

        # 4) Conv2: F.conv2d(y3, conv2_weight, stride=1, padding=1, bias=None)
        out2 = torch.nn.functional.conv2d(y3, conv2_weight, bias=None, stride=1, padding=1)

        # 5) GroupNorm2 (num_groups=32, per-channel affine), eps
        y5 = torch.empty_like(out2)
        groupnorm_affine_kernel[(N, num_groups)](
            out2, norm2_weight, norm2_bias, y5,
            N, out2.shape[1], out2.shape[2], out2.shape[3],
            num_groups, eps,
            num_warps=4, num_stages=2,
        )

        # 6) SiLU2
        y6 = torch.empty_like(y5)
        silu_kernel[(N, out2.shape[1], out2.shape[2], out2.shape[3])](
            y5, y6, N, out2.shape[1], out2.shape[2], out2.shape[3],
            num_warps=4, num_stages=2,
        )

        # 7) Add residual x
        y_out = torch.empty_like(y6)
        add_residual_kernel[(N, y6.shape[1], y6.shape[2], y6.shape[3])](
            y6, x, y_out,
            N, y6.shape[1], y6.shape[2], y6.shape[3],
            num_warps=4, num_stages=2,
        )

        return y_out


def run(*args):
    return ModelNew()(*args)
