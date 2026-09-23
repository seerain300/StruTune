import torch
import triton
import triton.language as tl

# 3x3 Convolution (NCHW, stride=1, padding=1, no bias) Triton kernel
@triton.jit
def conv3x3_nchw_nobias(x_ptr, w_ptr, y_ptr,
                         B, C_in, C_out, H, W, H_out, W_out,
                         x_stride_n, x_stride_c, x_stride_h, x_stride_w,
                         w_stride_co, w_stride_ci, w_stride_dh, w_stride_dw,
                         y_stride_n, y_stride_c, y_stride_h, y_stride_w,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)  # output height
    pid_w = tl.program_id(3)  # output width

    oh = pid_h
    ow = pid_w

    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels
    for ci in range(0, C_in):
        # loop over 3x3 neighborhood
        for dh in range(0, 3):
            for dw in range(0, 3):
                ih = oh + dh - 1  # kernel center at (1,1)
                iw = ow + dw - 1
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x_off = pid_n * x_stride_n + ci * x_stride_c + ih * x_stride_h + iw * x_stride_w
                x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)
                w_off = pid_co * w_stride_co + ci * w_stride_ci + dh * w_stride_dh + dw * w_stride_dw
                w_val = tl.load(w_ptr + w_off)
                acc += x_val * w_val

    y_off = pid_n * y_stride_n + pid_co * y_stride_c + oh * y_stride_h + ow * y_stride_w
    tl.store(y_ptr + y_off, acc)


# GroupNorm Triton kernel for fixed num_groups=32:
# For each (n, group), compute mean/var across all channels in the group and all spatial positions,
# then normalize and apply per-channel weight/bias.
@triton.jit
def group_norm_triton_fixed(y_ptr, weight_ptr, bias_ptr,
                             B, C, H_out, W_out,
                             eps,
                             CH_PER_GROUP: tl.constexpr,
                             y_stride_n, y_stride_c, y_stride_h, y_stride_w,
                             num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid_n = tl.program_id(0)  # batch
    pid_g = tl.program_id(1)  # group index [0..31]
    # First, compute mean and variance for each channel in the group
    for c_off in range(0, CH_PER_GROUP):
        c = pid_g * CH_PER_GROUP + c_off
        M = H_out * W_out
        sum_val = tl.zeros((), dtype=tl.float32)
        sum_sq = tl.zeros((), dtype=tl.float32)
        for k in range(0, M):
            h = k // W_out
            w = k % W_out
            y_off = pid_n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
            y_val = tl.load(y_ptr + y_off)
            sum_val += y_val
            sum_sq += y_val * y_val
        mean = sum_val / M
        var = sum_sq / M - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)
        gamma = tl.load(weight_ptr + c)
        beta = tl.load(bias_ptr + c)
        # Normalize and write back
        for k in range(0, M):
            h = k // W_out
            w = k % W_out
            y_off = pid_n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
            y_val = tl.load(y_ptr + y_off)
            y_norm = (y_val - mean) * rstd
            y_out = y_norm * gamma + beta
            tl.store(y_ptr + y_off, y_out)


# SiLU elementwise Triton kernel
@triton.jit
def silu_triton(x_ptr, y_ptr, N, num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid
    x_val = tl.load(x_ptr + idx)
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig
    tl.store(y_ptr + idx, y_val)


# Residual addition Triton kernel: y = y + x, elementwise
@triton.jit
def add_residual_triton(y_ptr, x_ptr, N, num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid
    y_val = tl.load(y_ptr + idx)
    x_val = tl.load(x_ptr + idx)
    y_sum = y_val + x_val
    tl.store(y_ptr + idx, y_sum)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor):
        # Enforce dtype and contiguity
        assert x.dtype == torch.float32, "Input x must be float32"
        x = x.contiguous()
        device = x.device

        B, C, H, W = x.shape
        assert C % 32 == 0, "C must be divisible by num_groups=32"
        CH_PER_GROUP = C // 32
        H1 = H - 2
        W1 = W - 2

        # First conv: y1 = conv3x3(x, conv1_weight) -> (B, C, H-2, W-2)
        y1 = torch.empty((B, C, H1, W1), dtype=torch.float32, device=device)
        grid1 = (B, C, H1, W1)
        conv3x3_nchw_nobias[grid1](
            x, conv1_weight, y1,
            B, C, C, H, W, H1, W1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=4, num_stages=2
        )

        # GroupNorm1 (num_groups=32), per-channel scale/bias, per-group reduction across channels and spatial
        y1_norm = torch.empty_like(y1, dtype=torch.float32, device=device)
        grid_g1 = (B, 32)
        group_norm_triton_fixed[grid_g1](
            y1, norm1_weight, norm1_bias,
            B, C, H1, W1, self.eps,
            CH_PER_GROUP=CH_PER_GROUP,
            y_stride_n=y1.stride(0), y_stride_c=y1.stride(1), y_stride_h=y1.stride(2), y_stride_w=y1.stride(3),
            num_warps=4, num_stages=2
        )
        # SiLU1
        y1_silu = torch.empty_like(y1_norm, dtype=torch.float32, device=device)
        N1 = y1_norm.numel()
        grid_s1 = (N1,)
        silu_triton[grid_s1](y1_norm, y1_silu, N1, num_warps=4, num_stages=2)

        # Second conv: y2 = conv3x3(y1_silu, conv2_weight) -> (B, C, H-4, W-4)
        H2 = H1 - 2
        W2 = W1 - 2
        y2 = torch.empty((B, C, H2, W2), dtype=torch.float32, device=device)
        grid2 = (B, C, H2, W2)
        conv3x3_nchw_nobias[grid2](
            y1_silu, conv2_weight, y2,
            B, C, C, H1, W1, H2, W2,
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=4, num_stages=2
        )

        # GroupNorm2
        y2_norm = torch.empty_like(y2, dtype=torch.float32, device=device)
        grid_g2 = (B, 32)
        group_norm_triton_fixed[grid_g2](
            y2, norm2_weight, norm2_bias,
            B, C, H2, W2, self.eps,
            CH_PER_GROUP=CH_PER_GROUP,
            y_stride_n=y2.stride(0), y_stride_c=y2.stride(1), y_stride_h=y2.stride(2), y_stride_w=y2.stride(3),
            num_warps=4, num_stages=2
        )

        # SiLU2
        y2_silu = torch.empty_like(y2_norm, dtype=torch.float32, device=device)
        N2 = y2_norm.numel()
        grid_s2 = (N2,)
        silu_triton[grid_s2](y2_norm, y2_silu, N2, num_warps=4, num_stages=2)

        # Residual add: y = y2_silu + x (elementwise)
        y_out = torch.empty_like(y2_silu, dtype=torch.float32, device=device)
        N_add = x.numel()
        grid_add = (N_add,)
        add_residual_triton[grid_add](y2_silu, x, N_add, num_warps=4, num_stages=2)
        # write back
        torch.copy_(y2_silu, y_out)

        return y_out


def run(*args):
    return ModelNew()(*args)
