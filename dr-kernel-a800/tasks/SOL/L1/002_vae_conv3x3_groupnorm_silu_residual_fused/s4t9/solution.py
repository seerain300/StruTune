import torch
import triton
import triton.language as tl

@triton.jit
def conv3x3_nchw_nobias(x_ptr, w_ptr, y_ptr,
                         B, C_in, C_out, H, W, H_out, W_out,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    # program ids
    pid_n = tl.program_id(0)  # batch
    pid_co = tl.program_id(1)  # output channel index
    pid_h = tl.program_id(2)   # output height index
    pid_w = tl.program_id(3)   # output width index

    # Accumulator for output
    acc = 0.0
    # Loop over input channels and 3x3 neighborhood
    for ci in range(C_in):
        for dh in range(3):
            for dw in range(3):
                # compute input spatial coordinate with padding=1
                h_in = pid_h + dh - 1
                w_in = pid_w + dw - 1
                # valid if h_in in [0..H-1] and w_in in [0..W-1]
                valid = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                # input pointer for x[n, ci, h_in, w_in]
                x_off = (pid_n * C_in + ci) * (H * W) + h_in * W + w_in
                x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                # weight pointer for w[co, ci, dh, dw]
                w_off = (co * C_in + ci) * 9 + (dh * 3 + dw)
                w_val = tl.load(w_ptr + w_off)
                acc += x_val * w_val
    # store result to y[n, co, h, w]
    y_off = (pid_n * C_out + pid_co) * (H_out * W_out) + pid_h * W_out + pid_w
    tl.store(y_ptr + y_off, acc)


@triton.jit
def group_norm_triton(y_ptr, weight_ptr, bias_ptr, y_norm_ptr,
                       B, C, H, W, num_groups, eps,
                       num_warps: tl.constexpr, num_stages: tl.constexpr):
    # One program per (n, group)
    n = tl.program_id(0)
    g = tl.program_id(1)
    channels_per_group = C // num_groups
    group_start_c = g * channels_per_group

    # First pass: compute per-channel sum and sumsq across the group's channels and spatial
    sum_c = tl.zeros((channels_per_group,), dtype=tl.float32)
    sumsq_c = tl.zeros((channels_per_group,), dtype=tl.float32)

    for ch in range(channels_per_group):
        c = group_start_c + ch
        base = (n * C + c) * (H * W)
        for h in range(H):
            for w in range(W):
                idx = base + h * W + w
                x = tl.load(y_ptr + idx)
                sum_c[ch] += x
                sumsq_c[ch] += x * x

    # Compute mean and variance per channel across group
    M = H * W
    mean = sum_c / M
    var = sumsq_c / M - mean * mean  # per-channel variance over group
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine per-channel
    for ch in range(channels_per_group):
        c = group_start_c + ch
        base = (n * C + c) * (H * W)
        scale = tl.load(weight_ptr + c)
        bias = tl.load(bias_ptr + c)
        for h in range(H):
            for w in range(W):
                idx = base + h * W + w
                x = tl.load(y_ptr + idx)
                y = (x - mean[ch]) * rstd[ch]
                y = y * scale + bias
                tl.store(y_norm_ptr + idx, y)


@triton.jit
def silu_triton(x_ptr, y_ptr, N,
                num_warps: tl.constexpr, num_stages: tl.constexpr):
    # Elementwise: y = x * sigmoid(x)
    for idx in range(N):
        x = tl.load(x_ptr + idx)
        sig = 1.0 / (1.0 + tl.exp(-x))
        y = x * sig
        tl.store(y_ptr + idx, y)


@triton.jit
def add_residual_triton(y_ptr, x_ptr, y_out_ptr, N,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    # Elementwise: y_out = y + x
    for idx in range(N):
        y = tl.load(y_ptr + idx)
        x = tl.load(x_ptr + idx)
        tl.store(y_out_ptr + idx, y + x)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        """
        Fused residual block:
        conv1: 3x3, no bias, stride=1, padding=1
        GroupNorm1: num_groups=32, per-channel scale/bias, per-channel variance
        SiLU1
        conv2: 3x3, no bias, stride=1, padding=1
        GroupNorm2: num_groups=32, per-channel scale/bias
        SiLU2
        Residual: +x
        All computations are done in Triton kernels; no torch operations.
        """
        # Ensure contiguity
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        B, C, H, W = x.shape
        assert C % 32 == 0, "C must be divisible by num_groups=32"
        # First conv output spatial size
        H_out1 = H - 2
        W_out1 = W - 2
        # Allocate y1
        y1 = torch.empty((B, C, H_out1, W_out1), dtype=torch.float32, device=x.device)

        # Launch conv1
        grid1 = (B, C, H_out1, W_out1)
        conv3x3_nchw_nobias[grid1](
            x, conv1_weight, y1,
            B, C, C, H, W, H_out1, W_out1,
            num_warps=4, num_stages=2
        )

        # GroupNorm1
        y1_norm = torch.empty_like(y1)
        grid_gn1 = (B, 32)
        group_norm_triton[grid_gn1](
            y1, norm1_weight, norm1_bias, y1_norm,
            B, C, H_out1, W_out1, 32, eps,
            num_warps=4, num_stages=2
        )

        # SiLU1
        y1_silu = torch.empty_like(y1_norm)
        N1 = y1_norm.numel()
        grid_silu1 = (N1,)
        silu_triton[grid_silu1](
            y1_norm, y1_silu, N1,
            num_warps=4, num_stages=2
        )

        # Second conv output spatial size
        H_out2 = H_out1 - 2
        W_out2 = W_out1 - 2
        # Allocate y2
        y2 = torch.empty((B, C, H_out2, W_out2), dtype=torch.float32, device=x.device)

        # Launch conv2
        grid2 = (B, C, H_out2, W_out2)
        conv3x3_nchw_nobias[grid2](
            y1_silu, conv2_weight, y2,
            B, C, C, H_out1, W_out1, H_out2, W_out2,
            num_warps=4, num_stages=2
        )

        # GroupNorm2
        y2_norm = torch.empty_like(y2)
        grid_gn2 = (B, 32)
        group_norm_triton[grid_gn2](
            y2, norm2_weight, norm2_bias, y2_norm,
            B, C, H_out2, W_out2, 32, eps,
            num_warps=4, num_stages=2
        )

        # SiLU2
        y2_silu = torch.empty_like(y2_norm)
        N2 = y2_norm.numel()
        grid_silu2 = (N2,)
        silu_triton[grid_silu2](
            y2_norm, y2_silu, N2,
            num_warps=4, num_stages=2
        )

        # Residual add: y = y2_silu + x (elementwise)
        y_out = torch.empty_like(x)  # output is same shape as input
        N_add = x.numel()
        grid_add = (N_add,)
        add_residual_triton[grid_add](
            y2_silu, x, y_out, N_add,
            num_warps=4, num_stages=2
        )

        return y_out


def run(*args):
    return ModelNew()(*args)
