import torch
import torch.nn as nn

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def conv3x3_no_bias_nchw(
    x_ptr,          # *f32, input tensor (B, C_in, H, W)
    w_ptr,          # *f32, weight tensor (C_out, C_in, 3, 3)
    out_ptr,        # *f32, output tensor (B, C_out, H, W)
    B: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    sH: tl.constexpr,  # stride (here 1)
    sW: tl.constexpr,  # stride (here 1)
    p: tl.constexpr,   # padding (here 1)
):
    # Program IDs: (n, co, h, w)
    n = tl.program_id(0)
    co = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    if (n >= B) or (co >= C_out) or (h >= H) or (w >= W):
        return

    # Accumulator for output value at (n, co, h, w)
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 neighborhood
    for ci in range(C_in):
        for dh in range(3):
            ih = h + dh - p  # ih can be negative or >= H; masked by bounds below
            for dw in range(3):
                iw = w + dw - p
                # Check bounds
                if (0 <= ih < H) and (0 <= iw < W):
                    # Linear index for x[n, ci, ih, iw]
                    x_idx = ((n * C_in + ci) * H + ih) * W + iw
                    # Load x
                    x_val = tl.load(x_ptr + x_idx)
                    # Linear index for w[co, ci, dh, dw]
                    w_idx = ((co * C_in) + ci) * 9 + (dh * 3 + dw)
                    w_val = tl.load(w_ptr + w_idx)
                    acc += x_val * w_val

    # Store output
    out_idx = ((n * C_out + co) * H + h) * W + w
    tl.store(out_ptr + out_idx, acc)


@triton.jit
def group_norm_triton_kernel(
    y_ptr,               # *f32, input
    out_ptr,             # *f32, output
    gamma_ptr,           # *f32, per-channel scale
    beta_ptr,            # *f32, per-channel bias
    B: tl.constexpr,     # int
    C: tl.constexpr,     # int
    H: tl.constexpr,     # int
    W: tl.constexpr,     # int
    num_groups: tl.constexpr,  # int, here 32
    eps: tl.constexpr,          # float
):
    # Each program instance handles one (n, group)
    n = tl.program_id(0)
    group_id = tl.program_id(1)
    if n >= B:
        return

    # Compute group size in channels (constant per group)
    channels_per_group = C // num_groups

    # First pass: compute sum and sum of squares per channel in the group
    sum_vec = tl.zeros((channels_per_group,), dtype=tl.float32)
    sumsq_vec = tl.zeros((channels_per_group,), dtype=tl.float32)

    # Loop over channels within the group
    for ch_in_group in range(channels_per_group):
        c = group_id * channels_per_group + ch_in_group

        hw_count = H * W
        total = 0.0
        total_sq = 0.0
        # Accumulate over spatial positions
        for hw in range(hw_count):
            h = hw // W
            w = hw % W
            idx = ((n * C + c) * (H * W)) + hw
            val = tl.load(y_ptr + idx)
            total += val
            total_sq += val * val

        sum_vec[ch_in_group] = total
        sumsq_vec[ch_in_group] = total_sq

    # Compute mean and rstd per channel in the group
    hw_per_group = H * W
    count_per_channel = hw_per_group * channels_per_group
    mean_vec = sum_vec / count_per_channel
    var_vec = sumsq_vec / count_per_channel - mean_vec * mean_vec
    rstd_vec = 1.0 / tl.sqrt(var_vec + eps)

    # Second pass: normalize and apply affine
    for ch_in_group in range(channels_per_group):
        c = group_id * channels_per_group + ch_in_group
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)

        for hw in range(hw_count):
            h = hw // W
            w = hw % W
            idx = ((n * C + c) * (H * W)) + hw
            val = tl.load(y_ptr + idx)
            normalized = (val - mean_vec[ch_in_group]) * rstd_vec[ch_in_group]
            out_val = normalized * gamma + beta
            tl.store(out_ptr + idx, out_val)


@triton.jit
def silu_triton_kernel(x_ptr, out_ptr, N: tl.constexpr):
    # Elementwise y = x * sigmoid(x)
    idx = tl.program_id(0)
    if idx >= N:
        return
    x = tl.load(x_ptr + idx)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + idx, y)


@triton.jit
def add_residual_triton_kernel(x_ptr, out_ptr, N: tl.constexpr):
    # Elementwise out = out + x
    idx = tl.program_id(0)
    if idx >= N:
        return
    x = tl.load(x_ptr + idx)
    out = tl.load(out_ptr + idx)
    out = out + x
    tl.store(out_ptr + idx, out)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        """
        Triton-only implementation of the fused residual block:
        Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        No torch.nn.functional calls; all compute is in Triton kernels.
        """
        assert TRITON_AVAILABLE, "Triton is not available"
        assert x.is_cuda, "Input tensor must be on CUDA for Triton"
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Weights must be on CUDA for Triton"
        assert x.dtype == torch.float32 and conv1_weight.dtype == torch.float32 and conv2_weight.dtype == torch.float32, "Expected float32 tensors"

        # Ensure contiguity
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        B, C, H, W = x.shape
        Cw1, Cin1, kH, kW = conv1_weight.shape
        Cw2, Cin2, kH2, kW2 = conv2_weight.shape
        assert Cin1 == C and kH == 3 and kW == 3, "conv1_weight must be (C, C, 3, 3)"
        assert Cin2 == Cw1 and kH2 == 3 and kW2 == 3, "conv2_weight must be (C, C, 3, 3)"

        # Output sizes for convolutions (stride=1, padding=1)
        H_out1 = (H + 2 * 1 - 3) // 1 + 1  # equals H
        W_out1 = (W + 2 * 1 - 3) // 1 + 1  # equals W
        H_out2 = (H_out1 + 2 * 1 - 3) // 1 + 1  # equals H_out1
        W_out2 = (W_out1 + 2 * 1 - 3) // 1 + 1  # equals W_out1

        # Allocate outputs
        out1 = torch.empty((B, C, H_out1, W_out1), dtype=torch.float32, device=x.device)
        out2 = torch.empty((B, C, H_out2, W_out2), dtype=torch.float32, device=x.device)

        # Launch first conv (no bias)
        grid1 = (B, C, H_out1, W_out1)
        conv3x3_no_bias_nchw[grid1](
            x, conv1_weight, out1,
            B, C, C, H, W,
            sH=1, sW=1, p=1,
            num_warps=4, num_stages=2
        )

        # GroupNorm 1 (Triton)
        out1_norm = torch.empty_like(out1)
        grid_gn1 = (B, 32)
        group_norm_triton_kernel[grid_gn1](
            out1, out1_norm, norm1_weight, norm1_bias,
            B, C, H_out1, W_out1, 32, eps,
            num_warps=4, num_stages=2
        )

        # SiLU 1 (Triton)
        out1_silu = torch.empty_like(out1_norm)
        N1 = out1_norm.numel()
        grid_silu1 = (N1,)
        silu_triton_kernel[grid_silu1](
            out1_norm, out1_silu,
            N1,
            num_warps=4, num_stages=2
        )

        # Launch second conv (no bias)
        grid2 = (B, C, H_out2, W_out2)
        conv3x3_no_bias_nchw[grid2](
            out1_silu, conv2_weight, out2,
            B, C, C, H_out1, W_out1,
            sH=1, sW=1, p=1,
            num_warps=4, num_stages=2
        )

        # GroupNorm 2 (Triton)
        out2_norm = torch.empty_like(out2)
        grid_gn2 = (B, 32)
        group_norm_triton_kernel[grid_gn2](
            out2, out2_norm, norm2_weight, norm2_bias,
            B, C, H_out2, W_out2, 32, eps,
            num_warps=4, num_stages=2
        )

        # SiLU 2 (Triton)
        out2_silu = torch.empty_like(out2_norm)
        N2 = out2_norm.numel()
        grid_silu2 = (N2,)
        silu_triton_kernel[grid_silu2](
            out2_norm, out2_silu,
            N2,
            num_warps=4, num_stages=2
        )

        # Residual add: out2_silu += x
        N = x.numel()
        grid_add = (N,)
        # Create a temporary buffer x_buf to pass to Triton
        x_buf = torch.empty_like(x)
        torch.copy_(x, x_buf)
        add_residual_triton_kernel[grid_add](
            x_buf, out2_silu,
            N,
            num_warps=4, num_stages=2
        )

        return out2_silu


def run(*args):
    return ModelNew()(*args)
