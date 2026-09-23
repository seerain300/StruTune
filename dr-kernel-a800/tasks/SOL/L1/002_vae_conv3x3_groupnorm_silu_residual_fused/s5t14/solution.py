import torch
import triton
import triton.language as tl


# Conv3x3: y[n, c_out, h, w] = sum_{c_in, dh, dw} x[n, c_in, h+dh, w+dw] * w[c_out, c_in, dh, dw]
# stride=1, padding=1, no bias
@triton.jit
def conv3x3_per_pixel_kernel(
    x_ptr,             # *f32, input [B, C_in, H, W]
    w_ptr,             # *f32, weight [C_out, C_in, 3, 3]
    y_ptr,             # *f32, output [B, C_out, H, W]
    B: tl.constexpr,   # int
    C_in: tl.constexpr,  # int
    C_out: tl.constexpr, # int
    H: tl.constexpr,   # int
    W: tl.constexpr,   # int
):
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    hw = tl.program_id(2)  # linear index over H*W

    h = hw // W
    w = hw % W

    acc = 0.0
    # Loop over input channels and 3x3 neighborhood
    for c_in in range(0, C_in):
        for dh in range(-1, 2):
            for dw in range(-1, 2):
                in_h = h + dh
                in_w = w + dw
                # NCHW layout: offset = (((n*C_in) + c_in) * H + in_h) * W + in_w
                offset = (((n * C_in) + c_in) * H + in_h) * W + in_w
                x_val = tl.load(x_ptr + offset)
                # weight layout: [C_out, C_in, 3, 3]; contiguous over C_in*9
                weight_idx = c_out * (C_in * 9) + c_in * 9 + (dh + 1) * 3 + (dw + 1)
                w_val = tl.load(w_ptr + weight_idx)
                acc += x_val * w_val

    # Store to output: y[n, c_out, h, w]
    y_offset = (((n * C_out) + c_out) * H + h) * W + w
    tl.store(y_ptr + y_offset, acc)


# GroupNorm reduction per channel: compute mean and rstd over H*W
@triton.jit
def group_norm_reduce_channel_kernel(
    x_ptr,           # *f32, input [B, C, H, W]
    mean_ptr,        # *f32, output [C]
    rstd_ptr,        # *f32, output [C]
    B: tl.constexpr, # int
    C: tl.constexpr, # int
    H: tl.constexpr, # int
    W: tl.constexpr, # int
):
    n = tl.program_id(0)  # not used in reduction; we iterate per channel
    c = tl.program_id(1)

    sum_val = 0.0
    sum_sq = 0.0

    # Iterate over all spatial positions
    for h in range(0, H):
        for w in range(0, W):
            idx = (((n * C) + c) * H + h) * W + w
            val = tl.load(x_ptr + idx)
            sum_val += val
            sum_sq += val * val

    N = H * W
    mean = sum_val / N
    var = sum_sq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)  # eps = 1e-5

    tl.store(mean_ptr + c, mean)
    tl.store(rstd_ptr + c, rstd)


# Apply GroupNorm (per channel) + affine + SiLU: y = silu(((x - mean) * rstd) * weight + bias)
@triton.jit
def group_norm_apply_silu_channel_kernel(
    x_ptr,            # *f32, input [B, C, H, W]
    mean_ptr,         # *f32, [C]
    rstd_ptr,         # *f32, [C]
    weight_ptr,       # *f32, [C] (affine scale)
    bias_ptr,         # *f32, [C] (affine bias)
    y_ptr,            # *f32, output [B, C, H, W]
    B: tl.constexpr,  # int
    C: tl.constexpr,  # int
    H: tl.constexpr,  # int
    W: tl.constexpr,  # int
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    hw = tl.program_id(2)  # linear index over H*W

    h = hw // W
    w = hw % W

    idx = (((n * C) + c) * H + h) * W + w
    x_val = tl.load(x_ptr + idx)

    mean = tl.load(mean_ptr + c)
    rstd = tl.load(rstd_ptr + c)
    scale = tl.load(weight_ptr + c)
    bias = tl.load(bias_ptr + c)

    norm = (x_val - mean) * rstd
    affine = norm * scale + bias

    # SiLU: y = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-affine))
    y_val = affine * sig

    tl.store(y_ptr + idx, y_val)


# SiLU elementwise kernel: y = x * sigmoid(x)
@triton.jit
def silu_kernel(
    x_ptr, y_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    hw = tl.program_id(2)
    h = hw // W
    w = hw % W
    idx = (((n * C) + c) * H + h) * W + w
    x_val = tl.load(x_ptr + idx)
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig
    tl.store(y_ptr + idx, y_val)


# Residual add: y = a + b, elementwise
@triton.jit
def residual_add_kernel(
    a_ptr, b_ptr, y_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    hw = tl.program_id(2)
    h = hw // W
    w = hw % W
    idx = (((n * C) + c) * H + h) * W + w
    a_val = tl.load(a_ptr + idx)
    b_val = tl.load(b_ptr + idx)
    tl.store(y_ptr + idx, a_val + b_val)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        x: (B, C, H, W), conv weights: (C_out, C_in, 3, 3), norm scales/bias: (C,)
        Returns: (B, C_out2, H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"
        # Ensure contiguous float32 for numerical stability
        B, C_in, H, W = x.shape
        x_f32 = x.contiguous().to(torch.float32)

        # First conv: conv1
        C_out1 = conv1_weight.shape[0]
        out1 = torch.empty((B, C_out1, H, W), device=x.device, dtype=torch.float32)
        grid_conv1 = (B, C_out1, H * W)
        conv3x3_per_pixel_kernel[grid_conv1](
            x_f32, conv1_weight.contiguous().to(torch.float32), out1,
            B=B, C_in=C_in, C_out=C_out1, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        # GroupNorm + SiLU for first block
        # GroupNorm in PyTorch requires C % num_groups == 0
        assert C_out1 % self.num_groups == 0, "C_out1 must be divisible by num_groups"
        channels_per_group = C_out1 // self.num_groups

        # Reduce per channel: mean and rstd
        mean1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        rstd1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        grid_reduce1 = (B, C_out1)
        group_norm_reduce_channel_kernel[grid_reduce1](
            out1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        # Apply GroupNorm + affine + SiLU
        out1_norm = torch.empty_like(out1)
        grid_apply1 = (B, C_out1, H * W)
        group_norm_apply_silu_channel_kernel[grid_apply1](
            out1, mean1, rstd1, norm1_weight.to(torch.float32).contiguous(), norm1_bias.to(torch.float32).contiguous(), out1_norm,
            B=B, C=C_out1, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        # Second conv: conv2
        C_out2 = conv2_weight.shape[0]
        out2 = torch.empty((B, C_out2, H, W), device=x.device, dtype=torch.float32)
        grid_conv2 = (B, C_out2, H * W)
        conv3x3_per_pixel_kernel[grid_conv2](
            out1_norm, conv2_weight.contiguous().to(torch.float32), out2,
            B=B, C_in=C_out1, C_out=C_out2, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        # GroupNorm + SiLU for second block
        assert C_out2 % self.num_groups == 0, "C_out2 must be divisible by num_groups"
        channels_per_group2 = C_out2 // self.num_groups

        mean2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        rstd2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        grid_reduce2 = (B, C_out2)
        group_norm_reduce_channel_kernel[grid_reduce2](
            out2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        out2_norm = torch.empty_like(out2)
        grid_apply2 = (B, C_out2, H * W)
        group_norm_apply_silu_channel_kernel[grid_apply2](
            out2, mean2, rstd2, norm2_weight.to(torch.float32).contiguous(), norm2_bias.to(torch.float32).contiguous(), out2_norm,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        # Final residual add
        out = torch.empty_like(out2_norm)
        grid_add = (B, C_out2, H * W)
        residual_add_kernel[grid_add](
            out2_norm, x_f32, out,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
