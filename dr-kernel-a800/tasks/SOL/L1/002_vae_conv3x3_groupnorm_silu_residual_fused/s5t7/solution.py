import torch
import triton
import triton.language as tl


# Conv3x3: y[n, c_out, h_out, w_out] = sum_{c_in=0..C_in-1} sum_{dh,dw in 3x3} x[n, c_in, h_out+dh, w_out+dw] * w[c_out, c_in, 3+dh, 3+dw]
# Stride=1, padding=1, no bias. Each program computes one output pixel.
@triton.jit
def conv3x3_simplified_kernel(
    x_ptr, w_ptr, y_ptr,
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    h_out = tl.program_id(2)
    w_out = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 neighborhood
    for c_in in range(C_in):
        for dh in range(3):
            h = h_out + dh
            for dw in range(3):
                w = w_out + dw
                x_offset = ((n * C_in) + c_in) * H * W + h * W + w
                w_offset = c_out * (C_in * 9) + (c_in * 9 + dh * 3 + dw)
                x_val = tl.load(x_ptr + x_offset)
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    y_offset = ((n * C_out) + c_out) * H * W + h_out * W + w_out
    tl.store(y_ptr + y_offset, acc)


# GroupNorm reduction: compute per-channel mean and rstd across spatial H*W for given (n, group, channel).
# Grid: (B, num_groups, channels_per_group), vectorized over HW without loops.
@triton.jit
def group_norm_reduce_kernel(
    x_ptr, mean_ptr, rstd_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, channels_per_group: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    c = tl.program_id(2)  # channel within the group

    c_idx = g * channels_per_group + c
    HW = H * W
    offs = tl.arange(0, HW)
    h = offs // W
    w = offs % W

    base = ((n * C) + c_idx) * HW
    x = tl.load(x_ptr + base + offs)

    s = tl.sum(x, axis=0)
    ss = tl.sum(x * x, axis=0)

    mean = s / (HW * 1.0)
    var = ss / (HW * 1.0) - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    tl.store(mean_ptr + c_idx, mean)
    tl.store(rstd_ptr + c_idx, rstd)


# GroupNorm apply: y = ((x - mean) * rstd) * gamma + beta, then SiLU
# Grid: (B, num_groups, channels_per_group), vectorized over HW.
@triton.jit
def group_norm_apply_kernel(
    x_ptr, mean_ptr, rstd_ptr, weight_ptr, bias_ptr, y_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, channels_per_group: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    c = tl.program_id(2)  # channel within the group
    c_idx = g * channels_per_group + c

    HW = H * W
    offs = tl.arange(0, HW)
    h = offs // W
    w = offs % W

    base = ((n * C) + c_idx) * HW
    x = tl.load(x_ptr + base + offs)
    mean = tl.load(mean_ptr + c_idx)
    rstd = tl.load(rstd_ptr + c_idx)
    gamma = tl.load(weight_ptr + c_idx)
    beta = tl.load(bias_ptr + c_idx)

    y = (x - mean) * rstd
    y = y * gamma + beta  # affine
    # SiLU: y = y * sigmoid(y)
    sig = 1.0 / (1.0 + tl.exp(-y))
    y = y * sig

    tl.store(y_ptr + base + offs, y)


# Elementwise residual add: out = y + x
@triton.jit
def residual_add_kernel(
    a_ptr, b_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    hw = tl.program_id(2)
    h = hw // W
    w = hw % W

    idx = ((n * C) + c) * H * W + h * W + w
    a = tl.load(a_ptr + idx)
    b = tl.load(b_ptr + idx)
    tl.store(out_ptr + idx, a + b)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        x: (B, C_in, H, W), conv weights: (C_out, C_in, 3, 3), norm scales/bias: (C,)
        Returns: (B, conv2_weight.shape[0], H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"
        B, C_in, H, W = x.shape

        # Cast to float32 for numerical stability
        x_f32 = x.contiguous().to(torch.float32)

        # Prepare weights as float32 contiguous
        conv1_w_f32 = conv1_weight.contiguous().to(torch.float32)  # (C_out1, C_in, 3, 3)
        conv2_w_f32 = conv2_weight.contiguous().to(torch.float32)  # (C_out2, C_out1, 3, 3)
        norm1_weight_f32 = norm1_weight.contiguous().to(torch.float32)
        norm1_bias_f32 = norm1_bias.contiguous().to(torch.float32)
        norm2_weight_f32 = norm2_weight.contiguous().to(torch.float32)
        norm2_bias_f32 = norm2_bias.contiguous().to(torch.float32)

        # First path: Conv3x3
        C_out1 = conv1_w_f32.shape[0]
        y1 = torch.empty((B, C_out1, H, W), device=x.device, dtype=torch.float32)
        grid_conv1 = (B, C_out1, H, W)
        conv3x3_simplified_kernel[grid_conv1](
            x_f32, conv1_w_f32, y1,
            B=B, C_in=C_in, C_out=C_out1, H=H, W=W,
            num_warps=1, num_stages=2,
        )

        # GroupNorm + SiLU for first block
        channels_per_group1 = C_out1 // self.num_groups
        mean1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        rstd1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        grid_reduce1 = (B, self.num_groups, channels_per_group1)
        group_norm_reduce_kernel[grid_reduce1](
            y1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group1,
            num_warps=4, num_stages=2,
        )
        y1_norm = torch.empty_like(y1)
        grid_apply1 = (B, self.num_groups, channels_per_group1)
        group_norm_apply_kernel[grid_apply1](
            y1, mean1, rstd1, norm1_weight_f32, norm1_bias_f32, y1_norm,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group1,
            num_warps=4, num_stages=2,
        )

        # Second path: Conv3x3
        C_out2 = conv2_w_f32.shape[0]
        y2 = torch.empty((B, C_out2, H, W), device=x.device, dtype=torch.float32)
        grid_conv2 = (B, C_out2, H, W)
        conv3x3_simplified_kernel[grid_conv2](
            y1_norm, conv2_w_f32, y2,
            B=B, C_in=C_out1, C_out=C_out2, H=H, W=W,
            num_warps=1, num_stages=2,
        )

        # GroupNorm + SiLU for second block
        mean2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        rstd2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        grid_reduce2 = (B, self.num_groups, C_out2 // self.num_groups)
        group_norm_reduce_kernel[grid_reduce2](
            y2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=(C_out2 // self.num_groups),
            num_warps=4, num_stages=2,
        )
        y2_norm = torch.empty_like(y2)
        grid_apply2 = (B, self.num_groups, C_out2 // self.num_groups)
        group_norm_apply_kernel[grid_apply2](
            y2, mean2, rstd2, norm2_weight_f32, norm2_bias_f32, y2_norm,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=(C_out2 // self.num_groups),
            num_warps=4, num_stages=2,
        )

        # Final residual add: y2_norm + x_f32
        out = torch.empty_like(y2_norm)
        grid_add = (B, C_out2, H * W)
        residual_add_kernel[grid_add](
            y2_norm, x_f32, out,
            B=B, C=C_out2, H=H, W=W,
            num_warps=2, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
