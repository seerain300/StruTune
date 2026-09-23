import torch
import triton
import triton.language as tl


# Conv3x3: y = x * w, stride=1, padding=1, no bias
@triton.jit
def conv3x3_kernel(
    x_ptr,           # *f32, input [B, C_in, H, W]
    w_ptr,           # *f32, weight [C_out, C_in, 3, 3]
    y_ptr,           # *f32, output [B, C_out, H, W]
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
):
    # Grid: (B, C_out, H*W)
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    pos = tl.program_id(2)  # linear index over H*W
    h_out = pos // W
    w_out = pos % W

    acc = tl.zeros((), dtype=tl.float32)

    # Accumulate over input channels and 3x3 neighborhood
    for c_in in range(0, C_in):
        for dh in range(-1, 2):
            h_in = h_out + dh
            for dw in range(-1, 2):
                w_in = w_out + dw
                # Compute input offset: (((n * C_in) + c_in) * H + h_in) * W + w_in
                x_off = (((n * C_in) + c_in) * H + h_in) * W + w_in
                x_val = tl.load(x_ptr + x_off)
                # Compute weight offset: (((c_out * C_in) + c_in) * 9 + (dh+1)*3 + (dw+1))
                w_off = (((c_out * C_in) + c_in) * 9) + (dh + 1) * 3 + (dw + 1)
                w_val = tl.load(w_ptr + w_off)
                acc += x_val * w_val

    # Store output
    y_off = (((n * C_out) + c_out) * H + h_out) * W + w_out
    tl.store(y_ptr + y_off, acc)


# GroupNorm reduction: per-channel (n,g) compute mean and rstd over all HW
@triton.jit
def group_norm_reduce_kernel(
    x_ptr,           # *f32, input [B, C, H, W]
    mean_ptr,        # *f32, output [C]
    rstd_ptr,        # *f32, output [C]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, eps: tl.constexpr,
):
    # grid: (B * num_groups * (C // num_groups))
    pid = tl.program_id(0)
    groups = num_groups
    channels_per_group = C // groups
    n = pid // (groups * channels_per_group)
    g = (pid % (groups * channels_per_group)) // channels_per_group
    c = (pid % channels_per_group) + g * channels_per_group

    total = H * W
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for h in range(0, H):
        for w in range(0, W):
            x_idx = (((n * C) + c) * H + h) * W + w
            x_val = tl.load(x_ptr + x_idx)
            sum_val += x_val
            sum_sq += x_val * x_val

    mean = sum_val / total
    var = sum_sq / total - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + c, mean)
    tl.store(rstd_ptr + c, rstd)


# GroupNorm apply: normalize + affine + SiLU
@triton.jit
def group_norm_apply_kernel(
    x_ptr,           # *f32, input [B, C, H, W]
    mean_ptr,        # *f32, [C]
    rstd_ptr,        # *f32, [C]
    gamma_ptr,       # *f32, [C]
    beta_ptr,        # *f32, [C]
    y_ptr,           # *f32, output [B, C, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
):
    # grid: (B * num_groups * (C // num_groups))
    pid = tl.program_id(0)
    groups = num_groups
    channels_per_group = C // groups
    n = pid // (groups * channels_per_group)
    g = (pid % (groups * channels_per_group)) // channels_per_group
    c = (pid % channels_per_group) + g * channels_per_group

    mean = tl.load(mean_ptr + c)
    rstd = tl.load(rstd_ptr + c)
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)

    for h in range(0, H):
        for w in range(0, W):
            x_idx = (((n * C) + c) * H + h) * W + w
            x_val = tl.load(x_ptr + x_idx)
            norm = (x_val - mean) * rstd
            # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
            silu = norm * tl.sigmoid(norm)
            y_val = silu * gamma + beta
            y_idx = x_idx  # same indexing
            tl.store(y_ptr + y_idx, y_val)


# Residual add: out = a + b, elementwise
@triton.jit
def residual_add_kernel(
    a_ptr, b_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    # Grid: (B, C, H*W)
    n = tl.program_id(0)
    c = tl.program_id(1)
    pos = tl.program_id(2)
    h = pos // W
    w = pos % W
    idx = (((n * C) + c) * H + h) * W + w
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
        x: (B, C, H, W), conv weights: (C_out, C_in, 3, 3), norm scales/bias: (C,)
        Returns: (B, C_out2, H, W)
        """
        # No torch ops for computation; just allocate outputs and launch kernels.
        # Shapes are passed as tl.constexpr to Triton via kwargs in kernel launch.
        B = x.shape[0]
        C_in = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]

        # First conv: out1 shape (B, C_out1, H, W)
        C_out1 = conv1_weight.shape[1]
        out1 = torch.empty((B, C_out1, H, W), device=x.device, dtype=x.dtype)

        grid1 = (B, C_out1, H * W)
        conv3x3_kernel[grid1](
            x, conv1_weight, out1,
            B=B, C_in=C_in, C_out=C_out1, H=H, W=W,
        )

        # First GroupNorm + SiLU
        assert C_out1 % self.num_groups == 0, "C_out1 must be divisible by num_groups"
        ch_per_group = C_out1 // self.num_groups
        mean1 = torch.empty(C_out1, device=x.device, dtype=x.dtype)
        rstd1 = torch.empty(C_out1, device=x.device, dtype=x.dtype)

        grid_reduce1 = (B * self.num_groups * ch_per_group,)
        group_norm_reduce_kernel[grid_reduce1](
            out1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups, eps=self.eps,
        )

        out1_norm = torch.empty_like(out1)
        grid_apply1 = (B * self.num_groups * ch_per_group,)
        group_norm_apply_kernel[grid_apply1](
            out1, mean1, rstd1, norm1_weight, norm1_bias, out1_norm,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups,
        )

        # Second conv: out2 shape (B, C_out2, H, W)
        C_in2 = C_out1
        C_out2 = conv2_weight.shape[1]
        out2 = torch.empty((B, C_out2, H, W), device=x.device, dtype=x.dtype)

        grid2 = (B, C_out2, H * W)
        conv3x3_kernel[grid2](
            out1_norm, conv2_weight, out2,
            B=B, C_in=C_in2, C_out=C_out2, H=H, W=W,
        )

        # Second GroupNorm + SiLU
        assert C_out2 % self.num_groups == 0, "C_out2 must be divisible by num_groups"
        ch_per_group2 = C_out2 // self.num_groups
        mean2 = torch.empty(C_out2, device=x.device, dtype=x.dtype)
        rstd2 = torch.empty(C_out2, device=x.device, dtype=x.dtype)

        grid_reduce2 = (B * self.num_groups * ch_per_group2,)
        group_norm_reduce_kernel[grid_reduce2](
            out2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups, eps=self.eps,
        )

        out2_norm = torch.empty_like(out2)
        grid_apply2 = (B * self.num_groups * ch_per_group2,)
        group_norm_apply_kernel[grid_apply2](
            out2, mean2, rstd2, norm2_weight, norm2_bias, out2_norm,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups,
        )

        # Final residual add: out2_norm + x
        out = torch.empty_like(out2_norm)
        grid_add = (B, C_out2, H * W)
        residual_add_kernel[grid_add](
            out2_norm, x, out,
            B=B, C=C_out2, H=H, W=W,
        )

        return out


def run(*args):
    return ModelNew()(*args)
