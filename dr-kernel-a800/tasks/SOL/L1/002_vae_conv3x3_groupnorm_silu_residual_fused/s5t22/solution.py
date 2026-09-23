import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_pixel_kernel(
    x_ptr,               # *f32, input [B, C_in, H, W]
    w_ptr,               # *f32, weight [C_out, C_in, 3, 3]
    y_ptr,               # *f32, output [B, C_out, H, W]
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    # Grid over (B, C_out, H*W). Each program computes one output pixel for (n, c_out, hw).
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    hw = tl.program_id(2)

    h = hw // W
    w = hw % W

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels
    for c_in in range(C_in):
        # Loop over 3x3 neighborhood (padding=1, stride=1)
        for dh in range(3):
            for dw in range(3):
                h_in = h + dh - 1
                w_in = w + dw - 1
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)

                x_idx = (((n * C_in) + c_in) * H + h_in) * W + w_in
                x_val = tl.load(x_ptr + x_idx, mask=in_bounds, other=0.0)

                # Weight index for (c_out, c_in, dh, dw)
                w_idx = (c_out * C_in + c_in) * (3 * 3) + (dh * 3 + dw)
                w_val = tl.load(w_ptr + w_idx)

                acc += x_val * w_val

    y_idx = (((n * C_out) + c_out) * H + h) * W + w
    tl.store(y_ptr + y_idx, acc)


@triton.jit
def group_norm_reduce_kernel(
    x_ptr,             # *f32, input [B, C, H, W]
    mean_ptr,          # *f32, per-(n,c) mean [B*C]
    rstd_ptr,          # *f32, per-(n,c) rstd [B*C]
    C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    # Grid: (B, groups, channels_per_group)
    n = tl.program_id(0)
    group = tl.program_id(1)
    c = tl.program_id(2)

    channels_per_group = C // 32  # num_groups is 32, as in original code
    base = n * C + group * channels_per_group + c

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Iterate over all spatial positions
    for i in range(H * W):
        idx = (((n * C) + c) * H + (i // W)) * W + (i % W)
        val = tl.load(x_ptr + idx)
        sum_val += val
        sumsq_val += val * val

    hw = H * W
    mean = sum_val / hw
    var = sumsq_val / hw - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)  # eps

    tl.store(mean_ptr + base, mean)
    tl.store(rstd_ptr + base, rstd)


@triton.jit
def apply_groupnorm_affine_silu_kernel(
    x_ptr,             # *f32, input [B, C, H, W] (normalized)
    mean_ptr,          # *f32, per-(n,c) mean
    rstd_ptr,          # *f32, per-(n,c) rstd
    weight_ptr,        # *f32, per-channel scale [C]
    bias_ptr,          # *f32, per-channel bias [C]
    y_ptr,             # *f32, output [B, C, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr, N_TILES: tl.constexpr,
):
    # Grid: (B, C, N_TILES)
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)

    offs = tile * 1024 + tl.arange(0, 1024)
    mask = offs < (H * W)
    h = offs // W
    w = offs % W

    # Load stats and affine parameters
    base = n * C + c
    mean = tl.load(mean_ptr + base)
    rstd = tl.load(rstd_ptr + base)
    gamma = tl.load(weight_ptr + c)
    beta = tl.load(bias_ptr + c)

    # Load input vector
    idx = (((n * C) + c) * H + h) * W + w
    x_val = tl.load(x_ptr + idx, mask=mask, other=0.0)

    # GroupNorm + affine
    norm = (x_val - mean) * rstd
    norm = norm * gamma + beta

    # SiLU
    sig = 1.0 / (1.0 + tl.exp(-norm))
    out = norm * sig

    tl.store(y_ptr + idx, out, mask=mask)


@triton.jit
def residual_add_kernel(
    a_ptr, b_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)

    offs = tile * 1024 + tl.arange(0, 1024)
    mask = offs < (H * W)
    h = offs // W
    w = offs % W

    idx = (((n * C) + c) * H + h) * W + w
    a_val = tl.load(a_ptr + idx, mask=mask, other=0.0)
    b_val = tl.load(b_ptr + idx, mask=mask, other=0.0)
    tl.store(out_ptr + idx, a_val + b_val, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        x: (B, C_in, H, W), float32, contiguous
        conv weights: (C_out, C_in, 3, 3), float32, contiguous
        norm scales/bias: (C_out,) float32
        Returns: (B, C_out2, H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"
        B, C_in, H, W = x.shape

        # Ensure float32 and contiguous for Triton
        x_f32 = x.contiguous().to(torch.float32)

        # First conv: y1 = conv3x3(x)
        C_out1 = conv1_weight.shape[0]
        out1 = torch.empty((B, C_out1, H, W), device=x.device, dtype=torch.float32)

        grid1 = (B, C_out1, H * W)
        conv3x3_pixel_kernel[grid1](
            x_f32, conv1_weight.contiguous().to(torch.float32), out1,
            B=B, C_in=C_in, C_out=C_out1, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        # GroupNorm + SiLU for first conv output
        out1_norm = torch.empty_like(out1)
        # Compute per-(n,c) mean/rstd
        mean1 = torch.empty((B * C_out1), device=x.device, dtype=torch.float32)
        rstd1 = torch.empty((B * C_out1), device=x.device, dtype=torch.float32)

        grid_r1 = (B, 32, C_out1 // 32)
        group_norm_reduce_kernel[grid_r1](
            out1, mean1, rstd1,
            C=C_out1, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        # Apply affine + SiLU
        N_TILES = triton.cdiv(H * W, 1024)
        apply_groupnorm_affine_silu_kernel[(B, C_out1, N_TILES)](
            out1, mean1, rstd1, norm1_weight.to(torch.float32), norm1_bias.to(torch.float32), out1_norm,
            B=B, C=C_out1, H=H, W=W, N_TILES=N_TILES,
            num_warps=4, num_stages=2,
        )

        # Second conv: y2 = conv3x3(out1_norm)
        C_out2 = conv2_weight.shape[0]
        out2 = torch.empty((B, C_out2, H, W), device=x.device, dtype=torch.float32)

        grid2 = (B, C_out2, H * W)
        conv3x3_pixel_kernel[grid2](
            out1_norm, conv2_weight.contiguous().to(torch.float32), out2,
            B=B, C_in=C_out1, C_out=C_out2, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        # GroupNorm + SiLU for second conv output
        out2_norm = torch.empty_like(out2)
        mean2 = torch.empty((B * C_out2), device=x.device, dtype=torch.float32)
        rstd2 = torch.empty((B * C_out2), device=x.device, dtype=torch.float32)

        grid_r2 = (B, 32, C_out2 // 32)
        group_norm_reduce_kernel[grid_r2](
            out2, mean2, rstd2,
            C=C_out2, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        N_TILES2 = triton.cdiv(H * W, 1024)
        apply_groupnorm_affine_silu_kernel[(B, C_out2, N_TILES2)](
            out2, mean2, rstd2, norm2_weight.to(torch.float32), norm2_bias.to(torch.float32), out2_norm,
            B=B, C=C_out2, H=H, W=W, N_TILES=N_TILES2,
            num_warps=4, num_stages=2,
        )

        # Residual add: out2_norm + x
        res = torch.empty_like(out2_norm)
        grid_add = (B, C_out2, triton.cdiv(H * W, 1024))
        residual_add_kernel[grid_add](
            out2_norm, x_f32, res,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        return res


def run(*args):
    return ModelNew()(*args)
