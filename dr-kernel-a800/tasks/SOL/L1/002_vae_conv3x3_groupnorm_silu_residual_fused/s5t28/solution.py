import torch
import triton
import triton.language as tl


# Conv3x3: y[n, co, h, w] = sum_{ci, dh,dw in 3x3} x[n, ci, h+dh, w+dw] * w[co, ci, 3+dh, 3+dw]
# Stride=1, padding=1, no bias. Each program handles one output pixel (n,h,w) and reduces over all input channels and 3x3 neighborhood.
@triton.jit
def conv3x3_pixel_kernel(
    x_ptr, w_ptr, y_ptr,
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    h = tl.program_id(1)
    w = tl.program_id(2)

    # Output channels vector
    co_vec = tl.arange(0, C_out)
    # Accumulator for all output channels
    acc_vec = tl.zeros([C_out], dtype=tl.float32)

    # Reduction over input channels and 3x3 neighborhood
    for ci in range(0, C_in):
        for dh in range(-1, 2):
            for dw in range(-1, 2):
                hi = h + dh
                wi = w + dw
                # Valid spatial indices due to padding=1
                x_idx = (((n * C_in) + ci) * H + hi) * W + wi
                x_val = tl.load(x_ptr + x_idx)  # scalar
                # Load corresponding weight vector for all co
                w_idx = (((co_vec * C_in) + ci) * 9) + (dh + 1) * 3 + (dw + 1)
                w_vec = tl.load(w_ptr + w_idx)  # shape [C_out]
                # Accumulate
                acc_vec += x_val * w_vec

    # Store result for all output channels (masked is unnecessary since co_vec < C_out always)
    for co in range(0, C_out):
        y_idx = (((n * C_out) + co) * H + h) * W + w
        tl.store(y_ptr + y_idx, acc_vec[co])


# GroupNorm reduction: per (n, group, channel), compute sum and sumsq across spatial HW.
# C_out must be divisible by num_groups (assumed 32). Each program processes one channel in one group.
@triton.jit
def group_norm_reduce_single(
    x_ptr, mean_ptr, rstd_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
):
    n = tl.program_id(0)
    group = tl.program_id(1)
    c = tl.program_id(2)

    channels_per_group = C // num_groups
    sum_val = 0.0
    sumsq_val = 0.0

    TILE_SIZE = 1024  # process H*W in chunks of 1024
    for tile in range(0, (H * W) // TILE_SIZE):
        base = tile * TILE_SIZE
        offs = base + tl.arange(0, TILE_SIZE)
        mask = offs < (H * W)

        h_vec = offs // W
        w_vec = offs % W
        idx = (((n * C) + c) * H + h_vec) * W + w_vec

        x_vec = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sum_tile = tl.sum(x_vec, axis=0)
        sumsq_tile = tl.sum(x_vec * x_vec, axis=0)

        sum_val += sum_tile
        sumsq_val += sumsq_tile

    M = H * W
    mean = sum_val / M
    var = sumsq_val / M - mean * mean
    rstd = 1.0 / tl.sqrt(var + 0.0)  # numerical stability

    # Store per-(n, c) mean and rstd
    tl.store(mean_ptr + (n * C + c), mean)
    tl.store(rstd_ptr + (n * C + c), rstd)


# GroupNorm apply: y = ((x - mean[c]) * rstd[c]) * gamma[c] + beta[c], then SiLU
@triton.jit
def group_norm_apply_kernel(
    x_ptr, mean_ptr, rstd_ptr, weight_ptr, bias_ptr, y_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    n = tl.program_id(0)
    group = tl.program_id(1)
    c = tl.program_id(2)
    tile = tl.program_id(3)

    channels_per_group = C // num_groups

    base = tile * TILE_SIZE
    offs = base + tl.arange(0, TILE_SIZE)
    mask = offs < (H * W)

    h_vec = offs // W
    w_vec = offs % W
    idx = (((n * C) + c) * H + h_vec) * W + w_vec

    x_vec = tl.load(x_ptr + idx, mask=mask, other=0.0)

    mean_c = tl.load(mean_ptr + (n * C + c))
    rstd_c = tl.load(rstd_ptr + (n * C + c))
    gamma_c = tl.load(weight_ptr + c)
    beta_c = tl.load(bias_ptr + c)

    norm_vec = (x_vec - mean_c) * rstd_c
    affine_vec = norm_vec * gamma_c + beta_c

    # SiLU: x * sigmoid(x) where sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-affine_vec))
    y_vec = affine_vec * sig

    tl.store(y_ptr + idx, y_vec, mask=mask)


# Elementwise residual add: out = a + b
@triton.jit
def residual_add_kernel(
    a_ptr, b_ptr, out_ptr,
    total_elems: tl.constexpr, CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * CHUNK
    offs = start + tl.arange(0, CHUNK)
    mask = offs < total_elems
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a + b, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        x: (B, C_in, H, W)
        conv weights: (C_out, C_in, 3, 3)
        norm scales/bias: (C_out,)
        Returns: (B, C_out2, H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"
        B, C_in, H, W = x.shape

        # Ensure contiguous and float32 for numerical stability
        x_f32 = x.contiguous().to(torch.float32)

        # First conv: out1 = Conv3x3(x)
        C_out1 = conv1_weight.shape[0]
        out1 = torch.empty((B, C_out1, H, W), device=x.device, dtype=torch.float32)
        grid_conv1 = (B, H, W)
        conv3x3_pixel_kernel[grid_conv1](
            x_f32, conv1_weight.contiguous().to(torch.float32), out1,
            B=B, C_in=C_in, C_out=C_out1, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        # GroupNorm + SiLU for first block
        mean1 = torch.empty((B, C_out1), device=x.device, dtype=torch.float32)
        rstd1 = torch.empty((B, C_out1), device=x.device, dtype=torch.float32)
        grid_reduce1 = (B, self.num_groups, C_out1 // self.num_groups)
        group_norm_reduce_single[grid_reduce1](
            out1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W, num_groups=self.num_groups,
            num_warps=4, num_stages=2,
        )

        out1_norm = torch.empty_like(out1)
        tiles_hw = triton.cdiv(H * W, 1024)
        grid_apply1 = (B, self.num_groups, C_out1 // self.num_groups, tiles_hw)
        group_norm_apply_kernel[grid_apply1](
            out1, mean1, rstd1, norm1_weight.contiguous().to(torch.float32), norm1_bias.contiguous().to(torch.float32), out1_norm,
            B=B, C=C_out1, H=H, W=W, num_groups=self.num_groups,
            TILE_SIZE=1024,
            num_warps=4, num_stages=2,
        )

        # Second conv: out2 = Conv3x3(out1_norm)
        C_out2 = conv2_weight.shape[0]
        out2 = torch.empty((B, C_out2, H, W), device=x.device, dtype=torch.float32)
        grid_conv2 = (B, H, W)
        conv3x3_pixel_kernel[grid_conv2](
            out1_norm, conv2_weight.contiguous().to(torch.float32), out2,
            B=B, C_in=C_out1, C_out=C_out2, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        # GroupNorm + SiLU for second block
        mean2 = torch.empty((B, C_out2), device=x.device, dtype=torch.float32)
        rstd2 = torch.empty((B, C_out2), device=x.device, dtype=torch.float32)
        grid_reduce2 = (B, self.num_groups, C_out2 // self.num_groups)
        group_norm_reduce_single[grid_reduce2](
            out2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W, num_groups=self.num_groups,
            num_warps=4, num_stages=2,
        )

        out2_norm = torch.empty_like(out2)
        tiles_hw2 = triton.cdiv(H * W, 1024)
        grid_apply2 = (B, self.num_groups, C_out2 // self.num_groups, tiles_hw2)
        group_norm_apply_kernel[grid_apply2](
            out2, mean2, rstd2, norm2_weight.contiguous().to(torch.float32), norm2_bias.contiguous().to(torch.float32), out2_norm,
            B=B, C=C_out2, H=H, W=W, num_groups=self.num_groups,
            TILE_SIZE=1024,
            num_warps=4, num_stages=2,
        )

        # Final residual add: out = out2_norm + x
        total_elems = B * C_out2 * H * W
        out = torch.empty_like(out2_norm)
        CHUNK = 4096
        grid_add = (triton.cdiv(total_elems, CHUNK),)
        residual_add_kernel[grid_add](
            out2_norm, x_f32, out,
            total_elems=total_elems, CHUNK=CHUNK,
            num_warps=4, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
