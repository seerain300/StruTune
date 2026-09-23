import torch
import triton
import triton.language as tl


# Conv3x3 via im2col + reduction over input channels, stride=1, padding=1, no bias.
# We implement two Triton kernels:
# 1) conv3x3_reduce: for each (n, c_out, tile) computes partial sums across input channels
# 2) conv3x3_write: writes accumulated output to y using a precomputed pointer to accum buffer
# This approach avoids per-pixel scalar loops and uses vectorized tiles.

# Kernel 1: compute reduction for one (n, c_out) over input channels for a tile of spatial positions
@triton.jit
def conv3x3_reduce_nc(
    x_ptr,            # *f32, input [B, C_in, H, W]
    w_ptr,            # *f32, weight [C_out, C_in, 3, 3]
    accum_ptr,        # *f32, accumulator [C_out, N_tiles], where N_tiles = H*W
    B: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    N_TILES: tl.constexpr,  # number of spatial tiles along flattened H*W
    BLOCK_HW: tl.constexpr, # number of spatial positions per tile
):
    # program ids: (n, c_out, tile)
    n = tl.program_id(0)
    co = tl.program_id(1)
    tile = tl.program_id(2)

    # vector of spatial indices within this tile
    offs = tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < (H * W)

    # map linear index to (h, w)
    h = offs // W
    w = offs % W

    # Accumulator per spatial position (vector length BLOCK_HW)
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Loop over input channels
    for ci in range(C_in):
        # Loop over 3x3 neighborhood
        for dh in range(3):
            for dw in range(3):
                # input coordinates
                hi = h + dh - 1
                wi = w + dw - 1
                # valid mask for input bounds
                in_bounds = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W) & mask
                # compute flattened input index: (((n * C_in) + ci) * H + hi) * W + wi
                x_idx = (((n * C_in) + ci) * H + hi) * W + wi
                # load input vector (masked)
                x_vals = tl.load(x_ptr + x_idx, mask=in_bounds, other=0.0)

                # load weight scalar for this (co, ci, dh, dw)
                w_idx = co * (C_in * 9) + (ci * 9) + (dh * 3 + dw)
                w_val = tl.load(w_ptr + w_idx)  # scalar

                # accumulate
                acc += x_vals * w_val

    # write to accumulator buffer: shape [C_out, N_TILES]
    # row index = co, column index = tile * BLOCK_HW + offs
    out_idx = co * N_TILES + offs
    tl.store(accum_ptr + out_idx, acc, mask=mask)


# Kernel 2: write accumulated output to y for all tiles
@triton.jit
def conv3x3_write(
    accum_ptr,        # *f32, accumulator [C_out, N_tiles]
    y_ptr,            # *f32, output [B, C_out, H, W]
    B: tl.constexpr,
    C_out: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    N_TILES: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # program ids: (n, c_out, tile)
    n = tl.program_id(0)
    co = tl.program_id(1)
    tile = tl.program_id(2)

    offs = tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < (H * W)

    # read accumulated values for this (n, co, tile)
    out_idx = co * N_TILES + offs
    vals = tl.load(accum_ptr + out_idx, mask=mask, other=0.0)

    # write to y at (n, co, h, w)
    h = offs // W
    w = offs % W
    # y linear index: (((n * C_out) + co) * H + h) * W + w
    y_idx = (((n * C_out) + co) * H + h) * W + w
    tl.store(y_ptr + y_idx, vals, mask=mask)


# GroupNorm kernels (no host-side loops)
# 1) Reduce sum and sum of squares per channel, compute mean and rstd
@triton.jit
def group_norm_reduce(
    x_ptr,           # *f32, input [B, C, H, W]
    mean_ptr,        # *f32, output [C]
    rstd_ptr,        # *f32, output [C]
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    num_groups: tl.constexpr,
    channels_per_group: tl.constexpr,
    N_TILES: tl.constexpr,   # number of spatial tiles in the group (H*W)
    BLOCK_HW: tl.constexpr,  # tile size along flattened H*W
):
    # program ids: (n, group, c_in_group)
    n = tl.program_id(0)
    group = tl.program_id(1)
    c_in_group = tl.program_id(2)

    c = group * channels_per_group + c_in_group
    # Accumulate sum and sumsq over all tiles
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for t in range(N_TILES):
        offs = t * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask = offs < (H * W)
        h = offs // W
        w = offs % W
        # linear index for x[n, c, h, w]
        x_idx = (((n * C) + c) * H + h) * W + w
        x_vec = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
        # reduce to scalars
        sum_val += tl.sum(x_vec, axis=0)
        sum_sq += tl.sum(x_vec * x_vec, axis=0)

    mean = sum_val / (H * W)
    var = sum_sq / (H * W) - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)  # use a small eps
    tl.store(mean_ptr + c, mean)
    tl.store(rstd_ptr + c, rstd)


# 2) Apply GroupNorm (affine + SiLU) per channel using precomputed mean/rstd
@triton.jit
def group_norm_apply(
    x_ptr,           # *f32, input [B, C, H, W]
    scale_ptr,       # *f32, weight [C] (gamma)
    bias_ptr,        # *f32, bias [C] (beta)
    mean_ptr,        # *f32, [C]
    rstd_ptr,        # *f32, [C]
    y_ptr,           # *f32, output [B, C, H, W]
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    num_groups: tl.constexpr,
    channels_per_group: tl.constexpr,
    N_TILES: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # program ids: (n, group, c_in_group, tile)
    n = tl.program_id(0)
    group = tl.program_id(1)
    c_in_group = tl.program_id(2)
    tile = tl.program_id(3)

    c = group * channels_per_group + c_in_group

    mean = tl.load(mean_ptr + c)
    rstd = tl.load(rstd_ptr + c)
    gamma = tl.load(scale_ptr + c)
    beta = tl.load(bias_ptr + c)

    offs = tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < (H * W)
    h = offs // W
    w = offs % W

    x_idx = (((n * C) + c) * H + h) * W + w
    x_vec = tl.load(x_ptr + x_idx, mask=mask, other=0.0)

    # GroupNorm affine
    y_vec = (x_vec - mean) * rstd
    y_vec = y_vec * gamma + beta

    # SiLU activation
    # silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-y_vec))
    y_vec = y_vec * sig

    y_idx = (((n * C) + c) * H + h) * W + w
    tl.store(y_ptr + y_idx, y_vec, mask=mask)


# Residual add (elementwise): out = out + x
@triton.jit
def residual_add_kernel(
    a_ptr, b_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    hw = tl.program_id(2)

    # hw is a linear index over H*W
    h = hw // W
    w = hw % W

    a_idx = (((n * C) + c) * H + h) * W + w
    b_idx = a_idx  # b_ptr is x
    out_idx = a_idx

    a_val = tl.load(a_ptr + a_idx)
    b_val = tl.load(b_ptr + b_idx)
    tl.store(out_ptr + out_idx, a_val + b_val)


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
        B, C_in, H, W = x.shape

        # Ensure C_out of conv1 equals C_in (typical), conv2 input is conv1 output
        C_out1 = conv1_weight.shape[0]
        C_out2 = conv2_weight.shape[0]

        # Make sure channels are divisible by num_groups
        if C_out1 % self.num_groups != 0 or C_out2 % self.num_groups != 0:
            raise ValueError(f"Channels ({C_out1} or {C_out2}) must be divisible by num_groups={self.num_groups}")
        channels_per_group1 = C_out1 // self.num_groups
        channels_per_group2 = C_out2 // self.num_groups

        # Cast to float32 for stable Triton math
        x_f32 = x.contiguous().to(torch.float32)
        conv1_w_f32 = conv1_weight.contiguous().to(torch.float32)
        conv2_w_f32 = conv2_weight.contiguous().to(torch.float32)
        # GroupNorm params
        norm1_weight_f32 = norm1_weight.contiguous().to(torch.float32)
        norm1_bias_f32 = norm1_bias.contiguous().to(torch.float32)
        norm2_weight_f32 = norm2_weight.contiguous().to(torch.float32)
        norm2_bias_f32 = norm2_bias.contiguous().to(torch.float32)

        # Conv1: im2col-like reduction over tiles
        out1 = torch.empty((B, C_out1, H, W), device=x.device, dtype=torch.float32)

        # Choose tiling parameters
        BLOCK_H = 32
        BLOCK_W = 32
        N_TILES = (H * W + BLOCK_H * BLOCK_W - 1) // (BLOCK_H * BLOCK_W)
        BLOCK_HW = BLOCK_H * BLOCK_W
        grid1_reduce = (B, C_out1, N_TILES)

        # Accumulator buffer [C_out1, N_TILES]
        accum1 = torch.zeros((C_out1, N_TILES), device=x.device, dtype=torch.float32)

        conv3x3_reduce_nc[grid1_reduce](
            x_f32, conv1_w_f32, accum1,
            B, C_in, C_out1, H, W, N_TILES, BLOCK_HW,
            num_warps=4, num_stages=2
        )

        # Write conv1 output
        conv3x3_write[(B, C_out1, N_TILES)](
            accum1, out1,
            B, C_out1, H, W, N_TILES, BLOCK_HW,
            num_warps=4, num_stages=2
        )

        # GroupNorm + SiLU for first block
        mean1 = torch.empty((C_out1,), device=x.device, dtype=torch.float32)
        rstd1 = torch.empty((C_out1,), device=x.device, dtype=torch.float32)
        # N_TILES_HW is the total number of spatial positions per channel
        N_TILES_HW = (H * W + BLOCK_H * BLOCK_W - 1) // (BLOCK_H * BLOCK_W)
        grid_reduce1 = (B, self.num_groups, channels_per_group1)
        group_norm_reduce[grid_reduce1](
            out1, mean1, rstd1,
            B, C_out1, H, W,
            self.num_groups, channels_per_group1, N_TILES_HW, BLOCK_H * BLOCK_W,
            num_warps=4, num_stages=2
        )

        out1_norm = torch.empty_like(out1)
        grid_apply1 = (B, self.num_groups, channels_per_group1, N_TILES_HW)
        group_norm_apply[grid_apply1](
            out1, norm1_weight_f32, norm1_bias_f32, mean1, rstd1, out1_norm,
            B, C_out1, H, W,
            self.num_groups, channels_per_group1, N_TILES_HW, BLOCK_H * BLOCK_W,
            num_warps=4, num_stages=2
        )

        # Conv2: im2col-like reduction
        out2 = torch.empty((B, C_out2, H, W), device=x.device, dtype=torch.float32)
        accum2 = torch.zeros((C_out2, N_TILES), device=x.device, dtype=torch.float32)

        conv3x3_reduce_nc[(B, C_out2, N_TILES)](
            out1_norm, conv2_w_f32, accum2,
            B, C_out1, C_out2, H, W, N_TILES, BLOCK_HW,
            num_warps=4, num_stages=2
        )
        conv3x3_write[(B, C_out2, N_TILES)](
            accum2, out2,
            B, C_out2, H, W, N_TILES, BLOCK_HW,
            num_warps=4, num_stages=2
        )

        # GroupNorm + SiLU for second block
        mean2 = torch.empty((C_out2,), device=x.device, dtype=torch.float32)
        rstd2 = torch.empty((C_out2,), device=x.device, dtype=torch.float32)
        group_norm_reduce[(B, self.num_groups, channels_per_group2)](
            out2, mean2, rstd2,
            B, C_out2, H, W,
            self.num_groups, channels_per_group2, N_TILES_HW, BLOCK_H * BLOCK_W,
            num_warps=4, num_stages=2
        )

        out2_norm = torch.empty_like(out2)
        group_norm_apply[(B, self.num_groups, channels_per_group2, N_TILES_HW)](
            out2, norm2_weight_f32, norm2_bias_f32, mean2, rstd2, out2_norm,
            B, C_out2, H, W,
            self.num_groups, channels_per_group2, N_TILES_HW, BLOCK_H * BLOCK_W,
            num_warps=4, num_stages=2
        )

        # Residual add
        out = torch.empty_like(out2_norm)
        residual_add_kernel[(B, C_out2, H * W)](
            out2_norm, x_f32, out,
            B, C_out2, H, W,
            num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
