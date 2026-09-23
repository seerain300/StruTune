import torch
import torch.nn.functional as F

# Triton kernels for GroupNorm + SiLU
import triton
import triton.language as tl


@triton.jit
def group_norm_silu_kernel(
    x_ptr,           # *const float, input pointer [B, C, H, W]
    weight_ptr,      # *const float, per-channel scale [C]
    bias_ptr,        # *const float, per-channel bias [C]
    y_ptr,           # *float, output pointer [B, C, H, W]
    B: tl.constexpr,  # int
    C: tl.constexpr,  # int
    H: tl.constexpr,  # int
    W: tl.constexpr,  # int
    NUM_GROUPS: tl.constexpr,  # int
    eps,              # float
    BLOCK_HW: tl.constexpr,    # tile size for HW
):
    # Each program handles one (n, group) pair and all channels in that group.
    n = tl.program_id(0)
    g = tl.program_id(1)

    # Number of channels per group and the starting channel index
    channels_per_group = C // NUM_GROUPS
    group_channel_start = g * channels_per_group

    # First pass: compute sum and sum of squares per channel in the group
    # We will use a vector over HW tiles.
    # For each channel c in the group, compute mean and rstd
    # Then in a second pass apply normalization and SiLU.

    # Initialize sum and sumsq arrays per channel
    # We'll do this by looping over channels in the group (fixed at compile-time),
    # and for each channel, loop over HW in tiles.
    # We'll use a list to keep it simple and Triton-friendly.

    # Note: Triton allows loops with dynamic bounds when using range, but
    # to keep it performant and predictable, we keep C and NUM_GROUPS constexpr.
    # We will compute the total number of tiles for a given channel: num_tiles = ceil_div(H*W, BLOCK_HW)
    HW = H * W
    num_tiles = (HW + BLOCK_HW - 1) // BLOCK_HW

    # We'll create arrays for sum and sumsq across channels in the group
    # but Triton arrays require constexpr sizes, so we iterate and allocate per channel.
    # Better approach: use a for-loop over channels and reduce per channel.
    # Implement a helper to reduce per channel:
    # We'll use a while loop to iterate over tiles.

    # For each channel in the group:
    # 1) sum = 0, sumsq = 0
    # 2) loop over tiles, load x[n, c, tile], accumulate
    # 3) mean = sum / (H*W), var = sumsq / (H*W) - mean^2, rstd = 1/sqrt(var + eps)
    # 4) apply y = x * weight[c] * rstd + bias[c]; silu(y) and store.

    # We'll do these steps in Triton using nested loops.

    # Triton allows Python-like loops with dynamic bounds (range), but here we keep everything constexpr-like
    # by treating HW, BLOCK_HW, C, NUM_GROUPS as constexpr in signature. However, n and g are runtime ints.
    # So we can loop over channels in the group and tiles over HW.

    # To keep it simple, we manually iterate:
    # channels_per_group is constexpr-like; we can use a for-loop with range(channels_per_group)
    for c_local in range(channels_per_group):
        c = group_channel_start + c_local

        # Initialize accumulators as scalars (float32)
        sum_val = 0.0
        sum_sq = 0.0

        # Loop over HW tiles
        # Triton supports while loops
        tile_idx = 0
        while tile_idx < num_tiles:
            # Compute linear offsets for this tile: [tile_idx*BLOCK_HW : (tile_idx+1)*BLOCK_HW)
            offs = tile_idx * BLOCK_HW + tl.arange(0, BLOCK_HW)
            mask = offs < HW  # mask for tail

            # Convert linear offs to (h, w)
            # h = offs // W
            # w = offs % W
            h = offs // W
            w = offs % W

            # Compute pointer offsets for x[n, c, h, w]
            # x layout is NCHW, so offset = n*C*H*W + c*H*W + h*W + w
            # For a given c, x_offset_base = n*C*H*W + c*H*W
            # Then element offset is h*W + w
            x_offset = n * C * H * W + c * H * W + h * W + w
            x_vals = tl.load(x_ptr + x_offset, mask=mask, other=0.0)

            # Accumulate sum and sumsq
            sum_val += tl.sum(x_vals, axis=0)
            sum_sq += tl.sum(x_vals * x_vals, axis=0)

            tile_idx += 1

        # Compute mean and rstd for this channel
        M = H * W
        mean = sum_val / M
        var = sum_sq / M - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)

        # Weight and bias for this channel
        wgt = tl.load(weight_ptr + c)
        bst = tl.load(bias_ptr + c)

        # Second pass: apply normalized + SiLU and store
        tile_idx = 0
        while tile_idx < num_tiles:
            offs = tile_idx * BLOCK_HW + tl.arange(0, BLOCK_HW)
            mask = offs < HW
            h = offs // W
            w = offs % W

            x_offset = n * C * H * W + c * H * W + h * W + w
            x_vals = tl.load(x_ptr + x_offset, mask=mask, other=0.0)

            # Normalize and affine
            y_norm = (x_vals - mean) * rstd
            y_affine = y_norm * wgt + bst

            # SiLU activation: y * sigmoid(y) with sigmoid(y) = 1 / (1 + exp(-y))
            sig = 1.0 / (1.0 + tl.exp(-y_affine))
            out = y_affine * sig

            y_offset = n * C * H * W + c * H * W + h * W + w
            tl.store(y_ptr + y_offset, out, mask=mask)

            tile_idx += 1


@triton.jit
def group_norm_silu_kernel_2(
    x_ptr,           # *const float, input pointer [B, C, H, W]
    weight_ptr,      # *const float, per-channel scale [C]
    bias_ptr,        # *const float, per-channel bias [C]
    y_ptr,           # *float, output pointer [B, C, H, W]
    B: tl.constexpr,  # int
    C: tl.constexpr,  # int
    H: tl.constexpr,  # int
    W: tl.constexpr,  # int
    NUM_GROUPS: tl.constexpr,  # int
    eps,              # float
    BLOCK_HW: tl.constexpr,    # tile size for HW
):
    # Same as above, just a duplicate for clarity. We can reuse the same kernel by passing different weight/bias.
    n = tl.program_id(0)
    g = tl.program_id(1)
    channels_per_group = C // NUM_GROUPS
    group_channel_start = g * channels_per_group

    HW = H * W
    num_tiles = (HW + BLOCK_HW - 1) // BLOCK_HW

    for c_local in range(channels_per_group):
        c = group_channel_start + c_local

        sum_val = 0.0
        sum_sq = 0.0

        tile_idx = 0
        while tile_idx < num_tiles:
            offs = tile_idx * BLOCK_HW + tl.arange(0, BLOCK_HW)
            mask = offs < HW
            h = offs // W
            w = offs % W

            x_offset = n * C * H * W + c * H * W + h * W + w
            x_vals = tl.load(x_ptr + x_offset, mask=mask, other=0.0)

            sum_val += tl.sum(x_vals, axis=0)
            sum_sq += tl.sum(x_vals * x_vals, axis=0)

            tile_idx += 1

        M = H * W
        mean = sum_val / M
        var = sum_sq / M - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)

        wgt = tl.load(weight_ptr + c)
        bst = tl.load(bias_ptr + c)

        tile_idx = 0
        while tile_idx < num_tiles:
            offs = tile_idx * BLOCK_HW + tl.arange(0, BLOCK_HW)
            mask = offs < HW
            h = offs // W
            w = offs % W

            x_offset = n * C * H * W + c * H * W + h * W + w
            x_vals = tl.load(x_ptr + x_offset, mask=mask, other=0.0)

            y_norm = (x_vals - mean) * rstd
            y_affine = y_norm * wgt + bst
            sig = 1.0 / (1.0 + tl.exp(-y_affine))
            out = y_affine * sig

            y_offset = n * C * H * W + c * H * W + h * W + w
            tl.store(y_ptr + y_offset, out, mask=mask)

            tile_idx += 1


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5, block_hw: int = 256):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        # BLOCK_HW controls tiling over H*W; 256 is a good default
        self.block_hw = block_hw

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float = None):
        """
        Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add

        Args:
            x: Input tensor of shape (B, C, H, W), expected float32, on CUDA
            conv1_weight: First conv weights (C, C, 3, 3), float32, on CUDA
            norm1_weight: First GroupNorm scale (C,), float32, on CUDA
            norm1_bias: First GroupNorm bias (C,), float32, on CUDA
            conv2_weight: Second conv weights (C, C, 3, 3), float32, on CUDA
            norm2_weight: Second GroupNorm scale (C,), float32, on CUDA
            norm2_bias: Second GroupNorm bias (C,), float32, on CUDA
            eps: Epsilon for GroupNorm numerical stability (float)

        Returns:
            Output tensor of shape (B, C, H, W)
        """
        # Ensure we are on CUDA and dtype is float32 for Triton kernels
        assert x.is_cuda, "Triton kernels require CUDA tensors"
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Conv weights must be on CUDA"
        assert norm1_weight.is_cuda and norm1_bias.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "Norm params must be on CUDA"
        # Enforce float32 for numerical stability
        if x.dtype != torch.float32:
            x = x.float()
        if conv1_weight.dtype != torch.float32:
            conv1_weight = conv1_weight.float()
        if conv2_weight.dtype != torch.float32:
            conv2_weight = conv2_weight.float()
        if norm1_weight.dtype != torch.float32:
            norm1_weight = norm1_weight.float()
        if norm1_bias.dtype != torch.float32:
            norm1_bias = norm1_bias.float()
        if norm2_weight.dtype != torch.float32:
            norm2_weight = norm2_weight.float()
        if norm2_bias.dtype != torch.float32:
            norm2_bias = norm2_bias.float()

        B, C, H, W = x.shape
        # Sanity checks for group norm
        assert C % self.num_groups == 0, "C must be divisible by num_groups (32) for GroupNorm"

        # Save residual
        residual = x

        # First path: Conv3x3 -> GroupNorm -> SiLU
        out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        # Triton GroupNorm + SiLU
        y1 = torch.empty_like(out)
        grid = (B, self.num_groups)
        group_norm_silu_kernel[grid](
            out, norm1_weight, norm1_bias, y1,
            B, C, H, W, self.num_groups, self.eps,
            BLOCK_HW=self.block_hw,
            num_warps=4,  # heuristic
            num_stages=2,
        )

        # Second path: Conv3x3 -> GroupNorm -> SiLU
        out = F.conv2d(y1, conv2_weight, bias=None, stride=1, padding=1)
        y2 = torch.empty_like(out)
        grid2 = (B, self.num_groups)
        group_norm_silu_kernel_2[grid2](
            out, norm2_weight, norm2_bias, y2,
            B, C, H, W, self.num_groups, self.eps,
            BLOCK_HW=self.block_hw,
            num_warps=4,
            num_stages=2,
        )

        # Residual connection
        out = y2 + residual

        return out


def run(*args):
    return ModelNew()(*args)
