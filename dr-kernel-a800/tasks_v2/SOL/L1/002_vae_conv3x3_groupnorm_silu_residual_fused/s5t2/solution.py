import torch
import triton
import triton.language as tl


# Conv3x3: y = x * w, stride=1, padding=1, no bias
@triton.jit
def conv3x3_kernel(
    x_ptr,           # *f32, input [B, C_in, H, W]
    w_ptr,           # *f32, weights [C_out, C_in, 3, 3]
    y_ptr,           # *f32, output [B, C_out, H, W]
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
):
    # program ids: (n, c_out, pixel_index)
    n = tl.program_id(0)        # batch
    c_out = tl.program_id(1)    # output channel
    pixel = tl.program_id(2)    # flattened spatial index
    h_out = pixel // W
    w_out = pixel % W

    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels and 3x3 neighborhood
    for c_in in range(C_in):
        for dh in range(3):
            for dw in range(3):
                in_h = h_out + dh - 1  # padding=1 -> index shifts
                in_w = w_out + dw - 1
                # input offset for x[n, c_in, in_h, in_w]
                x_offset = n * (C_in * H * W) + c_in * (H * W) + in_h * W + in_w
                x_val = tl.load(x_ptr + x_offset)
                # weight offset for w[c_out, c_in, dh, dw]
                w_offset = c_out * (C_in * 9) + c_in * 9 + dh * 3 + dw
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    # store y[n, c_out, h_out, w_out]
    y_offset = n * (C_out * H * W) + c_out * (H * W) + h_out * W + w_out
    tl.store(y_ptr + y_offset, acc)


# GroupNorm + SiLU Triton kernel.
# Each program handles one (n, group, channel-in-group) and one tile over HW.
# We pass N_TILES as a compile-time constant so tl.arange has a known width.
@triton.jit
def group_norm_silu_kernel(
    x_ptr,         # *f32, input [B, C, H, W]
    weight_ptr,    # *f32, scale [C]
    bias_ptr,      # *f32, bias [C]
    y_ptr,         # *f32, output [B, C, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, eps: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    N_TILES: tl.constexpr,     # number of tiles over HW
):
    # program ids: (n, group, combined channel-and-tile index)
    n = tl.program_id(0)
    g = tl.program_id(1)
    pid2 = tl.program_id(2)

    channels_per_group = C // num_groups
    # decode combined index into (c_rel, tile_index)
    c_rel = pid2 // N_TILES
    tile_index = pid2 % N_TILES

    c = g * channels_per_group + c_rel

    HW = H * W
    start = tile_index * BLOCK_HW

    # Vector of indices within the tile
    idx = start + tl.arange(0, BLOCK_HW)
    mask = idx < HW

    # Compute h and w for vector
    h = idx // W
    w = idx % W

    # First pass: reduction over this tile to update sum and sumsq (accumulate scalars)
    x_offsets = n * (C * HW) + c * HW + h * W + w
    x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
    # This kernel performs per-(n,group,channel) reduction over all tiles.
    # Here we only process one tile per program; we need to aggregate across tiles.
    # To implement full reduction, we would need to loop across N_TILES, but Triton
    # does not support dynamic loops. Therefore, we instead launch a separate reduction
    # kernel (not shown here) or structure the grid so each program handles all tiles
    # for a (n,group,channel) by iterating. Triton supports while loops. We will use
    # a small while loop across tiles to compute sum and sumsq, then a second while loop
    # to write outputs. However, Triton does not allow Python 'while' loops either in this
    # context. To keep the kernel simple and robust, we redesign: launch a dedicated
    # reduction kernel that computes mean/rstd, then apply kernel that reads mean/rstd
    # and writes normalized+SiLU. But since Triton doesn't allow arbitrary loops, the
    # simplest is to compute mean/rstd in PyTorch (which would violate Triton-only).
    #
    # Since we must use Triton, we will instead keep the previous two-pass vectorized
    # approach but fix the loop variable naming and avoid range that Triton can't
    # infer. We'll implement two kernels: reduction kernel that computes mean/rstd
    # per (n,group,channel) via a single-tile vectorized loop, and apply kernel that
    # applies normalization + affine + SiLU per tile using provided mean/rstd.
    #
    # For this code, we'll implement the two-pass approach correctly by decoding the
    # grid so each program handles all tiles for (n,group,channel). We'll keep the
    # reduction and apply in one kernel by looping across tiles using a Python for
    # loop with N_TILES passed as constexpr. This works because N_TILES is a
    # compile-time constant; Triton can unroll the loop.

    # Compute total sum and sumsq across all tiles for channel c
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # First pass: reduction across all tiles
    for t in range(N_TILES):
        idx = t * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask = idx < HW
        h = idx // W
        w = idx % W
        x_offsets = n * (C * HW) + c * HW + h * W + w
        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x_vals, axis=0)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    mean = sum_val / HW
    var = sum_sq / HW - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: apply GroupNorm + affine + SiLU across all tiles
    for t in range(N_TILES):
        idx = t * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask = idx < HW
        h = idx // W
        w = idx % W
        x_offsets = n * (C * HW) + c * HW + h * W + w
        x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        y = (x_vals - mean) * rstd
        scale = tl.load(weight_ptr + c)
        bias = tl.load(bias_ptr + c)
        y = y * scale + bias
        sig = 1.0 / (1.0 + tl.exp(-y))
        y = y * sig
        out_offsets = n * (C * HW) + c * HW + h * W + w
        tl.store(y_ptr + out_offsets, y, mask=mask)


# Residual add kernel: out = y + x
@triton.jit
def residual_add_kernel(
    y_ptr,   # *f32, [B, C, H, W]
    x_ptr,   # *f32, [B, C, H, W]
    out_ptr, # *f32, [B, C, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    pixel = tl.program_id(2)
    h = pixel // W
    w = pixel % W

    in_offset = n * (C * H * W) + c * (H * W) + h * W + w
    y_val = tl.load(y_ptr + in_offset)
    x_val = tl.load(x_ptr + in_offset)
    out_val = y_val + x_val
    tl.store(out_ptr + in_offset, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5, block_hw: int = 256):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.block_hw = block_hw

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor):
        """
        Fully Triton implementation:
          - conv1: 3x3 Conv (stride=1, padding=1, no bias)
          - group_norm_silu: GroupNorm + SiLU on conv1 output
          - conv2: 3x3 Conv (stride=1, padding=1, no bias)
          - group_norm_silu: GroupNorm + SiLU on conv2 output
          - residual add: conv2_out + x

        All math is done by Triton kernels; forward only orchestrates kernel launches.
        """
        # Triton kernels require CUDA and float32 for numerical stability
        assert x.is_cuda, "Triton kernels require CUDA tensors"
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

        B, C_in, H, W = x.shape
        C_out1 = conv1_weight.shape[0]
        C_out2 = conv2_weight.shape[0]

        # Sanity checks for conv weights
        assert conv1_weight.shape == (C_out1, C_in, 3, 3), f"conv1_weight shape must be (C_out1, C_in, 3, 3), got {conv1_weight.shape}"
        assert conv2_weight.shape == (C_out2, C_in, 3, 3), f"conv2_weight shape must be (C_out2, C_in, 3, 3), got {conv2_weight.shape}"

        # Ensure GroupNorm num_groups divides channels
        assert C_out1 % self.num_groups == 0, "conv1 output channels must be divisible by num_groups (32)"
        assert C_out2 % self.num_groups == 0, "conv2 output channels must be divisible by num_groups (32)"

        # Allocate intermediate outputs
        out1 = torch.empty((B, C_out1, H, W), device=x.device, dtype=x.dtype)
        out1_norm = torch.empty_like(out1)

        # Launch conv1 kernel
        grid1 = (B, C_out1, H * W)
        conv3x3_kernel[grid1](
            x, conv1_weight, out1,
            B=B, C_in=C_in, C_out=C_out1,
            H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        # GroupNorm + SiLU for first block
        grid_g1 = (B, self.num_groups, C_out1 // self.num_groups)  # each program handles one (n,group,channel)
        # For GroupNorm, we need to iterate across all tiles; Triton allows Python for-loops when loop count is constexpr.
        channels_per_group1 = C_out1 // self.num_groups
        N_TILES1 = ((H * W) + self.block_hw - 1) // self.block_hw
        group_norm_silu_kernel[grid_g1](
            out1, norm1_weight, norm1_bias, out1_norm,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups, eps=self.eps,
            BLOCK_HW=self.block_hw,
            N_TILES=N_TILES1,  # constexpr loop count
            num_warps=4,
            num_stages=2,
        )

        # Conv2: input is out1_norm, which has shape (B, C_out1, H, W)
        out2 = torch.empty((B, C_out2, H, W), device=x.device, dtype=x.dtype)
        grid2 = (B, C_out2, H * W)
        conv3x3_kernel[grid2](
            out1_norm, conv2_weight, out2,
            B=B, C_in=C_out1, C_out=C_out2,
            H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        # GroupNorm + SiLU for second block
        out2_norm = torch.empty_like(out2)
        channels_per_group2 = C_out2 // self.num_groups
        N_TILES2 = ((H * W) + self.block_hw - 1) // self.block_hw
        grid_g2 = (B, self.num_groups, channels_per_group2)  # per (n,group) and channel
        group_norm_silu_kernel[grid_g2](
            out2, norm2_weight, norm2_bias, out2_norm,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups, eps=self.eps,
            BLOCK_HW=self.block_hw,
            N_TILES=N_TILES2,  # constexpr loop count
            num_warps=4,
            num_stages=2,
        )

        # Final residual add: out2_norm + x
        out = torch.empty_like(out2_norm)
        residual_add_kernel[(B, C_out2, H * W)](
            out2_norm, x, out,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
