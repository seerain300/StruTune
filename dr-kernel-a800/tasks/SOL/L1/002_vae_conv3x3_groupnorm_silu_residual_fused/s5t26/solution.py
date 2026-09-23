import torch
import triton
import triton.language as tl


# Conv3x3: y = x * w, stride=1, padding=1, no bias.
# Each program handles one output pixel (n, h_out, w_out), accumulates over all input channels and 3x3 neighborhood,
# and writes into output channels in tiles.
@triton.jit
def conv3x3_pixel_kernel(
    x_ptr, w_ptr, y_ptr,
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_CO: tl.constexpr,  # number of output channels per tile
    TILE_SIZE: tl.constexpr,  # number of spatial elements per tile (e.g., 1024)
):
    n = tl.program_id(0)
    h_out = tl.program_id(1)
    w_out = tl.program_id(2)
    tile_id = tl.program_id(3)

    # Flattened spatial index within this tile
    base = tile_id * TILE_SIZE
    offs = base + tl.arange(0, TILE_SIZE)
    mask_hw = offs < (H * W)

    # Compute per-element h and w indices
    h_vec = offs // W
    w_vec = offs % W

    # Initialize accumulator over output channels in chunks
    # We iterate over output channels in tiles of size BLOCK_CO and write only valid ones.
    for co_start in range(0, C_out, BLOCK_CO):
        co_idx = co_start + tl.arange(0, BLOCK_CO)
        mask_co = co_idx < C_out

        # Accumulator for this tile of output channels
        acc = tl.zeros([BLOCK_CO], dtype=tl.float32)

        # Reduction over input channels and 3x3 neighborhood
        for ci in range(0, C_in):
            for dh in range(-1, 2):
                h_i = h_out + dh
                # Guard for h_i: if out-of-range, continue
                # In padding=1, h_i is always in [0, H-1], but we keep defensive logic
                valid_h = (h_i >= 0) & (h_i < H)
                for dw in range(-1, 2):
                    w_i = w_out + dw
                    valid_w = (w_i >= 0) & (w_i < W)
                    # Skip if either is invalid (shouldn't happen for padding=1)
                    if not valid_h or not valid_w:
                        continue

                    # Compute input index for this (n, ci, h_i, w_i)
                    in_idx = (((n * C_in) + ci) * H + h_i) * W + w_i

                    # Load x (scalar)
                    x_val = tl.load(x_ptr + in_idx)

                    # Load weight vector for current ci and all co in this tile
                    # Weight shape: (C_out, C_in, 3, 3)
                    # For each (co, ci), pick corresponding 3x3 index (dh, dw)
                    # Build weight pointers for all co in this tile
                    # Note: we can't build a 2D pointer vector, so we do a loop over co in tile
                    for co_i in range(0, BLOCK_CO):
                        co_curr = co_start + co_i
                        valid_co = co_curr < C_out
                        # If co invalid, skip
                        if not valid_co:
                            continue
                        # Compute weight index for weight[co_curr, ci, 3+dh, 3+dw]
                        # dh, dw are scalars, ci and co_curr are scalars
                        w_idx = ((co_curr * C_in + ci) * 9) + ((dh + 1) * 3 + (dw + 1)) - 1
                        w_val = tl.load(w_ptr + w_idx)
                        acc[co_i] += x_val * w_val

        # Write results to y for this tile of output channels
        # y layout: ((n * C_out + co) * H + h_out) * W + w_out
        # We map offs (spatial) and co_idx (channels) into linear indices.
        # Since acc is per-channel, we add the spatial offset to the channel-linear index.
        for co_i in range(0, BLOCK_CO):
            co_curr = co_start + co_i
            if co_curr >= C_out:
                break
            # Linear index for this (n, co_curr, h_out, w_out)
            y_idx = (((n * C_out) + co_curr) * H + h_out) * W + w_out
            # Store; masked for spatial and channel validity
            tl.store(y_ptr + y_idx, acc[co_i], mask=mask_co & (h_out < H) & (w_out < W))

        # After writing scalar, we can move to next co tile. No need to store vector; we process tiles sequentially.
        # The above loop writes scalar per co; we'll implement a vectorized store below.


# The above kernel writes scalar per co. A more vectorized approach: process vector of spatial positions and output channels.
# Conv3x3 kernel that handles BLOCK_HW spatial positions and accumulates into BLOCK_CO output channels in tiles.
@triton.jit
def conv3x3_block_hw_kernel(
    x_ptr, w_ptr, y_ptr,
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_HW: tl.constexpr, BLOCK_CO: tl.constexpr, TILE_SIZE: tl.constexpr,
):
    n = tl.program_id(0)
    tile_hw = tl.program_id(1)
    tile_co = tl.program_id(2)

    # Spatial offsets for this tile
    base = tile_hw * TILE_SIZE
    offs_hw = base + tl.arange(0, TILE_SIZE)
    mask_hw = offs_hw < (H * W)

    h_vec = offs_hw // W
    w_vec = offs_hw % W

    # Output channel tile
    co_start = tile_co * BLOCK_CO
    co_idx = co_start + tl.arange(0, BLOCK_CO)
    mask_co = co_idx < C_out

    # Initialize accumulator [BLOCK_CO, TILE_SIZE]
    acc = tl.zeros([BLOCK_CO, TILE_SIZE], dtype=tl.float32)

    # Reduction over input channels and 3x3 neighborhood
    for ci in range(0, C_in):
        for dh in range(-1, 2):
            h_i = h_vec + dh  # vector
            valid_h = (h_i >= 0) & (h_i < H)
            for dw in range(-1, 2):
                w_i = w_vec + dw  # vector
                valid_w = (w_i >= 0) & (w_i < W)
                valid = mask_hw & valid_h & valid_w

                # Compute input indices for vector of spatial positions: shape [TILE_SIZE]
                # in_idx = (((n * C_in) + ci) * H + h_i) * W + w_i
                # h_i is vector, w_i is vector, so we do elementwise ops
                in_idx = (((n * C_in) + ci) * H + h_i) * W + w_i

                # Load x vector
                x_vec = tl.load(x_ptr + in_idx, mask=valid, other=0.0)

                # Load weight matrix for this ci and all co in tile: shape [BLOCK_CO]
                # We loop over co in tile and compute weights
                for co_i in range(0, BLOCK_CO):
                    co_curr = co_start + co_i
                    valid_co = co_curr < C_out
                    if not valid_co:
                        continue
                    # w_idx = ((co_curr * C_in + ci) * 9) + ((dh + 1) * 3 + (dw + 1)) - 1
                    w_idx = ((co_curr * C_in + ci) * 9) + ((dh + 1) * 3 + (dw + 1)) - 1
                    w_val = tl.load(w_ptr + w_idx)
                    # acc[co_i, :] += x_vec * w_val
                    acc[co_i, :] += x_vec * w_val

    # Store results: y_ptr points to ((n * C_out + co) * H + h_out) * W + w_out
    # We need to build per-(co, hw) index. We'll do a loop over co in tile and store vector of hw.
    for co_i in range(0, BLOCK_CO):
        co_curr = co_start + co_i
        if co_curr >= C_out:
            break
        # y index vector for this co_curr and hw vector
        y_idx_vec = (((n * C_out + co_curr) * H + h_vec) * W + w_vec)
        # Store acc vector
        tl.store(y_ptr + y_idx_vec, acc[co_i, :], mask=mask_hw & (co_curr < C_out))

# Helper: we will use conv3x3_block_hw_kernel for conv operations; conv3x3_pixel_kernel kept for clarity.


# GroupNorm reduction: compute per-channel sum and sumsq across spatial (H*W) for each (n, group, channel).
@triton.jit
def group_norm_reduce_kernel(
    x_ptr, mean_ptr, rstd_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    # Each program handles one (n, group, channel)
    n = tl.program_id(0)
    group = tl.program_id(1)
    c = tl.program_id(2)

    channels_per_group = C // num_groups
    # Iterate over spatial tiles
    for tile in range(0, (H * W) // TILE_SIZE):
        base = tile * TILE_SIZE
        offs = base + tl.arange(0, TILE_SIZE)
        mask = offs < (H * W)

        # Compute linear index into x: ((n * C + c) * H + h) * W + w
        # First, we need h and w vectors
        h_vec = offs // W
        w_vec = offs % W
        idx = (((n * C) + c) * H + h_vec) * W + w_vec

        x_vec = tl.load(x_ptr + idx, mask=mask, other=0.0)
        # Reduce to scalars
        sum_tile = tl.sum(x_vec, axis=0)
        sumsq_tile = tl.sum(x_vec * x_vec, axis=0)

        # Accumulate into scalars
        # We need global sum and sumsq; we’ll atomic_add to per-(n,c) scalars.
        # Create per-(n,c) pointers
        # mean_ptr and rstd_ptr are of length C
        # Atomic add with pointer to specific element
        # Triton doesn’t have tl.atomic_add, so we’ll compute scalars and write once per program at the end.
        # To avoid atomics, we compute per-tile sums and use a second kernel to finalize. For simplicity,
        # we compute and store per-tile sum/sumsq in arrays on host, but Triton kernels must not use host arrays.
        # Alternative: do a single program per (n,c) to avoid tiling. Here, we redesign to single-program per (n,c).
    # Note: This kernel as written needs revision to avoid multiple programs per (n,c). We provide a corrected version below.


# Correct group norm reduction: one program per (n, group, channel), iterate all spatial tiles.
@triton.jit
def group_norm_reduce_single(
    x_ptr, mean_ptr, rstd_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    n = tl.program_id(0)
    group = tl.program_id(1)
    c = tl.program_id(2)

    channels_per_group = C // num_groups
    assert c < (group * channels_per_group + channels_per_group), "c out of range for group"

    sum_val = 0.0
    sumsq_val = 0.0

    # Iterate tiles over HW
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

    # Finalize mean and rstd for this (n, c)
    M = H * W
    mean = sum_val / M
    var = sumsq_val / M - mean * mean
    rstd = 1.0 / tl.sqrt(var + 0.0)  # rstd = 1/sqrt(var + eps), here eps is not needed in reduction kernel

    # Store to mean and rstd arrays (float32)
    # We write scalars; pointers are element-wise
    tl.store(mean_ptr + (n * C + c), mean)
    tl.store(rstd_ptr + (n * C + c), rstd)


# GroupNorm apply: y = ((x - mean[c]) * rstd[c]) * weight[c] + bias[c], then SiLU
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
    # We process spatial tiles
    base = tile * TILE_SIZE
    offs = base + tl.arange(0, TILE_SIZE)
    mask = offs < (H * W)

    h_vec = offs // W
    w_vec = offs % W
    idx = (((n * C) + c) * H + h_vec) * W + w_vec

    x_vec = tl.load(x_ptr + idx, mask=mask, other=0.0)

    mean = tl.load(mean_ptr + (n * C + c))
    rstd = tl.load(rstd_ptr + (n * C + c))
    gamma = tl.load(weight_ptr + c)
    beta = tl.load(bias_ptr + c)

    y_vec = (x_vec - mean) * rstd
    y_vec = y_vec * gamma + beta

    # SiLU: y = y * sigmoid(y)
    sig = 1.0 / (1.0 + tl.exp(-y_vec))
    y_vec = y_vec * sig

    tl.store(y_ptr + idx, y_vec, mask=mask)


# Elementwise add kernel over 1D flattened tensor
@triton.jit
def residual_add_1d_kernel(
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
        self.eps = eps  # not used directly; Triton kernels use provided mean/rstd

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        x: (B, C_in, H, W)
        conv1_weight: (C_out1, C_in, 3, 3)
        norm1_weight, norm1_bias: (C_out1,)
        conv2_weight: (C_out2, C_out1, 3, 3)
        norm2_weight, norm2_bias: (C_out2,)
        Returns: (B, C_out2, H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"
        B, C_in, H, W = x.shape

        # Ensure float32 for stable math
        x_f32 = x.contiguous().to(torch.float32)
        conv1_w_f32 = conv1_weight.contiguous().to(torch.float32)  # (C_out1, C_in, 3, 3)
        conv2_w_f32 = conv2_weight.contiguous().to(torch.float32)  # (C_out2, C_out1, 3, 3)
        norm1_weight_f32 = norm1_weight.contiguous().to(torch.float32)
        norm1_bias_f32 = norm1_bias.contiguous().to(torch.float32)
        norm2_weight_f32 = norm2_weight.contiguous().to(torch.float32)
        norm2_bias_f32 = norm2_bias.contiguous().to(torch.float32)

        # First conv
        C_out1 = conv1_w_f32.shape[0]
        out1 = torch.empty((B, C_out1, H, W), device=x.device, dtype=torch.float32)

        # Launch conv3x3_block_hw_kernel
        BLOCK_HW = 256
        TILE_SIZE = 1024
        grid_conv1 = (B, triton.cdiv(H * W, TILE_SIZE), triton.cdiv(C_out1, 128))
        conv3x3_block_hw_kernel[grid_conv1](
            x_f32, conv1_w_f32, out1,
            B, C_in, C_out1, H, W,
            BLOCK_HW, 128, TILE_SIZE,
            num_warps=4, num_stages=2,
        )

        # GroupNorm + SiLU for first block
        mean1 = torch.empty((B * C_out1), device=x.device, dtype=torch.float32)
        rstd1 = torch.empty_like(mean1)
        TILE_SIZE_reduce = 1024
        grid_reduce = (B, self.num_groups, C_out1)
        group_norm_reduce_single[grid_reduce](
            out1, mean1, rstd1,
            B, C_out1, H, W,
            self.num_groups, TILE_SIZE_reduce,
            num_warps=4, num_stages=2,
        )

        out1_norm = torch.empty_like(out1)
        grid_apply1 = (B, self.num_groups, C_out1, triton.cdiv(H * W, TILE_SIZE))
        group_norm_apply_kernel[grid_apply1](
            out1, mean1, rstd1, norm1_weight_f32, norm1_bias_f32, out1_norm,
            B, C_out1, H, W,
            self.num_groups, TILE_SIZE,
            num_warps=4, num_stages=2,
        )

        # Second conv
        C_out2 = conv2_w_f32.shape[0]
        out2 = torch.empty((B, C_out2, H, W), device=x.device, dtype=torch.float32)

        # Launch conv3x3_block_hw_kernel again for second conv
        grid_conv2 = (B, triton.cdiv(H * W, TILE_SIZE), triton.cdiv(C_out2, 128))
        conv3x3_block_hw_kernel[grid_conv2](
            out1_norm, conv2_w_f32, out2,
            B, C_out1, C_out2, H, W,
            BLOCK_HW, 128, TILE_SIZE,
            num_warps=4, num_stages=2,
        )

        # GroupNorm + SiLU for second block
        mean2 = torch.empty((B * C_out2), device=x.device, dtype=torch.float32)
        rstd2 = torch.empty_like(mean2)
        grid_reduce2 = (B, self.num_groups, C_out2)
        group_norm_reduce_single[grid_reduce2](
            out2, mean2, rstd2,
            B, C_out2, H, W,
            self.num_groups, TILE_SIZE_reduce,
            num_warps=4, num_stages=2,
        )

        out2_norm = torch.empty_like(out2)
        grid_apply2 = (B, self.num_groups, C_out2, triton.cdiv(H * W, TILE_SIZE))
        group_norm_apply_kernel[grid_apply2](
            out2, mean2, rstd2, norm2_weight_f32, norm2_bias_f32, out2_norm,
            B, C_out2, H, W,
            self.num_groups, TILE_SIZE,
            num_warps=4, num_stages=2,
        )

        # Final residual add: out2_norm + x
        total_elems = B * C_out2 * H * W
        out = torch.empty_like(out2_norm)
        CHUNK = 4096
        grid_add = (triton.cdiv(total_elems, CHUNK),)
        residual_add_1d_kernel[grid_add](
            out2_norm, x_f32, out,
            total_elems, CHUNK,
            num_warps=4, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
