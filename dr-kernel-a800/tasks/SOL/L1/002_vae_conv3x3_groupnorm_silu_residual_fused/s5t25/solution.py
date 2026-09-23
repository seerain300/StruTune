import torch
import triton
import triton.language as tl


# Conv3x3 via per-output-pixel reduction across input channels and 3x3 neighborhood.
# Grid: (B, H, W, tiles_along_co). Each program handles one output pixel (n, h_out, w_out) and a tile of output channels.
@triton.jit
def conv3x3_per_pixel_tile_kernel(
    x_ptr,            # *f32, input [B, C_in, H, W]
    w_ptr,            # *f32, weights [C_out, C_in, 3, 3]
    out_ptr,          # *f32, output [B, C_out, H, W]
    B: tl.constexpr,  C_in: tl.constexpr, C_out: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    co_start: tl.constexpr, BLOCK_CO: tl.constexpr, N_TILES: tl.constexpr,
):
    n = tl.program_id(0)
    h_out = tl.program_id(1)
    w_out = tl.program_id(2)
    tile_id = tl.program_id(3)  # which output-channel tile

    co_idx = co_start + tl.arange(0, BLOCK_CO)
    mask_co = co_idx < C_out

    # Accumulator for this output pixel across output channels
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # Loop over input channels
    for c_in in range(0, C_in):
        # 3x3 neighborhood around (h_out, w_out)
        for dh in range(-1, 2):
            for dw in range(-1, 2):
                h_in = h_out + dh
                w_in = w_out + dw
                # Load x[n, c_in, h_in, w_in] as a scalar; out-of-bounds due to padding are fine
                x_val = tl.load(
                    x_ptr + ((n * C_in + c_in) * H + h_in) * W + w_in,
                )
                # Load corresponding weights for this tile of output channels: w[co, c_in, 3+dh, 3+dw]
                # weight layout: w[co, c_in, kh, kw] with kh in [0..2], kw in [0..2]
                w_offsets = co_idx * (C_in * 9) + c_in * 9 + (dh + 1) * 3 + (dw + 1)
                w_vals = tl.load(
                    w_ptr + w_offsets,
                    mask=mask_co,
                    other=0.0,
                )
                acc += x_val * w_vals

    # Store results for this tile's output channels to out[n, co, h_out, w_out]
    for i in range(BLOCK_CO):
        co = co_start + i
        mask_store = co < C_out
        tl.store(out_ptr + ((n * C_out + co) * H + h_out) * W + w_out, acc[i], mask=mask_store)


# GroupNorm reduction: per channel, compute sum and sumsq over all elements (H*W) in its group.
# Grid: (B, num_groups, channels_per_group). Loops over tiles via N_TILES (constexpr).
@triton.jit
def group_norm_reduce_kernel(
    x_ptr,            # *f32, input [B, C, H, W]
    mean_ptr,         # *f32, output [C]
    rstd_ptr,         # *f32, output [C]
    B: tl.constexpr,  C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, channels_per_group: tl.constexpr,
    N_TILES: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    ch_in_group = tl.program_id(2)  # which channel in the group
    c = g * channels_per_group + ch_in_group

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # Loop over tiles of HW
    for t in range(0, N_TILES):
        tile_start = t * BLOCK_HW
        hw_idx = tile_start + tl.arange(0, BLOCK_HW)
        mask = hw_idx < (H * W)
        base = n * C + c
        addr = base * (H * W) + hw_idx
        x_vals = tl.load(x_ptr + addr, mask=mask, other=0.0)
        sum_val += tl.sum(x_vals, axis=0)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    m = H * W
    mean = sum_val / m
    var = sum_sq / m - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-12)
    tl.store(mean_ptr + c, mean)
    tl.store(rstd_ptr + c, rstd)


# GroupNorm apply: per channel, per tile, normalize and apply affine + SiLU.
# Grid: (B, num_groups, channels_per_group, N_TILES). Vectorizes across the tile (BLOCK_HW).
@triton.jit
def group_norm_apply_kernel(
    x_ptr,             # *f32, input [B, C, H, W]
    mean_ptr,          # *f32, [C]
    rstd_ptr,          # *f32, [C]
    scale_ptr,         # *f32, [C] (norm weight)
    bias_ptr,          # *f32, [C] (norm bias)
    y_ptr,             # *f32, output [B, C, H, W]
    B: tl.constexpr,   C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, channels_per_group: tl.constexpr,
    N_TILES: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    ch_in_group = tl.program_id(2)
    t = tl.program_id(3)  # tile index

    c = g * channels_per_group + ch_in_group

    tile_start = t * BLOCK_HW
    hw_idx = tile_start + tl.arange(0, BLOCK_HW)
    mask = hw_idx < (H * W)

    base = n * C + c
    addr = base * (H * W) + hw_idx

    x_vals = tl.load(x_ptr + addr, mask=mask, other=0.0)

    mean = tl.load(mean_ptr + c)
    rstd = tl.load(rstd_ptr + c)
    scale = tl.load(scale_ptr + c)
    bias = tl.load(bias_ptr + c)

    y_norm = (x_vals - mean) * rstd
    # SiLU: y = y * sigmoid(y) = y / (1 + exp(-y))
    y_silu = y_norm * (1.0 / (1.0 + tl.exp(-y_norm)))
    y_affine = y_silu * scale + bias

    tl.store(y_ptr + addr, y_affine, mask=mask)


# Elementwise residual add: out = a + b
@triton.jit
def residual_add_kernel_1d(
    a_ptr, b_ptr, out_ptr,
    total_elems: tl.constexpr, CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * CHUNK + tl.arange(0, CHUNK)
    mask = offs < total_elems
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a + b, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32):
        super().__init__()
        self.num_groups = num_groups

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

        # Ensure float32 for Triton kernels
        x_f32 = x.contiguous().to(torch.float32)

        # First conv: (B, C_out1, H, W)
        C_out1 = conv1_weight.shape[0]
        out1 = torch.empty((B, C_out1, H, W), device=x.device, dtype=torch.float32)

        # Launch conv kernel: grid over (B, H, W, tiles_along_co)
        BLOCK_CO = 64
        tiles_co = (C_out1 + BLOCK_CO - 1) // BLOCK_CO

        grid_conv1 = (B, H, W, tiles_co)
        conv3x3_per_pixel_tile_kernel[grid_conv1](
            x_f32, conv1_weight.contiguous().to(torch.float32), out1,
            B=B, C_in=C_in, C_out=C_out1, H=H, W=W,
            co_start=0, BLOCK_CO=BLOCK_CO, N_TILES=tiles_co,
            num_warps=4, num_stages=2,
        )

        # GroupNorm + SiLU for first block
        mean1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        rstd1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)

        channels_per_group1 = C_out1 // self.num_groups
        assert C_out1 % self.num_groups == 0, "C_out1 must be divisible by num_groups"

        # Reduction: grid (B, num_groups, channels_per_group), tile over HW
        N_TILES_1 = (H * W + 256 - 1) // 256
        BLOCK_HW = 256
        grid_reduce1 = (B, self.num_groups, channels_per_group1)
        group_norm_reduce_kernel[grid_reduce1](
            out1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group1,
            N_TILES=N_TILES_1, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2,
        )

        out1_norm = torch.empty_like(out1)
        grid_apply1 = (B, self.num_groups, channels_per_group1, N_TILES_1)
        group_norm_apply_kernel[grid_apply1](
            out1, mean1, rstd1, norm1_weight.contiguous().to(torch.float32), norm1_bias.contiguous().to(torch.float32),
            out1_norm,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group1,
            N_TILES=N_TILES_1, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2,
        )

        # Second conv: input is out1_norm, output (B, C_out2, H, W)
        C_out2 = conv2_weight.shape[0]
        out2 = torch.empty((B, C_out2, H, W), device=x.device, dtype=torch.float32)

        tiles_co2 = (C_out2 + BLOCK_CO - 1) // BLOCK_CO
        grid_conv2 = (B, H, W, tiles_co2)
        conv3x3_per_pixel_tile_kernel[grid_conv2](
            out1_norm, conv2_weight.contiguous().to(torch.float32), out2,
            B=B, C_in=C_out1, C_out=C_out2, H=H, W=W,
            co_start=0, BLOCK_CO=BLOCK_CO, N_TILES=tiles_co2,
            num_warps=4, num_stages=2,
        )

        # GroupNorm + SiLU for second block
        mean2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        rstd2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)

        channels_per_group2 = C_out2 // self.num_groups
        assert C_out2 % self.num_groups == 0, "C_out2 must be divisible by num_groups"

        N_TILES_2 = (H * W + 256 - 1) // 256
        grid_reduce2 = (B, self.num_groups, channels_per_group2)
        group_norm_reduce_kernel[grid_reduce2](
            out2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group2,
            N_TILES=N_TILES_2, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2,
        )

        out2_norm = torch.empty_like(out2)
        grid_apply2 = (B, self.num_groups, channels_per_group2, N_TILES_2)
        group_norm_apply_kernel[grid_apply2](
            out2, mean2, rstd2, norm2_weight.contiguous().to(torch.float32), norm2_bias.contiguous().to(torch.float32),
            out2_norm,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group2,
            N_TILES=N_TILES_2, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2,
        )

        # Final residual add: out2_norm + x
        total_elems = B * C_out2 * H * W
        out = torch.empty_like(out2_norm)
        CHUNK = 4096
        grid_add = ((total_elems + CHUNK - 1) // CHUNK,)
        residual_add_kernel_1d[grid_add](
            out2_norm, x_f32, out,
            total_elems=total_elems, CHUNK=CHUNK,
            num_warps=4, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
