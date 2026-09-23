import torch
import triton
import triton.language as tl


# Triton kernels for GroupNorm reduce and apply (no loops in host; loops in kernels are constexpr).
# We structure the grid to avoid overlapping reads within a group: each program handles one absolute channel c
# and its group g, so there's no cross-channel overlap.

# GroupNorm reduce: compute per-channel mean and rstd over spatial plane (H*W). Grid: (B, num_groups, channels_per_group).
@triton.jit
def group_norm_reduce_kernel(
    x_ptr,                # *f32, input [B, C, H, W]
    mean_ptr,             # *f32, per-channel mean [C]
    rstd_ptr,             # *f32, per-channel rstd [C]
    B: tl.constexpr,      # int
    C: tl.constexpr,      # int
    H: tl.constexpr,      # int
    W: tl.constexpr,      # int
    num_groups: tl.constexpr,   # int
    channels_per_group: tl.constexpr,  # int, typically 1 because we set grid to (..., channels_per_group=1)
    N_TILES: tl.constexpr,        # int, number of tiles over HW
    BLOCK_HW: tl.constexpr,       # int, tile size for HW
):
    n = tl.program_id(0)
    group = tl.program_id(1)
    c = tl.program_id(2)  # absolute channel index (one per program)

    HW = H * W
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for t in range(N_TILES):
        tile_start = t * BLOCK_HW
        idxs = tile_start + tl.arange(0, BLOCK_HW)
        mask = idxs < HW

        h = idxs // W
        w = idxs % W

        base = n * C * HW + c * HW
        vals = tl.load(x_ptr + base + idxs, mask=mask, other=0.0)

        sum_val += tl.sum(vals, axis=0)
        sum_sq += tl.sum(vals * vals, axis=0)

    mean = sum_val / HW
    var = sum_sq / HW - mean * mean
    var = tl.maximum(var, 0.0)  # numerical stability
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    tl.store(mean_ptr + c, mean)
    tl.store(rstd_ptr + c, rstd)


# GroupNorm apply: y = ((x - mean[c]) * rstd[c]) * scale[c] + bias[c], then SiLU. Grid: (B, num_groups, channels_per_group, N_TILES).
@triton.jit
def group_norm_apply_kernel(
    x_ptr,           # *f32, input [B, C, H, W]
    mean_ptr,        # *f32, mean [C]
    rstd_ptr,        # *f32, rstd [C]
    scale_ptr,       # *f32, scale [C]
    bias_ptr,        # *f32, bias [C]
    y_ptr,           # *f32, output [B, C, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, channels_per_group: tl.constexpr, N_TILES: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    group = tl.program_id(1)
    c = tl.program_id(2)  # absolute channel index
    t = tl.program_id(3)  # tile index over HW

    HW = H * W
    tile_start = t * BLOCK_HW
    idxs = tile_start + tl.arange(0, BLOCK_HW)
    mask = idxs < HW

    h = idxs // W
    w = idxs % W

    base = n * C * HW
    chan_offset = c * HW

    x_ptrs = x_ptr + base + chan_offset + idxs
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

    mean_c = tl.load(mean_ptr + c)
    rstd_c = tl.load(rstd_ptr + c)
    scale_c = tl.load(scale_ptr + c)
    bias_c = tl.load(bias_ptr + c)

    norm = (x_vals - mean_c) * rstd_c
    y_vals = norm * scale_c + bias_c

    # SiLU(z) = z * sigmoid(z)
    sig = 1.0 / (1.0 + tl.exp(-y_vals))
    y_vals = y_vals * sig

    y_ptrs = y_ptr + base + chan_offset + idxs
    tl.store(y_ptrs, y_vals, mask=mask)


# Triton elementwise residual add: out = y + x
@triton.jit
def residual_add_kernel(
    y_ptr, x_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    hw = tl.program_id(2)

    HW = H * W
    idxs = hw + tl.arange(0, 1)  # single element per program in this simple case
    mask = hw < HW  # ensure we don't go out of bounds

    base = n * C * HW
    chan_offset = c * HW

    y_ptrs = y_ptr + base + chan_offset + hw
    x_ptrs = x_ptr + base + chan_offset + hw
    out_ptrs = out_ptr + base + chan_offset + hw

    y_val = tl.load(y_ptrs, mask=mask, other=0.0)
    x_val = tl.load(x_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, y_val + x_val, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        x: (B, C, H, W)
        conv1_weight: (C_out1, C, 3, 3)
        norm1_weight, norm1_bias: (C_out1,)
        conv2_weight: (C_out2, C_out1, 3, 3)
        norm2_weight, norm2_bias: (C_out2,)
        Returns: (B, C_out2, H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"
        B, C_in, H, W = x.shape

        # Ensure contiguous float32 for Triton math
        x_f32 = x.contiguous().to(torch.float32)

        # First conv: robust PyTorch conv2d
        out1 = torch.nn.functional.conv2d(
            x_f32,
            conv1_weight.contiguous().to(torch.float32),
            bias=None, stride=1, padding=1
        )

        # GroupNorm + SiLU for first block
        C_out1 = out1.shape[1]
        mean1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        rstd1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)

        HW1 = H * W
        BLOCK_HW = 1024
        N_TILES1 = (HW1 + BLOCK_HW - 1) // BLOCK_HW

        # We set channels_per_group=1 to avoid any overlap: one program per absolute channel c
        grid_reduce1 = (B, self.num_groups, C_out1 // self.num_groups)  # here, C_out1 // num_groups should be 1 to match original; if not, this still works because we reduce over all channels; but for GroupNorm with 32 groups, C must be divisible by 32; we enforce it by using only those C_out1 channels.
        # However, in PyTorch, GroupNorm groups are over C, not over output channels. We should reduce over input channels. Correction: We need to compute per-channel statistics over spatial for each output channel. The above reduce kernel currently reduces over input channels. To match PyTorch GroupNorm, we must reduce per output channel over spatial (H*W), not input channels. Therefore, we need to adjust the reduce kernel to iterate over HW, not input channels.

        # Correction: Implement per-output-channel GroupNorm stats (reduction over H*W). We need to change reduce kernel to reduce per c_out over spatial.
        # Define a new reduce kernel specialized for per-output-channel stats.

        # We will instead do: GroupNorm requires reducing over spatial for each channel. In our code, out1 has shape (B, C_out1, H, W). We need stats per channel of out1. So reduce per output channel c in [0..C_out1-1].
        # Launch reduce kernel with grid (B, C_out1), channels_per_group=1, num_groups=1? No, PyTorch GroupNorm uses num_groups over input channels. Our model applies GroupNorm to the conv output, which has C_out channels. PyTorch GroupNorm expects C divisible by num_groups. Here, C is C_out1 or C_out2. To match PyTorch, C_out1 must be divisible by 32. The original code uses GroupNorm with num_groups=32, so C_out1 must be divisible by 32. Similarly for C_out2.

        # Therefore, to be correct, we must ensure that C_out1 % 32 == 0 and C_out2 % 32 == 0. If not, we can fall back to PyTorch GroupNorm, but since we must use Triton, we will enforce divisibility and structure the reduce/apply accordingly.

        # Adjust reduce kernel to reduce per output channel c (i.e., over spatial H*W), and apply kernel will be per (B, group, c). For per-output-channel GroupNorm, we need to map channels to groups. Since channels are absolute, we can set num_groups=32 and channels_per_group = C_out1 // 32. If C_out1 is not divisible by 32, we cannot map channels into groups without overlap. The original code uses num_groups=32, implying C_out1 is divisible by 32. We will assert this.

        # Assert divisibility to match PyTorch GroupNorm behavior.
        assert C_out1 % self.num_groups == 0, "C_out1 must be divisible by num_groups for GroupNorm"
        assert C_out2 % self.num_groups == 0, "C_out2 must be divisible by num_groups for GroupNorm"

        # Now launch the reduce with grid over (B, num_groups, channels_per_group) where channels_per_group = C_out // num_groups
        channels_per_group1 = C_out1 // self.num_groups

        # For reduce: we need one program per (n, group, c in group). Each program reduces over spatial.
        grid_reduce1 = (B, self.num_groups, channels_per_group1)
        group_norm_reduce_kernel[grid_reduce1](
            out1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group1, N_TILES=N_TILES1, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2,
        )

        # Apply GroupNorm + affine + SiLU
        out1_norm = torch.empty_like(out1)
        grid_apply1 = (B, self.num_groups, channels_per_group1, N_TILES1)
        group_norm_apply_kernel[grid_apply1](
            out1, mean1, rstd1, norm1_weight.to(torch.float32), norm1_bias.to(torch.float32),
            out1_norm,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group1, N_TILES=N_TILES1, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2,
        )

        # Second conv
        out2 = torch.nn.functional.conv2d(
            out1_norm,
            conv2_weight.contiguous().to(torch.float32),
            bias=None, stride=1, padding=1
        )

        # GroupNorm + SiLU for second block
        C_out2 = out2.shape[1]
        mean2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        rstd2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)

        assert C_out2 % self.num_groups == 0, "C_out2 must be divisible by num_groups for GroupNorm"

        channels_per_group2 = C_out2 // self.num_groups
        N_TILES2 = (H * W + BLOCK_HW - 1) // BLOCK_HW

        grid_reduce2 = (B, self.num_groups, channels_per_group2)
        group_norm_reduce_kernel[grid_reduce2](
            out2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group2, N_TILES=N_TILES2, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2,
        )

        out2_norm = torch.empty_like(out2)
        grid_apply2 = (B, self.num_groups, channels_per_group2, N_TILES2)
        group_norm_apply_kernel[grid_apply2](
            out2, mean2, rstd2, norm2_weight.to(torch.float32), norm2_bias.to(torch.float32),
            out2_norm,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group2, N_TILES=N_TILES2, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2,
        )

        # Residual add in Triton: out = out2_norm + x_f32
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
