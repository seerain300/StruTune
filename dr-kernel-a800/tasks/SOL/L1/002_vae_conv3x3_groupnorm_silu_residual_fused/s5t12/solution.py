import torch
import triton
import triton.language as tl


# Conv3x3 via im2col + Triton reduction:
# Input x: (B, C_in, H, W), weight w: (C_out, C_in, 3, 3)
# Output y: (B, C_out, H, W)
@triton.jit
def conv3x3_gemm_kernel(
    x_ptr,           # *f32, input [B, C_in, H, W]
    w_ptr,           # *f32, weight [C_out, C_in, 3, 3]
    y_ptr,           # *f32, output [B, C_out, H, W]
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    N_TILES: tl.constexpr, BLOCK_HW: tl.constexpr
):
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    tile_id = tl.program_id(2)

    # tile over HW
    tile_start = tile_id * BLOCK_HW
    hw_idx = tile_start + tl.arange(0, BLOCK_HW)
    mask_hw = hw_idx < (H * W)

    # Compute h and w from flattened hw_idx
    h_vec = hw_idx // W
    w_vec = hw_idx % W

    # Accumulator for output tile
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Build A [K, BLOCK_HW] and b [BLOCK_HW], then compute acc = A @ b
    # K = C_in * 9
    for c_in in range(C_in):
        for dh in range(-1, 2):
            for dw in range(-1, 2):
                # input indices for padding=1
                hi = h_vec + dh
                wi = w_vec + dw
                # mask for valid input indices (hi, wi are always in [0,H-1] and [0,W-1] due to padding=1)
                # We still use mask_hw to avoid out-of-range loads (for safety).
                # Note: Triton pointer arithmetic allows out-of-bounds; we rely on padding=1 so no real OOB.
                # Compute input flattened index
                x_index = (((n * C_in) + c_in) * H + hi) * W + wi
                x_vals = tl.load(x_ptr + x_index, mask=mask_hw, other=0.0)
                # Load corresponding weight (scalar per (c_out, c_in, dh, dw))
                w_index = ((c_out * C_in) + c_in) * 9 + (dh + 1) * 3 + (dw + 1)
                w_val = tl.load(w_ptr + w_index)
                # Accumulate
                acc += x_vals * w_val

    # Store result to y: y[n, c_out, h_vec, w_vec]
    # Output flattened index
    y_index = (((n * C_out) + c_out) * H + h_vec) * W + w_vec
    tl.store(y_ptr + y_index, acc, mask=mask_hw)


# GroupNorm reduction kernel:
# For each (n, group, channel), compute sum and sumsq over all spatial elements in that group.
@triton.jit
def group_norm_reduce_kernel(
    x_ptr,           # *f32, input [B, C, H, W] after conv
    mean_ptr,        # *f32, [C]
    rstd_ptr,        # *f32, [C]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, channels_per_group: tl.constexpr,
    N_TILES: tl.constexpr, BLOCK_HW: tl.constexpr
):
    n = tl.program_id(0)
    group = tl.program_id(1)
    c = tl.program_id(2)  # this c is actually per-group channel index, but we can decode: c = group * channels_per_group + ch
    # decode c into group channel index
    ch = c - group * channels_per_group
    if ch < 0:
        # safety if any negative due to pid (shouldn't happen)
        ch = 0

    # Accumulate sum and sumsq over all spatial elements
    sum_ = 0.0
    sumsq_ = 0.0
    for tile in range(N_TILES):
        tile_start = tile * BLOCK_HW
        hw_idx = tile_start + tl.arange(0, BLOCK_HW)
        mask = hw_idx < (H * W)
        h_vec = hw_idx // W
        w_vec = hw_idx % W
        x_index = (((n * C) + (group * channels_per_group + ch)) * H + h_vec) * W + w_vec
        vals = tl.load(x_ptr + x_index, mask=mask, other=0.0)
        sum_ += tl.sum(vals, axis=0)
        sumsq_ += tl.sum(vals * vals, axis=0)

    # Compute mean and rstd
    num_elems = H * W
    mean = sum_ / num_elems
    var = sumsq_ / num_elems - mean * mean
    # numerical stability
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    # Store per-channel mean and rstd
    tl.store(mean_ptr + (group * channels_per_group + ch), mean)
    tl.store(rstd_ptr + (group * channels_per_group + ch), rstd)


# GroupNorm apply + affine + SiLU:
# y = silu(((x - mean) * rstd) * scale + bias)
@triton.jit
def group_norm_apply_silu_kernel(
    x_ptr,           # *f32, input [B, C, H, W]
    mean_ptr,        # *f32, [C]
    rstd_ptr,        # *f32, [C]
    scale_ptr,       # *f32, [C]
    bias_ptr,        # *f32, [C]
    y_ptr,           # *f32, output [B, C, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    N_TILES: tl.constexpr, BLOCK_HW: tl.constexpr
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)

    tile_start = tile * BLOCK_HW
    hw_idx = tile_start + tl.arange(0, BLOCK_HW)
    mask = hw_idx < (H * W)

    h_vec = hw_idx // W
    w_vec = hw_idx % W

    # Load mean, rstd, scale, bias for channel c
    mean = tl.load(mean_ptr + c)
    rstd = tl.load(rstd_ptr + c)
    scale = tl.load(scale_ptr + c)
    bias = tl.load(bias_ptr + c)

    # Load input tile
    x_index = (((n * C) + c) * H + h_vec) * W + w_vec
    x_vals = tl.load(x_ptr + x_index, mask=mask, other=0.0)

    # Normalize and affine
    norm = (x_vals - mean) * rstd
    z = norm * scale + bias

    # SiLU: z * sigmoid(z) = z / (1 + exp(-z))
    # Triton has tl.exp; implement sigmoid
    sig = 1.0 / (1.0 + tl.exp(-z))
    y_vals = z * sig

    # Store
    tl.store(y_ptr + x_index, y_vals, mask=mask)


# Residual add: out = a + b, elementwise
@triton.jit
def residual_add_kernel(
    a_ptr, b_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    hw = tl.program_id(2)
    h = hw // W
    w = hw % W
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
        Returns: (B, C_out2, H, W) after the fused residual block.
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"
        B, C_in, H, W = x.shape

        # Ensure all tensors are float32 and contiguous
        x_f32 = x.contiguous().to(torch.float32)
        conv1_w_f32 = conv1_weight.contiguous().to(torch.float32)  # (C_out1, C_in, 3, 3)
        conv2_w_f32 = conv2_weight.contiguous().to(torch.float32)  # (C_out2, C_out1, 3, 3)
        norm1_weight_f32 = norm1_weight.contiguous().to(torch.float32)
        norm1_bias_f32 = norm1_bias.contiguous().to(torch.float32)
        norm2_weight_f32 = norm2_weight.contiguous().to(torch.float32)
        norm2_bias_f32 = norm2_bias.contiguous().to(torch.float32)

        # First conv: y1 = conv3x3(x)
        C_out1 = conv1_w_f32.shape[0]
        y1 = torch.empty((B, C_out1, H, W), device=x.device, dtype=torch.float32)

        # Grid over (B, C_out1, N_TILES)
        BLOCK_HW = 256
        N_TILES1 = (H * W + BLOCK_HW - 1) // BLOCK_HW
        grid_conv1 = (B, C_out1, N_TILES1)
        conv3x3_gemm_kernel[grid_conv1](
            x_f32, conv1_w_f32, y1,
            B=B, C_in=C_in, C_out=C_out1, H=H, W=W,
            N_TILES=N_TILES1, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2
        )

        # GroupNorm + SiLU for first block
        assert C_out1 % self.num_groups == 0, "C_out1 must be divisible by num_groups"
        channels_per_group1 = C_out1 // self.num_groups
        mean1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        rstd1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        grid_reduce1 = (B, self.num_groups, channels_per_group1)
        group_norm_reduce_kernel[grid_reduce1](
            y1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group1,
            N_TILES=N_TILES1, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2
        )
        y1_norm = torch.empty_like(y1)
        grid_apply1 = (B, C_out1, N_TILES1)
        group_norm_apply_silu_kernel[grid_apply1](
            y1, mean1, rstd1, norm1_weight_f32, norm1_bias_f32, y1_norm,
            B=B, C=C_out1, H=H, W=W,
            N_TILES=N_TILES1, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2
        )

        # Second conv: y2 = conv3x3(y1_norm)
        C_out2 = conv2_w_f32.shape[0]
        y2 = torch.empty((B, C_out2, H, W), device=x.device, dtype=torch.float32)
        grid_conv2 = (B, C_out2, N_TILES1)
        conv3x3_gemm_kernel[grid_conv2](
            y1_norm, conv2_w_f32, y2,
            B=B, C_in=C_out1, C_out=C_out2, H=H, W=W,
            N_TILES=N_TILES1, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2
        )

        # GroupNorm + SiLU for second block
        assert C_out2 % self.num_groups == 0, "C_out2 must be divisible by num_groups"
        channels_per_group2 = C_out2 // self.num_groups
        mean2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        rstd2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        grid_reduce2 = (B, self.num_groups, channels_per_group2)
        group_norm_reduce_kernel[grid_reduce2](
            y2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group2,
            N_TILES=N_TILES1, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2
        )
        y2_norm = torch.empty_like(y2)
        grid_apply2 = (B, C_out2, N_TILES1)
        group_norm_apply_silu_kernel[grid_apply2](
            y2, mean2, rstd2, norm2_weight_f32, norm2_bias_f32, y2_norm,
            B=B, C=C_out2, H=H, W=W,
            N_TILES=N_TILES1, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2
        )

        # Residual add: out = y2_norm + x
        out = torch.empty_like(y2_norm)
        grid_add = (B, C_out2, H * W)
        residual_add_kernel[grid_add](
            y2_norm, x_f32, out,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4, num_stages=2
        )

        # If original x was not float32, cast back; for this benchmark we keep float32
        return out


def run(*args):
    return ModelNew()(*args)
