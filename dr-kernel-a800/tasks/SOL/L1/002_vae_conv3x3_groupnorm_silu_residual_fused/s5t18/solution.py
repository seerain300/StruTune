import torch
import triton
import triton.language as tl


# Conv3x3: y = x * w, stride=1, padding=1, no bias
# Grid: (B, C_out, H, N_W_TILES)
@triton.jit
def conv3x3_kernel_hw_tiled(
    x_ptr,           # *f32, input [B, C_in, H, W]
    w_ptr,           # *f32, weight [C_out, C_in, 3, 3]
    y_ptr,           # *f32, output [B, C_out, H, W]
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    BLOCK_W: tl.constexpr, N_W_TILES: tl.constexpr,
):
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    h_out = tl.program_id(2)
    tile_id = tl.program_id(3)  # along W tiles

    w_offsets = tile_id * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W

    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    # Accumulate over input channels and 3x3 neighborhood, masked for padding
    for c_in in range(0, C_in):
        for dh in range(-1, 2):
            h_in = h_out + dh
            in_bounds_h = (h_in >= 0) & (h_in < H)
            for dw in range(-1, 2):
                w_in = w_offsets + dw
                in_bounds = mask_w & (w_in >= 0) & (w_in < W) & in_bounds_h
                # Input offset: (((n * C_in) + c_in) * H + h_in) * W + w_in
                x_base = (((n * C_in) + c_in) * H + h_in) * W
                x_vec = tl.load(x_ptr + x_base + w_in, mask=in_bounds, other=0.0)
                # Weight offset: (((c_out * C_in) + c_in) * 9) + (dh+1)*3 + (dw+1)
                w_off = (((c_out * C_in) + c_in) * 9) + (dh + 1) * 3 + (dw + 1)
                w_val = tl.load(w_ptr + w_off)
                acc += x_vec * w_val

    # Store output vector along W for this (n, c_out, h_out)
    y_base = (((n * C_out) + c_out) * H + h_out) * W
    tl.store(y_ptr + y_base + w_offsets, acc, mask=mask_w)


# Triton elementwise kernel: apply per-channel affine and SiLU: y = (x * gamma + beta) * sigmoid(x)
@triton.jit
def affine_silu_kernel(
    x_ptr, gamma_ptr, beta_ptr, y_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    hw = tl.program_id(2)

    h = hw // W
    w = hw % W

    idx = (((n * C) + c) * H + h) * W + w
    x_val = tl.load(x_ptr + idx)
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig * gamma + beta
    tl.store(y_ptr + idx, y_val)


# Elementwise residual add: out = a + b, both (B, C, H, W)
@triton.jit
def residual_add_kernel(
    a_ptr, b_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
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


# GroupNorm reduction: per-channel (n,g) compute sum and sumsq over all HW tiles
@triton.jit
def group_norm_reduce_kernel(
    x_ptr,           # *f32, input [B, C, H, W]
    mean_ptr,        # *f32, output [C]
    rstd_ptr,        # *f32, output [C]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, N_TILES: tl.constexpr,
):
    # One program per (n, group, channel)
    total_groups = num_groups
    ch_per_group = C // total_groups
    pid = tl.program_id(0)
    n = pid // (total_groups * ch_per_group)
    g = (pid % (total_groups * ch_per_group)) // ch_per_group
    ch = pid % ch_per_group
    c = g * ch_per_group + ch

    s = tl.zeros((), dtype=tl.float32)
    ss = tl.zeros((), dtype=tl.float32)

    for t in range(0, N_TILES):
        hw_start = t * (H * W)
        hw_offsets = hw_start + tl.arange(0, H * W)
        mask = (hw_offsets >= hw_start) & (hw_offsets < hw_start + H * W)
        # Linear index into (B, C, H, W) flattened over HW
        idx = (((n * C) + c) * (H * W)) + (hw_offsets % (H * W))
        x_vec = tl.load(x_ptr + idx, mask=mask, other=0.0)
        s += tl.sum(x_vec, axis=0)
        ss += tl.sum(x_vec * x_vec, axis=0)

    numel = H * W
    mean = s / numel
    var = ss / numel - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)  # use eps=1e-5 to match common practice

    tl.store(mean_ptr + c, mean)
    tl.store(rstd_ptr + c, rstd)


# GroupNorm apply: normalize + affine + SiLU using precomputed mean/rstd
@triton.jit
def group_norm_apply_kernel(
    x_ptr, mean_ptr, rstd_ptr, gamma_ptr, beta_ptr, y_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, N_TILES: tl.constexpr,
):
    # One program per (n, group, channel, tile)
    total_groups = num_groups
    ch_per_group = C // total_groups
    pid = tl.program_id(0)
    n = pid // (total_groups * ch_per_group * N_TILES)
    g = (pid % (total_groups * ch_per_group * N_TILES)) // (ch_per_group * N_TILES)
    tile = (pid % (ch_per_group * N_TILES)) // ch_per_group
    ch = pid % ch_per_group
    c = g * ch_per_group + ch

    mean = tl.load(mean_ptr + c)
    rstd = tl.load(rstd_ptr + c)
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)

    hw_start = tile * (H * W)
    hw_offsets = hw_start + tl.arange(0, H * W)
    mask = (hw_offsets >= hw_start) & (hw_offsets < hw_start + H * W)
    # Linear index into (B, C, H, W) flattened over HW
    idx = (((n * C) + c) * (H * W)) + (hw_offsets % (H * W))
    x_vec = tl.load(x_ptr + idx, mask=mask, other=0.0)
    y_vec = (x_vec - mean) * rstd
    # SiLU
    sig = 1.0 / (1.0 + tl.exp(-y_vec))
    y_vec = y_vec * sig * gamma + beta
    tl.store(y_ptr + idx, y_vec, mask=mask)


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

        # Ensure contiguous and float32 for Triton
        x_f32 = x.contiguous().to(torch.float32)

        # First conv: C_in -> C_out1
        C_out1 = conv1_weight.shape[0]
        out1 = torch.empty((B, C_out1, H, W), device=x.device, dtype=torch.float32)

        BLOCK_W = 128  # tile size along W; tuneable
        N_W_TILES = (W + BLOCK_W - 1) // BLOCK_W
        grid1 = (B, C_out1, H, N_W_TILES)
        conv3x3_kernel_hw_tiled[grid1](
            x_f32, conv1_weight.contiguous().to(torch.float32), out1,
            B=B, C_in=C_in, C_out=C_out1, H=H, W=W,
            BLOCK_W=BLOCK_W, N_W_TILES=N_W_TILES,
            num_warps=4, num_stages=2,
        )

        # GroupNorm + SiLU for first block
        C = C_out1
        mean1 = torch.empty(C, device=x.device, dtype=torch.float32)
        rstd1 = torch.empty(C, device=x.device, dtype=torch.float32)
        assert C % self.num_groups == 0, "C_out1 must be divisible by num_groups"
        ch_per_group = C // self.num_groups
        N_TILES1 = (H * W + 1024 - 1) // 1024  # reduce over entire HW in tiles; 1024 elements per tile
        grid_reduce1 = (B * self.num_groups * ch_per_group,)
        group_norm_reduce_kernel[grid_reduce1](
            out1, mean1, rstd1,
            B=B, C=C, H=H, W=W,
            num_groups=self.num_groups, N_TILES=N_TILES1,
            num_warps=4, num_stages=2,
        )
        out1_norm = torch.empty_like(out1)
        grid_apply1 = (B * self.num_groups * ch_per_group,)
        group_norm_apply_kernel[grid_apply1](
            out1, mean1, rstd1, norm1_weight_f32, norm1_bias_f32, out1_norm,
            B=B, C=C, H=H, W=W,
            num_groups=self.num_groups, N_TILES=N_TILES1,
            num_warps=4, num_stages=2,
        )

        # Second conv: C_out1 -> C_out2
        C_in2 = C_out1
        C_out2 = conv2_weight.shape[0]
        out2 = torch.empty((B, C_out2, H, W), device=x.device, dtype=torch.float32)

        BLOCK_W2 = 128
        N_W_TILES2 = (W + BLOCK_W2 - 1) // BLOCK_W2
        grid2 = (B, C_out2, H, N_W_TILES2)
        conv3x3_kernel_hw_tiled[grid2](
            out1_norm, conv2_weight.contiguous().to(torch.float32), out2,
            B=B, C_in=C_in2, C_out=C_out2, H=H, W=W,
            BLOCK_W=BLOCK_W2, N_W_TILES=N_W_TILES2,
            num_warps=4, num_stages=2,
        )

        # GroupNorm + SiLU for second block
        C = C_out2
        mean2 = torch.empty(C, device=x.device, dtype=torch.float32)
        rstd2 = torch.empty(C, device=x.device, dtype=torch.float32)
        assert C % self.num_groups == 0, "C_out2 must be divisible by num_groups"
        ch_per_group2 = C // self.num_groups
        N_TILES2 = (H * W + 1024 - 1) // 1024
        grid_reduce2 = (B * self.num_groups * ch_per_group2,)
        group_norm_reduce_kernel[grid_reduce2](
            out2, mean2, rstd2,
            B=B, C=C, H=H, W=W,
            num_groups=self.num_groups, N_TILES=N_TILES2,
            num_warps=4, num_stages=2,
        )
        out2_norm = torch.empty_like(out2)
        grid_apply2 = (B * self.num_groups * ch_per_group2,)
        group_norm_apply_kernel[grid_apply2](
            out2, mean2, rstd2, norm2_weight_f32, norm2_bias_f32, out2_norm,
            B=B, C=C, H=H, W=W,
            num_groups=self.num_groups, N_TILES=N_TILES2,
            num_warps=4, num_stages=2,
        )

        # Final residual add: out2_norm + x_f32
        out = torch.empty_like(out2_norm)
        grid_res = (B, C_out2, H * W)
        residual_add_kernel[grid_res](
            out2_norm, x_f32, out,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
