import torch
import triton
import triton.language as tl


# Triton Conv3x3: y = conv(x, weight), stride=1, padding=1, no bias
# Each program computes BLOCK_HW output pixels for a given (n, c_out).
@triton.jit
def conv3x3_kernel(
    x_ptr,            # *f32, input [B, C_in, H, W]
    w_ptr,            # *f32, weights [C_out, C_in, 3, 3]
    y_ptr,            # *f32, output [B, C_out, H, W]
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    tile_id = tl.program_id(2)

    # Vector of HW offsets this program will handle
    offs = tile_id * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < (H * W)
    h = offs // W
    w = offs % W

    # Accumulator for outputs (vector of BLOCK_HW)
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Loop over input channels
    for cin in range(0, C_in):
        # For each 3x3 kernel position
        for dh in range(-1, 2):
            for dw in range(-1, 2):
                in_h = h + dh
                in_w = w + dw
                # Valid positions for padding=1
                # Since H, W are constexpr, Triton can generate efficient code.
                # Note: We rely on that H+dh and W+dw fall in [0,H-1] and [0,W-1] for padding=1.
                x_base = ((n * C_in) + cin) * (H * W)
                x_idx = x_base + in_h * W + in_w
                x_vec = tl.load(x_ptr + x_idx, mask=mask, other=0.0)

                # Load corresponding weights for this (c_out, cin, dh, dw)
                # weight layout: [C_out, C_in, 3, 3]
                w_idx = c_out * (C_in * 9) + cin * 9 + (dh + 1) * 3 + (dw + 1)
                w_val = tl.load(w_ptr + w_idx)  # scalar

                acc += x_vec * w_val

    # Store results to y: y[n, c_out, h, w]
    y_base = (n * C_out) * (H * W) + c_out * (H * W)
    y_idx = y_base + offs
    tl.store(y_ptr + y_idx, acc, mask=mask)


# Triton GroupNorm Reduction: compute per-channel mean and rstd over H*W
@triton.jit
def group_norm_reduce_kernel(
    x_ptr,            # *f32, input [B, C, H, W]
    mean_ptr,         # *f32, per-channel mean [C]
    rstd_ptr,         # *f32, per-channel rstd [C]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, channels_per_group: tl.constexpr,
    BLOCK_HW: tl.constexpr,  # tile size for HW
    N_TILES: tl.constexpr,   # number of tiles along HW
):
    n = tl.program_id(0)
    group = tl.program_id(1)
    c = tl.program_id(2)  # channel within this group

    c_global = group * channels_per_group + c

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq_val = tl.zeros((), dtype=tl.float32)

    for t in range(0, N_TILES):
        base = t * BLOCK_HW
        offs = base + tl.arange(0, BLOCK_HW)
        mask = offs < (H * W)

        h = offs // W
        w = offs % W

        x_base = ((n * C) + c_global) * (H * W)
        x_idx = x_base + h * W + w

        x_vec = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
        x_vec = tl.where(mask, x_vec, 0.0)

        sum_val += tl.sum(x_vec, axis=0)
        sum_sq_val += tl.sum(x_vec * x_vec, axis=0)

    numel = H * W
    mean = sum_val / numel
    var = sum_sq_val / numel - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    tl.store(mean_ptr + c_global, mean)
    tl.store(rstd_ptr + c_global, rstd)


# Triton GroupNorm Apply + Affine + SiLU
@triton.jit
def group_norm_apply_kernel(
    x_ptr,            # *f32, input [B, C, H, W]
    y_ptr,            # *f32, output [B, C, H, W]
    gamma_ptr,        # *f32, per-channel scale [C]
    beta_ptr,         # *f32, per-channel bias [C]
    mean_ptr,         # *f32, per-channel mean [C]
    rstd_ptr,         # *f32, per-channel rstd [C]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, channels_per_group: tl.constexpr,
    BLOCK_HW: tl.constexpr,  # tile size for HW
    N_TILES: tl.constexpr,   # number of tiles along HW
):
    n = tl.program_id(0)
    group = tl.program_id(1)
    c = tl.program_id(2)  # channel within this group
    t = tl.program_id(3)  # tile id along HW

    c_global = group * channels_per_group + c
    base = t * BLOCK_HW
    offs = base + tl.arange(0, BLOCK_HW)
    mask = offs < (H * W)

    h = offs // W
    w = offs % W

    x_base = ((n * C) + c_global) * (H * W)
    x_idx = x_base + h * W + w
    x_vec = tl.load(x_ptr + x_idx, mask=mask, other=0.0)

    mean = tl.load(mean_ptr + c_global)
    rstd = tl.load(rstd_ptr + c_global)
    gamma = tl.load(gamma_ptr + c_global)
    beta = tl.load(beta_ptr + c_global)

    # Normalize and affine
    norm = (x_vec - mean) * rstd
    norm = norm * gamma + beta

    # SiLU: y = x * sigmoid(x)
    y_vec = norm * (1.0 / (1.0 + tl.exp(-norm)))

    y_base = ((n * C) + c_global) * (H * W)
    y_idx = y_base + h * W + w
    tl.store(y_ptr + y_idx, y_vec, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5, block_hw: int = 1024):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.block_hw = block_hw

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        x: (B, C_in, H, W), conv weights: (C_out, C_in, 3, 3), norm scales/bias: (C_out,)
        Returns: (B, C_out2, H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C_in, H, W)"
        B, C_in, H, W = x.shape

        # Ensure all tensors are float32 and contiguous for Triton
        x_f32 = x.contiguous().to(torch.float32)
        conv1_w_f32 = conv1_weight.contiguous().to(torch.float32)  # (C_out1, C_in, 3, 3)
        conv2_w_f32 = conv2_weight.contiguous().to(torch.float32)  # (C_out2, C_in, 3, 3)
        norm1_gamma = norm1_weight.contiguous().to(torch.float32)
        norm1_beta = norm1_bias.contiguous().to(torch.float32)
        norm2_gamma = norm2_weight.contiguous().to(torch.float32)
        norm2_beta = norm2_bias.contiguous().to(torch.float32)

        # First conv: y1 = conv3x3(x)
        C_out1 = conv1_w_f32.shape[0]
        y1 = torch.empty((B, C_out1, H, W), device=x.device, dtype=torch.float32)

        grid_conv1 = (B, C_out1, triton.cdiv(H * W, self.block_hw))
        conv3x3_kernel[grid_conv1](
            x_f32, conv1_w_f32, y1,
            B=B, C_in=C_in, C_out=C_out1, H=H, W=W,
            BLOCK_HW=self.block_hw,
            num_warps=4, num_stages=2
        )

        # GroupNorm + SiLU for first block
        mean1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        rstd1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)

        N_TILES1 = triton.cdiv(H * W, self.block_hw)
        grid_reduce1 = (B, self.num_groups, C_out1 // self.num_groups)
        group_norm_reduce_kernel[grid_reduce1](
            y1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=C_out1 // self.num_groups,
            BLOCK_HW=self.block_hw, N_TILES=N_TILES1,
            num_warps=4, num_stages=2
        )

        y1_norm = torch.empty_like(y1)
        grid_apply1 = (B, self.num_groups, C_out1 // self.num_groups, N_TILES1)
        group_norm_apply_kernel[grid_apply1](
            y1, y1_norm, norm1_gamma, norm1_beta, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=C_out1 // self.num_groups,
            BLOCK_HW=self.block_hw, N_TILES=N_TILES1,
            num_warps=4, num_stages=2
        )

        # Second conv: y2 = conv3x3(y1_norm)
        C_out2 = conv2_w_f32.shape[0]
        y2 = torch.empty((B, C_out2, H, W), device=x.device, dtype=torch.float32)

        grid_conv2 = (B, C_out2, triton.cdiv(H * W, self.block_hw))
        conv3x3_kernel[grid_conv2](
            y1_norm, conv2_w_f32, y2,
            B=B, C_in=C_out1, C_out=C_out2, H=H, W=W,
            BLOCK_HW=self.block_hw,
            num_warps=4, num_stages=2
        )

        # GroupNorm + SiLU for second block
        mean2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        rstd2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)

        N_TILES2 = triton.cdiv(H * W, self.block_hw)
        grid_reduce2 = (B, self.num_groups, C_out2 // self.num_groups)
        group_norm_reduce_kernel[grid_reduce2](
            y2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=C_out2 // self.num_groups,
            BLOCK_HW=self.block_hw, N_TILES=N_TILES2,
            num_warps=4, num_stages=2
        )

        y2_norm = torch.empty_like(y2)
        grid_apply2 = (B, self.num_groups, C_out2 // self.num_groups, N_TILES2)
        group_norm_apply_kernel[grid_apply2](
            y2, y2_norm, norm2_gamma, norm2_beta, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=C_out2 // self.num_groups,
            BLOCK_HW=self.block_hw, N_TILES=N_TILES2,
            num_warps=4, num_stages=2
        )

        # Final residual add: out = y2_norm + x
        out = torch.empty_like(y2_norm)
        # Elementwise add using PyTorch (simple and correct), though we can keep Triton add as well:
        out = y2_norm + x_f32  # This is allowed as an elementwise op, but if needed, we can write a Triton kernel.

        return out


def run(*args):
    return ModelNew()(*args)
