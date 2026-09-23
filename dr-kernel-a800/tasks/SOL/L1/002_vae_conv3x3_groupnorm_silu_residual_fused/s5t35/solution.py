import torch
import triton
import triton.language as tl


# Kernel 1: compute per-channel mean and rstd for GroupNorm
# Grid: (B, num_groups, channels_per_group)
@triton.jit
def group_norm_reduce_kernel(
    x_ptr,              # *f32, input [B, C, H, W]
    mean_ptr,           # *f32, output [C] per-channel mean
    rstd_ptr,           # *f32, output [C] per-channel rstd
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
    BLOCK_HW: tl.constexpr,            # tile size along HW
    N_TILES: tl.constexpr,             # number of tiles = ceil((H*W)/BLOCK_HW)
):
    n = tl.program_id(0)
    g = tl.program_id(1)                # group id
    ch = tl.program_id(2)               # channel within the group

    total_ch = C
    ch_per_group = total_ch // num_groups

    # this channel index within the overall channel set
    c = g * ch_per_group + ch

    # accumulators for sum and sumsq (scalars)
    sum_val = 0.0
    sum_sq = 0.0

    # loop over tiles along flattened HW
    for t in range(N_TILES):
        start = t * BLOCK_HW
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < (H * W)
        h = offs // W
        w = offs % W

        # index into x: (((n*C) + c) * H + h) * W + w
        base = (n * total_ch + c) * H * W
        idx = base + h * W + w

        x_vec = tl.load(x_ptr + idx, mask=mask, other=0.0)
        # reduce sum and sumsq
        sum_val += tl.sum(x_vec, axis=0)
        sum_sq += tl.sum(x_vec * x_vec, axis=0)

    numel = H * W
    mean = sum_val / numel
    var = sum_sq / numel - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)  # eps for stability

    # store mean and rstd
    tl.store(mean_ptr + c, mean)
    tl.store(rstd_ptr + c, rstd)


# Kernel 2: apply GroupNorm + affine + SiLU
# Grid: (B, num_groups, channels_per_group, N_TILES)
@triton.jit
def group_norm_apply_kernel(
    x_ptr, y_ptr,
    mean_ptr, rstd_ptr, weight_ptr, bias_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
    BLOCK_HW: tl.constexpr, N_TILES: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    ch = tl.program_id(2)
    t = tl.program_id(3)

    total_ch = C
    ch_per_group = total_ch // num_groups
    c = g * ch_per_group + ch

    start = t * BLOCK_HW
    offs = start + tl.arange(0, BLOCK_HW)
    mask = offs < (H * W)
    h = offs // W
    w = offs % W

    base_x = (n * total_ch + c) * H * W
    idx_x = base_x + h * W + w

    # load per-channel params
    mean_c = tl.load(mean_ptr + c)
    rstd_c = tl.load(rstd_ptr + c)
    gamma_c = tl.load(weight_ptr + c)
    beta_c = tl.load(bias_ptr + c)

    x_vec = tl.load(x_ptr + idx_x, mask=mask, other=0.0)
    # normalize and apply affine
    y_vec = (x_vec - mean_c) * rstd_c * gamma_c + beta_c
    # SiLU: y * sigmoid(y)
    sig = 1.0 / (1.0 + tl.exp(-y_vec))
    y_vec = y_vec * sig

    base_y = (n * total_ch + c) * H * W
    idx_y = base_y + h * W + w
    tl.store(y_ptr + idx_y, y_vec, mask=mask)


# Kernel 3: elementwise residual add
# Grid: (B, C, H*W)
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


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps  # not directly used in kernels; rstd uses 1e-5 for stability

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

        # Ensure float32 and contiguous
        x_f32 = x.contiguous().to(torch.float32)

        # First Conv
        C_out1 = conv1_weight.shape[0]
        out1 = torch.nn.functional.conv2d(x_f32, conv1_weight.to(torch.float32), bias=None, stride=1, padding=1)

        # First GroupNorm + SiLU using Triton
        mean1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        rstd1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)

        BLOCK_HW = 1024  # tile along HW
        N_TILES1 = (H * W + BLOCK_HW - 1) // BLOCK_HW

        grid_reduce1 = (B, self.num_groups, C_out1 // self.num_groups)
        group_norm_reduce_kernel[grid_reduce1](
            out1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups,
            BLOCK_HW=BLOCK_HW, N_TILES=N_TILES1,
            num_warps=4, num_stages=2,
        )

        out1_norm = torch.empty_like(out1)

        grid_apply1 = (B, self.num_groups, C_out1 // self.num_groups, N_TILES1)
        group_norm_apply_kernel[grid_apply1](
            out1, out1_norm, mean1, rstd1, norm1_weight.to(torch.float32), norm1_bias.to(torch.float32),
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups,
            BLOCK_HW=BLOCK_HW, N_TILES=N_TILES1,
            num_warps=4, num_stages=2,
        )

        # Second Conv
        C_out2 = conv2_weight.shape[0]
        out2 = torch.nn.functional.conv2d(out1_norm, conv2_weight.to(torch.float32), bias=None, stride=1, padding=1)

        # Second GroupNorm + SiLU using Triton
        mean2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        rstd2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)

        N_TILES2 = (H * W + BLOCK_HW - 1) // BLOCK_HW

        grid_reduce2 = (B, self.num_groups, C_out2 // self.num_groups)
        group_norm_reduce_kernel[grid_reduce2](
            out2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups,
            BLOCK_HW=BLOCK_HW, N_TILES=N_TILES2,
            num_warps=4, num_stages=2,
        )

        out2_norm = torch.empty_like(out2)

        grid_apply2 = (B, self.num_groups, C_out2 // self.num_groups, N_TILES2)
        group_norm_apply_kernel[grid_apply2](
            out2, out2_norm, mean2, rstd2, norm2_weight.to(torch.float32), norm2_bias.to(torch.float32),
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups,
            BLOCK_HW=BLOCK_HW, N_TILES=N_TILES2,
            num_warps=4, num_stages=2,
        )

        # Final residual add: out2_norm + x_f32
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
