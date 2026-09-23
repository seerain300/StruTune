import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton GroupNorm reduction kernel:
# For each (n, group, c in group), compute sum and sum of squares over all H*W elements.
# Inputs:
#   x_ptr: *f32, input tensor flattened to [B*C, H*W]
#   mean_ptr: *f32, output mean per channel [C]
#   rstd_ptr: *f32, output rstd per channel [C]
# Meta-parameters:
#   C, H, W, NUM_GROUPS, GROUP_SIZE, N_TILES, BLOCK_HW, all tl.constexpr
@triton.jit
def group_norm_reduce_kernel(
    x_ptr,
    mean_ptr,
    rstd_ptr,
    C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    NUM_GROUPS: tl.constexpr, GROUP_SIZE: tl.constexpr,
    N_TILES: tl.constexpr, BLOCK_HW: tl.constexpr
):
    n = tl.program_id(0)
    g = tl.program_id(1)  # group id
    c = tl.program_id(2)  # channel id within this group

    group_hw = GROUP_SIZE * H * W  # total elements for this channel across the group

    sum_val = 0.0
    sum_sq = 0.0

    # Iterate over spatial tiles to accumulate sum and sum of squares
    for t in range(N_TILES):
        start = t * BLOCK_HW
        hw_vec = start + tl.arange(0, BLOCK_HW)
        mask = hw_vec < (H * W)

        base_nc = (n * C + c) * (H * W)
        x_vals = tl.load(x_ptr + base_nc + hw_vec, mask=mask, other=0.0)
        sum_val += tl.sum(x_vals)
        sum_sq += tl.sum(x_vals * x_vals)

    mean = sum_val / group_hw
    var = sum_sq / group_hw - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    tl.store(mean_ptr + c, mean)
    tl.store(rstd_ptr + c, rstd)


# Triton GroupNorm apply + SiLU:
# For each (n, group, c in group, tile), apply y = ((x - mean[c]) * rstd[c] * scale[c] + bias[c]) then SiLU
@triton.jit
def group_norm_apply_silu_kernel(
    x_ptr,
    mean_ptr,
    rstd_ptr,
    scale_ptr,
    bias_ptr,
    y_ptr,
    C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    NUM_GROUPS: tl.constexpr, GROUP_SIZE: tl.constexpr,
    N_TILES: tl.constexpr, BLOCK_HW: tl.constexpr
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    c = tl.program_id(2)
    t = tl.program_id(3)

    mean = tl.load(mean_ptr + c)
    rstd = tl.load(rstd_ptr + c)
    scale = tl.load(scale_ptr + c)
    beta = tl.load(bias_ptr + c)

    start = t * BLOCK_HW
    hw_vec = start + tl.arange(0, BLOCK_HW)
    mask = hw_vec < (H * W)

    base_nc = (n * C + c) * (H * W)
    x_vals = tl.load(x_ptr + base_nc + hw_vec, mask=mask, other=0.0)
    norm = (x_vals - mean) * rstd
    lin = norm * scale + beta
    silu = lin * tl.sigmoid(lin)
    tl.store(y_ptr + base_nc + hw_vec, silu, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5, block_hw: int = 256):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.block_hw = block_hw

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        x: (B, C_in, H, W)
        conv weights: (C_out, C_in, 3, 3)
        norm params: (C,)
        Returns: (B, C_out2, H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"
        B, C_in, H, W = x.shape

        # First path: Conv3x3 -> GroupNorm -> SiLU
        # Use PyTorch conv for correctness and performance
        out1 = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # GroupNorm reduction in Triton
        C_out1 = out1.shape[1]
        assert C_out1 % self.num_groups == 0, "C_out1 must be divisible by num_groups"
        GROUP_SIZE1 = C_out1 // self.num_groups
        N_TILES1 = (H * W + self.block_hw - 1) // self.block_hw

        # Flatten for Triton kernels
        out1_flat = out1.contiguous().view(B * C_out1, H * W).to(torch.float32)
        mean1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        rstd1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)

        grid_reduce1 = (B, self.num_groups, GROUP_SIZE1)
        group_norm_reduce_kernel[grid_reduce1](
            out1_flat, mean1, rstd1,
            C=C_out1, H=H, W=W,
            NUM_GROUPS=self.num_groups, GROUP_SIZE=GROUP_SIZE1,
            N_TILES=N_TILES1, BLOCK_HW=self.block_hw,
            num_warps=4, num_stages=2
        )

        # Apply GroupNorm + SiLU in Triton
        out1_norm = torch.empty_like(out1)
        out1_norm_flat = out1_norm.view(B * C_out1, H * W)
        grid_apply1 = (B, self.num_groups, GROUP_SIZE1, N_TILES1)
        group_norm_apply_silu_kernel[grid_apply1](
            out1_flat, mean1, rstd1, norm1_weight.to(torch.float32), norm1_bias.to(torch.float32),
            out1_norm_flat,
            C=C_out1, H=H, W=W,
            NUM_GROUPS=self.num_groups, GROUP_SIZE=GROUP_SIZE1,
            N_TILES=N_TILES1, BLOCK_HW=self.block_hw,
            num_warps=4, num_stages=2
        )

        # Second path: Conv3x3 -> GroupNorm -> SiLU
        out2 = F.conv2d(out1_norm, conv2_weight, bias=None, stride=1, padding=1)

        C_out2 = out2.shape[1]
        assert C_out2 % self.num_groups == 0, "C_out2 must be divisible by num_groups"
        GROUP_SIZE2 = C_out2 // self.num_groups
        N_TILES2 = (H * W + self.block_hw - 1) // self.block_hw

        out2_flat = out2.contiguous().view(B * C_out2, H * W).to(torch.float32)
        mean2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        rstd2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)

        grid_reduce2 = (B, self.num_groups, GROUP_SIZE2)
        group_norm_reduce_kernel[grid_reduce2](
            out2_flat, mean2, rstd2,
            C=C_out2, H=H, W=W,
            NUM_GROUPS=self.num_groups, GROUP_SIZE=GROUP_SIZE2,
            N_TILES=N_TILES2, BLOCK_HW=self.block_hw,
            num_warps=4, num_stages=2
        )

        out2_norm = torch.empty_like(out2)
        out2_norm_flat = out2_norm.view(B * C_out2, H * W)
        grid_apply2 = (B, self.num_groups, GROUP_SIZE2, N_TILES2)
        group_norm_apply_silu_kernel[grid_apply2](
            out2_flat, mean2, rstd2, norm2_weight.to(torch.float32), norm2_bias.to(torch.float32),
            out2_norm_flat,
            C=C_out2, H=H, W=W,
            NUM_GROUPS=self.num_groups, GROUP_SIZE=GROUP_SIZE2,
            N_TILES=N_TILES2, BLOCK_HW=self.block_hw,
            num_warps=4, num_stages=2
        )

        # Residual add: out2_norm + x
        out = out2_norm + x

        return out


def run(*args):
    return ModelNew()(*args)
