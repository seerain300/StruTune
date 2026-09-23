import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def groupnorm_affine_kernel(
    in_ptr, out_ptr,
    weight_ptr, bias_ptr,
    N, C, H, W, NUM_GROUPS, EPS,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles one (n, group)
    n = tl.program_id(0)
    g = tl.program_id(1)

    GROUP_SIZE = C // NUM_GROUPS
    c0 = g * GROUP_SIZE

    # Accumulate sum and sum of squares across channels in group and all spatial positions
    sum_val = 0.0
    sum_sq = 0.0

    for ic in range(GROUP_SIZE):
        c = c0 + ic
        hw = H * W
        for start in range(0, hw, BLOCK_HW):
            offs = start + tl.arange(0, BLOCK_HW)
            mask = offs < hw
            h = offs // W
            w = offs % W
            in_offs = (n * C + c) * hw + offs
            x = tl.load(in_ptr + in_offs, mask=mask, other=0.0)
            sum_val += tl.sum(x, axis=0)
            sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / (GROUP_SIZE * H * W)
    var = sum_sq / (GROUP_SIZE * H * W) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + EPS)

    # Normalize and apply affine for each element in the group
    for ic in range(GROUP_SIZE):
        c = c0 + ic
        hw = H * W
        for start in range(0, hw, BLOCK_HW):
            offs = start + tl.arange(0, BLOCK_HW)
            mask = offs < hw
            h = offs // W
            w = offs % W
            in_offs = (n * C + c) * hw + offs
            x = tl.load(in_ptr + in_offs, mask=mask, other=0.0)
            y = (x - mean) * inv_std
            scale = tl.load(weight_ptr + c)
            bias = tl.load(bias_ptr + c)
            y = y * scale + bias
            out_offs = (n * C + c) * hw + offs
            tl.store(out_ptr + out_offs, y, mask=mask)


@triton.jit
def silu_kernel(in_ptr, out_ptr, TOTAL_ELEMS, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL_ELEMS
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def add_residual_kernel(a_ptr, b_ptr, out_ptr, TOTAL_ELEMS, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL_ELEMS
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    y = a + b
    tl.store(out_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5, block_hw: int = 1024, silu_block: int = 2048, add_block: int = 2048):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.block_hw = block_hw
        self.silu_block = silu_block
        self.add_block = add_block

    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        B, C, H, W = x.shape
        assert C % self.num_groups == 0, f"Channels {C} must be divisible by num_groups {self.num_groups}"
        device = x.device
        dtype = torch.float32

        # Save residual
        residual = x.to(dtype).contiguous()

        # First conv: NCHW, stride=1, padding=1
        out = F.conv2d(x.to(dtype).contiguous(), conv1_weight.to(dtype).contiguous(), bias=None, stride=1, padding=1)

        # GroupNorm 1 with affine
        in_flat = out.view(B, C, H * W).contiguous()
        gn_out = torch.empty_like(in_flat, device=device, dtype=dtype)
        grid_gn1 = (B, self.num_groups)
        groupnorm_affine_kernel[grid_gn1](
            in_flat, gn_out,
            norm1_weight.to(dtype), norm1_bias.to(dtype),
            B, C, H, W, self.num_groups, self.eps,
            BLOCK_HW=self.block_hw,
        )

        # SiLU 1
        silu_out = torch.empty_like(gn_out, device=device, dtype=dtype)
        total = C * H * W
        grid_silu1 = (triton.cdiv(total, self.silu_block),)
        silu_kernel[grid_silu1](gn_out, silu_out, total, BLOCK=self.silu_block)

        # Second conv
        out2 = F.conv2d(silu_out.view(B, C, H, W), conv2_weight.to(dtype).contiguous(), bias=None, stride=1, padding=1)
        out2 = out2.contiguous().to(dtype)

        # GroupNorm 2 with affine
        in_flat2 = out2.view(B, C, H * W).contiguous()
        gn_out2 = torch.empty_like(in_flat2, device=device, dtype=dtype)
        grid_gn2 = (B, self.num_groups)
        groupnorm_affine_kernel[grid_gn2](
            in_flat2, gn_out2,
            norm2_weight.to(dtype), norm2_bias.to(dtype),
            B, C, H, W, self.num_groups, self.eps,
            BLOCK_HW=self.block_hw,
        )

        # SiLU 2
        silu_out2 = torch


def run(*args):
    return ModelNew()(*args)
