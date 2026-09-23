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
    # program ids: one per (n, group)
    n = tl.program_id(0)
    g = tl.program_id(1)

    # size per group in channels
    GROUP_SIZE = C // NUM_GROUPS

    # base channel index for this group
    c0 = g * GROUP_SIZE

    # Accumulate sum and sum of squares across all elements of this (n, group)
    sum_val = 0.0
    sum_sq = 0.0

    # loop over channels in the group
    # Note: Triton requires static loops; we use a Python range which Triton can unroll.
    for ic in range(GROUP_SIZE):
        c = c0 + ic
        # loop over spatial HW
        hw = H * W
        # iterate in chunks
        for start in range(0, hw, BLOCK_HW):
            offs = start + tl.arange(0, BLOCK_HW)
            mask = offs < hw
            # compute linear index for NCHW contiguous: (((n*C + c)*H + (offs // W)) * W + (offs % W))
            # but since tensor is contiguous, we can just flatten and rely on NCHW linearization:
            # For a fixed (n, c), elements across H*W are contiguous in memory as NCHW has H,W innermost.
            # We can compute index as ((n*C + c) * hw) + offs.
            # However, the tensor is (B, C, H, W) contiguous, not necessarily flattened as (N,C) concatenated.
            # Better approach: flatten the (H,W) plane for each channel and batch separately.
            # We will treat the input as [N, C, H*W] contiguous; PyTorch conv output is contiguous NCHW,
            # but to be safe, we'll make it contiguous and then treat as [N, C, HW] for this kernel.
            # We pass the tensor as such to simplify. So we assume caller made a view or copy to [N, C, HW].
            # Given the provided code, conv outputs are (N, C, H, W) contiguous; we'll rely on that and
            # reindex as (n*C + c) * HW + offs. This requires the out_ptr to be of shape [N, C, HW].
            # Since we cannot reshape inside Triton easily, we require the host to pass a contiguous
            # tensor whose layout matches N*C*HW (we'll do that in forward by creating a view).
            # The following comment explains how we map back to NCHW: idx = ((n*C + c) * HW) + offs.
            idx = ((n * C + c) * hw) + offs

            # Load values; compute in fp32
            x = tl.load(in_ptr + idx, mask=mask, other=0.0)
            x32 = x.to(tl.float32)
            sum_val += tl.sum(x32, axis=0)
            sum_sq += tl.sum(x32 * x32, axis=0)

    # Compute mean and variance
    total = GROUP_SIZE * hw
    mean = sum_val / total
    var = sum_sq / total - mean * mean
    inv_std = 1.0 / tl.sqrt(var + EPS)

    # Second pass: write normalized with affine
    for ic in range(GROUP_SIZE):
        c = c0 + ic
        for start in range(0, hw, BLOCK_HW):
            offs = start + tl.arange(0, BLOCK_HW)
            mask = offs < hw
            idx = ((n * C + c) * hw) + offs
            x = tl.load(in_ptr + idx, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(weight_ptr + c).to(tl.float32)
            b = tl.load(bias_ptr + c).to(tl.float32)
            y = (x - mean) * inv_std
            y = y * w + b
            tl.store(out_ptr + idx, y, mask=mask)


@triton.jit
def silu_kernel(in_ptr, out_ptr, total_elems, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    # SiLU: x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def add_kernel(in_ptr, add_ptr, out_ptr, total_elems, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems
    a = tl.load(in_ptr + offs, mask=mask, other=0.0)
    b = tl.load(add_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a + b, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        Fused residual block:
        Conv3x3 -> GroupNorm(num_groups=32) -> SiLU
        Conv3x3 -> GroupNorm(num_groups=32) -> SiLU
        Add residual
        """
        assert x.is_cuda, "Triton kernels require CUDA tensors."
        B, C, H, W = x.shape
        # Ensure conv weights are on the same device/dtype as x
        conv1_weight = conv1_weight.to(device=x.device, dtype=x.dtype)
        conv2_weight = conv2_weight.to(device=x.device, dtype=x.dtype)

        # Save residual
        residual = x

        # First conv: F.conv2d (stride=1, padding=1)
        out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # Ensure out is contiguous and in fp32 for Triton kernels
        out = out.contiguous()
        out32 = out.to(torch.float32)

        # Allocate output for GroupNorm
        gn_out = torch.empty_like(out32)

        # GroupNorm: Triton kernel
        # We need to pass out32 as [N, C, H*W] contiguous for the kernel. Create that view.
        # Conv output is already contiguous NCHW; we can flatten H*W per (n,c) and treat as [N, C, HW].
        # Let's make a contiguous view [N, C, HW] explicitly.
        HW = H * W
        out_flat = out32.view(B, C, HW)
        gn_flat = gn_out.view(B, C, HW)

        num_groups = self.num_groups
        assert C % num_groups == 0, f"num_groups={num_groups} must divide C={C} for GroupNorm."
        GROUP_SIZE = C // num_groups

        grid_groupnorm = (B, num_groups)
        groupnorm_affine_kernel[grid_groupnorm](
            out_flat, gn_flat,
            norm1_weight.to(torch.float32), norm1_bias.to(torch.float32),
            B, C, H, W, num_groups, self.eps,
            BLOCK_HW=1024,
        )

        # SiLU: Triton elementwise
        silu_out = torch.empty_like(gn_flat)
        total_elems = B * C * HW
        grid_silu = (triton.cdiv(total_elems, 1024),)
        silu_kernel[grid_silu](gn_flat, silu_out, total_elems, BLOCK=1024)

        # Second conv
        out = F.conv2d(silu_out.view(B, C, H, W), conv2_weight, bias=None, stride=1, padding=1)
        out = out.contiguous().to(torch.float32)
        out_flat2 = out.view(B, C, HW)
        gn_out2 = torch.empty_like(out_flat2)

        # Second GroupNorm
        groupnorm_affine_kernel[grid_groupnorm](
            out_flat2, gn_out2,
            norm2_weight.to(torch.float32), norm2_bias.to(torch.float32),
            B, C, H, W, num_groups, self.eps,
            BLOCK_HW=1024,
        )

        # SiLU
        silu_out2 = torch.empty_like(gn_out2)
        grid_silu2 = (triton.cdiv(B * C * HW, 1024),)
        silu_kernel[grid_silu2](gn_out2, silu_out2, B * C * HW, BLOCK=1024)

        # Add residual
        total_elems_add = B * C * HW
        add_out = torch.empty(total_elems_add, device=x.device, dtype=torch.float32)
        add_kernel[grid_silu2](silu_out2, residual.view(B, C, HW).to(torch.float32), add_out, total_elems_add, BLOCK=1024)
        out_final = add_out.view(B, C, H, W)

        return out_final


def run(*args):
    return ModelNew()(*args)
