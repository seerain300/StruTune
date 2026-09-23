import torch
import triton
import triton.language as tl


@triton.jit
def groupnorm_affine_kernel(
    in_ptr,           # *f32, shape (B, C, H*W), contiguous
    out_ptr,          # *f32, shape (B, C, H*W), contiguous
    weight_ptr,       # *f32, shape (C,), affine scale
    bias_ptr,         # *f32, shape (C,), affine bias
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    NUM_GROUPS: tl.constexpr, EPS: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    # One program per (n, group)
    n = tl.program_id(0)
    g = tl.program_id(1)
    GROUP_SIZE = C // NUM_GROUPS
    c_start = g * GROUP_SIZE

    # First pass: compute sum and sum of squares across group channels and all spatial positions
    sum_val = 0.0
    sum_sq = 0.0
    for ic in range(GROUP_SIZE):
        c = c_start + ic
        hw = H * W
        for start in range(0, hw, BLOCK_HW):
            offs = start + tl.arange(0, BLOCK_HW)
            mask = offs < hw
            in_offs = (n * C + c) * hw + offs
            x = tl.load(in_ptr + in_offs, mask=mask, other=0.0)
            sum_val += tl.sum(x, axis=0)
            sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / (GROUP_SIZE * H * W)
    var = sum_sq / (GROUP_SIZE * H * W) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + EPS)

    # Second pass: normalize and apply affine, then store
    for ic in range(GROUP_SIZE):
        c = c_start + ic
        hw = H * W
        for start in range(0, hw, BLOCK_HW):
            offs = start + tl.arange(0, BLOCK_HW)
            mask = offs < hw
            in_offs = (n * C + c) * hw + offs
            x = tl.load(in_ptr + in_offs, mask=mask, other=0.0)
            y = (x - mean) * inv_std
            scale = tl.load(weight_ptr + c, mask=True, other=1.0)
            bias = tl.load(bias_ptr + c, mask=True, other=0.0)
            y = y * scale + bias
            out_offs = (n * C + c) * hw + offs
            tl.store(out_ptr + out_offs, y, mask=mask)


@triton.jit
def silu_kernel(in_ptr, out_ptr, TOTAL_ELEMS, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL_ELEMS
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
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
    c = a + b
    tl.store(out_ptr + offs, c, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Triton meta-parameters; can be tuned if needed
        self.groupnorm_block_hw = 1024
        self.silu_block = 1024

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        # Ensure shapes are 4D NCHW and contiguity
        assert x.ndim == 4, "Input must be 4D (N, C, H, W)"
        B, C, H, W = x.shape
        # We assume num_groups = 32 as in the original code; enforce divisibility
        num_groups = 32
        assert C % num_groups == 0, f"Channels {C} must be divisible by num_groups {num_groups}"

        device = x.device
        # Make input and weights contiguous and cast to float32 for computation
        x = x.contiguous().to(torch.float32)
        conv1_weight = conv1_weight.contiguous().to(torch.float32)
        conv2_weight = conv2_weight.contiguous().to(torch.float32)
        norm1_weight = norm1_weight.contiguous().to(torch.float32)
        norm1_bias = norm1_bias.contiguous().to(torch.float32)
        norm2_weight = norm2_weight.contiguous().to(torch.float32)
        norm2_bias = norm2_bias.contiguous().to(torch.float32)

        # 1) First conv: NCHW, stride=1, padding=1, no bias (use PyTorch for correctness and speed)
        out1 = torch.nn.functional.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        # 2) GroupNorm 1 with affine
        out1_flat = out1.view(B, C, H * W).contiguous()
        gn_out1 = torch.empty((B, C, H * W), device=device, dtype=torch.float32)
        grid_gn1 = (B, num_groups)
        groupnorm_affine_kernel[grid_gn1](
            out1_flat, gn_out1,
            norm1_weight, norm1_bias,
            B, C, H, W,
            num_groups, 1e-5,
            BLOCK_HW=self.groupnorm_block_hw
        )
        # 3) SiLU 1
        silu_out1 = torch.empty_like(gn_out1, device=device, dtype=torch.float32)
        total1 = C * H * W
        grid_silu1 = (triton.cdiv(total1, self.silu_block),)
        silu_kernel[grid_silu1](gn_out1, silu_out1, total1, BLOCK=self.silu_block)

        # 4) Second conv: NCHW, stride=1, padding=1, no bias (PyTorch)
        out2_input = silu_out1.view(B, C, H, W)
        out2 = torch.nn.functional.conv2d(out2_input, conv2_weight, bias=None, stride=1, padding=1)

        # 5) GroupNorm 2 with affine
        out2_flat = out2.view(B, C, H * W).contiguous()
        gn_out2 = torch.empty((B, C, H * W), device=device, dtype=torch.float32)
        grid_gn2 = (B, num_groups)
        groupnorm_affine_kernel[grid_gn2](
            out2_flat, gn_out2,
            norm2_weight, norm2_bias,
            B, C, H, W,
            num_groups, 1e-5,
            BLOCK_HW=self.groupnorm_block_hw
        )
        # 6) SiLU 2
        silu_out2 = torch.empty_like(gn_out2, device=device, dtype=torch.float32)
        total2 = C * H * W
        grid_silu2 = (triton.cdiv(total2, self.silu_block),)
        silu_kernel[grid_silu2](gn_out2, silu_out2, total2, BLOCK=self.silu_block)

        # 7) Residual addition: add original x to final processed output
        residual = x  # original input, cast to fp32 above
        final = silu_out2.view(B, C, H, W).contiguous()
        out_final = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        total_add = B * C * H * W
        grid_add = (triton.cdiv(total_add, self.silu_block),)
        add_residual_kernel[grid_add](final, residual, out_final, total_add, BLOCK=self.silu_block)

        return out_final


def run(*args):
    return ModelNew()(*args)
