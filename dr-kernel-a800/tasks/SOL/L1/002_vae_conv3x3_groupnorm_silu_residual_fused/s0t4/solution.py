import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def group_norm_kernel(
    x_ptr,            # *const float
    gamma_ptr,        # *const float (length C)
    beta_ptr,         # *const float (length C)
    out_ptr,          # *float
    B: tl.int32,
    C: tl.int32,
    H: tl.int32,
    W: tl.int32,
    G: tl.int32,      # num_groups = 32
    eps: tl.float32,
    BLOCK: tl.constexpr,
):
    # One program per (n, group)
    pid = tl.program_id(0)
    group_size = C // G
    n = pid // G
    g = pid % G

    c_start = g * group_size
    HW = H * W
    total_elems = group_size * HW

    # First pass: compute sum and sum of squares over group
    sum_val = 0.0
    sum_sq = 0.0
    for start in range(0, total_elems, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < total_elems
        ci = c_start + (offs // HW)
        sp = offs % HW
        h = sp // W
        w = sp % W
        idx = ((n * C + ci) * H + h) * W + w
        x_val = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x_val, axis=0)
        sum_sq += tl.sum(x_val * x_val, axis=0)

    mean = sum_val / total_elems
    var = sum_sq / total_elems - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize, apply affine, store
    for start in range(0, total_elems, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < total_elems
        ci = c_start + (offs // HW)
        sp = offs % HW
        h = sp // W
        w = sp % W
        idx = ((n * C + ci) * H + h) * W + w
        x_val = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        norm = (x_val - mean) * inv_std
        gamma = tl.load(gamma_ptr + ci, mask=mask, other=1.0).to(tl.float32)
        beta = tl.load(beta_ptr + ci, mask=mask, other=0.0).to(tl.float32)
        y = norm * gamma + beta
        tl.store(out_ptr + idx, y, mask=mask)


@triton.jit
def silu_kernel(x_ptr, out_ptr, N: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def add_residual_kernel(x_ptr, out_ptr, N: tl.int32, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(out_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = y + x
    tl.store(out_ptr + offs, z, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        # Ensure CUDA tensors
        assert x.is_cuda, "Input x must be on CUDA device for Triton kernels."
        B, C, H, W = x.shape
        assert C % self.num_groups == 0, f"Channels {C} must be divisible by num_groups {self.num_groups}."

        # 1) First conv
        out1 = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # 2) Triton GroupNorm for out1
        out1_gn = torch.empty_like(out1, dtype=torch.float32)
        N1 = out1.numel()
        grid_gn1 = (B * self.num_groups,)
        group_norm_kernel[grid_gn1](
            out1, norm1_weight, norm1_bias, out1_gn,
            B, C, H, W, self.num_groups, self.eps,
            BLOCK=1024,
        )

        # 3) Triton SiLU for out1_gn
        out1_silu = torch.empty_like(out1_gn, dtype=torch.float32)
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](out1_gn, out1_silu, N1, BLOCK=1024)

        # 4) Second conv
        out2 = F.conv2d(out1_silu, conv2_weight, bias=None, stride=1, padding=1)

        # 5) Triton GroupNorm for out2
        out2_gn = torch.empty_like(out2, dtype=torch.float32)
        N2 = out2.numel()
        grid_gn2 = (B * self.num_groups,)
        group_norm_kernel[grid_gn2](
            out2, norm2_weight, norm2_bias, out2_gn,
            B, C, H, W, self.num_groups, self.eps,
            BLOCK=1024,
        )

        # 6) Triton SiLU for out2_gn
        out2_silu = torch.empty_like(out2_gn, dtype=torch.float32)
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_kernel[grid_silu2](out2_gn, out2_silu, N2, BLOCK=1024)

        # 7) Residual addition via Triton
        # Note: x is original input, out2_silu is the second conv output after SiLU and GroupNorm
        final_out = torch.empty_like(out2_silu, dtype=torch.float32)
        Nfinal = out2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 1024),)
        add_residual_kernel[x, out2_silu, final_out](x, out2_silu, Nfinal, BLOCK=1024)

        return final_out


def run(*args):
    return ModelNew()(*args)
