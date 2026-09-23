import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton GroupNorm forward: two-pass per (n, group)
# out_ptr: input [B, C, H, W] (pre-normalized or raw), gamma_ptr: per-channel scale [C], beta_ptr: per-channel bias [C]
# out_norm_ptr: output normalized [B, C, H, W]
@triton.jit
def group_norm_forward_two_pass(
    out_ptr,            # *const float
    gamma_ptr,          # *const float
    beta_ptr,           # *const float
    out_norm_ptr,       # *float
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    num_groups: tl.constexpr,
    eps: tl.constexpr,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,  # same strides for out_norm
):
    pid_n = tl.program_id(0)  # sample
    pid_g = tl.program_id(1)  # group id
    group_size = C // num_groups
    group_start_c = pid_g * group_size

    # First pass: sum and sumsq over group
    sum_val = 0.0
    sum_sq = 0.0
    for c_off in tl.static_range(group_size):
        c = group_start_c + c_off
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                ptr = out_ptr + pid_n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
                x_val = tl.load(ptr)
                sum_val += x_val
                sum_sq += x_val * x_val
    M = group_size * H * W
    mean = sum_val / M
    var = sum_sq / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize + affine and store
    for c_off in tl.static_range(group_size):
        c = group_start_c + c_off
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                ptr_in = out_ptr + pid_n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
                x_val = tl.load(ptr_in)
                y = (x_val - mean) * inv_std
                y = y * gamma + beta
                ptr_out = out_norm_ptr + pid_n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
                tl.store(ptr_out, y)


# Triton SiLU: elementwise on flattened buffer
@triton.jit
def silu_kernel(
    inp_ptr,       # *const float
    out_ptr,       # *float
    N,             # int
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offs, y, mask=mask)


# Triton residual add: out = out + inp (elementwise)
@triton.jit
def add_residual_kernel(
    out_ptr,       # *const float
    inp_ptr,       # *const float
    N,             # int
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    out_val = tl.load(out_ptr + offs, mask=mask, other=0.0)
    inp_val = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, out_val + inp_val, mask=mask)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_groups = 32
        self.eps = 1e-5

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        # Ensure contiguous and float32
        x = x.contiguous().to(torch.float32)
        device = x.device

        # First conv using PyTorch (bias=None, stride=1, padding=1)
        out1 = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # GroupNorm1 via Triton
        out1_gn = torch.empty_like(out1)
        grid_gn1 = (out1.shape[0], self.num_groups)
        group_norm_forward_two_pass[grid_gn1](
            out1, norm1_weight, norm1_bias, out1_gn,
            B=out1.shape[0], C=out1.shape[1], H=out1.shape[2], W=out1.shape[3],
            num_groups=self.num_groups, eps=self.eps,
            x_stride_n=out1_gn.stride(0), x_stride_c=out1_gn.stride(1), x_stride_h=out1_gn.stride(2), x_stride_w=out1_gn.stride(3),
        )

        # SiLU1 via Triton
        out1_silu = torch.empty_like(out1_gn)
        N1 = out1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](out1_gn, out1_silu, N1, BLOCK=1024)

        # Second conv using PyTorch
        out2_pre = F.conv2d(out1_silu, conv2_weight, bias=None, stride=1, padding=1)

        # GroupNorm2 via Triton
        out2_gn = torch.empty_like(out2_pre)
        grid_gn2 = (out2_pre.shape[0], self.num_groups)
        group_norm_forward_two_pass[grid_gn2](
            out2_pre, norm2_weight, norm2_bias, out2_gn,
            B=out2_pre.shape[0], C=out2_pre.shape[1], H=out2_pre.shape[2], W=out2_pre.shape[3],
            num_groups=self.num_groups, eps=self.eps,
            x_stride_n=out2_gn.stride(0), x_stride_c=out2_gn.stride(1), x_stride_h=out2_gn.stride(2), x_stride_w=out2_gn.stride(3),
        )

        # SiLU2 via Triton
        out2_silu = torch.empty_like(out2_gn)
        N2 = out2_gn.numel()
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_kernel[grid_silu2](out2_gn, out2_silu, N2, BLOCK=1024)

        # Add residual x via Triton
        out = torch.empty_like(out2_silu)
        Nfinal = out2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 1024),)
        add_residual_kernel[grid_add](out2_silu, x, Nfinal, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
