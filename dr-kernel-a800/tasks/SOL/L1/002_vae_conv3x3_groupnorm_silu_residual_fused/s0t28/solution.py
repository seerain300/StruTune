import torch
import triton
import triton.language as tl


# Triton GroupNorm with affine: per (n, group), two-pass reduction + normalization + affine.
# Assumes num_groups=32. C must be divisible by 32.
@triton.jit
def group_norm_two_pass(
    x_ptr,            # *const float, input [B, C, H, W]
    gamma_ptr,        # *const float, per-channel scale [C]
    beta_ptr,         # *const float, per-channel bias [C]
    out_ptr,          # *float, output [B, C, H, W]
    B, C, H, W, G,    # B=batch, C=channels, H,W spatial, G=num_groups
    eps,              # float eps
    stride_n, stride_c, stride_h, stride_w,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    # number of channels per group
    M = C // G
    HW = H * W

    # pass 1: compute mean and rstd across the group's channels and spatial
    total_sum = tl.zeros((), dtype=tl.float32)
    total_sq_sum = tl.zeros((), dtype=tl.float32)

    for m in tl.static_range(M):
        c = pid_g * M + m
        for p in tl.static_range(HW):
            h = p // W
            w = p % W
            base_x = pid_n * stride_n + c * stride_c
            ptr_x = x_ptr + base_x + h * stride_h + w * stride_w
            x_val = tl.load(ptr_x).to(tl.float32)
            total_sum += x_val
            total_sq_sum += x_val * x_val

    mean = total_sum / (M * HW)
    var = total_sq_sum / (M * HW) - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # pass 2: normalize and apply affine
    for m in tl.static_range(M):
        c = pid_g * M + m
        base_out = pid_n * stride_n + c * stride_c
        # gamma and beta are per-channel scalars
        gamma = tl.load(gamma_ptr + c).to(tl.float32)
        beta = tl.load(beta_ptr + c).to(tl.float32)
        for p in tl.static_range(HW):
            h = p // W
            w = p % W
            in_ptr = x_ptr + base_out + h * stride_h + w * stride_w
            out_ptr_p = out_ptr + base_out + h * stride_h + w * stride_w
            x_val = tl.load(in_ptr).to(tl.float32)
            y = (x_val - mean) * rstd * gamma + beta
            tl.store(out_ptr_p, y)


# Triton elementwise SiLU: y = x * sigmoid(x)
@triton.jit
def silu_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(out_ptr + offs, y, mask=mask)


# Triton elementwise residual addition: out = x1 + x2
@triton.jit
def add_residual_kernel(x1_ptr, x2_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(x1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = a + b
    tl.store(out_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups=32, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias):
        # conv1: PyTorch for robustness
        out1 = torch.nn.functional.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        # Triton GroupNorm1 (num_groups=32, affine) over out1
        out1_gn = torch.empty_like(out1)
        B, C, H, W = out1.shape
        grid_gn1 = (B, self.num_groups)
        group_norm_two_pass[grid_gn1](
            out1, norm1_weight, norm1_bias, out1_gn,
            B, C, H, W, self.num_groups, self.eps,
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        )
        # Triton SiLU1
        out1_silu = torch.empty_like(out1_gn)
        N1 = out1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](out1_gn, out1_silu, N1, BLOCK=1024)

        # conv2: PyTorch for robustness
        out2_pre = torch.nn.functional.conv2d(out1_silu, conv2_weight, bias=None, stride=1, padding=1)
        # Triton GroupNorm2 (num_groups=32, affine) over out2_pre
        out2_gn = torch.empty_like(out2_pre)
        B2, C2, H2, W2 = out2_pre.shape
        grid_gn2 = (B2, self.num_groups)
        group_norm_two_pass[grid_gn2](
            out2_pre, norm2_weight, norm2_bias, out2_gn,
            B2, C2, H2, W2, self.num_groups, self.eps,
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        )
        # Triton SiLU2
        out2_silu = torch.empty_like(out2_gn)
        N2 = out2_gn.numel()
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_kernel[grid_silu2](out2_gn, out2_silu, N2, BLOCK=1024)

        # Residual add: Triton
        out = torch.empty_like(out2_silu)
        Nfinal = out2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 1024),)
        add_residual_kernel[grid_add](out2_silu, x, out, Nfinal, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
