import torch
import triton
import triton.language as tl


# Conv3x3 stride=1, padding=1, no bias: one program per (n, co, h_out, w_out)
@triton.jit
def conv3x3_nobias_one_elem(
    x_ptr,        # *const float, input [B, C_in, H, W]
    w_ptr,        # *const float, weights [C_out, C_in, 3, 3], flattened
    out_ptr,      # *float, output [B, C_out, H, W]
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    h_out = pid_h
    w_out = pid_w

    acc = 0.0
    for ci in tl.static_range(C_in):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                h_in = h_out - 1 + kh  # padding=1
                w_in = w_out - 1 + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                base_x = pid_n * x_stride_n + ci * x_stride_c
                ptr_x = x_ptr + base_x + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(ptr_x, mask=in_bounds, other=0.0)
                w_idx = pid_co * (C_in * 9) + ci * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_idx)
                acc += x_val * w_val
    out_offset = pid_n * out_stride_n + pid_co * out_stride_c + h_out * out_stride_h + w_out * out_stride_w
    tl.store(out_ptr + out_offset, acc)


# GroupNorm two-pass per (n, group): compute sum and sum of squares
@triton.jit
def group_norm_reduce_two_pass(
    x_ptr,         # *const float, input tensor [B, C, H, W]
    sum_ptr,       # *float, per (n, group) sum
    sumsq_ptr,     # *float, per (n, group) sum of squares
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    group_size: tl.constexpr,   # C // num_groups, passed from host
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)
    c_start = pid_g * group_size

    total = group_size * H * W
    s = 0.0
    ss = 0.0
    for i in tl.static_range(total):
        ci = c_start + (i // (H * W))
        pos = i % (H * W)
        h = pos // W
        w = pos % W
        base = pid_n * (C * H * W) + ci * (H * W) + h * W + w
        x_val = tl.load(x_ptr + base)
        s += x_val
        ss += x_val * x_val
    out_idx = pid_n * 32 + pid_g  # 32 is num_groups
    tl.store(sum_ptr + out_idx, s)
    tl.store(sumsq_ptr + out_idx, ss)


# GroupNorm normalize and apply affine per (n, group)
@triton.jit
def group_norm_norm_apply_affine(
    x_ptr,         # *const float, input tensor [B, C, H, W]
    y_ptr,         # *float, output tensor [B, C, H, W]
    sum_ptr,       # *float, per (n, group) sum
    sumsq_ptr,     # *float, per (n, group) sum of squares
    gamma_ptr,     # *const float, per-channel scale [C]
    beta_ptr,      # *const float, per-channel bias [C]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    group_size: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)
    c_start = pid_g * group_size

    total = group_size * H * W
    for i in tl.static_range(total):
        ci = c_start + (i // (H * W))
        pos = i % (H * W)
        h = pos // W
        w = pos % W
        base = pid_n * (C * H * W) + ci * (H * W) + h * W + w
        x_val = tl.load(x_ptr + base)
        s = tl.load(sum_ptr + pid_n * 32 + pid_g)
        ss = tl.load(sumsq_ptr + pid_n * 32 + pid_g)
        M = group_size * H * W
        mean = s / M
        var = ss / M - mean * mean
        inv_std = 1.0 / tl.sqrt(var + 1e-5)  # eps-like for stability
        gamma = tl.load(gamma_ptr + ci)
        beta = tl.load(beta_ptr + ci)
        y_val = ((x_val - mean) * inv_std) * gamma + beta
        tl.store(y_ptr + base, y_val)


# SiLU elementwise: y = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
@triton.jit
def silu_kernel(in_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    # sigmoid
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offs, y, mask=mask)


# Elementwise residual add: y = out + residual
@triton.jit
def add_residual_kernel(out_ptr, res_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(out_ptr + offs, mask=mask, other=0.0)
    b = tl.load(res_ptr + offs, mask=mask, other=0.0)
    y = a + b
    tl.store(out_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups=32, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        Triton-only implementation.
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Weights must be on CUDA device."
        B, C, H, W = x.shape
        device = x.device
        dtype = torch.float32

        # Ensure contiguous and float32
        x = x.contiguous().to(dtype)
        # First path
        out1 = torch.empty((B, C, H, W), device=device, dtype=dtype)
        # grid for conv: (B, C_out, H, W)
        grid_conv1 = (B, C, H, W)
        conv3x3_nobias_one_elem[grid_conv1](
            x, conv1_weight, out1,
            B, C, H, W, C, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        )

        # GroupNorm1: num_groups=32
        # We need C % 32 == 0. The original example uses multiples of 32, but we guard here.
        assert C % self.num_groups == 0, "C must be divisible by num_groups for GroupNorm."
        group_size = C // self.num_groups
        # sum and sumsq buffers: shape (B, num_groups)
        sum1 = torch.empty((B, self.num_groups), device=device, dtype=dtype)
        sumsq1 = torch.empty((B, self.num_groups), device=device, dtype=dtype)
        # Reduction kernel grid: (B, num_groups)
        grid_reduce1 = (B, self.num_groups)
        group_norm_reduce_two_pass[grid_reduce1](
            out1, sum1, sumsq1,
            B, C, H, W, group_size,
        )

        # Normalize and apply affine
        out1_gn = torch.empty_like(out1)
        group_norm_norm_apply_affine[grid_reduce1](
            out1, out1_gn, sum1, sumsq1, norm1_weight, norm1_bias,
            B, C, H, W, group_size,
        )

        # SiLU1
        out1_silu = torch.empty_like(out1_gn)
        N1 = out1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](out1_gn, out1_silu, N1, BLOCK=1024)

        # Second conv: out2_pre = conv3x3(out1_silu)
        out2_pre = torch.empty((B, C, H, W), device=device, dtype=dtype)
        grid_conv2 = (B, C, H, W)
        conv3x3_nobias_one_elem[grid_conv2](
            out1_silu, conv2_weight, out2_pre,
            B, C, H, W, C, H, W,
            out1_silu.stride(0), out1_silu.stride(1), out1_silu.stride(2), out1_silu.stride(3),
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        )

        # GroupNorm2
        sum2 = torch.empty((B, self.num_groups), device=device, dtype=dtype)
        sumsq2 = torch.empty((B, self.num_groups), device=device, dtype=dtype)
        grid_reduce2 = (B, self.num_groups)
        group_norm_reduce_two_pass[grid_reduce2](
            out2_pre, sum2, sumsq2,
            B, C, H, W, group_size,
        )
        out2_gn = torch.empty_like(out2_pre)
        group_norm_norm_apply_affine[grid_reduce2](
            out2_pre, out2_gn, sum2, sumsq2, norm2_weight, norm2_bias,
            B, C, H, W, group_size,
        )

        # SiLU2
        out2_silu = torch.empty_like(out2_gn)
        N2 = out2_gn.numel()
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_kernel[grid_silu2](out2_gn, out2_silu, N2, BLOCK=1024)

        # Residual connection
        out = torch.empty_like(out2_silu)
        Nfinal = out2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 1024),)
        add_residual_kernel[grid_add](out2_silu, x, Nfinal, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
