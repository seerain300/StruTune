import torch
import triton
import triton.language as tl


# Simple conv3x3 (stride=1, padding=1, no bias) per output element
@triton.jit
def conv3x3_nobias_one_elem(
    x_ptr,        # *const float32, input [B, C_in, H, W]
    w_ptr,        # *const float32, weights [C_out, C_in, 3, 3]
    out_ptr,      # *float32, output [B, C_out, H_out, W_out]
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_ho = tl.program_id(2)
    pid_wo = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    for ci in tl.static_range(0, C_in):
        for kh in tl.static_range(0, 3):
            for kw in tl.static_range(0, 3):
                h_in = pid_ho - 1 + kh
                w_in = pid_wo - 1 + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                base_x = pid_n * x_stride_n + ci * x_stride_c
                ptr_x = x_ptr + base_x + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(ptr_x, mask=in_bounds, other=0.0)
                # weight index: (co * (C_in * 9)) + (ci * 9) + (kh * 3 + kw)
                w_idx = pid_co * (C_in * 9) + ci * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_idx)
                acc += x_val * w_val

    base_out = pid_n * out_stride_n + pid_co * out_stride_c
    ptr_out = out_ptr + base_out + pid_ho * out_stride_h + pid_wo * out_stride_w
    tl.store(ptr_out, acc)


# GroupNorm: two-pass per (n, group) - compute sum/sumsq then normalize+affine
@triton.jit
def group_norm_two_pass(
    x_ptr,      # *const float32, input to norm [B, C, H, W]
    gamma_ptr,  # *const float32, per-channel scale [C]
    beta_ptr,   # *const float32, per-channel bias [C]
    out_ptr,    # *float32, output [B, C, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, eps: tl.constexpr,
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)  # group id
    group_size = (C + num_groups - 1) // num_groups  # channels per group

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # First pass: compute sum and sumsq over channels in group and all spatial locations
    for gc in tl.static_range(0, group_size):
        c = pid_g * group_size + gc
        if c >= C:
            continue
        for h in tl.static_range(0, H):
            for w in tl.static_range(0, W):
                base_x = pid_n * x_stride_n + c * x_stride_c
                ptr_x = x_ptr + base_x + h * x_stride_h + w * x_stride_w
                x_val = tl.load(ptr_x).to(tl.float32)
                sum_val += x_val
                sum_sq += x_val * x_val

    M = group_size * H * W
    mean = sum_val / M
    var = sum_sq / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for gc in tl.static_range(0, group_size):
        c = pid_g * group_size + gc
        if c >= C:
            continue
        gamma = tl.load(gamma_ptr + c).to(tl.float32)
        beta = tl.load(beta_ptr + c).to(tl.float32)
        for h in tl.static_range(0, H):
            for w in tl.static_range(0, W):
                base_x = pid_n * x_stride_n + c * x_stride_c
                ptr_x = x_ptr + base_x + h * x_stride_h + w * x_stride_w
                x_val = tl.load(ptr_x).to(tl.float32)
                y = (x_val - mean) * inv_std
                y = y * gamma + beta
                base_out = pid_n * out_stride_n + c * out_stride_c
                ptr_out = out_ptr + base_out + h * out_stride_h + w * out_stride_w
                tl.store(ptr_out, y)


# Elementwise SiLU: y = x * sigmoid(x) = x / (1 + exp(-x))
@triton.jit
def silu_kernel(
    inp_ptr,  # *const float32
    out_ptr,  # *float32
    N,        # total number of elements
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offsets, y, mask=mask)


# Elementwise residual add: out = y + x
@triton.jit
def add_residual_kernel(
    y_ptr,    # *const float32, already processed tensor
    x_ptr,    # *const float32, original input x
    out_ptr,  # *float32, output
    N,        # total number of elements
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = y + x
    tl.store(out_ptr + offsets, out, mask=mask)


# ModelNew: Triton-only forward, no torch ops
class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor):
        """
        Triton-only fused residual block:
        Conv3x3 -> GroupNorm(num_groups=32) -> SiLU -> Conv3x3 -> GroupNorm(num_groups=32) -> SiLU -> Add(x)
        """
        device = x.device
        # Ensure float32 and contiguous
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = x.contiguous()

        # conv weights: [C_out, C_in, 3, 3]
        C = x.shape[1]
        if C % self.num_groups != 0:
            raise ValueError(f"Channel count {C} must be divisible by num_groups {self.num_groups} for GroupNorm.")

        # First conv: out1_pre
        B, C_in, H, W = x.shape
        C_out = conv1_weight.shape[0]
        H_out = H
        W_out = W
        out1_pre = torch.empty((B, C_out, H_out, W_out), device=device, dtype=torch.float32)

        grid_conv1 = (B, C_out, H_out, W_out)
        conv3x3_nobias_one_elem[grid_conv1](
            x, conv1_weight, out1_pre,
            B, C_in, H, W, C_out, H_out, W_out,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out1_pre.stride(0), out1_pre.stride(1), out1_pre.stride(2), out1_pre.stride(3),
        )

        # GroupNorm1
        out1_gn = torch.empty_like(out1_pre)
        grid_gn1 = (B, self.num_groups)
        group_norm_two_pass[grid_gn1](
            out1_pre, norm1_weight, norm1_bias, out1_gn,
            B, C_out, H_out, W_out,
            self.num_groups, self.eps,
            out1_pre.stride(0), out1_pre.stride(1), out1_pre.stride(2), out1_pre.stride(3),
            out1_gn.stride(0), out1_gn.stride(1), out1_gn.stride(2), out1_gn.stride(3),
        )

        # SiLU1
        out1_silu = torch.empty_like(out1_gn)
        N1 = out1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](out1_gn, out1_silu, N1, BLOCK=1024)

        # Second conv: out2_pre = conv3x3(out1_silu)
        C_in2 = out1_silu.shape[1]
        C_out2 = conv2_weight.shape[0]
        H2 = out1_silu.shape[2]
        W2 = out1_silu.shape[3]
        out2_pre = torch.empty((B, C_out2, H2, W2), device=device, dtype=torch.float32)

        grid_conv2 = (B, C_out2, H2, W2)
        conv3x3_nobias_one_elem[grid_conv2](
            out1_silu, conv2_weight, out2_pre,
            B, C_in2, H2, W2, C_out2, H2, W2,
            out1_silu.stride(0), out1_silu.stride(1), out1_silu.stride(2), out1_silu.stride(3),
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        )

        # GroupNorm2
        out2_gn = torch.empty_like(out2_pre)
        grid_gn2 = (B, self.num_groups)
        group_norm_two_pass[grid_gn2](
            out2_pre, norm2_weight, norm2_bias, out2_gn,
            B, C_out2, H2, W2,
            self.num_groups, self.eps,
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
            out2_gn.stride(0), out2_gn.stride(1), out2_gn.stride(2), out2_gn.stride(3),
        )

        # SiLU2
        out2_silu = torch.empty_like(out2_gn)
        N2 = out2_gn.numel()
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_kernel[grid_silu2](out2_gn, out2_silu, N2, BLOCK=1024)

        # Add residual x
        out = torch.empty_like(out2_silu)
        Nfinal = out2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 1024),)
        add_residual_kernel[grid_add](out2_silu, x, out, Nfinal, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
