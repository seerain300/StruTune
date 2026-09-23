import torch
import triton
import triton.language as tl


# Conv3x3 (stride=1, padding=1, no bias): per-output-element kernel
@triton.jit
def conv3x3_nobias_one_elem(
    x_ptr,        # *const float32, input [B, C_in, H, W]
    w_ptr,        # *const float32, weights [C_out, C_in, 3, 3], flattened to [C_out, C_in, 9]
    out_ptr,      # *float32, output [B, C_out, H, W]
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C_out: tl.constexpr,
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch
    pid_co = tl.program_id(1) # output channel
    pid_h = tl.program_id(2)  # output height
    pid_w = tl.program_id(3)  # output width

    h_out = pid_h
    w_out = pid_w

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel positions with padding=1
    for ci in tl.static_range(C_in):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                h_in = h_out - 1 + kh
                w_in = w_out - 1 + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                base_x = pid_n * x_stride_n + ci * x_stride_c
                ptr_x = x_ptr + base_x + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(ptr_x, mask=in_bounds, other=0.0).to(tl.float32)
                # weight index: (co * (C_in * 9)) + (ci * 9) + (kh * 3 + kw)
                w_idx = pid_co * (C_in * 9) + ci * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_idx).to(tl.float32)
                acc += x_val * w_val

    # Store result to out[n, co, h_out, w_out]
    base_out = pid_n * out_stride_n + pid_co * out_stride_c
    ptr_out = out_ptr + base_out + h_out * out_stride_h + w_out * out_stride_w
    tl.store(ptr_out, acc)


# GroupNorm (num_groups fixed, affine=True) in two passes per (n, group)
@triton.jit
def group_norm_two_pass(
    in_ptr,      # *const float32, input tensor to be normalized
    scale_ptr,   # *const float32, per-channel scale (group_size elements)
    bias_ptr,    # *const float32, per-channel bias (group_size elements)
    out_ptr,     # *float32, output tensor
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
    eps: tl.constexpr,
    in_stride_n: tl.constexpr, in_stride_c: tl.constexpr, in_stride_h: tl.constexpr, in_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch
    pid_g = tl.program_id(1)  # group id
    group_size = C // num_groups
    group_start = pid_g * group_size

    # First pass: compute sum and sum of squares over the group's elements
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for ci in tl.static_range(group_size):
        c = group_start + ci
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                base = pid_n * in_stride_n + c * in_stride_c
                ptr = in_ptr + base + h * in_stride_h + w * in_stride_w
                val = tl.load(ptr).to(tl.float32)
                sum_val += val
                sum_sq += val * val

    M = group_size * H * W
    mean = sum_val / M
    var = sum_sq / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for ci in tl.static_range(group_size):
        c = group_start + ci
        gamma = tl.load(scale_ptr + c).to(tl.float32)
        beta = tl.load(bias_ptr + c).to(tl.float32)
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                base_in = pid_n * in_stride_n + c * in_stride_c
                ptr_in = in_ptr + base_in + h * in_stride_h + w * in_stride_w
                val = tl.load(ptr_in).to(tl.float32)
                norm = (val - mean) * inv_std
                out_val = norm * gamma + beta
                base_out = pid_n * out_stride_n + c * out_stride_c
                ptr_out = out_ptr + base_out + h * out_stride_h + w * out_stride_w
                tl.store(ptr_out, out_val)


# Elementwise SiLU: y = x * sigmoid(x)
@triton.jit
def silu_kernel(in_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offsets, y, mask=mask)


# Elementwise residual add: out = in_ptr + res_ptr
@triton.jit
def add_residual_kernel(in_ptr, res_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(res_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(out_ptr + offsets, c, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups=32, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias):
        device = x.device
        B, C, H, W = x.shape
        assert C % self.num_groups == 0, "C must be divisible by num_groups for GroupNorm"

        # Ensure contiguous and float32 for Triton
        x_f = x.contiguous().to(torch.float32)
        conv1_weight_f = conv1_weight.contiguous().to(torch.float32)
        norm1_weight_f = norm1_weight.contiguous().to(torch.float32)
        norm1_bias_f = norm1_bias.contiguous().to(torch.float32)
        conv2_weight_f = conv2_weight.contiguous().to(torch.float32)
        norm2_weight_f = norm2_weight.contiguous().to(torch.float32)
        norm2_bias_f = norm2_bias.contiguous().to(torch.float32)

        # 1) First conv: out1 = conv3x3(x, conv1_weight, stride=1, padding=1, no bias)
        out1 = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        grid_conv1 = (B, C, H, W)
        conv3x3_nobias_one_elem[grid_conv1](
            x_f, conv1_weight_f, out1,
            B, C, H, W, C,
            x_f.stride(0), x_f.stride(1), x_f.stride(2), x_f.stride(3),
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        )

        # 2) GroupNorm1 (num_groups=self.num_groups, affine=True) and SiLU1
        out1_gn = torch.empty_like(out1)
        grid_gn1 = (B, self.num_groups)
        group_norm_two_pass[grid_gn1](
            out1, norm1_weight_f, norm1_bias_f, out1_gn,
            B, C, H, W, self.num_groups, self.eps,
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
            out1_gn.stride(0), out1_gn.stride(1), out1_gn.stride(2), out1_gn.stride(3),
        )
        N1 = out1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        out1_silu = torch.empty_like(out1_gn)
        silu_kernel[grid_silu1](out1_gn, out1_silu, N1, BLOCK=1024)

        # 3) Second conv: out2_pre = conv3x3(out1_silu, conv2_weight, no bias)
        out2_pre = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        grid_conv2 = (B, C, H, W)
        conv3x3_nobias_one_elem[grid_conv2](
            out1_silu, conv2_weight_f, out2_pre,
            B, C, H, W, C,
            out1_silu.stride(0), out1_silu.stride(1), out1_silu.stride(2), out1_silu.stride(3),
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        )

        # 4) GroupNorm2 (num_groups=self.num_groups, affine=True)
        out2_gn = torch.empty_like(out2_pre)
        grid_gn2 = (B, self.num_groups)
        group_norm_two_pass[grid_gn2](
            out2_pre, norm2_weight_f, norm2_bias_f, out2_gn,
            B, C, H, W, self.num_groups, self.eps,
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
            out2_gn.stride(0), out2_gn.stride(1), out2_gn.stride(2), out2_gn.stride(3),
        )

        # 5) SiLU2
        N2 = out2_gn.numel()
        out2_silu = torch.empty_like(out2_gn)
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_kernel[grid_silu2](out2_gn, out2_silu, N2, BLOCK=1024)

        # 6) Add residual x
        out = torch.empty_like(out2_silu)
        Nfinal = out2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 1024),)
        add_residual_kernel[grid_add](out2_silu, x_f, out, Nfinal, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
