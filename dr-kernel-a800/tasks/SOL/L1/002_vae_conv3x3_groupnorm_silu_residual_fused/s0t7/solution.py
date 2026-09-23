import torch
import triton
import triton.language as tl


# Triton conv3x3 stride=1, padding=1, no bias: one program computes one output element.
@triton.jit
def conv3x3_nobias_one_elem(
    x_ptr,        # *const float, input [B, C_in, H, W]
    w_ptr,        # *const float, weights flattened [C_out * C_in * 9]
    out_ptr,      # *float, output [B, C_out, H, W]
    B: tl.constexpr,
    C_in: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C_out: tl.constexpr,
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch index
    pid_co = tl.program_id(1) # output channel index
    pid_h = tl.program_id(2)  # output height index
    pid_w = tl.program_id(3)  # output width index

    h_out = pid_h
    w_out = pid_w

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel with padding
    for ci in tl.static_range(C_in):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                h_in = h_out - 1 + kh
                w_in = w_out - 1 + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                base_x = pid_n * x_stride_n + ci * x_stride_c
                ptr_x = x_ptr + base_x + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(ptr_x, mask=in_bounds, other=0.0).to(tl.float32)
                # weight index: co * (C_in * 9) + ci * 9 + (kh * 3 + kw)
                w_idx = pid_co * (C_in * 9) + ci * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_idx).to(tl.float32)
                acc += x_val * w_val

    out_ptr_idx = pid_n * out_stride_n + pid_co * out_stride_c + h_out * out_stride_h + w_out * out_stride_w
    tl.store(out_ptr + out_ptr_idx, acc)


# Triton GroupNorm: two-pass per (n, group), using group_size = C // num_groups
@triton.jit
def group_norm_two_pass(
    out_ptr,      # *const float, input tensor to normalize [B, C, H, W]
    gamma_ptr,    # *const float, per-channel scale [C]
    beta_ptr,     # *const float, per-channel bias [C]
    out_norm_ptr, # *float, normalized output [B, C, H, W]
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    num_groups: tl.constexpr,
    eps: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    n = tl.program_id(0)  # batch index
    g = tl.program_id(1)  # group index

    group_size = C // num_groups  # integer division, valid since C % num_groups == 0
    total_elems = group_size * H * W

    # First pass: compute sum and sum of squares
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # Loop over channels in this group and all spatial positions
    for ci in tl.static_range(group_size * C):
        ch = ci // (C // group_size)  # maps 0..group_size*C-1 to channel index
        if (ch >= g * group_size) & (ch < (g + 1) * group_size):
            for h in tl.static_range(H):
                for w in tl.static_range(W):
                    ptr = out_ptr + n * out_stride_n + ch * out_stride_c + h * out_stride_h + w * out_stride_w
                    x = tl.load(ptr).to(tl.float32)
                    sum_val += x
                    sum_sq += x * x

    mean = sum_val / total_elems
    var = sum_sq / total_elems - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize, apply affine, and store
    for ci in tl.static_range(group_size * C):
        ch = ci // (C // group_size)
        if (ch >= g * group_size) & (ch < (g + 1) * group_size):
            for h in tl.static_range(H):
                for w in tl.static_range(W):
                    ptr_in = out_ptr + n * out_stride_n + ch * out_stride_c + h * out_stride_h + w * out_stride_w
                    x = tl.load(ptr_in).to(tl.float32)
                    y = (x - mean) * inv_std
                    gamma = tl.load(gamma_ptr + ch).to(tl.float32)
                    beta = tl.load(beta_ptr + ch).to(tl.float32)
                    y = y * gamma + beta
                    ptr_out = out_norm_ptr + n * out_stride_n + ch * out_stride_c + h * out_stride_h + w * out_stride_w
                    tl.store(ptr_out, y)


# Triton SiLU elementwise: y = x * sigmoid(x)
@triton.jit
def silu_kernel(in_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    x = tl.load(in_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + idx, y, mask=mask)


# Triton add_residual elementwise: out = out + x
@triton.jit
def add_residual_kernel(out_ptr, x_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N
    out = tl.load(out_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    y = out + res
    tl.store(out_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5, num_groups: int = 32):
        super().__init__()
        self.eps = eps
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor) -> torch.Tensor:
        # Move to CUDA and float32, make contiguous
        device = x.device
        x_f = x.to(device=device, dtype=torch.float32).contiguous()
        B, C, H, W = x_f.shape

        # Prepare weights as float32 and flatten conv weights
        conv1_weight_f = conv1_weight.to(device=device, dtype=torch.float32).contiguous().view(-1)
        norm1_weight_f = norm1_weight.to(device=device, dtype=torch.float32).contiguous()
        norm1_bias_f = norm1_bias.to(device=device, dtype=torch.float32).contiguous()

        conv2_weight_f = conv2_weight.to(device=device, dtype=torch.float32).contiguous().view(-1)
        norm2_weight_f = norm2_weight.to(device=device, dtype=torch.float32).contiguous()
        norm2_bias_f = norm2_bias.to(device=device, dtype=torch.float32).contiguous()

        # First conv: out1
        out1 = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        grid_conv1 = (B, C, H, W)
        conv3x3_nobias_one_elem[grid_conv1](
            x_f, conv1_weight_f, out1,
            B, C, H, W, C,
            x_f.stride(0), x_f.stride(1), x_f.stride(2), x_f.stride(3),
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        )

        # GroupNorm1 (num_groups=32)
        out1_gn = torch.empty_like(out1)
        grid_gn1 = (B, self.num_groups)
        group_norm_two_pass[grid_gn1](
            out1, norm1_weight_f, norm1_bias_f, out1_gn,
            B, C, H, W, self.num_groups, self.eps,
            out1_gn.stride(0), out1_gn.stride(1), out1_gn.stride(2), out1_gn.stride(3),
        )

        # SiLU1
        out1_silu = torch.empty_like(out1_gn)
        N1 = out1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](out1_gn, out1_silu, N1, BLOCK=1024)

        # Second conv: out2_pre
        out2_pre = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        grid_conv2 = (B, C, H, W)
        conv3x3_nobias_one_elem[grid_conv2](
            out1_silu, conv2_weight_f, out2_pre,
            B, C, H, W, C,
            out1_silu.stride(0), out1_silu.stride(1), out1_silu.stride(2), out1_silu.stride(3),
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        )

        # GroupNorm2
        out2_gn = torch.empty_like(out2_pre)
        grid_gn2 = (B, self.num_groups)
        group_norm_two_pass[grid_gn2](
            out2_pre, norm2_weight_f, norm2_bias_f, out2_gn,
            B, C, H, W, self.num_groups, self.eps,
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
        add_residual_kernel[grid_add](out2_silu, x_f, Nfinal, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
