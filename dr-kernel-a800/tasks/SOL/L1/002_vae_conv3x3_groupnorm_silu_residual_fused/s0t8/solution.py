import torch
import triton
import triton.language as tl


# Triton conv3x3 stride=1, padding=1, no bias.
# One program computes one output element: out[n, co, h_out, w_out]
@triton.jit
def conv3x3_nobias_one_elem(
    x_ptr,        # *const float, input [B, C_in, H, W]
    w_ptr,        # *const float, weights [C_out, C_in, 3, 3], contiguous
    out_ptr,      # *float, output [B, C_out, H_out, W_out]
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)  # output height
    pid_w = tl.program_id(3)  # output width

    h_out = pid_h
    w_out = pid_w

    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels and 3x3 kernel
    for ci in tl.static_range(C_in):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                h_in = h_out - 1 + kh  # padding=1
                w_in = w_out - 1 + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                base_x = pid_n * x_stride_n + ci * x_stride_c
                ptr_x = x_ptr + base_x + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(ptr_x, mask=in_bounds, other=0.0).to(tl.float32)

                # weight index: ((co * C_in) + ci) * (3*3) + (kh * 3 + kw)
                w_idx = (pid_co * C_in + ci) * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_idx).to(tl.float32)

                acc += x_val * w_val

    # store result
    base_out = pid_n * out_stride_n + pid_co * out_stride_c
    ptr_out = out_ptr + base_out + h_out * out_stride_h + w_out * out_stride_w
    tl.store(ptr_out, acc)


# Triton GroupNorm: two-pass, per (n, group), affine with gamma/beta, elementwise
@triton.jit
def group_norm_two_pass(
    x_ptr,       # *const float, input tensor [B, C, H_out, W_out]
    gamma_ptr,   # *const float, weight [C]
    beta_ptr,    # *const float, bias [C]
    out_ptr,     # *float, output [B, C, H_out, W_out]
    B: tl.constexpr, C: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    num_groups: tl.constexpr, eps: tl.constexpr,
    x_stride0: tl.constexpr, x_stride1: tl.constexpr, x_stride2: tl.constexpr, x_stride3: tl.constexpr,
    out_stride0: tl.constexpr, out_stride1: tl.constexpr, out_stride2: tl.constexpr, out_stride3: tl.constexpr,
):
    # program per (n, group)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_size = C // num_groups
    channel_start = pid_g * group_size
    L = group_size * H_out * W_out  # number of elements in this group for sample n

    # first pass: compute sum and sum of squares over the group's elements
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # We perform a simple loop over the group in chunks of 1 (since it's robust). Triton will compile it.
    for idx in tl.static_range(L):
        ch = channel_start + (idx // (H_out * W_out))
        hw = idx % (H_out * W_out)
        h = hw // W_out
        w = hw % W_out
        # pointer to x[n, ch, h, w]
        base_x = pid_n * x_stride0 + ch * x_stride1
        ptr_x = x_ptr + base_x + h * x_stride2 + w * x_stride3
        x_val = tl.load(ptr_x).to(tl.float32)
        sum_val += x_val
        sum_sq += x_val * x_val

    mean = sum_val / L
    var = sum_sq / L - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # second pass: normalize, apply affine, store
    for idx in tl.static_range(L):
        ch = channel_start + (idx // (H_out * W_out))
        hw = idx % (H_out * W_out)
        h = hw // W_out
        w = hw % W_out

        base_x = pid_n * x_stride0 + ch * x_stride1
        ptr_x = x_ptr + base_x + h * x_stride2 + w * x_stride3
        x_val = tl.load(ptr_x).to(tl.float32)

        gamma = tl.load(gamma_ptr + ch).to(tl.float32)
        beta = tl.load(beta_ptr + ch).to(tl.float32)

        y = (x_val - mean) * inv_std
        y = y * gamma + beta

        base_out = pid_n * out_stride0 + ch * out_stride1
        ptr_out = out_ptr + base_out + h * out_stride2 + w * out_stride3
        tl.store(ptr_out, y)


# Triton elementwise SiLU: y = x * sigmoid(x)
@triton.jit
def silu_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offsets, y, mask=mask)


# Triton elementwise add_residual: out = out + x
@triton.jit
def add_residual_kernel(out_ptr, x_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    c = a + b
    tl.store(out_ptr + offsets, c, mask=mask)


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
        Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        Assumes: C divisible by num_groups=32 for GroupNorm.
        """
        assert x.is_cuda, "Input tensor must be on CUDA for Triton kernels."
        device = x.device
        dtype = torch.float32

        B, C, H, W = x.shape

        # Ensure weights and params are on device and float32
        conv1_weight_f = conv1_weight.to(device=device, dtype=torch.float32).contiguous()
        norm1_weight_f = norm1_weight.to(device=device, dtype=torch.float32).contiguous()
        norm1_bias_f = norm1_bias.to(device=device, dtype=torch.float32).contiguous()

        conv2_weight_f = conv2_weight.to(device=device, dtype=torch.float32).contiguous()
        norm2_weight_f = norm2_weight.to(device=device, dtype=torch.float32).contiguous()
        norm2_bias_f = norm2_bias.to(device=device, dtype=torch.float32).contiguous()

        # Cast input to float32 for computation
        x_f = x.to(device=device, dtype=torch.float32).contiguous()

        # First conv: out1_pre = conv3x3_nobias(x_f)
        H_out = H
        W_out = W
        out1_pre = torch.empty((B, C, H_out, W_out), device=device, dtype=torch.float32)

        grid_conv1 = (B, C, H_out, W_out)
        conv3x3_nobias_one_elem[grid_conv1](
            x_f, conv1_weight_f, out1_pre,
            B, C, H, W, C, H_out, W_out,
            x_f.stride(0), x_f.stride(1), x_f.stride(2), x_f.stride(3),
            out1_pre.stride(0), out1_pre.stride(1), out1_pre.stride(2), out1_pre.stride(3),
        )

        # GroupNorm1 (num_groups=32)
        out1_gn = torch.empty_like(out1_pre)
        grid_gn1 = (B, self.num_groups)
        group_norm_two_pass[grid_gn1](
            out1_pre, norm1_weight_f, norm1_bias_f, out1_gn,
            B, C, H_out, W_out, self.num_groups, self.eps,
            out1_pre.stride(0), out1_pre.stride(1), out1_pre.stride(2), out1_pre.stride(3),
            out1_gn.stride(0), out1_gn.stride(1), out1_gn.stride(2), out1_gn.stride(3),
        )

        # SiLU1
        out1_silu = torch.empty_like(out1_gn)
        N1 = out1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](out1_gn, out1_silu, N1, BLOCK=1024)

        # Second conv: out2_pre = conv3x3_nobias(out1_silu)
        out2_pre = torch.empty((B, C, H_out, W_out), device=device, dtype=torch.float32)

        grid_conv2 = (B, C, H_out, W_out)
        conv3x3_nobias_one_elem[grid_conv2](
            out1_silu, conv2_weight_f, out2_pre,
            B, C, H_out, W_out, C, H_out, W_out,
            out1_silu.stride(0), out1_silu.stride(1), out1_silu.stride(2), out1_silu.stride(3),
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        )

        # GroupNorm2 (num_groups=32)
        out2_gn = torch.empty_like(out2_pre)

        grid_gn2 = (B, self.num_groups)
        group_norm_two_pass[grid_gn2](
            out2_pre, norm2_weight_f, norm2_bias_f, out2_gn,
            B, C, H_out, W_out, self.num_groups, self.eps,
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
        add_residual_kernel[grid_add](out2_silu, x_f, Nfinal, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
