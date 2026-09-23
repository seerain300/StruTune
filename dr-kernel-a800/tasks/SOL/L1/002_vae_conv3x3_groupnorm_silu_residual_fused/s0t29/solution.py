import torch
import triton
import triton.language as tl


# Triton conv3x3 stride=1, padding=1, no bias.
# One program computes one output element y[n, co, h_out, w_out].
@triton.jit
def conv3x3_nobias_one_elem(
    x_ptr,        # *const float, input [B, C_in, H, W]
    w_ptr,        # *const float, weights [C_out, C_in, 3, 3], contiguous
    out_ptr,      # *float, output [B, C_out, H, W]
    B, C_in, H, W, C_out, H_out, W_out,
    # input strides
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    # output strides
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    h_out = pid_h
    w_out = pid_w

    acc = tl.zeros((), dtype=tl.float32)

    # iterate over input channels and 3x3 kernel
    for ci in tl.static_range(C_in):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                h_in = h_out - 1 + kh
                w_in = w_out - 1 + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                base_x = pid_n * x_stride_n + ci * x_stride_c
                ptr_x = x_ptr + base_x + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(ptr_x, mask=in_bounds, other=0.0).to(tl.float32)
                # weight index: w[co, ci, kh, kw] in contiguous layout of [C_out, C_in, 3, 3]
                w_idx = pid_co * (C_in * 9) + ci * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_idx).to(tl.float32)
                acc += x_val * w_val

    # store result
    base_out = pid_n * out_stride_n + pid_co * out_stride_c
    ptr_out = out_ptr + base_out + h_out * out_stride_h + w_out * out_stride_w
    tl.store(ptr_out, acc)


# Triton GroupNorm with affine: per (n, group), two-pass reduction + normalization + affine
@triton.jit
def group_norm_two_pass(
    x_ptr,            # *const float, input to normalize [B, C, H, W]
    gamma_ptr,        # *const float, per-channel scale [C]
    beta_ptr,         # *const float, per-channel bias [C]
    out_ptr,          # *float, output [B, C, H, W]
    B, C, H, W, G,    # B=batch, C=channels, H,W spatial, G=num_groups
    eps,              # float eps
    # strides for x/out (assume same strides for x and out; out is allocated like x)
    stride_n, stride_c, stride_h, stride_w,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    # number of channels per group
    M = C // G
    HW = H * W

    # pass 1: compute sum and sum of squares across the group's channels and spatial
    total_sum = tl.zeros((), dtype=tl.float32)
    total_sq_sum = tl.zeros((), dtype=tl.float32)

    for m in tl.static_range(M):
        c = pid_g * M + m
        # iterate over spatial H*W
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
        gamma = tl.load(gamma_ptr + c).to(tl.float32)
        beta = tl.load(beta_ptr + c).to(tl.float32)
        for p in tl.static_range(HW):
            h = p // W
            w = p % W
            base_x = pid_n * stride_n + c * stride_c
            ptr_x = x_ptr + base_x + h * stride_h + w * stride_w
            x_val = tl.load(ptr_x).to(tl.float32)
            y = (x_val - mean) * rstd
            y = y * gamma + beta
            base_out = pid_n * stride_n + c * stride_c
            ptr_out = out_ptr + base_out + h * stride_h + w * stride_w
            tl.store(ptr_out, y)


# Triton elementwise SiLU: y = x * sigmoid(x)
@triton.jit
def silu_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offsets, y, mask=mask)


# Triton elementwise residual addition: out = x1 + x2
@triton.jit
def add_residual_kernel(x_ptr, residual_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    res = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    y = x + res
    tl.store(out_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5, num_groups: int = 32):
        super().__init__()
        self.eps = eps
        self.num_groups = num_groups

    def forward(self,
                x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float = 1e-5):
        # Ensure dtype and contiguity
        device = x.device
        dtype = torch.float32

        # 1) First convolution in Triton: out1 = conv3x3(x, conv1_weight)
        B, C_in, H, W = x.shape
        C_out = conv1_weight.shape[0]  # conv weights: (C_out, C_in, 3, 3)
        H_out, W_out = H, W  # padding=1, stride=1 -> same spatial size
        out1 = torch.empty((B, C_out, H_out, W_out), device=device, dtype=dtype)
        x_in = x.contiguous().to(dtype)
        w1 = conv1_weight.contiguous().to(dtype)

        grid_conv1 = (B, C_out, H_out, W_out)
        conv3x3_nobias_one_elem[grid_conv1](
            x_in, w1, out1,
            B, C_in, H, W, C_out, H_out, W_out,
            x_in.stride(0), x_in.stride(1), x_in.stride(2), x_in.stride(3),
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        )

        # 2) GroupNorm1 (num_groups=32, affine)
        # Prepare per-channel gamma/beta tensors on device
        gamma1 = norm1_weight.contiguous().to(dtype)
        beta1 = norm1_bias.contiguous().to(dtype)
        out1_gn = torch.empty_like(out1)

        grid_gn1 = (B, self.num_groups)
        group_norm_two_pass[grid_gn1](
            out1, gamma1, beta1, out1_gn,
            B, out1.shape[1], out1.shape[2], out1.shape[3], self.num_groups,
            self.eps,
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        )

        # 3) SiLU1
        out1_silu = torch.empty_like(out1_gn)
        N1 = out1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](out1_gn, out1_silu, N1, BLOCK=1024)

        # 4) Second convolution in Triton: out2_pre = conv3x3(out1_silu)
        B, C_in2, H2, W2 = out1_silu.shape  # out1_silu shape is (B, C_out, H, W)
        C_out2 = conv2_weight.shape[0]      # conv2 weight: (C_out, C_in, 3, 3)
        H_out2, W_out2 = H2, W2
        out2_pre = torch.empty((B, C_out2, H_out2, W_out2), device=device, dtype=dtype)
        x2 = out1_silu.contiguous().to(dtype)
        w2 = conv2_weight.contiguous().to(dtype)

        grid_conv2 = (B, C_out2, H_out2, W_out2)
        conv3x3_nobias_one_elem[grid_conv2](
            x2, w2, out2_pre,
            B, C_in2, H2, W2, C_out2, H_out2, W_out2,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        )

        # 5) GroupNorm2 (num_groups=32, affine)
        gamma2 = norm2_weight.contiguous().to(dtype)
        beta2 = norm2_bias.contiguous().to(dtype)
        out2_gn = torch.empty_like(out2_pre)

        grid_gn2 = (B, self.num_groups)
        group_norm_two_pass[grid_gn2](
            out2_pre, gamma2, beta2, out2_gn,
            B, out2_pre.shape[1], out2_pre.shape[2], out2_pre.shape[3], self.num_groups,
            self.eps,
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        )

        # 6) SiLU2
        out2_silu = torch.empty_like(out2_gn)
        N2 = out2_gn.numel()
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_kernel[grid_silu2](out2_gn, out2_silu, N2, BLOCK=1024)

        # 7) Add residual x
        out = torch.empty_like(out2_silu)
        Nfinal = out2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 1024),)
        add_residual_kernel[grid_add](out2_silu, x.to(dtype), out, Nfinal, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
