import torch
import triton
import triton.language as tl

# Constants for Triton
EPS = 1e-5  # epsilon for numerical stability in GroupNorm

# 1) Conv3x3 stride=1, padding=1, no bias. One output element per program.
@triton.jit
def conv3x3_nobias_one_elem(
    x_ptr,        # *const float, input [B, C_in, H, W]
    w_ptr,        # *const float, weights [C_out, C_in, 3, 3] flattened
    out_ptr,      # *float, output [B, C_out, H, W]
    B: tl.constexpr,
    C_in: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C_out: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    # program ids
    pid_n = tl.program_id(0)  # batch index
    pid_co = tl.program_id(1) # output channel index
    pid_h = tl.program_id(2)  # output height index
    pid_w = tl.program_id(3)  # output width index

    h_out = pid_h
    w_out = pid_w

    acc = 0.0  # accumulate in float32

    # loop over input channels and 3x3 kernel
    for ci in tl.static_range(C_in):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                h_in = h_out - 1 + kh  # padding=1
                w_in = w_out - 1 + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                # base pointer for x at (n, ci, h_in, w_in)
                base_x = pid_n * x_stride_n + ci * x_stride_c
                ptr_x = x_ptr + base_x + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(ptr_x, mask=in_bounds, other=0.0)  # float32 input
                # weight index: (co * (C_in * 9)) + (ci * 9) + (kh * 3 + kw)
                w_idx = pid_co * (C_in * 9) + ci * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_idx)  # float32 weights
                acc += x_val * w_val

    # store output
    out_ptr_addr = out_ptr + pid_n * out_stride_n + pid_co * out_stride_c + h_out * out_stride_h + w_out * out_stride_w
    tl.store(out_ptr_addr, acc)


# 2) GroupNorm: per sample, per group, two-pass. In-place affine normalized output.
@triton.jit
def group_norm_two_pass(
    x_ptr,            # *const float, input tensor [B, C, H, W]
    gamma_ptr,        # *const float, weight [C], per-channel scale
    beta_ptr,         # *const float, bias [C], per-channel bias
    out_ptr,          # *float, output tensor [B, C, H, W]
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    eps,              # float32 epsilon
    group_size_per_sample: tl.constexpr,  # C // num_groups
    num_groups: tl.constexpr,             # 32
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch index
    pid_g = tl.program_id(1)  # group index

    # compute group channel range
    group_start = pid_g * group_size_per_sample
    group_end = (pid_g + 1) * group_size_per_sample

    # first pass: compute sum and sum of squares for this group
    sum_val = 0.0
    sum_sq = 0.0
    # iterate over channels in group and all spatial positions
    for ci in tl.static_range(group_start, group_end):
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                x_val = tl.load(
                    x_ptr + pid_n * x_stride_n + ci * x_stride_c + h * x_stride_h + w * x_stride_w
                ).to(tl.float32)
                sum_val += x_val
                sum_sq += x_val * x_val

    group_count = group_size_per_sample * H * W
    mean = sum_val / group_count
    var = sum_sq / group_count - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # second pass: normalize and apply affine
    for ci in tl.static_range(group_start, group_end):
        gamma = tl.load(gamma_ptr + ci).to(tl.float32)
        beta = tl.load(beta_ptr + ci).to(tl.float32)
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                x_val = tl.load(
                    x_ptr + pid_n * x_stride_n + ci * x_stride_c + h * x_stride_h + w * x_stride_w
                ).to(tl.float32)
                y = (x_val - mean) * inv_std
                y = y * gamma + beta
                out_ptr_addr = out_ptr + pid_n * out_stride_n + ci * out_stride_c + h * out_stride_h + w * out_stride_w
                tl.store(out_ptr_addr, y)


# 3) SiLU elementwise: y = x * sigmoid(x)
@triton.jit
def silu_kernel(
    x_ptr,  # *const float
    y_ptr,  # *float
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # sigmoid
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(y_ptr + offsets, y, mask=mask)


# 4) Residual add: out = out + x (elementwise)
@triton.jit
def add_residual_kernel(
    out_ptr,  # *float
    x_ptr,    # *const float
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    out = tl.load(out_ptr + offsets, mask=mask, other=0.0)
    add = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = out + add
    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_groups = 32
        self.eps = 1e-5

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor):
        """
        Triton-only implementation of:
          out = SiLU(GroupNorm(SiLU(GroupNorm(Conv3x3(x, conv1_weight), num_groups=32, norm1_weight, norm1_bias))) + x)
                 + SiLU(GroupNorm(Conv3x3(...), num_groups=32, norm2_weight, norm2_bias)) + x
        """
        # Ensure CUDA and contiguous
        device = x.device
        x = x.contiguous().to(torch.float32)
        B, C, H, W = x.shape
        H_out = H  # conv2d with padding=1 keeps spatial dims (PyTorch default)
        W_out = W

        # First convolution: conv1
        out1 = torch.empty((B, C, H_out, W_out), device=device, dtype=torch.float32)
        # Flattened weights for conv1: (C, C, 3, 3) -> (C, C, 9)
        conv1_weight_f = conv1_weight.contiguous().to(torch.float32).view(C, C, 9)
        grid_conv1 = (B, C, H_out, W_out)
        conv3x3_nobias_one_elem[grid_conv1](
            x, conv1_weight_f, out1,
            B=B, C_in=C, H=H, W=W, C_out=C, H_out=H_out, W_out=W_out,
            x_stride_n=x.stride(0), x_stride_c=x.stride(1), x_stride_h=x.stride(2), x_stride_w=x.stride(3),
            out_stride_n=out1.stride(0), out_stride_c=out1.stride(1), out_stride_h=out1.stride(2), out_stride_w=out1.stride(3),
        )

        # GroupNorm1
        out1_gn = torch.empty_like(out1)
        group_size_per_sample = C // self.num_groups
        grid_gn1 = (B, self.num_groups)
        group_norm_two_pass[grid_gn1](
            out1, norm1_weight.contiguous().to(torch.float32), norm1_bias.contiguous().to(torch.float32), out1_gn,
            B=B, C=C, H=H_out, W=W_out, eps=self.eps,
            group_size_per_sample=group_size_per_sample, num_groups=self.num_groups,
            x_stride_n=out1.stride(0), x_stride_c=out1.stride(1), x_stride_h=out1.stride(2), x_stride_w=out1.stride(3),
            out_stride_n=out1_gn.stride(0), out_stride_c=out1_gn.stride(1), out_stride_h=out1_gn.stride(2), out_stride_w=out1_gn.stride(3),
        )

        # SiLU1
        out1_silu = torch.empty_like(out1_gn)
        N1 = out1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](out1_gn, out1_silu, N1, BLOCK=1024)

        # Second convolution: conv2 on out1_silu
        out2_pre = torch.empty((B, C, H_out, W_out), device=device, dtype=torch.float32)
        conv2_weight_f = conv2_weight.contiguous().to(torch.float32).view(C, C, 9)
        grid_conv2 = (B, C, H_out, W_out)
        conv3x3_nobias_one_elem[grid_conv2](
            out1_silu, conv2_weight_f, out2_pre,
            B=B, C_in=C, H=H_out, W=W_out, C_out=C, H_out=H_out, W_out=W_out,
            x_stride_n=out1_silu.stride(0), x_stride_c=out1_silu.stride(1), x_stride_h=out1_silu.stride(2), x_stride_w=out1_silu.stride(3),
            out_stride_n=out2_pre.stride(0), out_stride_c=out2_pre.stride(1), out_stride_h=out2_pre.stride(2), out_stride_w=out2_pre.stride(3),
        )

        # GroupNorm2
        out2_gn = torch.empty_like(out2_pre)
        grid_gn2 = (B, self.num_groups)
        group_norm_two_pass[grid_gn2](
            out2_pre, norm2_weight.contiguous().to(torch.float32), norm2_bias.contiguous().to(torch.float32), out2_gn,
            B=B, C=C, H=H_out, W=W_out, eps=self.eps,
            group_size_per_sample=group_size_per_sample, num_groups=self.num_groups,
            x_stride_n=out2_pre.stride(0), x_stride_c=out2_pre.stride(1), x_stride_h=out2_pre.stride(2), x_stride_w=out2_pre.stride(3),
            out_stride_n=out2_gn.stride(0), out_stride_c=out2_gn.stride(1), out_stride_h=out2_gn.stride(2), out_stride_w=out2_gn.stride(3),
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
        add_residual_kernel[grid_add](out2_silu, x, Nfinal, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
