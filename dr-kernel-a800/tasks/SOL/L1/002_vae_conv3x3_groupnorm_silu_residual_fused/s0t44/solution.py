import torch
import triton
import triton.language as tl


# Simple conv3x3 (stride=1, padding=1, no bias) per output element.
# One Triton program computes one output element (n, co, h_out, w_out).
@triton.jit
def conv3x3_nobias_one_elem(
    x_ptr,        # *const float32, input [B, C_in, H, W]
    w_ptr,        # *const float32, weights [C_out, C_in, 3, 3]
    out_ptr,      # *float32, output [B, C_out, H_out, W_out]
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch
    pid_co = tl.program_id(1) # output channel
    pid_h = tl.program_id(2)  # output height index
    pid_w = tl.program_id(3)  # output width index

    h_out = pid_h
    w_out = pid_w

    acc = 0.0

    # loop over input channels and 3x3 kernel with padding=1
    for ci in tl.static_range(C_in):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                h_in = h_out - 1 + kh
                w_in = w_out - 1 + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                base_x = pid_n * x_stride_n + ci * x_stride_c
                ptr_x = x_ptr + base_x + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(ptr_x, mask=in_bounds, other=0.0)
                # weight index: (co * (C_in * 9)) + (ci * 9) + (kh * 3 + kw)
                w_idx = pid_co * (C_in * 9) + ci * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_idx)
                acc += x_val * w_val

    # store to output
    out_ptr_out = out_ptr + pid_n * out_stride_n + pid_co * out_stride_c + h_out * out_stride_h + w_out * out_stride_w
    tl.store(out_ptr_out, acc)


# GroupNorm (num_groups) with two passes: compute sum and sumsq, then normalize + affine.
# One Triton program per (batch, group).
@triton.jit
def group_norm_two_pass(
    inp_ptr,            # *const float32 input tensor [B, C, H, W]
    weight_ptr,         # *const float32 per-channel scale [C]
    bias_ptr,           # *const float32 per-channel bias [C]
    out_ptr,            # *float32 output tensor [B, C, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, eps: tl.constexpr,
    inp_stride_n: tl.constexpr, inp_stride_c: tl.constexpr, inp_stride_h: tl.constexpr, inp_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch
    pid_group = tl.program_id(1)  # group id

    group_size = C // num_groups
    group_start = pid_group * group_size

    # First pass: compute sum and sumsq over this group's elements
    sum_val = 0.0
    sumsq_val = 0.0
    for ci in tl.static_range(group_start, group_start + group_size):
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                in_ptr = inp_ptr + pid_n * inp_stride_n + ci * inp_stride_c + h * inp_stride_h + w * inp_stride_w
                x_val = tl.load(in_ptr)
                sum_val += x_val
                sumsq_val += x_val * x_val

    group_elems = group_size * H * W
    mean = sum_val / group_elems
    var = sumsq_val / group_elems - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for ci in tl.static_range(group_start, group_start + group_size):
        gamma = tl.load(weight_ptr + ci)
        beta = tl.load(bias_ptr + ci)
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                in_ptr = inp_ptr + pid_n * inp_stride_n + ci * inp_stride_c + h * inp_stride_h + w * inp_stride_w
                x_val = tl.load(in_ptr)
                y = (x_val - mean) * inv_std
                y = y * gamma + beta
                out_ptr = out_ptr + pid_n * out_stride_n + ci * out_stride_c + h * out_stride_h + w * out_stride_w
                tl.store(out_ptr, y)


# Elementwise SiLU over flattened buffer
@triton.jit
def silu_kernel_flat(
    inp_ptr,      # *const float32 input flattened
    out_ptr,      # *float32 output flattened
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offs, y, mask=mask)


# Elementwise add residual over flattened buffer (out = out + inp)
@triton.jit
def add_residual_flat(
    out_ptr,      # *float32 output flattened
    inp_ptr,      # *const float32 residual flattened (same shape as out)
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(out_ptr + offs, mask=mask, other=0.0)
    b = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    c = a + b
    tl.store(out_ptr + offs, c, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups=32, eps=1e-5):
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

        Args:
            x: Input tensor of shape (B, C, H, W), contiguous, float32.
            conv1_weight: First conv weights (C, C, 3, 3), contiguous, float32.
            norm1_weight: First GroupNorm scale (C,), contiguous, float32.
            norm1_bias: First GroupNorm bias (C,), contiguous, float32.
            conv2_weight: Second conv weights (C, C, 3, 3), contiguous, float32.
            norm2_weight: Second GroupNorm scale (C,), contiguous, float32.
            norm2_bias: Second GroupNorm bias (C,), contiguous, float32.
        Returns:
            Output tensor of shape (B, C, H, W), contiguous, float32.
        """
        # Ensure all inputs are contiguous and float32
        x = x.contiguous().to(torch.float32)
        device = x.device

        B, C, H, W = x.shape
        # First conv
        out1 = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        grid_conv1 = (B, C, H, W)
        conv3x3_nobias_one_elem[grid_conv1](
            x, conv1_weight, out1,
            B, C, H, W, C, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        )

        # GroupNorm1 (num_groups=32)
        out1_gn = torch.empty_like(out1)
        grid_gn1 = (B, self.num_groups)
        group_norm_two_pass[grid_gn1](
            out1, norm1_weight, norm1_bias, out1_gn,
            B, C, H, W, self.num_groups, self.eps,
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
            out1_gn.stride(0), out1_gn.stride(1), out1_gn.stride(2), out1_gn.stride(3),
        )

        # SiLU1
        out1_silu = torch.empty_like(out1_gn)
        N1 = out1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel_flat[grid_silu1](out1_gn, out1_silu, N1, BLOCK=1024)

        # Second conv
        out2_pre = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        grid_conv2 = (B, C, H, W)
        conv3x3_nobias_one_elem[grid_conv2](
            out1_silu, conv2_weight, out2_pre,
            B, C, H, W, C, H, W,
            out1_silu.stride(0), out1_silu.stride(1), out1_silu.stride(2), out1_silu.stride(3),
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        )

        # GroupNorm2 (num_groups=32)
        out2_gn = torch.empty_like(out2_pre)
        grid_gn2 = (B, self.num_groups)
        group_norm_two_pass[grid_gn2](
            out2_pre, norm2_weight, norm2_bias, out2_gn,
            B, C, H, W, self.num_groups, self.eps,
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
            out2_gn.stride(0), out2_gn.stride(1), out2_gn.stride(2), out2_gn.stride(3),
        )

        # SiLU2
        out2_silu = torch.empty_like(out2_gn)
        N2 = out2_gn.numel()
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_kernel_flat[grid_silu2](out2_gn, out2_silu, N2, BLOCK=1024)

        # Add residual x
        out = torch.empty_like(out2_silu)
        Nfinal = out2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 1024),)
        add_residual_flat[grid_add](out2_silu, x, Nfinal, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
