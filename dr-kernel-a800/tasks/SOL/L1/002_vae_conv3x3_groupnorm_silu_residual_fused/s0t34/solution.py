import torch
import triton
import triton.language as tl


# Triton conv3x3 stride=1, padding=1, no bias: one program per (n, co, h_out, w_out)
@triton.jit
def conv3x3_nobias_one_elem(
    x_ptr,         # *const float, input [B, C_in, H, W]
    w_ptr,         # *const float, weights [C_out, C_in, 3, 3], flattened to [C_out, C_in, 9]
    out_ptr,       # *float, output [B, C_out, H, W]
    B: tl.constexpr,
    C_in: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C_out: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
):
    # program ids
    pid_n = tl.program_id(0)  # batch
    pid_co = tl.program_id(1) # output channel
    pid_h = tl.program_id(2)  # output height index
    pid_w = tl.program_id(3)  # output width index

    h_out = pid_h
    w_out = pid_w

    # accumulate
    acc = 0.0

    # loop over input channels and 3x3 kernel
    for ci in tl.static_range(C_in):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                h_in = h_out - 1 + kh  # padding=1
                w_in = w_out - 1 + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                # base pointer for x at (n, ci, h_in, w_in)
                ptr_x = x_ptr + pid_n * x_stride_n + ci * x_stride_c + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(ptr_x, mask=in_bounds, other=0.0)
                # weight index: (co * (C_in * 9)) + (ci * 9) + (kh * 3 + kw)
                w_idx = pid_co * (C_in * 9) + ci * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_idx)
                acc += x_val * w_val

    # store output
    out_ptr_elt = out_ptr + pid_n * out_stride_n + pid_co * out_stride_c + h_out * out_stride_h + w_out * out_stride_w
    tl.store(out_ptr_elt, acc)


# Triton GroupNorm forward: two-pass per (n, group)
@triton.jit
def group_norm_forward_two_pass(
    inp_ptr,        # *const float, input [B, C, H, W]
    gamma_ptr,      # *const float, [C] per-channel scale
    beta_ptr,       # *const float, [C] per-channel bias
    out_ptr,        # *float, output [B, C, H, W]
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    num_groups: tl.constexpr,  # e.g., 32
    eps: tl.constexpr,         # e.g., 1e-5
    inp_stride_n, inp_stride_c, inp_stride_h, inp_stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
):
    # program ids
    pid_b = tl.program_id(0)  # batch
    pid_g = tl.program_id(1)  # group id in [0, num_groups)

    group_size = C // num_groups
    c_start = pid_g * group_size

    # First pass: compute sum and sum of squares over this group for sample pid_b
    sum_val = 0.0
    sumsq_val = 0.0

    # Iterate over channels in the group and all spatial locations
    for ci in tl.static_range(c_start, c_start + group_size):
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                ptr = inp_ptr + pid_b * inp_stride_n + ci * inp_stride_c + h * inp_stride_h + w * inp_stride_w
                x_val = tl.load(ptr)
                sum_val += x_val
                sumsq_val += x_val * x_val

    # Compute mean and variance
    M = group_size * H * W
    mean = sum_val / M
    var = sumsq_val / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, then store
    for ci in tl.static_range(c_start, c_start + group_size):
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                ptr_in = inp_ptr + pid_b * inp_stride_n + ci * inp_stride_c + h * inp_stride_h + w * inp_stride_w
                ptr_out = out_ptr + pid_b * out_stride_n + ci * out_stride_c + h * out_stride_h + w * out_stride_w
                x_val = tl.load(ptr_in)
                gamma_val = tl.load(gamma_ptr + ci)
                beta_val = tl.load(beta_ptr + ci)
                y_val = (x_val - mean) * inv_std
                y_val = y_val * gamma_val + beta_val
                tl.store(ptr_out, y_val)


# Triton SiLU elementwise: y = x * sigmoid(x)
@triton.jit
def silu_kernel(
    inp_ptr,  # *const float
    out_ptr,  # *float
    N: tl.constexpr,
    BLOCK: tl.constexpr = 1024,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offs, y, mask=mask)


# Triton elementwise residual add: out = out + x
@triton.jit
def add_residual_kernel(
    out_ptr,  # *float
    x_ptr,    # *const float
    N: tl.constexpr,
    BLOCK: tl.constexpr = 1024,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    out_val = tl.load(out_ptr + offs, mask=mask, other=0.0)
    x_val = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, out_val + x_val, mask=mask)


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
        Args:
            x: Input tensor of shape (B, C, H, W), float32, contiguous
            conv1_weight: First conv weights (C, C, 3, 3), float32, contiguous
            norm1_weight: First GroupNorm scale (C,), float32, contiguous
            norm1_bias: First GroupNorm bias (C,), float32, contiguous
            conv2_weight: Second conv weights (C, C, 3, 3), float32, contiguous
            norm2_weight: Second GroupNorm scale (C,), float32, contiguous
            norm2_bias: Second GroupNorm bias (C,), float32, contiguous
        Returns:
            Output tensor of shape (B, C, H, W), float32
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        B, C, H, W = x.shape
        # Ensure float32 and contiguous
        x = x.contiguous().to(torch.float32)

        # Prepare weight tensors contiguous and float32
        conv1_weight = conv1_weight.contiguous().to(torch.float32)
        norm1_weight = norm1_weight.contiguous().to(torch.float32)
        norm1_bias = norm1_bias.contiguous().to(torch.float32)

        conv2_weight = conv2_weight.contiguous().to(torch.float32)
        norm2_weight = norm2_weight.contiguous().to(torch.float32)
        norm2_bias = norm2_bias.contiguous().to(torch.float32)

        # Check GroupNorm constraint: C must be divisible by num_groups
        assert C % self.num_groups == 0, "C must be divisible by num_groups (32) for GroupNorm."

        # 1) First convolution: y1 = conv3x3(x)
        y1 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid_conv1 = (B, C, H, W)
        conv3x3_nobias_one_elem[grid_conv1](
            x, conv1_weight, y1,
            B, C, H, W, C, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        )

        # 2) GroupNorm1 (num_groups=32, eps)
        y1_gn = torch.empty_like(y1)
        grid_gn1 = (B, self.num_groups)
        group_norm_forward_two_pass[grid_gn1](
            y1, norm1_weight, norm1_bias, y1_gn,
            B, C, H, W, self.num_groups, self.eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_gn.stride(0), y1_gn.stride(1), y1_gn.stride(2), y1_gn.stride(3),
        )

        # 3) SiLU1
        y1_silu = torch.empty_like(y1_gn)
        N1 = y1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](y1_gn, y1_silu, N1, BLOCK=1024)

        # 4) Second convolution: y2_pre = conv3x3(y1_silu)
        y2_pre = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid_conv2 = (B, C, H, W)
        conv3x3_nobias_one_elem[grid_conv2](
            y1_silu, conv2_weight, y2_pre,
            B, C, H, W, C, H, W,
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            y2_pre.stride(0), y2_pre.stride(1), y2_pre.stride(2), y2_pre.stride(3),
        )

        # 5) GroupNorm2 (num_groups=32, eps)
        y2_gn = torch.empty_like(y2_pre)
        grid_gn2 = (B, self.num_groups)
        group_norm_forward_two_pass[grid_gn2](
            y2_pre, norm2_weight, norm2_bias, y2_gn,
            B, C, H, W, self.num_groups, self.eps,
            y2_pre.stride(0), y2_pre.stride(1), y2_pre.stride(2), y2_pre.stride(3),
            y2_gn.stride(0), y2_gn.stride(1), y2_gn.stride(2), y2_gn.stride(3),
        )

        # 6) SiLU2
        y2_silu = torch.empty_like(y2_gn)
        N2 = y2_gn.numel()
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_kernel[grid_silu2](y2_gn, y2_silu, N2, BLOCK=1024)

        # 7) Add residual x
        out = torch.empty_like(y2_silu)
        Nfinal = y2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 1024),)
        add_residual_kernel[grid_add](y2_silu, x, Nfinal, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
