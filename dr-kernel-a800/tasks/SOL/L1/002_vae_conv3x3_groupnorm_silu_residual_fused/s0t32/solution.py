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
                # weight index: (co * (C_in * 9)) + (ci * 9 + kh*3 + kw)
                w_idx = pid_co * (C_in * 9) + ci * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_idx)
                acc += x_val * w_val

    # store output
    out_ptr_elt = out_ptr + pid_n * out_stride_n + pid_co * out_stride_c + h_out * out_stride_h + w_out * out_stride_w
    tl.store(out_ptr_elt, acc)


# Triton GroupNorm over groups: two-pass per (n, group)
@triton.jit
def group_norm_forward_two_pass(
    inp_ptr,           # *const float, input [B, C, H, W]
    gamma_ptr,         # *const float, per-channel gamma [C]
    beta_ptr,          # *const float, per-channel beta [C]
    out_ptr,           # *float, output normalized [B, C, H, W]
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    NUM_GROUPS: tl.constexpr,  # e.g., 32
    eps: tl.constexpr,         # e.g., 1e-5
    in_stride_n, in_stride_c, in_stride_h, in_stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
):
    n = tl.program_id(0)
    group = tl.program_id(1)

    # compute channel range for this group
    group_size = C // NUM_GROUPS
    c_start = group * group_size
    c_end = c_start + group_size

    # first pass: compute sum and sumsq over (channels in group) * (H*W)
    total_sum = 0.0
    total_sumsq = 0.0
    for ch in tl.static_range(C):
        # only this group's channels
        if ch < c_start or ch >= c_end:
            continue
        # iterate H and W
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                ptr_in = inp_ptr + n * in_stride_n + ch * in_stride_c + h * in_stride_h + w * in_stride_w
                x_val = tl.load(ptr_in)
                total_sum += x_val
                total_sumsq += x_val * x_val

    N = group_size * H * W
    mean = total_sum / N
    var = total_sumsq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # second pass: normalize and apply affine
    for ch in tl.static_range(C):
        if ch < c_start or ch >= c_end:
            continue
        gamma = tl.load(gamma_ptr + ch)
        beta = tl.load(beta_ptr + ch)
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                ptr_in = inp_ptr + n * in_stride_n + ch * in_stride_c + h * in_stride_h + w * in_stride_w
                x_val = tl.load(ptr_in)
                y = (x_val - mean) * rstd
                y = y * gamma + beta
                ptr_out = out_ptr + n * out_stride_n + ch * out_stride_c + h * out_stride_h + w * out_stride_w
                tl.store(ptr_out, y)


# Triton elementwise SiLU: y = x * sigmoid(x)
@triton.jit
def silu_kernel(inp_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    y = x / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offs, y, mask=mask)


# Triton elementwise add residual: out = out + inp
@triton.jit
def add_residual_kernel(out_ptr, inp_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    out_val = tl.load(out_ptr + offs, mask=mask, other=0.0)
    res_val = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    y = out_val + res_val
    tl.store(out_ptr + offs, y, mask=mask)


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
        # Ensure dtype and contiguity
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels"
        x = x.contiguous().to(torch.float32)
        conv1_weight = conv1_weight.contiguous().to(torch.float32)
        conv2_weight = conv2_weight.contiguous().to(torch.float32)
        norm1_weight = norm1_weight.contiguous().to(torch.float32)
        norm1_bias = norm1_bias.contiguous().to(torch.float32)
        norm2_weight = norm2_weight.contiguous().to(torch.float32)
        norm2_bias = norm2_bias.contiguous().to(torch.float32)

        B, C, H, W = x.shape
        C_in1 = conv1_weight.shape[1]  # input channels for first conv
        C_out1 = conv1_weight.shape[0] # output channels for first conv
        C_in2 = conv2_weight.shape[1]  # input channels for second conv
        C_out2 = conv2_weight.shape[0] # output channels for second conv

        # Output of first conv (before GroupNorm)
        out1 = torch.empty((B, C_out1, H, W), device=x.device, dtype=torch.float32)

        # First conv3x3: bias=None, stride=1, padding=1
        grid_conv1 = (B, C_out1, H, W)
        conv3x3_nobias_one_elem[grid_conv1](
            x, conv1_weight, out1,
            B, C_in1, H, W, C_out1, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        )

        # GroupNorm1: num_groups=32, per-channel affine
        out1_gn = torch.empty_like(out1)
        grid_gn1 = (B, self.num_groups)
        group_norm_forward_two_pass[grid_gn1](
            out1, norm1_weight, norm1_bias, out1_gn,
            B, C_out1, H, W, self.num_groups, self.eps,
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
            out1_gn.stride(0), out1_gn.stride(1), out1_gn.stride(2), out1_gn.stride(3),
        )

        # SiLU1
        out1_silu = torch.empty_like(out1_gn)
        N1 = out1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](out1_gn, out1_silu, N1, 1024)

        # Second conv3x3: bias=None, stride=1, padding=1 on out1_silu
        out2_pre = torch.empty((B, C_out2, H, W), device=x.device, dtype=torch.float32)

        grid_conv2 = (B, C_out2, H, W)
        conv3x3_nobias_one_elem[grid_conv2](
            out1_silu, conv2_weight, out2_pre,
            B, C_in2, H, W, C_out2, H, W,
            out1_silu.stride(0), out1_silu.stride(1), out1_silu.stride(2), out1_silu.stride(3),
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        )

        # GroupNorm2: num_groups=32, per-channel affine
        out2_gn = torch.empty_like(out2_pre)

        grid_gn2 = (B, self.num_groups)
        group_norm_forward_two_pass[grid_gn2](
            out2_pre, norm2_weight, norm2_bias, out2_gn,
            B, C_out2, H, W, self.num_groups, self.eps,
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
            out2_gn.stride(0), out2_gn.stride(1), out2_gn.stride(2), out2_gn.stride(3),
        )

        # SiLU2
        out2_silu = torch.empty_like(out2_gn)
        N2 = out2_gn.numel()
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_kernel[grid_silu2](out2_gn, out2_silu, N2, 1024)

        # Add residual x
        out = torch.empty_like(out2_silu)
        Nfinal = out2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 1024),)
        add_residual_kernel[grid_add](out2_silu, x, Nfinal, 1024)

        return out


def run(*args):
    return ModelNew()(*args)
