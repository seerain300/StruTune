import torch
import triton
import triton.language as tl


# Triton conv3x3 stride=1, padding=1, no bias: one program per (n, co, h_out, w_out)
@triton.jit
def conv3x3_nobias_one_elem(
    x_ptr,         # *const float, input [B, C_in, H, W]
    w_ptr,         # *const float, weights [C_out, C_in, 3, 3], flattened to [C_out, C_in, 9]
    out_ptr,       # *float, output [B, C_out, H, W]
    B,             # int, batch size (runtime)
    C_in,          # tl.constexpr, input channels
    H,             # tl.constexpr, input height
    W,             # tl.constexpr, input width
    C_out,         # tl.constexpr, output channels
    H_out,         # tl.constexpr, output height
    W_out,         # tl.constexpr, output width
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

    # accumulate in float32
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


# Triton GroupNorm forward with affine: two-pass per (n, group)
@triton.jit
def group_norm_forward_two_pass(
    inp_ptr,       # *const float, input [B, C, H, W]
    gamma_ptr,     # *const float, [C] scale (weight)
    beta_ptr,      # *const float, [C] bias (bias)
    out_ptr,       # *float, output [B, C, H, W]
    B,             # int
    C,             # tl.constexpr, channels
    H,             # tl.constexpr
    W,             # tl.constexpr
    num_groups: tl.constexpr,   # e.g., 32
    eps: tl.constexpr,          # e.g., 1e-5
    inp_stride_n, inp_stride_c, inp_stride_h, inp_stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
):
    n = tl.program_id(0)
    group_id = tl.program_id(1)
    group_size = C // num_groups

    # First pass: compute sum and sum of squares over channels in group and all H*W
    sum_val = 0.0
    sumsq_val = 0.0

    for ch in tl.static_range(group_id * group_size, (group_id + 1) * group_size):
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                ptr_in = inp_ptr + n * inp_stride_n + ch * inp_stride_c + h * inp_stride_h + w * inp_stride_w
                x = tl.load(ptr_in)
                sum_val += x
                sumsq_val += x * x

    group_numel = group_size * H * W
    mean = sum_val / group_numel
    var = sumsq_val / group_numel - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, then store
    for ch in tl.static_range(group_id * group_size, (group_id + 1) * group_size):
        gamma = tl.load(gamma_ptr + ch)
        beta = tl.load(beta_ptr + ch)
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                ptr_in = inp_ptr + n * inp_stride_n + ch * inp_stride_c + h * inp_stride_h + w * inp_stride_w
                x = tl.load(ptr_in)
                y = (x - mean) * rstd
                y = y * gamma + beta
                ptr_out = out_ptr + n * out_stride_n + ch * out_stride_c + h * out_stride_h + w * out_stride_w
                tl.store(ptr_out, y)


# Triton SiLU elementwise
@triton.jit
def silu_kernel(inp_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(out_ptr + offs, y, mask=mask)


# Triton elementwise add residual
@triton.jit
def add_residual_kernel(out_ptr, res_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    y = tl.load(out_ptr + offs, mask=mask, other=0.0)
    r = tl.load(res_ptr + offs, mask=mask, other=0.0)
    z = y + r
    tl.store(out_ptr + offs, z, mask=mask)


# ModelNew: forward uses only Triton kernels
class ModelNew(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.num_groups = 32

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                        conv2_weight, norm2_weight, norm2_bias):
        # Ensure float32 and contiguous
        x = x.contiguous().to(torch.float32)
        device = x.device

        # Dimensions
        B, C, H, W = x.shape
        # Check GroupNorm requirement
        assert C % self.num_groups == 0, "C must be divisible by num_groups=32 for GroupNorm."

        # First conv: out1 = conv3x3(x, conv1_weight)
        out1 = torch.empty((B, C, H, W), device=device, dtype=torch.float32)

        grid_conv1 = (B, C, H, W)
        conv3x3_nobias_one_elem[grid_conv1](
            x, conv1_weight, out1,
            B, C, H, W, C, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        )

        # GroupNorm1 (num_groups=32), affine with weight/bias
        out1_gn = torch.empty_like(out1)

        grid_gn1 = (B, self.num_groups)
        group_norm_forward_two_pass[grid_gn1](
            out1, norm1_weight, norm1_bias, out1_gn,
            B, C, H, W,
            self.num_groups, self.eps,
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
            out1_gn.stride(0), out1_gn.stride(1), out1_gn.stride(2), out1_gn.stride(3),
        )

        # SiLU1
        out1_silu = torch.empty_like(out1_gn)
        N1 = out1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](out1_gn, out1_silu, N1, BLOCK=1024)

        # Second conv: out2_pre = conv3x3(out1_silu)
        out2_pre = torch.empty((B, C, H, W), device=device, dtype=torch.float32)

        grid_conv2 = (B, C, H, W)
        conv3x3_nobias_one_elem[grid_conv2](
            out1_silu, conv2_weight, out2_pre,
            B, C, H, W, C, H, W,
            out1_silu.stride(0), out1_silu.stride(1), out1_silu.stride(2), out1_silu.stride(3),
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        )

        # GroupNorm2 (num_groups=32), affine with weight/bias
        out2_gn = torch.empty_like(out2_pre)

        grid_gn2 = (B, self.num_groups)
        group_norm_forward_two_pass[grid_gn2](
            out2_pre, norm2_weight, norm2_bias, out2_gn,
            B, C, H, W,
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
        add_residual_kernel[grid_add](out2_silu, x, Nfinal, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
