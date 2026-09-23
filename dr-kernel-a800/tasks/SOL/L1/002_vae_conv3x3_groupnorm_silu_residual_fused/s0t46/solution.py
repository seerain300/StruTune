import torch
import triton
import triton.language as tl


# 1) First convolution: y1[n, co, h_out, w_out] = sum_{ci,kh,kw} x[n, ci, h_out+kh-1, w_out+kw-1] * w[co, ci, kh, kw], padding=1, bias=None
@triton.jit
def conv3x3_nobias_one_elem(
    x_ptr,       # *const float, input [B, C_in, H, W]
    w_ptr,       # *const float, weights [C_out, C_in, 3, 3] flattened to [C_out, C_in, 9]
    out_ptr,     # *float, output [B, C_out, H_out, W_out]
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C_out: tl.constexpr,
    H_out: tl.constexpr, W_out: tl.constexpr,
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch
    pid_co = tl.program_id(1) # output channel
    pid_h = tl.program_id(2)  # output height index
    pid_w = tl.program_id(3)  # output width index

    # Compute accumulation
    acc = 0.0
    # Loop over input channels and 3x3 kernel
    for ci in tl.static_range(C_in):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                # padding=1
                h_in = pid_h - 1 + kh
                w_in = pid_w - 1 + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                base_x = pid_n * x_stride_n + ci * x_stride_c
                ptr_x = x_ptr + base_x + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(ptr_x, mask=in_bounds, other=0.0)
                # weight index: ((co * C_in) + ci) * 9 + (kh * 3 + kw)
                w_idx = (pid_co * C_in + ci) * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_idx)
                acc += x_val * w_val
    # Store result
    out_ptr_base = out_ptr + pid_n * out_stride_n + pid_co * out_stride_c
    out_ptr_val = out_ptr_base + pid_h * out_stride_h + pid_w * out_stride_w
    tl.store(out_ptr_val, acc)


# 2) GroupNorm in Triton: two-pass per (n, group) across C_out channels (and spatial dims H_out*W_out).
#    We assume input is [B, C, H_out, W_out], contiguous.
@triton.jit
def group_norm_two_pass(
    x_ptr,        # *const float, input [B, C, H_out, W_out]
    gamma_ptr,    # *const float, per-channel weight (C,)
    beta_ptr,     # *const float, per-channel bias (C,)
    out_ptr,      # *float, output [B, C, H_out, W_out]
    B: tl.constexpr, C: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr, num_groups: tl.constexpr, eps: tl.constexpr,
    x_stride_n: tl.constexpr, x_stride_c: tl.constexpr, x_stride_h: tl.constexpr, x_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    # program ids: (batch, group)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)
    group_size = C // num_groups
    channel_start = pid_g * group_size

    # First pass: compute sum and sum of squares for this (n, group)
    sum_val = 0.0
    sumsq_val = 0.0
    for ci in tl.static_range(group_size):
        c = channel_start + ci
        # iterate over H_out * W_out elements
        for h in tl.static_range(H_out):
            for w in tl.static_range(W_out):
                base_x = pid_n * x_stride_n + c * x_stride_c
                ptr_x = x_ptr + base_x + h * x_stride_h + w * x_stride_w
                x_val = tl.load(ptr_x)
                sum_val += x_val
                sumsq_val += x_val * x_val

    m = group_size * H_out * W_out
    mean = sum_val / m
    var = sumsq_val / m - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine gamma/beta
    for ci in tl.static_range(group_size):
        c = channel_start + ci
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)
        for h in tl.static_range(H_out):
            for w in tl.static_range(W_out):
                base_x = pid_n * x_stride_n + c * x_stride_c
                ptr_x = x_ptr + base_x + h * x_stride_h + w * x_stride_w
                x_val = tl.load(ptr_x)
                norm = (x_val - mean) * inv_std
                out_val = norm * gamma + beta
                base_out = out_ptr + pid_n * out_stride_n + c * out_stride_c
                ptr_out = base_out + h * out_stride_h + w * out_stride_w
                tl.store(ptr_out, out_val)


# 3) Elementwise SiLU over flattened tensor
@triton.jit
def silu_kernel(
    x_ptr, out_ptr, N, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offsets, y, mask=mask)


# 4) Elementwise residual addition over flattened tensor
@triton.jit
def add_residual_kernel(
    x_ptr, res_ptr, out_ptr, N, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    r = tl.load(res_ptr + offsets, mask=mask, other=0.0)
    y = x + r
    tl.store(out_ptr + offsets, y, mask=mask)


# Forward function: Triton-only, no torch ops
class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        conv1_weight: torch.Tensor,
        norm1_weight: torch.Tensor,
        norm1_bias: torch.Tensor,
        conv2_weight: torch.Tensor,
        norm2_weight: torch.Tensor,
        norm2_bias: torch.Tensor,
    ):
        # Ensure all tensors are contiguous and float32
        device = x.device
        B, C, H, W = x.shape
        C1 = conv1_weight.shape[0]
        C2 = conv2_weight.shape[0]
        # Sanity checks for conv weights
        assert conv1_weight.shape == (C1, C, 3, 3), "conv1_weight must be (C_out, C_in, 3, 3)"
        assert conv2_weight.shape == (C2, C, 3, 3), "conv2_weight must be (C_out, C_in, 3, 3)"
        # GroupNorm requires C divisible by num_groups
        assert C % self.num_groups == 0, "C must be divisible by num_groups (32)"
        assert C1 == C and C2 == C, "conv weights output channels must equal input channels for this residual block"

        # 1) First conv: out1 = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
        out1 = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        # Flatten conv1_weight to [C, C*9]
        w1 = conv1_weight.contiguous().view(C, C * 9).to(torch.float32)
        x_f = x.contiguous().to(torch.float32)

        grid_conv1 = (B, C, H, W)
        conv3x3_nobias_one_elem[grid_conv1](
            x_f, w1, out1,
            B, C, H, W, C, H, W, H, W,
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        )

        # 2) GroupNorm1 (num_groups=32)
        out1_gn = torch.empty_like(out1)
        grid_gn1 = (B, self.num_groups)
        group_norm_two_pass[grid_gn1](
            out1, norm1_weight.to(torch.float32), norm1_bias.to(torch.float32), out1_gn,
            B, C, H, W, self.num_groups, self.eps,
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
            out1_gn.stride(0), out1_gn.stride(1), out1_gn.stride(2), out1_gn.stride(3),
        )

        # 3) SiLU1
        out1_silu = torch.empty_like(out1_gn)
        N1 = out1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](out1_gn, out1_silu, N1, BLOCK=1024)

        # 4) Second convolution: out2_pre = conv3x3(out1_silu)
        out2_pre = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        w2 = conv2_weight.contiguous().view(C, C * 9).to(torch.float32)

        grid_conv2 = (B, C, H, W)
        conv3x3_nobias_one_elem[grid_conv2](
            out1_silu, w2, out2_pre,
            B, C, H, W, C, H, W, H, W,
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        )

        # 5) GroupNorm2 (num_groups=32)
        out2_gn = torch.empty_like(out2_pre)
        grid_gn2 = (B, self.num_groups)
        group_norm_two_pass[grid_gn2](
            out2_pre, norm2_weight.to(torch.float32), norm2_bias.to(torch.float32), out2_gn,
            B, C, H, W, self.num_groups, self.eps,
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
            out2_gn.stride(0), out2_gn.stride(1), out2_gn.stride(2), out2_gn.stride(3),
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
        add_residual_kernel[grid_add](out2_silu, x_f, out, Nfinal, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
