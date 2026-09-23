import torch
import triton
import triton.language as tl

# Conv3x3, stride=1, padding=1, bias=None
# One program per output element: (n, co, h_out, w_out)
@triton.jit
def conv3x3_nobias_one_elem(
    x_ptr,       # *const float, input [B, C_in, H, W], contiguous
    w_ptr,       # *const float, weights [C_out, C_in, 3, 3], contiguous
    out_ptr,     # *float, output [B, C_out, H_out, W_out], contiguous
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # Accumulator
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel with padding=1
    for ci in tl.static_range(C_in):
        for kh in tl.static_range(3):
            for kw in tl.static_range(3):
                h_in = pid_h - 1 + kh
                w_in = pid_w - 1 + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                # Flattened indexing for x: ((n * C_in + ci) * H + h_in) * W + w_in
                x_index = (pid_n * C_in + ci) * H * W + h_in * W + w_in
                x_val = tl.load(x_ptr + x_index, mask=in_bounds, other=0.0)
                # weights [C_out, C_in, 3, 3] contiguous; flatten to [C_out, C_in*9]
                w_index = pid_co * (C_in * 9) + ci * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_index)
                acc += x_val * w_val

    # Store to output
    out_index = (pid_n * C_out + pid_co) * H_out * W_out + pid_h * W_out + pid_w
    tl.store(out_ptr + out_index, acc)

# Two-pass GroupNorm per (n, group): compute sum and sumsq, then normalize and affine
@triton.jit
def group_norm_two_pass(
    in_ptr,    # *const float, input tensor [B, C, H, W], contiguous
    gamma_ptr, # *const float, per-channel gamma [C]
    beta_ptr,  # *const float, per-channel beta [C]
    out_ptr,   # *float, output tensor [B, C, H, W], contiguous
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, eps: tl.constexpr,
    in_stride0, in_stride1, in_stride2, in_stride3,
    out_stride0, out_stride1, out_stride2, out_stride3,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)  # group id
    group_size = C // num_groups  # channels per group

    # First pass: compute sum and sum of squares for this (n, group)
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)
    start = pid_g * group_size
    for c in tl.static_range(group_size):
        chan = start + c
        base = pid_n * in_stride0 + chan * in_stride1
        total = H * W
        for i in tl.static_range(total):
            ptr = in_ptr + base + i
            val = tl.load(ptr)
            sum_val += val
            sum_sq += val * val

    m = group_size
    mean = sum_val / m
    var = sum_sq / m - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and affine, then store
    for c in tl.static_range(group_size):
        chan = start + c
        base_in = pid_n * in_stride0 + chan * in_stride1
        base_out = pid_n * out_stride0 + chan * out_stride1
        gamma = tl.load(gamma_ptr + chan)
        beta = tl.load(beta_ptr + chan)
        total = H * W
        for i in tl.static_range(total):
            ptr_in = in_ptr + base_in + i
            x = tl.load(ptr_in)
            y = (x - mean) * rstd
            y = y * gamma + beta
            ptr_out = out_ptr + base_out + i
            tl.store(ptr_out, y)

# Elementwise SiLU: y = x * sigmoid(x)
@triton.jit
def silu_kernel(in_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offs, y, mask=mask)

# Elementwise add: out = in_ptr + add_ptr
@triton.jit
def add_residual_kernel(in_ptr, add_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(in_ptr + offs, mask=mask, other=0.0)
    b = tl.load(add_ptr + offs, mask=mask, other=0.0)
    c = a + b
    tl.store(out_ptr + offs, c, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        Triton-only forward:
        - Both convolutions (3x3, stride=1, padding=1, bias=None) via Triton.
        - Both GroupNorms (num_groups=self.num_groups) via Triton.
        - SiLU via Triton elementwise.
        - Add residual x via Triton elementwise.
        """
        device = x.device

        # Ensure dtype and contiguity; operate in float32
        x_f = x.contiguous().to(torch.float32)
        conv1_weight_f = conv1_weight.contiguous().to(torch.float32)
        norm1_weight_f = norm1_weight.contiguous().to(torch.float32)
        norm1_bias_f = norm1_bias.contiguous().to(torch.float32)
        conv2_weight_f = conv2_weight.contiguous().to(torch.float32)
        norm2_weight_f = norm2_weight.contiguous().to(torch.float32)
        norm2_bias_f = norm2_bias.contiguous().to(torch.float32)

        B, C_in, H, W = x_f.shape
        C_out = conv1_weight_f.shape[0]
        H_out = H  # padding=1, stride=1 -> same spatial size
        W_out = W

        # 1) First conv: out1
        out1 = torch.empty((B, C_out, H_out, W_out), device=device, dtype=torch.float32)
        grid_conv1 = (B, C_out, H_out, W_out)
        conv3x3_nobias_one_elem[grid_conv1](
            x_f, conv1_weight_f, out1,
            B, C_in, H, W, C_out, H_out, W_out,
            num_warps=1, num_stages=1,
        )

        # 2) GroupNorm1 (num_groups)
        out1_gn = torch.empty_like(out1)
        grid_gn1 = (B, self.num_groups)
        group_norm_two_pass[grid_gn1](
            out1, norm1_weight_f, norm1_bias_f, out1_gn,
            B, C_out, H_out, W_out, self.num_groups, self.eps,
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
            out1_gn.stride(0), out1_gn.stride(1), out1_gn.stride(2), out1_gn.stride(3),
            num_warps=1, num_stages=1,
        )

        # 3) SiLU1
        out1_silu = torch.empty_like(out1_gn)
        N1 = out1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](out1_gn, out1_silu, N1, BLOCK=1024, num_warps=4)

        # 4) Second convolution: out2_pre
        out2_pre = torch.empty((B, C_out, H_out, W_out), device=device, dtype=torch.float32)
        grid_conv2 = (B, C_out, H_out, W_out)
        conv3x3_nobias_one_elem[grid_conv2](
            out1_silu, conv2_weight_f, out2_pre,
            B, C_out, H_out, W_out, C_out, H_out, W_out,
            num_warps=1, num_stages=1,
        )

        # 5) GroupNorm2 (num_groups)
        out2_gn = torch.empty_like(out2_pre)
        grid_gn2 = (B, self.num_groups)
        group_norm_two_pass[grid_gn2](
            out2_pre, norm2_weight_f, norm2_bias_f, out2_gn,
            B, C_out, H_out, W_out, self.num_groups, self.eps,
            out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
            out2_gn.stride(0), out2_gn.stride(1), out2_gn.stride(2), out2_gn.stride(3),
            num_warps=1, num_stages=1,
        )

        # 6) SiLU2
        out2_silu = torch.empty_like(out2_gn)
        N2 = out2_gn.numel()
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_kernel[grid_silu2](out2_gn, out2_silu, N2, BLOCK=1024, num_warps=4)

        # 7) Add residual x
        out = torch.empty_like(out2_silu)
        Nfinal = out2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 1024),)
        add_residual_kernel[grid_add](out2_silu, x_f, out, Nfinal, BLOCK=1024, num_warps=4)

        return out


def run(*args):
    return ModelNew()(*args)
