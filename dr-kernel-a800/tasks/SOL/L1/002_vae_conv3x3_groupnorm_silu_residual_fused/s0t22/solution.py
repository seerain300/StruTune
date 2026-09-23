import torch
import triton
import triton.language as tl


# Triton GroupNorm (num_groups, per-channel affine gamma/beta, eps) - two-pass on flattened buffer
# Grid is (B, G), i.e., one program per (batch, group)
@triton.jit
def group_norm_two_pass_flat(
    in_ptr,         # *const float, input flattened
    out_ptr,        # *float, output flattened
    gamma_ptr,      # *const float, per-channel gamma [C]
    beta_ptr,       # *const float, per-channel beta [C]
    N: tl.constexpr,          # total number of elements = B * C * H * W
    C: tl.constexpr,          # channels
    H: tl.constexpr,          # height
    W: tl.constexpr,          # width
    G: tl.constexpr,          # num_groups
    eps,                      # float epsilon
    BLOCK: tl.constexpr,      # block size for processing
):
    n = tl.program_id(0)  # batch
    g = tl.program_id(1)  # group index

    group_size = C // G
    group_start = g * group_size
    total_elems = group_size * H * W

    # First pass: compute sum and sum of squares for this (n, group)
    sum_val = 0.0
    sum_sq = 0.0
    for ci in tl.static_range(group_start, group_start + group_size):
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                idx = ((n * C + ci) * H + h) * W + w
                x = tl.load(in_ptr + idx)
                sum_val += x
                sum_sq += x * x

    mean = sum_val / total_elems
    var = sum_sq / total_elems - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, write to out
    for ci in tl.static_range(group_start, group_start + group_size):
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                idx = ((n * C + ci) * H + h) * W + w
                x = tl.load(in_ptr + idx)
                gamma = tl.load(gamma_ptr + ci)
                beta = tl.load(beta_ptr + ci)
                y = ((x - mean) * inv_std) * gamma + beta
                tl.store(out_ptr + idx, y)


# Triton SiLU elementwise: y = x * sigmoid(x)
@triton.jit
def silu_kernel_flat(
    x_ptr,        # *const float, input flattened
    y_ptr,        # *float, output flattened
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


# Triton residual add elementwise: y = x1 + x2 (x1 is post-SiLU/GroupNorm, x2 is residual x)
@triton.jit
def add_residual_flat(
    x1_ptr,        # *const float, input flattened (post-processing result)
    x2_ptr,        # *const float, input flattened (residual x)
    out_ptr,       # *float, output flattened
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x1 = tl.load(x1_ptr + offs, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offs, mask=mask, other=0.0)
    y = x1 + x2
    tl.store(out_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        Fused residual block:
        Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add(residual)

        Convolutions are computed via torch.nn.functional.conv2d for robustness.
        GroupNorm, SiLU, and residual addition are implemented in Triton kernels.
        """
        # Ensure dtype and contiguity
        x = x.contiguous().to(torch.float32)

        # 1) First conv: out1 = conv3x3(x, conv1_weight, stride=1, padding=1, bias=None)
        out1 = torch.nn.functional.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # 2) GroupNorm1: out1_gn = GroupNorm(num_groups=self.num_groups, weight=norm1_weight, bias=norm1_bias, eps=self.eps)
        out1_gn = torch.empty_like(out1)
        B, C, H, W = out1.shape
        assert C % self.num_groups == 0, "C must be divisible by num_groups=32"
        N1 = out1.numel()
        grid_gn1 = (B, self.num_groups)
        group_norm_two_pass_flat[grid_gn1](
            out1.view(-1),
            out1_gn.view(-1),
            norm1_weight.to(torch.float32),
            norm1_bias.to(torch.float32),
            N1,
            C,
            H,
            W,
            self.num_groups,
            self.eps,
            BLOCK=8192,
            num_warps=4,
        )

        # 3) SiLU1
        out1_silu = torch.empty_like(out1_gn)
        N1 = out1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 8192),)
        silu_kernel_flat[grid_silu1](out1_gn.view(-1), out1_silu.view(-1), N1, BLOCK=8192, num_warps=4)

        # 4) Second conv: out2_pre = conv3x3(out1_silu, conv2_weight, stride=1, padding=1, bias=None)
        out2_pre = torch.nn.functional.conv2d(out1_silu, conv2_weight, bias=None, stride=1, padding=1)

        # 5) GroupNorm2
        out2_gn = torch.empty_like(out2_pre)
        B2, C2, H2, W2 = out2_pre.shape
        assert C2 % self.num_groups == 0, "C must be divisible by num_groups=32"
        N2 = out2_pre.numel()
        grid_gn2 = (B2, self.num_groups)
        group_norm_two_pass_flat[grid_gn2](
            out2_pre.view(-1),
            out2_gn.view(-1),
            norm2_weight.to(torch.float32),
            norm2_bias.to(torch.float32),
            N2,
            C2,
            H2,
            W2,
            self.num_groups,
            self.eps,
            BLOCK=8192,
            num_warps=4,
        )

        # 6) SiLU2
        out2_silu = torch.empty_like(out2_gn)
        N2 = out2_gn.numel()
        grid_silu2 = (triton.cdiv(N2, 8192),)
        silu_kernel_flat[grid_silu2](out2_gn.view(-1), out2_silu.view(-1), N2, BLOCK=8192, num_warps=4)

        # 7) Add residual x
        out = torch.empty_like(out2_silu)
        Nfinal = out2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 8192),)
        add_residual_flat[grid_add](out2_silu.view(-1), x.view(-1), out.view(-1), Nfinal, BLOCK=8192, num_warps=4)

        return out


def run(*args):
    return ModelNew()(*args)
