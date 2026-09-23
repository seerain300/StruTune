import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def group_norm_two_pass(
    inp_ptr,        # *const float, input flattened
    gamma_ptr,      # *const float, per-channel scale (C,)
    beta_ptr,       # *const float, per-channel bias (C,)
    out_ptr,        # *float, output flattened
    N,              # total number of elements in inp_ptr/out_ptr (B*C*H*W)
    C,              # channels
    H,              # height
    W,              # width
    num_groups,     # number of groups (32)
):
    # Each program handles one (n, group). We derive n from the number of programs in axis 0.
    n = tl.program_id(0)
    group_id = tl.program_id(1)
    group_size = C // num_groups
    start_channel = group_id * group_size

    # First pass: compute mean and variance over the group
    sum_val = 0.0
    sum_sq = 0.0
    for ci in tl.static_range(start_channel, start_channel + group_size):
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                idx = ((n * C + ci) * H + h) * W + w
                x = tl.load(inp_ptr + idx)
                sum_val += x
                sum_sq += x * x

    numel_group = group_size * H * W
    mean = sum_val / numel_group
    var = sum_sq / numel_group - mean * mean
    inv_std = 1.0 / tl.sqrt(var + 1e-5)  # use a small epsilon for stability

    # Second pass: normalize and apply affine
    for ci in tl.static_range(start_channel, start_channel + group_size):
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                idx = ((n * C + ci) * H + h) * W + w
                x = tl.load(inp_ptr + idx)
                gamma = tl.load(gamma_ptr + ci)
                beta = tl.load(beta_ptr + ci)
                y = (x - mean) * inv_std * gamma + beta
                tl.store(out_ptr + idx, y)


@triton.jit
def silu_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # SiLU: x * sigmoid(x)
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def add_residual_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(out_ptr + offsets, c, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5, num_groups: int = 32):
        super().__init__()
        self.eps = eps
        self.num_groups = num_groups

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
        """
        Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        Triton kernels implement GroupNorm (two-pass), SiLU, and residual add. Conv3x3 is done via PyTorch for robustness.
        """
        # Ensure dtype float32 and contiguity for Triton
        device = x.device

        # conv1: PyTorch (stride=1, padding=1, bias=None)
        out1 = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)

        # Triton GroupNorm1
        B, C, H, W = out1.shape
        out1_g = out1.contiguous().view(-1)
        gamma1 = norm1_weight.to(device=device, dtype=torch.float32)
        beta1 = norm1_bias.to(device=device, dtype=torch.float32)
        out1_gn = torch.empty_like(out1_g)
        grid_gn1 = (B, self.num_groups)
        group_norm_two_pass[grid_gn1](
            out1_g,
            gamma1,
            beta1,
            out1_gn,
            out1_g.numel(),
            C,
            H,
            W,
            self.num_groups,
            num_warps=4,
        )
        out1_gn = out1_gn.view(B, C, H, W)

        # SiLU1 via Triton
        N1 = out1_gn.numel()
        out1_silu = torch.empty_like(out1_gn)
        out1_silu_flat = out1_silu.view(-1)
        grid_silu1 = (triton.cdiv(N1, 4096),)
        silu_kernel[grid_silu1](out1_gn.view(-1), out1_silu_flat, N1, BLOCK=4096, num_warps=8)

        # conv2: PyTorch
        out2_pre = F.conv2d(out1_silu, conv2_weight, bias=None, stride=1, padding=1)

        # Triton GroupNorm2
        B2, C2, H2, W2 = out2_pre.shape
        out2_g = out2_pre.contiguous().view(-1)
        gamma2 = norm2_weight.to(device=device, dtype=torch.float32)
        beta2 = norm2_bias.to(device=device, dtype=torch.float32)
        out2_gn = torch.empty_like(out2_g)
        grid_gn2 = (B2, self.num_groups)
        group_norm_two_pass[grid_gn2](
            out2_g,
            gamma2,
            beta2,
            out2_gn,
            out2_g.numel(),
            C2,
            H2,
            W2,
            self.num_groups,
            num_warps=4,
        )
        out2_gn = out2_gn.view(B2, C2, H2, W2)

        # SiLU2 via Triton
        N2 = out2_gn.numel()
        out2_silu = torch.empty_like(out2_gn)
        out2_silu_flat = out2_silu.view(-1)
        grid_silu2 = (triton.cdiv(N2, 4096),)
        silu_kernel[grid_silu2](out2_gn.view(-1), out2_silu_flat, N2, BLOCK=4096, num_warps=8)

        # Residual add via Triton
        out = torch.empty_like(out2_silu)
        Nfinal = out2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 4096),)
        add_residual_kernel[grid_add](out2_silu_flat, x.contiguous().view(-1), out.view(-1), Nfinal, BLOCK=4096, num_warps=8)

        return out


def run(*args):
    return ModelNew()(*args)
