import torch
import triton
import triton.language as tl

# -------------------------------
# Triton kernels
# -------------------------------

@triton.jit
def group_norm_two_pass(
    in_ptr,       # *const float, input tensor [B, C, H, W] (after conv)
    gamma_ptr,    # *const float, per-channel scale [C]
    beta_ptr,     # *const float, per-channel bias [C]
    out_ptr,      # *float, output tensor [B, C, H, W]
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    eps: tl.constexpr,
    in_stride_n: tl.constexpr, in_stride_c: tl.constexpr, in_stride_h: tl.constexpr, in_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr, out_stride_c: tl.constexpr, out_stride_h: tl.constexpr, out_stride_w: tl.constexpr,
):
    # one program per (n, group)
    n = tl.program_id(0)  # batch index
    g = tl.program_id(1)  # group index
    # channels per group
    CG = C // 32  # num_groups is fixed to 32 in original code
    c_start = g * CG
    c_end = c_start + CG

    # compute sum and sum of squares over group channels for all H*W
    total_sum = 0.0
    total_sq = 0.0
    for ci in tl.static_range(C):
        if ci < c_start or ci >= c_end:
            continue
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                in_base = n * in_stride_n + ci * in_stride_c + h * in_stride_h + w * in_stride_w
                x = tl.load(in_ptr + in_base).to(tl.float32)
                total_sum += x
                total_sq += x * x

    M = CG * H * W
    mean = total_sum / M
    var = total_sq / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # second pass: normalize and apply affine
    for ci in tl.static_range(C):
        if ci < c_start or ci >= c_end:
            continue
        gamma = tl.load(gamma_ptr + ci).to(tl.float32)
        beta = tl.load(beta_ptr + ci).to(tl.float32)
        for h in tl.static_range(H):
            for w in tl.static_range(W):
                in_base = n * in_stride_n + ci * in_stride_c + h * in_stride_h + w * in_stride_w
                x = tl.load(in_ptr + in_base).to(tl.float32)
                y = (x - mean) * inv_std
                y = y * gamma + beta
                out_base = n * out_stride_n + ci * out_stride_c + h * out_stride_h + w * out_stride_w
                tl.store(out_ptr + out_base, y)


@triton.jit
def silu_kernel(in_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    # simple elementwise SiLU over flat buffer
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    x32 = x.to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-x32))
    y32 = x32 * sig
    tl.store(out_ptr + offs, y32, mask=mask)


@triton.jit
def add_residual_kernel(in_ptr, res_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(in_ptr + offs, mask=mask, other=0.0)
    b = tl.load(res_ptr + offs, mask=mask, other=0.0)
    c = a + b
    tl.store(out_ptr + offs, c, mask=mask)


# -------------------------------
# ModelNew forward (Triton for GroupNorm and SiLU, PyTorch for convs)
# -------------------------------
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
        Forward path:
        - Two convs via PyTorch (conv2d, stride=1, padding=1, bias=None)
        - Triton GroupNorm twice (num_groups=32)
        - Triton SiLU twice
        - Triton residual addition
        """
        device = x.device
        dtype = x.dtype

        # Ensure float32 and contiguous for stable GroupNorm/activation
        x_f = x.to(device=device, dtype=torch.float32).contiguous()

        # First conv: y1 = conv3x3(x)
        y1 = torch.nn.functional.conv2d(
            x_f, conv1_weight.to(device=device, dtype=torch.float32).contiguous(),
            bias=None, stride=1, padding=1
        )

        # GroupNorm1 (num_groups=32), then SiLU1
        y1_gn = torch.empty_like(y1)

        grid_gn1 = (y1.shape[0], self.num_groups)
        group_norm_two_pass[grid_gn1](
            y1, norm1_weight.to(device=device, dtype=torch.float32).contiguous(),
            norm1_bias.to(device=device, dtype=torch.float32).contiguous(),
            y1_gn,
            y1.shape[0], y1.shape[1], y1.shape[2], y1.shape[3], self.eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_gn.stride(0), y1_gn.stride(1), y1_gn.stride(2), y1_gn.stride(3),
        )

        # SiLU1
        y1_silu = torch.empty_like(y1_gn)
        N1 = y1_gn.numel()
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_kernel[grid_silu1](y1_gn, y1_silu, N1, BLOCK=1024)

        # Second conv: y2_pre = conv3x3(y1_silu)
        y2_pre = torch.nn.functional.conv2d(
            y1_silu, conv2_weight.to(device=device, dtype=torch.float32).contiguous(),
            bias=None, stride=1, padding=1
        )

        # GroupNorm2 (num_groups=32), then SiLU2
        y2_gn = torch.empty_like(y2_pre)

        grid_gn2 = (y2_pre.shape[0], self.num_groups)
        group_norm_two_pass[grid_gn2](
            y2_pre, norm2_weight.to(device=device, dtype=torch.float32).contiguous(),
            norm2_bias.to(device=device, dtype=torch.float32).contiguous(),
            y2_gn,
            y2_pre.shape[0], y2_pre.shape[1], y2_pre.shape[2], y2_pre.shape[3], self.eps,
            y2_pre.stride(0), y2_pre.stride(1), y2_pre.stride(2), y2_pre.stride(3),
            y2_gn.stride(0), y2_gn.stride(1), y2_gn.stride(2), y2_gn.stride(3),
        )

        # SiLU2
        y2_silu = torch.empty_like(y2_gn)
        N2 = y2_gn.numel()
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_kernel[grid_silu2](y2_gn, y2_silu, N2, BLOCK=1024)

        # Add residual x
        out = torch.empty_like(y2_silu)
        Nfinal = y2_silu.numel()
        grid_add = (triton.cdiv(Nfinal, 1024),)
        add_residual_kernel[grid_add](y2_silu, x_f, out, Nfinal, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
