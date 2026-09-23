import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_no_bias_nchw(x_ptr, w_ptr, y_ptr,
                          B, C, Cin, H, W,
                          sH, sW, p,
                          num_warps: tl.constexpr, num_stages: tl.constexpr):
    """
    3x3 convolution (no bias), stride=(sH, sW), padding=p, NCHW layout.
    Each program computes one output element y[n, co, h, w].
    """
    n = tl.program_id(0)
    co = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    if n >= B or co >= C or h >= H or w >= W:
        return

    acc = tl.zeros((), dtype=tl.float32)
    # Cin must be C when weights are (C, C, 3, 3)
    for ci in range(Cin):
        for dh in range(3):
            ih = h + dh - p
            valid_h = (ih >= 0) & (ih < H)
            for dw in range(3):
                iw = w + dw - p
                valid_w = (iw >= 0) & (iw < W)
                valid = valid_h & valid_w
                x_off = ((n * Cin + ci) * H + ih) * W + iw
                x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                # weight layout: w[co, ci, dh, dw]
                # since Cin=C, index simplifies
                w_off = (co * Cin * 9) + (dh * 3 + dw) * Cin + ci
                w_val = tl.load(w_ptr + w_off)
                acc += x_val * w_val

    y_off = ((n * C + co) * H + h) * W + w
    tl.store(y_ptr + y_off, acc)


@triton.jit
def group_norm_triton(y_ptr, y_norm_ptr, weight_ptr, bias_ptr,
                       B, C, H, W, num_groups, eps,
                       num_warps: tl.constexpr, num_stages: tl.constexpr):
    """
    GroupNorm with num_groups groups per sample, NCHW layout, per-channel scale and bias.
    We normalize across all elements of each (n, group), i.e., across all channels in that group and all H*W positions.
    Then apply per-channel affine (scale=weight_ptr[c], bias=bias_ptr[c]).
    Assumes C % num_groups == 0.
    """
    n = tl.program_id(0)
    group = tl.program_id(1)

    channels_per_group = C // num_groups
    group_start_c = group * channels_per_group
    M = H * W  # spatial elements per channel per sample
    total = channels_per_group * M  # total elements in this group for sample n

    # First pass: compute per-channel sum and sum of squares across the group
    sum_c = tl.zeros((channels_per_group,), dtype=tl.float32)
    sumsq_c = tl.zeros((channels_per_group,), dtype=tl.float32)

    for ch in range(channels_per_group):
        c = group_start_c + ch
        base = (n * C + c) * M
        for h in range(H):
            for w in range(W):
                idx = base + h * W + w
                x = tl.load(y_ptr + idx)
                sum_c[ch] += x
                sumsq_c[ch] += x * x

    # Mean and variance per channel across group
    mean = sum_c / float(M)
    var = sumsq_c / float(M) - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)  # per channel

    # Second pass: normalize and apply affine
    for ch in range(channels_per_group):
        c = group_start_c + ch
        base = (n * C + c) * M
        scale = tl.load(weight_ptr + c)
        bias = tl.load(bias_ptr + c)
        for h in range(H):
            for w in range(W):
                idx = base + h * W + w
                x = tl.load(y_ptr + idx)
                y = (x - mean[ch]) * rstd[ch]
                y = y * scale + bias
                tl.store(y_norm_ptr + idx, y)


@triton.jit
def silu_triton(x_ptr, y_ptr, N,
                num_warps: tl.constexpr, num_stages: tl.constexpr):
    """
    Elementwise SiLU: y = x * sigmoid(x), where sigmoid(x) = 1 / (1 + exp(-x))
    N is total number of elements; we use a 1D grid over N.
    """
    idx = tl.program_id(0)
    if idx >= N:
        return
    x = tl.load(x_ptr + idx)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + idx, y)


@triton.jit
def add_residual_triton(x_ptr, y_ptr, N,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    """
    Elementwise add residual: y += x
    N is total number of elements; we use a 1D grid over N.
    """
    idx = tl.program_id(0)
    if idx >= N:
        return
    a = tl.load(y_ptr + idx)
    b = tl.load(x_ptr + idx)
    c = a + b
    tl.store(y_ptr + idx, c)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        conv1_weight: torch.Tensor,
        norm1_weight: torch.Tensor,
        norm1_bias: torch.Tensor,
        conv2_weight: torch.Tensor,
        norm2_weight: torch.Tensor,
        norm2_bias: torch.Tensor,
        eps: float,
    ):
        """
        Fused residual block using Triton kernels:
        Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        All computation is performed by Triton kernels; no torch ops in forward.
        """
        # Ensure CUDA and dtype
        assert x.is_cuda and conv1_weight.is_cuda and conv2_weight.is_cuda, "Tensors must be on CUDA"
        assert x.dtype == torch.float32 and conv1_weight.dtype == torch.float32 and conv2_weight.dtype == torch.float32, "Use float32"
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        B, C, H, W = x.shape
        assert C % 32 == 0, "num_groups=32 requires C to be divisible by 32"

        # Intermediate outputs (allocated by forward and written by Triton)
        # Conv1 output
        y1 = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)
        # GroupNorm 1 output
        y1_norm = torch.empty_like(y1)
        # SiLU 1 output
        y1_silu = torch.empty_like(y1)

        # Conv2 output
        y2 = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)
        # GroupNorm 2 output
        y2_norm = torch.empty_like(y2)
        # SiLU 2 output (final before residual add)
        y2_silu = torch.empty_like(y2)

        # Total elements for elementwise kernels
        N = B * C * H * W

        # Launch conv1
        grid1 = (B, C, H, W)
        conv3x3_no_bias_nchw[grid1](
            x, conv1_weight, y1,
            B, C, C, H, W,
            sH=1, sW=1, p=1,
            num_warps=4, num_stages=2
        )

        # GroupNorm 1
        grid_gn1 = (B, 32)
        group_norm_triton[grid_gn1](
            y1, y1_norm, norm1_weight, norm1_bias,
            B, C, H, W, 32, eps,
            num_warps=4, num_stages=2
        )

        # SiLU 1
        grid_silu1 = (N,)
        silu_triton[grid_silu1](
            y1_norm, y1_silu,
            N,
            num_warps=4, num_stages=2
        )

        # Conv2
        grid2 = (B, C, H, W)
        conv3x3_no_bias_nchw[grid2](
            y1_silu, conv2_weight, y2,
            B, C, C, H, W,
            sH=1, sW=1, p=1,
            num_warps=4, num_stages=2
        )

        # GroupNorm 2
        grid_gn2 = (B, 32)
        group_norm_triton[grid_gn2](
            y2, y2_norm, norm2_weight, norm2_bias,
            B, C, H, W, 32, eps,
            num_warps=4, num_stages=2
        )

        # SiLU 2
        grid_silu2 = (N,)
        silu_triton[grid_silu2](
            y2_norm, y2_silu,
            N,
            num_warps=4, num_stages=2
        )

        # Add residual x
        grid_add = (N,)
        add_residual_triton[grid_add](
            x, y2_silu,
            N,
            num_warps=4, num_stages=2
        )

        return y2_silu


def run(*args):
    return ModelNew()(*args)
