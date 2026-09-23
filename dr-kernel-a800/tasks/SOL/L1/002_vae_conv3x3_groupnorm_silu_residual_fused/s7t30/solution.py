import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_kernel(
    x_ptr,           # *float or *half input tensor (B, C_in, H, W)
    w_ptr,           # *float or *half weight tensor (C_out, C_in, 3, 3)
    y_ptr,           # *float output tensor (B, C_out, H, W)
    N,               # int
    C_in,            # int
    H,               # int
    W,               # int
    C_out,           # int
    BLOCK_OC: tl.constexpr,  # tile size for output channels per program
):
    # Grid: (N, ceil_div(C_out, BLOCK_OC))
    n = tl.program_id(0)
    oc_block_id = tl.program_id(1)
    oc_start = oc_block_id * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_out

    # Accumulator for the tile of output channels
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel taps
    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                # Output spatial dimensions match input due to padding=1, stride=1
                for h in range(H):
                    ih = h + kh - 1
                    valid_h = (ih >= 0) & (ih < H)
                    for wj in range(W):
                        iw = wj + kw - 1
                        valid = valid_h & (iw >= 0) & (iw < W)
                        # Input linear index: (((n * C_in + cin) * H + ih) * W + iw)
                        in_index = (((n * C_in + cin) * H + ih) * W + iw)
                        x_val = tl.load(x_ptr + in_index, mask=valid, other=0.0).to(tl.float32)
                        # Weight linear index: (((oc * C_in + cin) * 9) + (kh * 3 + kw))
                        for j in range(BLOCK_OC):
                            if oc_mask[oc_offsets[j]]:
                                w_index = (((oc_offsets[j] * C_in + cin) * 9) + (kh * 3 + kw))
                                w_val = tl.load(w_ptr + w_index).to(tl.float32)
                                acc[j] += x_val * w_val

    # Store results for all spatial positions (h, w)
    for j in range(BLOCK_OC):
        if oc_mask[oc_offsets[j]]:
            for h in range(H):
                for wj in range(W):
                    out_index = (((n * C_out + oc_offsets[j]) * H + h) * W + wj)
                    tl.store(y_ptr + out_index, acc[j])


@triton.jit
def group_norm_affine_kernel(
    x_ptr,           # *float input tensor (B, C, H, W)
    scale_ptr,       # *float per-channel scale (C,)
    bias_ptr,        # *float per-channel bias (C,)
    y_ptr,           # *float output tensor (B, C, H, W)
    N,               # int
    C,               # int
    H,               # int
    W,               # int
    num_groups: tl.constexpr,  # e.g., 32
    eps,             # float epsilon
):
    # Grid: (N, num_groups)
    n = tl.program_id(0)
    g = tl.program_id(1)
    channels_per_group = C // num_groups
    group_start = g * channels_per_group

    # Compute sum and sum of squares over group
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for ch in range(channels_per_group):
        c = group_start + ch
        for h in range(H):
            for w in range(W):
                x_index = (((n * C + c) * H + h) * W + w)
                x_val = tl.load(x_ptr + x_index).to(tl.float32)
                sum_val += x_val
                sum_sq += x_val * x_val

    M = channels_per_group * H * W
    mean = sum_val / M
    var = sum_sq / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for ch in range(channels_per_group):
        c = group_start + ch
        scale = tl.load(scale_ptr + c).to(tl.float32)
        beta = tl.load(bias_ptr + c).to(tl.float32)
        for h in range(H):
            for w in range(W):
                x_index = (((n * C + c) * H + h) * W + w)
                x_val = tl.load(x_ptr + x_index).to(tl.float32)
                y_val = (x_val - mean) * inv_std * scale + beta
                tl.store(y_ptr + x_index, y_val)


@triton.jit
def silu_kernel(x_ptr, y_ptr, N, C, H, W):
    # Elementwise: y = x * sigmoid(x)
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    index = (((n * C + c) * H + h) * W + w)
                    x_val = tl.load(x_ptr + index).to(tl.float32)
                    sig = 1.0 / (1.0 + tl.exp(-x_val))
                    y_val = x_val * sig
                    tl.store(y_ptr + index, y_val)


@triton.jit
def add_residual_kernel(x_ptr, y_ptr, out_ptr, N, C, H, W):
    # out = y + x
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    index = (((n * C + c) * H + h) * W + w)
                    x_val = tl.load(x_ptr + index).to(tl.float32)
                    y_val = tl.load(y_ptr + index).to(tl.float32)
                    out_val = y_val + x_val
                    tl.store(out_ptr + index, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        """
        Triton-only fused residual block:
        Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        Assumes num_groups = 32. Enforce that C % 32 == 0.
        """

        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Weights must be on CUDA device."
        assert norm1_weight.is_cuda and norm1_bias.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "GroupNorm params must be on CUDA device."

        B, C, H, W = x.shape
        # Ensure contiguous
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()

        # 1) conv1: (B, C, H, W) -> (B, C, H, W)
        y1 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        BLOCK_OC = 32  # tile over output channels
        grid_conv1 = (B, triton.cdiv(C, BLOCK_OC))
        conv3x3_stride1_pad1_kernel[grid_conv1](
            x, conv1_weight, y1, B, C, H, W, C, BLOCK_OC=BLOCK_OC, num_warps=4, num_stages=2
        )

        # 2) GroupNorm1 (C % 32 must be 0)
        assert C % 32 == 0, "GroupNorm requires channels divisible by num_groups=32."
        y1_gn = torch.empty_like(y1, dtype=torch.float32)
        grid_gn1 = (B, 32)
        group_norm_affine_kernel[grid_gn1](
            y1, norm1_weight, norm1_bias, y1_gn, B, C, H, W,
            num_groups=32, eps=eps, num_warps=4, num_stages=2
        )

        # 3) SiLU1
        y1_silu = torch.empty_like(y1_gn, dtype=torch.float32)
        grid_silu1 = (B, C, H, W)
        silu_kernel[grid_silu1](y1_gn, y1_silu, B, C, H, W, num_warps=1, num_stages=1)

        # 4) conv2: (B, C, H, W) -> (B, C, H, W)
        y2 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid_conv2 = (B, triton.cdiv(C, BLOCK_OC))
        conv3x3_stride1_pad1_kernel[grid_conv2](
            y1_silu, conv2_weight, y2, B, C, H, W, C, BLOCK_OC=BLOCK_OC, num_warps=4, num_stages=2
        )

        # 5) GroupNorm2
        assert C % 32 == 0, "GroupNorm requires channels divisible by num_groups=32."
        y2_gn = torch.empty_like(y2, dtype=torch.float32)
        grid_gn2 = (B, 32)
        group_norm_affine_kernel[grid_gn2](
            y2, norm2_weight, norm2_bias, y2_gn, B, C, H, W,
            num_groups=32, eps=eps, num_warps=4, num_stages=2
        )

        # 6) SiLU2
        y2_silu = torch.empty_like(y2_gn, dtype=torch.float32)
        grid_silu2 = (B, C, H, W)
        silu_kernel[grid_silu2](y2_gn, y2_silu, B, C, H, W, num_warps=1, num_stages=1)

        # 7) Add residual x
        out = torch.empty_like(y2_silu, dtype=torch.float32)
        grid_add = (B, C, H, W)
        add_residual_kernel[grid_add](y2_silu, x, out, B, C, H, W, num_warps=1, num_stages=1)

        return out


def run(*args):
    return ModelNew()(*args)
