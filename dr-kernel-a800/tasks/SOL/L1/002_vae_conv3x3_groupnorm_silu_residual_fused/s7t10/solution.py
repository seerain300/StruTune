import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_full_channel_kernel(
    x_ptr,          # *float32 input tensor (B, C_in, H, W)
    w_ptr,          # *float32 weights tensor (C_out, C_in, 3, 3)
    y_ptr,          # *float32 output tensor (B, C_out, H, W)
    N,              # int32
    C_in,           # int32
    H,              # int32
    W,              # int32
    C_out,          # int32
):
    # program ids: one program per (n, c_out)
    n = tl.program_id(0)
    c_out = tl.program_id(1)

    # Accumulator for the full output tensor for this (n, c_out)
    # We'll fill y[n, c_out, :, :] by iterating oh, ow and computing acc for each position.
    for oh in range(H):
        for ow in range(W):
            acc = 0.0
            # loop over input channels and 3x3 taps
            for cin in range(C_in):
                for kh in range(3):
                    for kw in range(3):
                        ih = oh + kh - 1
                        iw = ow + kw - 1
                        valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                        # Load input scalar
                        x_index = (((n * C_in + cin) * H + ih) * W + iw)
                        x_val = tl.load(x_ptr + x_index, mask=valid, other=0.0)
                        # Load weight scalar (weight layout: w[c_out, cin, kh, kw])
                        w_index = (((c_out * C_in + cin) * 9) + (kh * 3 + kw))
                        w_val = tl.load(w_ptr + w_index)
                        acc += x_val * w_val
            # Store result to y[n, c_out, oh, ow]
            y_index = (((n * C_out + c_out) * H + oh) * W + ow)
            tl.store(y_ptr + y_index, acc)


@triton.jit
def group_norm_affine_kernel(
    x_ptr,          # *float32 input tensor (B, C, H, W)
    gamma_ptr,      # *float32 per-channel scale (C,)
    beta_ptr,       # *float32 per-channel bias (C,)
    y_ptr,          # *float32 output tensor (B, C, H, W)
    N,              # int32
    C,              # int32
    H,              # int32
    W,              # int32
    num_groups: tl.constexpr,  # 32
    eps,            # float32
):
    # Grid: (N, num_groups)
    n = tl.program_id(0)
    g = tl.program_id(1)
    channels_per_group = C // num_groups
    group_start = g * channels_per_group

    # First pass: compute sum and sum of squares over the group
    sum_ = 0.0
    sumsq_ = 0.0
    for ch in range(channels_per_group):
        c = group_start + ch
        for h in range(H):
            for w in range(W):
                index = (((n * C + c) * H + h) * W + w)
                val = tl.load(x_ptr + index)
                sum_ += val
                sumsq_ += val * val

    numel = channels_per_group * H * W
    mean = sum_ / numel
    var = sumsq_ / numel - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for ch in range(channels_per_group):
        c = group_start + ch
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)
        for h in range(H):
            for w in range(W):
                index_in = (((n * C + c) * H + h) * W + w)
                x = tl.load(x_ptr + index_in)
                norm = (x - mean) * inv_std
                y = norm * gamma + beta
                index_out = (((n * C + c) * H + h) * W + w)
                tl.store(y_ptr + index_out, y)


@triton.jit
def silu_kernel_4d(
    x_ptr,          # *float32 input tensor (B, C, H, W)
    y_ptr,          # *float32 output tensor (B, C, H, W)
    N,              # int32
    C,              # int32
    H,              # int32
    W,              # int32
):
    # Grid: (N, C, H, W)
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    x_index = (((n * C + c) * H + h) * W + w)
    x_val = tl.load(x_ptr + x_index)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig
    y_index = (((n * C + c) * H + h) * W + w)
    tl.store(y_ptr + y_index, y_val)


@triton.jit
def add_residual_kernel_4d(
    y_ptr,          # *float32 y tensor (B, C, H, W)
    x_ptr,          # *float32 x tensor (residual) (B, C, H, W)
    N,              # int32
    C,              # int32
    H,              # int32
    W,              # int32
):
    # Grid: (N, C, H, W)
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    y_index = (((n * C + c) * H + h) * W + w)
    x_index = (((n * C + c) * H + h) * W + w)
    y_val = tl.load(y_ptr + y_index)
    x_val = tl.load(x_ptr + x_index)
    out_val = y_val + x_val
    tl.store(y_ptr + y_index, out_val)


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
        Triton-optimized fused residual block:
        Conv3x3 -> GroupNorm(num_groups=32) -> SiLU -> Conv3x3 -> GroupNorm(num_groups=32) -> SiLU -> Add(residual)

        All heavy lifting is done by Triton kernels. No torch ops in forward.
        """
        # Ensure CUDA and float32 for stability
        assert x.is_cuda, "Inputs must be on CUDA device for Triton kernels."
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Conv weights must be on CUDA."
        assert norm1_weight.is_cuda and norm1_bias.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "GroupNorm params must be on CUDA."

        # Make inputs contiguous
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()

        B, C, H, W = x.shape
        C1 = conv1_weight.shape[0]
        C2 = conv2_weight.shape[0]
        assert C1 == C and C2 == C, "Output channels of convs must match input channels (C)."

        # 1) First Conv (3x3, stride=1, pad=1, bias=None) — Triton
        out1 = torch.empty((B, C1, H, W), dtype=torch.float32, device=x.device)
        grid_conv1 = (B, C1)
        conv3x3_stride1_pad1_full_channel_kernel[grid_conv1](
            x.to(torch.float32), conv1_weight.to(torch.float32), out1,
            B, C, H, W, C1,
            num_warps=4,
            num_stages=2,
        )

        # 2) GroupNorm1 (num_groups=32) — Triton
        # Check GroupNorm requirement
        assert C % 32 == 0, "GroupNorm requires C % num_groups == 0. Got C={} and num_groups=32.".format(C)
        out1_norm = torch.empty_like(out1, dtype=torch.float32, device=out1.device)
        grid_gn1 = (B, 32)
        group_norm_affine_kernel[grid_gn1](
            out1, norm1_weight.to(torch.float32), norm1_bias.to(torch.float32), out1_norm,
            B, C, H, W,
            num_groups=32,
            eps=eps,
            num_warps=4,
            num_stages=2,
        )

        # 3) SiLU1 — Triton (elementwise, 4D grid)
        out1_silu = torch.empty_like(out1_norm, dtype=torch.float32, device=out1_norm.device)
        grid_silu1 = (B, C, H, W)
        silu_kernel_4d[grid_silu1](
            out1_norm, out1_silu,
            B, C, H, W,
            num_warps=1,
            num_stages=1,
        )

        # 4) First Residual Save: residual = x
        residual = x

        # 5) Second Conv (3x3, stride=1, pad=1, bias=None) — Triton
        out2 = torch.empty((B, C2, H, W), dtype=torch.float32, device=x.device)
        grid_conv2 = (B, C2)
        conv3x3_stride1_pad1_full_channel_kernel[grid_conv2](
            out1_silu, conv2_weight.to(torch.float32), out2,
            B, C1, H, W, C2,
            num_warps=4,
            num_stages=2,
        )

        # 6) GroupNorm2 (num_groups=32) — Triton
        out2_norm = torch.empty_like(out2, dtype=torch.float32, device=out2.device)
        grid_gn2 = (B, 32)
        group_norm_affine_kernel[grid_gn2](
            out2, norm2_weight.to(torch.float32), norm2_bias.to(torch.float32), out2_norm,
            B, C, H, W,
            num_groups=32,
            eps=eps,
            num_warps=4,
            num_stages=2,
        )

        # 7) SiLU2 — Triton (elementwise, 4D grid)
        out2_silu = torch.empty_like(out2_norm, dtype=torch.float32, device=out2_norm.device)
        grid_silu2 = (B, C, H, W)
        silu_kernel_4d[grid_silu2](
            out2_norm, out2_silu,
            B, C, H, W,
            num_warps=1,
            num_stages=1,
        )

        # 8


def run(*args):
    return ModelNew()(*args)
