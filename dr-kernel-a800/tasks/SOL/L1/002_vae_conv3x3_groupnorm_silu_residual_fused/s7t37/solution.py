import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_single_elem(
    x_ptr,         # *float32, input tensor (B, C_in, H, W)
    w_ptr,         # *float32, weight tensor (C_out, C_in, 3, 3)
    y_ptr,         # *float32, output tensor (B, C_out, H, W)
    N, C_in, H, W, C_out,
    BLOCK_SCALAR: tl.constexpr,
):
    # Grid: (N * C_out * H * W,)
    linear = tl.program_id(0)
    c_out = linear % C_out
    tmp = linear // C_out
    h = tmp % H
    w = tmp // H
    n = tmp // (H * W)

    acc = 0.0

    # For each input channel and 3x3 tap
    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                ih = h + kh - 1
                iw = w + kw - 1
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                in_index = (((n * C_in + cin) * H + ih) * W + iw)
                x_val = tl.load(x_ptr + in_index, mask=in_bounds, other=0.0)
                w_index = (((c_out * C_in + cin) * 9) + (kh * 3 + kw))
                w_val = tl.load(w_ptr + w_index)
                acc += x_val * w_val

    out_index = (((n * C_out + c_out) * H + h) * W + w)
    tl.store(y_ptr + out_index, acc)


@triton.jit
def group_norm_affine_kernel(
    x_ptr,          # *float32, input tensor (B, C, H, W)
    gamma_ptr,      # *float32, per-channel scale (C,)
    beta_ptr,       # *float32, per-channel bias (C,)
    y_ptr,          # *float32, output tensor (B, C, H, W)
    N, C, H, W,
    num_groups: tl.constexpr,  # e.g., 32
    eps: tl.constexpr,         # e.g., 1e-5
):
    # Grid: (N, num_groups)
    n = tl.program_id(0)
    g = tl.program_id(1)

    channels_per_group = C // num_groups
    group_start = g * channels_per_group

    # First pass: compute sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    for c_off in range(channels_per_group):
        c = group_start + c_off
        for h in range(H):
            for w in range(W):
                x_index = (((n * C + c) * H + h) * W + w)
                x_val = tl.load(x_ptr + x_index)
                sum_val += x_val
                sum_sq += x_val * x_val

    M = float(channels_per_group * H * W)
    mean = sum_val / M
    var = sum_sq / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for c_off in range(channels_per_group):
        c = group_start + c_off
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)
        for h in range(H):
            for w in range(W):
                x_index = (((n * C + c) * H + h) * W + w)
                x_val = tl.load(x_ptr + x_index)
                y_val = (x_val - mean) * inv_std
                y_val = y_val * gamma + beta
                y_index = (((n * C + c) * H + h) * W + w)
                tl.store(y_ptr + y_index, y_val)


@triton.jit
def silu_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
):
    # Grid: (N, C, H, W)
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    x_index = (((n * C + c) * H + h) * W + w)
    x_val = tl.load(x_ptr + x_index)
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig
    y_index = (((n * C + c) * H + h) * W + w)
    tl.store(y_ptr + y_index, y_val)


@triton.jit
def add_residual_kernel(
    y_ptr, x_ptr, out_ptr,
    N, C, H, W,
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
    out_index = (((n * C + c) * H + h) * W + w)
    tl.store(out_ptr + out_index, out_val)


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
        # Enforce GroupNorm constraint
        B, C, H, W = x.shape
        assert C % 32 == 0, "GroupNorm requires channels divisible by num_groups=32"
        device = x.device
        dtype = torch.float32

        # Ensure inputs are contiguous and float32
        x1 = x.contiguous().to(dtype)
        conv1_w = conv1_weight.contiguous().to(dtype)
        conv2_w = conv2_weight.contiguous().to(dtype)
        norm1_w = norm1_weight.contiguous().to(dtype)
        norm1_b = norm1_bias.contiguous().to(dtype)
        norm2_w = norm2_weight.contiguous().to(dtype)
        norm2_b = norm2_bias.contiguous().to(dtype)

        C_in = x1.shape[1]
        C_out = conv1_w.shape[0]

        # 1) First conv: Triton kernel
        y1 = torch.empty((B, C_out, H, W), dtype=dtype, device=device)
        grid = (B * C_out * H * W,)
        conv3x3_stride1_pad1_single_elem[grid](
            x1, conv1_w, y1,
            B, C_in, H, W, C_out,
            BLOCK_SCALAR=1,
            num_warps=1, num_stages=1,
        )

        # 2) GroupNorm1: Triton kernel
        y1_gn = torch.empty_like(y1)
        group_norm_affine_kernel[(B, 32)](
            y1, norm1_w, norm1_b, y1_gn, B, C_out, H, W,
            num_groups=32, eps=1e-5,
            num_warps=4, num_stages=2,
        )

        # 3) SiLU1: Triton kernel
        y1_silu = torch.empty_like(y1_gn)
        silu_kernel[(B, C_out, H, W)](
            y1_gn, y1_silu,
            B, C_out, H, W,
            num_warps=4, num_stages=2,
        )

        # Save residual (original x, cast to float32)
        residual = x1

        # 4) Second conv: Triton kernel
        y2 = torch.empty((B, C_out, H, W), dtype=dtype, device=device)
        conv3x3_stride1_pad1_single_elem[grid](
            y1_silu, conv2_w, y2,
            B, C_in, H, W, C_out,
            BLOCK_SCALAR=1,
            num_warps=1, num_stages=1,
        )

        # 5) GroupNorm2: Triton kernel
        y2_gn = torch.empty_like(y2)
        group_norm_affine_kernel[(B, 32)](
            y2, norm2_w, norm2_b, y2_gn, B, C_out, H, W,
            num_groups=32, eps=1e-5,
            num_warps=4, num_stages=2,
        )

        # 6) SiLU2: Triton kernel
        y2_silu = torch.empty_like(y2_gn)
        silu_kernel[(B, C_out, H, W)](
            y2_gn, y2_silu,
            B, C_out, H, W,
            num_warps=4, num_stages=2,
        )

        # 7) Add residual: Triton kernel
        y_out = torch.empty_like(y2_silu)
        add_residual_kernel[(B, C_out, H, W)](
            y2_silu, residual, y_out,
            B, C_out, H, W,
            num_warps=4, num_stages=2,
        )

        return y_out


def run(*args):
    return ModelNew()(*args)
