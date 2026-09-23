import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 Conv (NCHW, stride=1, padding=1, no bias)
@triton.jit
def conv3x3_nchw_nobias(x_ptr, w_ptr, y_ptr,
                         B, C_in, C_out, H, W, H_out, W_out,
                         x_stride_n, x_stride_c, x_stride_h, x_stride_w,
                         w_stride_co, w_stride_ci, w_stride_dh, w_stride_dw,
                         y_stride_n, y_stride_c, y_stride_h, y_stride_w,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    # program ids: (batch, output_channel, output_h, output_w)
    pid_b = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # accumulate in fp32
    acc = 0.0

    # loop over input channels
    for ci in range(0, C_in):
        # loop over 3x3 neighborhood with padding
        for dh in range(0, 3):
            hi = pid_h + dh - 1
            for dw in range(0, 3):
                wi = pid_w + dw - 1
                in_bounds = (hi >= 0) & (hi < H_out) & (wi >= 0) & (wi < W_out)
                # compute x pointer for this location
                x_offset = pid_b * x_stride_n + ci * x_stride_c + hi * x_stride_h + wi * x_stride_w
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)

                # compute weight for this (co, ci, dh, dw)
                w_offset = pid_co * w_stride_co + ci * w_stride_ci + dh * w_stride_dh + dw * w_stride_dw
                w_val = tl.load(w_ptr + w_offset)

                # accumulate
                acc += x_val * w_val

    # store result
    y_offset = pid_b * y_stride_n + pid_co * y_stride_c + pid_h * y_stride_h + pid_w * y_stride_w
    tl.store(y_ptr + y_offset, acc)


# Triton kernel: GroupNorm per channel (num_groups=32), per-channel stats over spatial, affine apply
# grid: (B, C)
@triton.jit
def groupnorm_triton_per_channel(y_in_ptr, weight_ptr, bias_ptr, y_out_ptr,
                                  B, C, H, W,
                                  y_in_stride_n, y_in_stride_c, y_in_stride_h, y_in_stride_w,
                                  y_out_stride_n, y_out_stride_c, y_out_stride_h, y_out_stride_w,
                                  num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    N = H * W  # number of spatial elements per channel
    # accumulate sum and sum of squares over spatial plane
    sum_val = 0.0
    sum_sq = 0.0
    for h in range(0, H):
        for w in range(0, W):
            ptr = pid_n * y_in_stride_n + pid_c * y_in_stride_c + h * y_in_stride_h + w * y_in_stride_w
            x = tl.load(y_in_ptr + ptr)
            sum_val += x
            sum_sq += x * x

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    # apply normalization and affine
    for h in range(0, H):
        for w in range(0, W):
            ptr_in = pid_n * y_in_stride_n + pid_c * y_in_stride_c + h * y_in_stride_h + w * y_in_stride_w
            x = tl.load(y_in_ptr + ptr_in)
            norm = (x - mean) * rstd
            gamma = tl.load(weight_ptr + pid_c)
            beta = tl.load(bias_ptr + pid_c)
            y = norm * gamma + beta
            ptr_out = pid_n * y_out_stride_n + pid_c * y_out_stride_c + h * y_out_stride_h + w * y_out_stride_w
            tl.store(y_out_ptr + ptr_out, y)


# Triton elementwise SiLU: y = x * sigmoid(x)
@triton.jit
def silu_triton(x_ptr, y_ptr, N,
                 num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * num_warps + tl.arange(0, num_warps)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)


# Triton elementwise residual add: y = y + x
@triton.jit
def add_residual_triton(y_ptr, x_ptr, out_ptr, N,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * num_warps + tl.arange(0, num_warps)
    mask = offs < N
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, y + x, mask=mask)


def _launch_conv_nchw(x, w, out_shape):
    B, C_in, H, W = x.shape
    C_out = w.shape[0]
    H_out, W_out = out_shape
    y = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = (B, C_out, H_out, W_out)
    # Launch conv Triton kernel
    conv3x3_nchw_nobias[grid](
        x, w, y,
        B, C_in, C_out, H, W, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        num_warps=1, num_stages=2
    )
    return y


def _launch_groupnorm_per_channel(y_in, weight, bias, y_out):
    B, C, H, W = y_in.shape
    y_in = y_in.contiguous()
    y_out = y_out.contiguous()
    grid = (B, C)
    groupnorm_triton_per_channel[grid](
        y_in, weight, bias, y_out,
        B, C, H, W,
        y_in.stride(0), y_in.stride(1), y_in.stride(2), y_in.stride(3),
        y_out.stride(0), y_out.stride(1), y_out.stride(2), y_out.stride(3),
        num_warps=1, num_stages=2
    )
    return y_out


def _launch_silu_triton(x):
    B, C, H, W = x.shape
    x_flat = x.reshape(-1)
    N = x_flat.numel()
    y_flat = torch.empty_like(x_flat, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(N, 1024),)
    silu_triton[grid](x_flat, y_flat, N, num_warps=4, num_stages=2)
    return y_flat.reshape(B, C, H, W)


def _launch_add_residual(y, x):
    B, C, H, W = y.shape
    y_flat = y.reshape(-1)
    x_flat = x.reshape(-1)
    N = y_flat.numel()
    out_flat = torch.empty_like(y_flat, device=y.device, dtype=y.dtype)
    grid = (triton.cdiv(N, 1024),)
    add_residual_triton[grid](y_flat, x_flat, out_flat, N, num_warps=4, num_stages=2)
    return out_flat.reshape(B, C, H, W)


@torch.no_grad()
def run_triton(
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
    Fused residual block: Conv3x3 -> GroupNorm (num_groups=32, per-channel stats) -> SiLU
    -> Conv3x3 -> GroupNorm -> SiLU -> Add (with input residual)
    """
    assert x.is_cuda and x.dtype == torch.float32
    B, C, H, W = x.shape
    # Ensure inputs are contiguous
    x = x.contiguous()
    conv1_weight = conv1_weight.contiguous()
    conv2_weight = conv2_weight.contiguous()
    norm1_weight = norm1_weight.contiguous()
    norm1_bias = norm1_bias.contiguous()
    norm2_weight = norm2_weight.contiguous()
    norm2_bias = norm2_bias.contiguous()

    # First conv: output spatial size (H-2, W-2)
    y1 = _launch_conv_nchw(x, conv1_weight, (H - 2, W - 2))

    # GroupNorm1: per-channel stats over spatial, num_groups=32 (per-channel normalization)
    y1_gn = _launch_groupnorm_per_channel(y1, norm1_weight, norm1_bias, torch.empty_like(y1))

    # SiLU1
    y1_silu = _launch_silu_triton(y1_gn)

    # Second conv: output spatial size (H-4, W-4)
    y2 = _launch_conv_nchw(y1_silu, conv2_weight, (H - 4, W - 4))

    # GroupNorm2
    y2_gn = _launch_groupnorm_per_channel(y2, norm2_weight, norm2_bias, torch.empty_like(y2))

    # SiLU2
    y2_silu = _launch_silu_triton(y2_gn)

    # Residual add: y = y2_silu + x
    y_out = _launch_add_residual(y2_silu, x)

    return y_out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # ModelNew.forward must launch Triton kernels; no torch ops here.
        if len(args) != 7:
            raise RuntimeError("ModelNew.forward expects 7 arguments: x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps")
        x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps = args
        return run_triton(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)


def run(*args):
    return ModelNew()(*args)
