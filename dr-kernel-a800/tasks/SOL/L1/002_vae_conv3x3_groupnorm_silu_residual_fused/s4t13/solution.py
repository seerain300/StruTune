import torch
import triton
import triton.language as tl

# Triton kernel: 3x3 Convolution (NCHW, stride=1, padding=1, no bias)
@triton.jit
def conv3x3_nchw_nobias(x_ptr, w_ptr, y_ptr,
                         B, C_in, C_out, H, W, H_out, W_out,
                         x_stride_n, x_stride_c, x_stride_h, x_stride_w,
                         w_stride_co, w_stride_ci, w_stride_dh, w_stride_dw,
                         y_stride_n, y_stride_c, y_stride_h, y_stride_w,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid_n = tl.program_id(0)  # batch
    pid_co = tl.program_id(1)  # output channel
    pid_h = tl.program_id(2)  # output height index
    pid_w = tl.program_id(3)  # output width index

    # Compute output base pointer
    y_offset = pid_n * y_stride_n + pid_co * y_stride_c + pid_h * y_stride_h + pid_w * y_stride_w

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 neighborhood
    for ci in range(0, C_in):
        for dh in range(0, 3):
            for dw in range(0, 3):
                hi = pid_h + dh - 1
                wi = pid_w + dw - 1
                # In-bounds mask for padding
                in_bounds = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W)
                x_offset = pid_n * x_stride_n + ci * x_stride_c + hi * x_stride_h + wi * x_stride_w
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
                w_offset = pid_co * w_stride_co + ci * w_stride_ci + dh * w_stride_dh + dw * w_stride_dw
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    tl.store(y_ptr + y_offset, acc)

# Triton kernel: GroupNorm for fixed num_groups=32, per-channel scale/bias
@triton.jit
def group_norm_triton_fixed(y_ptr, weight_ptr, bias_ptr, eps,
                             B, C, H_out, W_out,
                             y_stride_n, y_stride_c, y_stride_h, y_stride_w,
                             num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid_n = tl.program_id(0)  # batch
    pid_g = tl.program_id(1)  # group id in 0..31

    # channels per group
    CH_PER_GROUP = C // 32
    group_ch_start = pid_g * CH_PER_GROUP

    # First pass: compute sum and sumsq over the group for each channel
    sum_total = tl.zeros((), dtype=tl.float32)
    sumsq_total = tl.zeros((), dtype=tl.float32)

    for gc in range(0, CH_PER_GROUP):
        c = group_ch_start + gc
        H_elems = H_out * W_out
        for h in range(0, H_out):
            for w in range(0, W_out):
                y_offset = pid_n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
                val = tl.load(y_ptr + y_offset)
                sum_total += val
                sumsq_total += val * val

    mean = sum_total / (CH_PER_GROUP * H_elems)
    var = sumsq_total / (CH_PER_GROUP * H_elems) - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply per-channel scale/bias, then store back
    for gc in range(0, CH_PER_GROUP):
        c = group_ch_start + gc
        for h in range(0, H_out):
            for w in range(0, W_out):
                y_offset = pid_n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
                val = tl.load(y_ptr + y_offset)
                w_val = tl.load(weight_ptr + c)
                b_val = tl.load(bias_ptr + c)
                norm = (val - mean) * rstd
                new_val = norm * w_val + b_val
                tl.store(y_ptr + y_offset, new_val)

# Triton kernel: SiLU activation elementwise
@triton.jit
def silu_triton(x_ptr, y_ptr, N,
                num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + idx, y, mask=mask)

# Triton kernel: elementwise residual add y = y + x
@triton.jit
def add_residual_triton(y_ptr, x_ptr, N,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    y = tl.load(y_ptr + idx, mask=mask, other=0.0)
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    y = y + x
    tl.store(y_ptr + idx, y, mask=mask)

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
    Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
    Implements all ops in Triton; forward does not use torch ops.
    """
    assert x.is_cuda, "Input tensor must be on CUDA device for Triton kernels."
    # Ensure contiguous tensors
    x = x.contiguous()
    conv1_weight = conv1_weight.contiguous()
    norm1_weight = norm1_weight.contiguous()
    norm1_bias = norm1_bias.contiguous()
    conv2_weight = conv2_weight.contiguous()
    norm2_weight = norm2_weight.contiguous()
    norm2_bias = norm2_bias.contiguous()

    B, C, H, W = x.shape
    H1 = H - 2
    W1 = W - 2
    H2 = H1 - 2
    W2 = W1 - 2

    # Allocate intermediate and final outputs
    y1 = torch.empty((B, C, H1, W1), device=x.device, dtype=x.dtype)
    y1_norm = torch.empty((B, C, H1, W1), device=x.device, dtype=x.dtype)
    y1_silu = torch.empty((B, C, H1, W1), device=x.device, dtype=x.dtype)

    y2 = torch.empty((B, C, H2, W2), device=x.device, dtype=x.dtype)
    y2_norm = torch.empty((B, C, H2, W2), device=x.device, dtype=x.dtype)
    y2_silu = torch.empty((B, C, H2, W2), device=x.device, dtype=x.dtype)

    y_out = torch.empty_like(x)

    # Launch conv1 Triton kernel
    grid_conv1 = (B, C, H1, W1)
    conv3x3_nchw_nobias[grid_conv1](
        x, conv1_weight, y1,
        B, C, C, H, W, H1, W1,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        num_warps=4, num_stages=2
    )

    # Launch GroupNorm1 Triton kernel
    grid_gn1 = (B, 32)
    group_norm_triton_fixed[grid_gn1](
        y1, norm1_weight, norm1_bias, eps,
        B, C, H1, W1,
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        num_warps=4, num_stages=2
    )

    # SiLU1 Triton
    N1 = y1.numel()
    grid_silu1 = (triton.cdiv(N1, 1024),)
    silu_triton[grid_silu1](y1, y1_silu, N1, num_warps=4, num_stages=2)

    # Conv2 Triton
    grid_conv2 = (B, C, H2, W2)
    conv3x3_nchw_nobias[grid_conv2](
        y1_silu, conv2_weight, y2,
        B, C, C, H1, W1, H2, W2,
        y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
        conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        num_warps=4, num_stages=2
    )

    # GroupNorm2 Triton
    grid_gn2 = (B, 32)
    group_norm_triton_fixed[grid_gn2](
        y2, norm2_weight, norm2_bias, eps,
        B, C, H2, W2,
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        num_warps=4, num_stages=2
    )

    # SiLU2 Triton
    N2 = y2.numel()
    grid_silu2 = (triton.cdiv(N2, 1024),)
    silu_triton[grid_silu2](y2, y2_silu, N2, num_warps=4, num_stages=2)

    # Residual addition: y_out = y2_silu + x, Triton kernel (launch it)
    N_add = x.numel()
    grid_add = (triton.cdiv(N_add, 1024),)
    add_residual_triton[grid_add](y2_silu, x, y_out, N_add, num_warps=4, num_stages=2)

    return y_out

# Triton-based model entry point
class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        # Run everything in Triton; no torch ops here
        return run_triton(
            x, conv1_weight, norm1_weight, norm1_bias,
            conv2_weight, norm2_weight, norm2_bias,
            eps
        )


def run(*args):
    return ModelNew()(*args)
