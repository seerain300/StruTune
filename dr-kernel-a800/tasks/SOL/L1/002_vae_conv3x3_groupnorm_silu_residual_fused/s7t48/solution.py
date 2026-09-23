import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_single(
    x_ptr, w_ptr, y_ptr,
    N, C_in: tl.constexpr, C_out: tl.constexpr, H, W,
    K_H: tl.constexpr, K_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # strides
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cout, w_stride_cin, w_stride_kh, w_stride_kw,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
):
    # One program per (n, c_out, oh, ow)
    pid = tl.program_id(axis=0)

    # Map pid -> (n, c_out, oh, ow)
    tiles_per_n = C_out * H * W
    n = pid // tiles_per_n
    rem = pid % tiles_per_n
    c_out = rem // (H * W)
    rem2 = rem % (H * W)
    oh = rem2 // W
    ow = rem2 % W

    acc = tl.zeros((), dtype=tl.float32)

    # Iterate over input channels and 3x3 kernel taps
    for cin in range(C_in):
        for kh in range(K_H):
            for kw in range(K_W):
                ih = oh + kh - PAD_H
                iw = ow + kw - PAD_W
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

                # Load input x[n, cin, ih, iw] (masked)
                x_off = n * x_stride_n + cin * x_stride_c + ih * x_stride_h + iw * x_stride_w
                x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)

                # Load weight w[c_out, cin, kh, kw]
                w_off = c_out * w_stride_cout + cin * w_stride_cin + kh * w_stride_kh + kw * w_stride_kw
                w_val = tl.load(w_ptr + w_off)

                acc += x_val * w_val

    # Store output y[n, c_out, oh, ow]
    y_off = n * y_stride_n + c_out * y_stride_c + oh * y_stride_h + ow * y_stride_w
    tl.store(y_ptr + y_off, acc)


@triton.jit
def group_norm_forward(
    x_ptr, gamma_ptr, beta_ptr, y_ptr,
    N, C: tl.constexpr, H, W,
    num_groups: tl.constexpr, eps: tl.constexpr,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
):
    # One program per (n, group)
    pid = tl.program_id(axis=0)
    n = pid // num_groups
    g = pid % num_groups
    group_size = C // num_groups
    c_start = g * group_size

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # Accumulate sum and sum of squares over channels in group and all spatial positions
    for c in range(group_size):
        c_idx = c_start + c
        for h in range(H):
            for w in range(W):
                x_off = n * x_stride_n + c_idx * x_stride_c + h * x_stride_h + w * x_stride_w
                x_val = tl.load(x_ptr + x_off)
                sum_val += x_val
                sum_sq += x_val * x_val

    mean = sum_val / (group_size * H * W)
    var = sum_sq / (group_size * H * W) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine
    for c in range(group_size):
        c_idx = c_start + c
        for h in range(H):
            for w in range(W):
                x_off = n * x_stride_n + c_idx * x_stride_c + h * x_stride_h + w * x_stride_w
                x_val = tl.load(x_ptr + x_off)
                norm = (x_val - mean) * inv_std
                gamma = tl.load(gamma_ptr + c_idx)
                beta = tl.load(beta_ptr + c_idx)
                y_val = norm * gamma + beta
                y_off = n * y_stride_n + c_idx * y_stride_c + h * y_stride_h + w * y_stride_w
                tl.store(y_ptr + y_off, y_val)


@triton.jit
def silu_kernel(
    x_ptr, y_ptr,
    N, C: tl.constexpr, H, W,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
):
    pid = tl.program_id(axis=0)
    # One program per element in N*C*H*W
    total = N * C * H * W
    idx = pid
    if idx >= total:
        return
    n = idx // (C * H * W)
    rem = idx % (C * H * W)
    c = rem // (H * W)
    rem2 = rem % (H * W)
    h = rem2 // W
    w = rem2 % W

    x_off = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
    x_val = tl.load(x_ptr + x_off)
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig

    y_off = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
    tl.store(y_ptr + y_off, y_val)


@triton.jit
def add_residual_kernel(
    x_ptr, y_ptr, out_ptr,
    N, C: tl.constexpr, H, W,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
):
    pid = tl.program_id(axis=0)
    total = N * C * H * W
    idx = pid
    if idx >= total:
        return
    n = idx // (C * H * W)
    rem = idx % (C * H * W)
    c = rem // (H * W)
    rem2 = rem % (H * W)
    h = rem2 // W
    w = rem2 % W

    x_off = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
    y_off = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
    out_off = n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w

    x_val = tl.load(x_ptr + x_off)
    y_val = tl.load(y_ptr + y_off)
    out_val = x_val + y_val
    tl.store(out_ptr + out_off, out_val)


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
    # Ensure contiguous float32
    x = x.contiguous().to(torch.float32)
    conv1_weight = conv1_weight.contiguous().to(torch.float32)
    norm1_weight = norm1_weight.contiguous().to(torch.float32)
    norm1_bias = norm1_bias.contiguous().to(torch.float32)
    conv2_weight = conv2_weight.contiguous().to(torch.float32)
    norm2_weight = norm2_weight.contiguous().to(torch.float32)
    norm2_bias = norm2_bias.contiguous().to(torch.float32)

    N, C_in, H, W = x.shape
    C_out1, C_in_w1, K_H, K_W = conv1_weight.shape
    assert C_in_w1 == C_in and K_H == 3 and K_W == 3, "conv1_weight must be (C_out1, C_in, 3, 3)"
    C_out2, C_in_w2, K_H2, K_W2 = conv2_weight.shape
    assert C_in_w2 == C_out1 and K_H2 == 3 and K_W2 == 3, "conv2_weight must be (C_out2, C_out1, 3, 3)"

    # First conv
    y1 = torch.empty((N, C_out1, H, W), device=x.device, dtype=torch.float32)
    grid_conv1 = (N * C_out1 * H * W,)
    conv3x3_stride1_pad1_single[grid_conv1](
        x, conv1_weight, y1,
        N, C_in, C_out1, H, W,
        K_H=3, K_W=3, PAD_H=1, PAD_W=1,
        x_stride_n=x.stride(0), x_stride_c=x.stride(1), x_stride_h=x.stride(2), x_stride_w=x.stride(3),
        w_stride_cout=conv1_weight.stride(0), w_stride_cin=conv1_weight.stride(1), w_stride_kh=conv1_weight.stride(2), w_stride_kw=conv1_weight.stride(3),
        y_stride_n=y1.stride(0), y_stride_c=y1.stride(1), y_stride_h=y1.stride(2), y_stride_w=y1.stride(3),
        num_warps=2, num_stages=2
    )

    # First GroupNorm: require C_out1 % 32 == 0
    assert (C_out1 % 32) == 0, "C_out1 must be divisible by num_groups=32 for GroupNorm"
    y1_norm = torch.empty_like(y1)
    grid_gn1 = (N * 32,)
    group_norm_forward[grid_gn1](
        y1, norm1_weight, norm1_bias, y1_norm,
        N, C_out1, H, W,
        num_groups=32, eps=eps,
        x_stride_n=y1.stride(0), x_stride_c=y1.stride(1), x_stride_h=y1.stride(2), x_stride_w=y1.stride(3),
        y_stride_n=y1_norm.stride(0), y_stride_c=y1_norm.stride(1), y_stride_h=y1_norm.stride(2), y_stride_w=y1_norm.stride(3),
        num_warps=2, num_stages=2
    )

    # First SiLU
    y1_silu = torch.empty_like(y1_norm)
    grid_silu1 = (N * C_out1 * H * W,)
    silu_kernel[grid_silu1](
        y1_norm, y1_silu,
        N, C_out1, H, W,
        x_stride_n=y1_norm.stride(0), x_stride_c=y1_norm.stride(1), x_stride_h=y1_norm.stride(2), x_stride_w=y1_norm.stride(3),
        y_stride_n=y1_silu.stride(0), y_stride_c=y1_silu.stride(1), y_stride_h=y1_silu.stride(2), y_stride_w=y1_silu.stride(3),
        num_warps=2, num_stages=2
    )

    # Second conv
    y2 = torch.empty((N, C_out2, H, W), device=x.device, dtype=torch.float32)
    grid_conv2 = (N * C_out2 * H * W,)
    conv3x3_stride1_pad1_single[grid_conv2](
        y1_silu, conv2_weight, y2,
        N, C_out1, C_out2, H, W,
        K_H=3, K_W=3, PAD_H=1, PAD_W=1,
        x_stride_n=y1_silu.stride(0), x_stride_c=y1_silu.stride(1), x_stride_h=y1_silu.stride(2), x_stride_w=y1_silu.stride(3),
        w_stride_cout=conv2_weight.stride(0), w_stride_cin=conv2_weight.stride(1), w_stride_kh=conv2_weight.stride(2), w_stride_kw=conv2_weight.stride(3),
        y_stride_n=y2.stride(0), y_stride_c=y2.stride(1), y_stride_h=y2.stride(2), y_stride_w=y2.stride(3),
        num_warps=2, num_stages=2
    )

    # Second GroupNorm: require C_out2 % 32 == 0
    assert (C_out2 % 32) == 0, "C_out2 must be divisible by num_groups=32 for GroupNorm"
    y2_norm = torch.empty_like(y2)
    grid_gn2 = (N * 32,)
    group_norm_forward[grid_gn2](
        y2, norm2_weight, norm2_bias, y2_norm,
        N, C_out2, H, W,
        num_groups=32, eps=eps,
        x_stride_n=y2.stride(0), x_stride_c=y2.stride(1), x_stride_h=y2.stride(2), x_stride_w=y2.stride(3),
        y_stride_n=y2_norm.stride(0), y_stride_c=y2_norm.stride(1), y_stride_h=y2_norm.stride(2), y_stride_w=y2_norm.stride(3),
        num_warps=2, num_stages=2
    )

    # Second SiLU
    y2_silu = torch.empty_like(y2_norm)
    grid_silu2 = (N * C_out2 * H * W,)
    silu_kernel[grid_silu2](
        y2_norm, y2_silu,
        N, C_out2, H, W,
        x_stride_n=y2_norm.stride(0), x_stride_c=y2_norm.stride(1), x_stride_h=y2_norm.stride(2), x_stride_w=y2_norm.stride(3),
        y_stride_n=y2_silu.stride(0), y_stride_c=y2_silu.stride(1), y_stride_h=y2_silu.stride(2), y_stride_w=y2_silu.stride(3),
        num_warps=2, num_stages=2
    )

    # Residual add
    out = torch.empty_like(y2_silu)
    grid_add = (N * C_out2 * H * W,)
    add_residual_kernel[grid_add](
        x, y2_silu, out,
        N, C_out2, H, W,
        x_stride_n=x.stride(0), x_stride_c=x.stride(1), x_stride_h=x.stride(2), x_stride_w=x.stride(3),
        y_stride_n=y2_silu.stride(0), y_stride_c=y2_silu.stride(1), y_stride_h=y2_silu.stride(2), y_stride_w=y2_silu.stride(3),
        out_stride_n=out.stride(0), out_stride_c=out.stride(1), out_stride_h=out.stride(2), out_stride_w=out.stride(3),
        num_warps=2, num_stages=2
    )

    return out


class ModelNew(torch.nn.Module):
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        return run_triton(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)


def run(*args):
    return ModelNew()(*args)
