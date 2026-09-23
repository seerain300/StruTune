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

    # Accumulator for output
    acc = tl.zeros((), dtype=tl.float32)

    # Iterate over input channels and 3x3 kernel taps
    for cin in range(C_in):
        for kh in range(K_H):
            for kw in range(K_W):
                ih = oh + kh - PAD_H
                iw = ow + kw - PAD_W
                # in-bounds check
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

                # Load input x[n, cin, ih, iw] (masked)
                x_off = n * x_stride_n + cin * x_stride_c + ih * x_stride_h + iw * x_stride_w
                x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)

                # Load weight w[c_out, cin, kh, kw]
                w_off = c_out * w_stride_cout + cin * w_stride_cin + kh * w_stride_kh + kw * w_stride_kw
                w_val = tl.load(w_ptr + w_off)

                # FMA
                acc += x_val * w_val

    # Store output y[n, c_out, oh, ow]
    y_off = n * y_stride_n + c_out * y_stride_c + oh * y_stride_h + ow * y_stride_w
    tl.store(y_ptr + y_off, acc)


@triton.jit
def group_norm_kernel(
    x_ptr, gamma_ptr, beta_ptr, y_ptr,
    N, C: tl.constexpr, H, W,
    num_groups: tl.constexpr, eps: tl.constexpr,
    # strides
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
):
    # One program per (n, group)
    pid = tl.program_id(axis=0)
    n = pid // num_groups
    group = pid % num_groups

    group_size = C // num_groups
    # First pass: compute mean and variance over group
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for ch in range(group_size * group, group_size * group + group_size):
        for h in range(H):
            for w in range(W):
                x_off = n * x_stride_n + ch * x_stride_c + h * x_stride_h + w * x_stride_w
                x_val = tl.load(x_ptr + x_off)
                sum_val += x_val
                sum_sq += x_val * x_val

    mean = sum_val / (group_size * H * W)
    var = sum_sq / (group_size * H * W) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for ch in range(group_size * group, group_size * group + group_size):
        for h in range(H):
            for w in range(W):
                x_off = n * x_stride_n + ch * x_stride_c + h * x_stride_h + w * x_stride_w
                x_val = tl.load(x_ptr + x_off)
                norm = (x_val - mean) * inv_std
                gamma = tl.load(gamma_ptr + ch)
                beta = tl.load(beta_ptr + ch)
                y_val = norm * gamma + beta
                y_off = n * y_stride_n + ch * y_stride_c + h * y_stride_h + w * y_stride_w
                tl.store(y_ptr + y_off, y_val)


@triton.jit
def silu_kernel(
    x_ptr, y_ptr,
    N, C: tl.constexpr, H, W,
    # strides
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
):
    pid = tl.program_id(axis=0)
    size = N * C * H * W
    # One element per program; simple loop-safe mapping
    for i in range(size):
        # compute n, c, h, w from i
        tmp = i
        w = tmp % W
        tmp = tmp // W
        h = tmp % H
        tmp = tmp // H
        c = tmp % C
        n = tmp // C

        x_off = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
        x_val = tl.load(x_ptr + x_off)
        # SiLU: x * sigmoid(x)
        sig = 1.0 / (1.0 + tl.exp(-x_val))
        y_val = x_val * sig
        y_off = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
        tl.store(y_ptr + y_off, y_val)


@triton.jit
def add_residual_kernel(
    x_ptr, y_ptr, out_ptr,
    N, C: tl.constexpr, H, W,
    # strides
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
):
    pid = tl.program_id(axis=0)
    size = N * C * H * W
    for i in range(size):
        w = i % W
        tmp = i // W
        h = tmp % H
        tmp = tmp // H
        c = tmp % C
        n = tmp // C

        x_off = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
        y_off = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
        out_off = n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w

        x_val = tl.load(x_ptr + x_off)
        y_val = tl.load(y_ptr + y_off)
        out_val = x_val + y_val
        tl.store(out_ptr + out_off, out_val)


def triton_conv3x3_stride1_pad1(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Triton conv2d: 3x3, stride=1, padding=1, bias=None
    x: (N, C_in, H, W), w: (C_out, C_in, 3, 3)
    returns y: (N, C_out, H, W)
    """
    assert x.is_cuda and w.is_cuda, "Inputs must be CUDA tensors"
    x = x.contiguous().float()
    w = w.contiguous().float()
    N, C_in, H, W = x.shape
    C_out, C_in_w, K_H, K_W = w.shape
    assert C_in == C_in_w and K_H == 3 and K_W == 3, "Expect (C_out, C_in, 3, 3) weights and H=W=3 for conv"
    y = torch.empty((N, C_out, H, W), device=x.device, dtype=torch.float32)

    grid = (N * C_out * H * W,)
    conv3x3_stride1_pad1_single[grid](
        x, w, y,
        N, C_in, C_out, H, W,
        K_H=3, K_W=3, PAD_H=1, PAD_W=1,
        x_stride_n=x.stride(0), x_stride_c=x.stride(1), x_stride_h=x.stride(2), x_stride_w=x.stride(3),
        w_stride_cout=w.stride(0), w_stride_cin=w.stride(1), w_stride_kh=w.stride(2), w_stride_kw=w.stride(3),
        y_stride_n=y.stride(0), y_stride_c=y.stride(1), y_stride_h=y.stride(2), y_stride_w=y.stride(3),
        num_warps=2, num_stages=2
    )
    return y


def triton_group_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Triton GroupNorm: C must be divisible by num_groups (default 32).
    x: (N, C, H, W), weight, bias: (C,)
    returns y: (N, C, H, W)
    """
    N, C, H, W = x.shape
    assert C % 32 == 0, "C must be divisible by num_groups (32)"
    y = torch.empty_like(x)
    grid = (N * 32,)
    group_norm_kernel[grid](
        x, weight, bias, y,
        N, C, H, W, num_groups=32, eps=eps,
        x_stride_n=x.stride(0), x_stride_c=x.stride(1), x_stride_h=x.stride(2), x_stride_w=x.stride(3),
        y_stride_n=y.stride(0), y_stride_c=y.stride(1), y_stride_h=y.stride(2), y_stride_w=y.stride(3),
        num_warps=2, num_stages=2
    )
    return y


def triton_silu(x: torch.Tensor) -> torch.Tensor:
    """
    Triton SiLU: y = x * sigmoid(x)
    x: (N, C, H, W)
    returns y: (N, C, H, W)
    """
    N, C, H, W = x.shape
    y = torch.empty_like(x)
    grid = (N * C * H * W,)
    silu_kernel[grid](
        x, y,
        N, C, H, W,
        x_stride_n=x.stride(0), x_stride_c=x.stride(1), x_stride_h=x.stride(2), x_stride_w=x.stride(3),
        y_stride_n=y.stride(0), y_stride_c=y.stride(1), y_stride_h=y.stride(2), y_stride_w=y.stride(3),
        num_warps=2, num_stages=2
    )
    return y


def triton_add_residual(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Triton elementwise add: out = x + y
    x, y: (N, C, H, W)
    returns out: (N, C, H, W)
    """
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    grid = (N * C * H * W,)
    add_residual_kernel[grid](
        x, y, out,
        N, C, H, W,
        x_stride_n=x.stride(0), x_stride_c=x.stride(1), x_stride_h=x.stride(2), x_stride_w=x.stride(3),
        y_stride_n=y.stride(0), y_stride_c=y.stride(1), y_stride_h=y.stride(2), y_stride_w=y.stride(3),
        out_stride_n=out.stride(0), out_stride_c=out.stride(1), out_stride_h=out.stride(2), out_stride_w=out.stride(3),
        num_warps=2, num_stages=2
    )
    return out


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
    Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
    Triton-only implementation. No torch ops in forward.
    """
    assert x.is_cuda and conv1_weight.is_cuda and norm1_weight.is_cuda and norm1_bias.is_cuda \
        and conv2_weight.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "All tensors must be CUDA"

    # 1) First conv
    out1 = triton_conv3x3_stride1_pad1(x, conv1_weight)  # (N, C1_out, H, W)

    # 2) First GroupNorm
    out1 = triton_group_norm(out1, norm1_weight, norm1_bias, eps)

    # 3) First SiLU
    out1 = triton_silu(out1)

    # 4) Second conv
    out2 = triton_conv3x3_stride1_pad1(out1, conv2_weight)  # (N, C2_out, H, W)

    # 5) Second GroupNorm
    out2 = triton_group_norm(out2, norm2_weight, norm2_bias, eps)

    # 6) Second SiLU
    out2 = triton_silu(out2)

    # 7) Residual add
    final = triton_add_residual(x, out2)

    return final


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect same signature as original: (x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)
        # Note: torch.no_grad() is not used; Triton kernels handle forward computation.
        return run_triton(*args)


def run(*args):
    return ModelNew()(*args)
