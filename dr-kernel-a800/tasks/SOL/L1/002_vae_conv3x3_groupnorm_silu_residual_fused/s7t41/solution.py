import torch
import triton
import triton.language as tl


# Conv2d 3x3 stride=1, padding=1, bias=None
# Each program computes one output element y[n, c_out, oh, ow] for fixed n, oh, ow.
@triton.jit
def conv3x3_stride1_pad1_element_kernel(
    x_ptr, w_ptr, y_ptr,
    N, C_in, H, W,
    C_out, K_h, K_w,
    pad_h, pad_w,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    w_stride_co, w_stride_ci, w_stride_kh, w_stride_kw,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    MAX_C_IN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    total_per_batch = C_out * H * W
    n = pid // total_per_batch
    rem = pid % total_per_batch
    c_out = rem // (H * W)
    rem2 = rem % (H * W)
    oh = rem2 // W
    ow = rem2 % W

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels up to MAX_C_IN (mask beyond actual C_in)
    for ci in range(0, MAX_C_IN):
        valid_ci = ci < C_in
        sum_val = tl.zeros((), dtype=tl.float32)
        # Unroll over 3x3 taps
        for kh in range(0, K_h):
            for kw in range(0, K_w):
                ih = oh + kh - pad_h
                iw = ow + kw - pad_w
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & valid_ci
                x_offset = n * x_stride_n + ci * x_stride_c + ih * x_stride_h + iw * x_stride_w
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)

                w_offset = c_out * w_stride_co + ci * w_stride_ci + kh * w_stride_kh + kw * w_stride_kw
                w_val = tl.load(w_ptr + w_offset, mask=True, other=0.0)
                sum_val += x_val * w_val

        acc += sum_val

    y_offset = n * y_stride_n + c_out * y_stride_c + oh * y_stride_h + ow * y_stride_w
    tl.store(y_ptr + y_offset, acc)


# GroupNorm per (n, group): C % G == 0, affine per channel.
@triton.jit
def group_norm_kernel(
    inp_ptr, y_ptr, gn_w_ptr, gn_b_ptr,
    N, C, H, W,
    G: tl.constexpr,  # num_groups
    eps,
    inp_stride_n, inp_stride_c, inp_stride_h, inp_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    MAX_C: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    total_groups = N * G
    n = pid // G
    group = pid % G
    c_start = group * (C // G)
    C_group = C // G

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # First pass: accumulate sum and sum of squares across group channels and spatial positions
    for ci in range(0, MAX_C):
        valid_ci = ci < C_group + c_start
        for h in range(0, H):
            for w in range(0, W):
                inp_offset = n * inp_stride_n + (c_start + ci) * inp_stride_c + h * inp_stride_h + w * inp_stride_w
                x_val = tl.load(inp_ptr + inp_offset)
                sum_val += x_val
                sum_sq += x_val * x_val

    total = C_group * H * W
    mean = sum_val / total
    var = sum_sq / total - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, then store
    for ci in range(0, MAX_C):
        valid_ci = ci < C_group + c_start
        for h in range(0, H):
            for w in range(0, W):
                inp_offset = n * inp_stride_n + (c_start + ci) * inp_stride_c + h * inp_stride_h + w * inp_stride_w
                x_val = tl.load(inp_ptr + inp_offset)
                norm = (x_val - mean) * inv_std
                gamma = tl.load(gn_w_ptr + (c_start + ci))
                beta = tl.load(gn_b_ptr + (c_start + ci))
                y_val = norm * gamma + beta
                y_offset = n * y_stride_n + (c_start + ci) * y_stride_c + h * y_stride_h + w * y_stride_w
                tl.store(y_ptr + y_offset, y_val)


# SiLU elementwise: y = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
@triton.jit
def silu_kernel(
    x_ptr, y_ptr,
    N, C, H, W,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    MAX_C: tl.constexpr,
):
    total = N * C * H * W
    pid = tl.program_id(axis=0)
    if pid < total:
        n = pid // (C * H * W)
        rem = pid % (C * H * W)
        c = rem // (H * W)
        rem2 = rem % (H * W)
        h = rem2 // W
        w = rem2 % W

        x_offset = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
        y_offset = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
        x_val = tl.load(x_ptr + x_offset)
        sig = 1.0 / (1.0 + tl.exp(-x_val))
        y_val = x_val * sig
        tl.store(y_ptr + y_offset, y_val)


# Elementwise add: out = y + x
@triton.jit
def add_kernel(
    x_ptr, y_ptr, out_ptr,
    N, C, H, W,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    MAX_C: tl.constexpr,
):
    total = N * C * H * W
    pid = tl.program_id(axis=0)
    if pid < total:
        n = pid // (C * H * W)
        rem = pid % (C * H * W)
        c = rem // (H * W)
        rem2 = rem % (H * W)
        h = rem2 // W
        w = rem2 % W

        x_offset = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
        y_offset = n * y_stride_n + c * y_stride_c + h * y_stride_h + w * y_stride_w
        out_offset = n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w

        x_val = tl.load(x_ptr + x_offset)
        y_val = tl.load(y_ptr + y_offset)
        tl.store(out_ptr + out_offset, x_val + y_val)


def _launch_conv(x, w, out, N, C_in, H, W, C_out):
    # x: (N, C_in, H, W), w: (C_out, C_in, 3, 3), out: (N, C_out, H, W)
    x_f32 = x.contiguous().float()
    w_f32 = w.contiguous().float()
    out_f32 = out  # float32 buffer

    total = N * C_out * H * W
    grid = (total,)
    conv3x3_stride1_pad1_element_kernel[grid](
        x_f32, w_f32, out_f32,
        N, C_in, H, W,
        C_out, 3, 3, 1, 1,
        x_f32.stride(0), x_f32.stride(1), x_f32.stride(2), x_f32.stride(3),
        w_f32.stride(0), w_f32.stride(1), w_f32.stride(2), w_f32.stride(3),
        out_f32.stride(0), out_f32.stride(1), out_f32.stride(2), out_f32.stride(3),
        MAX_C_IN=64,  # upper bound for input channels; adjust if needed
        BLOCK=9,
        num_warps=4,
        num_stages=2,
    )
    return out_f32


def _launch_group_norm(inp, y, gn_weight, gn_bias, N, C, H, W, G, eps):
    assert C % G == 0, "C must be divisible by num_groups"
    inp_f32 = inp.contiguous().float()
    y_f32 = y  # float32 buffer
    total_groups = N * G
    grid = (total_groups,)
    group_norm_kernel[grid](
        inp_f32, y_f32, gn_weight.contiguous().float(), gn_bias.contiguous().float(),
        N, C, H, W,
        G=G, eps=eps,
        inp_stride_n=inp_f32.stride(0), inp_stride_c=inp_f32.stride(1), inp_stride_h=inp_f32.stride(2), inp_stride_w=inp_f32.stride(3),
        y_stride_n=y_f32.stride(0), y_stride_c=y_f32.stride(1), y_stride_h=y_f32.stride(2), y_stride_w=y_f32.stride(3),
        MAX_C=256,  # loop bound; more than enough for typical C
        num_warps=4,
        num_stages=2,
    )
    return y_f32


def _launch_silu(x, y, N, C, H, W):
    x_f32 = x.contiguous().float()
    y_f32 = y  # float32 buffer
    total = N * C * H * W
    grid = (total,)
    silu_kernel[grid](
        x_f32, y_f32,
        N, C, H, W,
        x_f32.stride(0), x_f32.stride(1), x_f32.stride(2), x_f32.stride(3),
        y_f32.stride(0), y_f32.stride(1), y_f32.stride(2), y_f32.stride(3),
        MAX_C=256,
        num_warps=4,
        num_stages=2,
    )
    return y_f32


def _launch_add(x, y, out, N, C, H, W):
    x_f32 = x.contiguous().float()
    y_f32 = y.contiguous().float()
    out_f32 = out  # float32 buffer
    total = N * C * H * W
    grid = (total,)
    add_kernel[grid](
        x_f32, y_f32, out_f32,
        N, C, H, W,
        x_f32.stride(0), x_f32.stride(1), x_f32.stride(2), x_f32.stride(3),
        y_f32.stride(0), y_f32.stride(1), y_f32.stride(2), y_f32.stride(3),
        out_f32.stride(0), out_f32.stride(1), out_f32.stride(2), out_f32.stride(3),
        MAX_C=256,
        num_warps=4,
        num_stages=2,
    )
    return out_f32


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        # x: (N, C, H, W)
        N, C, H, W = x.shape

        # 1) conv1
        out1 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)
        _launch_conv(x, conv1_weight, out1, N, C, H, W, C)

        # 2) GroupNorm1 (num_groups=32)
        assert C % 32 == 0, "C must be divisible by num_groups=32 for GroupNorm"
        out1_gn = torch.empty_like(out1)
        _launch_group_norm(out1, out1_gn, norm1_weight, norm1_bias, N, C, H, W, G=32, eps=eps)

        # 3) SiLU1
        out1_silu = torch.empty_like(out1_gn)
        _launch_silu(out1_gn, out1_silu, N, C, H, W)

        # 4) conv2
        out2 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)
        _launch_conv(out1_silu, conv2_weight, out2, N, C, H, W, C)

        # 5) GroupNorm2
        out2_gn = torch.empty_like(out2)
        _launch_group_norm(out2, out2_gn, norm2_weight, norm2_bias, N, C, H, W, G=32, eps=eps)

        # 6) SiLU2
        out2_silu = torch.empty_like(out2_gn)
        _launch_silu(out2_gn, out2_silu, N, C, H, W)

        # 7) Add residual x
        out = torch.empty_like(out2_silu)
        _launch_add(out2_silu, x, out, N, C, H, W)

        return out


def run(*args):
    return ModelNew()(*args)
