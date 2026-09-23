import torch
import triton
import triton.language as tl


# Triton conv3x3: stride=1, padding=1, NCHW layout, one output per program
@triton.jit
def conv3x3_nchw_4d(x_ptr, w_ptr, y_ptr,
                     N, C_in, H, W,
                     x_sN, x_sC, x_sH, x_sW,
                     w_sCo, w_sCi, w_sKh, w_sKw,
                     y_sN, y_sC, y_sH, y_sW,
                     NUM_CI: tl.constexpr, NUM_H: tl.constexpr, NUM_W: tl.constexpr):
    # Grid: (N, C_out, H, W)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    for ci in range(0, NUM_CI):  # compile-time unrolled for given C_in
        for kh in range(0, NUM_H):  # 3x3 kernel
            for kw in range(0, NUM_W):
                h_in = pid_h + kh
                w_in = pid_w + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                x_offset = pid_n * x_sN + ci * x_sC + h_in * x_sH + w_in * x_sW
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
                w_offset = pid_co * w_sCo + ci * w_sCi + kh * w_sKh + kw * w_sKw
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    y_offset = pid_n * y_sN + pid_co * y_sC + pid_h * y_sH + pid_w * y_sW
    tl.store(y_ptr + y_offset, acc)


# GroupNorm: reduce sums per (N, group)
@triton.jit
def groupnorm_reduce_sums(y_ptr, sums_ptr,
                           N, C, H, W, groups,
                           y_sN, y_sC, y_sH, y_sW):
    # grid = (N, groups)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_size_c = C // groups
    start_c = pid_g * group_size_c

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    for c in range(start_c, start_c + group_size_c):
        for h in range(0, H):
            for w in range(0, W):
                y_offset = pid_n * y_sN + c * y_sC + h * y_sH + w * y_sW
                val = tl.load(y_ptr + y_offset)
                sum_val += val
                sumsq_val += val * val

    base = pid_n * groups + pid_g
    tl.store(sums_ptr + base * 2 + 0, sum_val)
    tl.store(sums_ptr + base * 2 + 1, sumsq_val)


# GroupNorm: apply normalization + per-channel affine + SiLU per (N, C) using Triton
@triton.jit
def groupnorm_apply_affine_silu(y_ptr, out_ptr, sums_ptr, scale_ptr, bias_ptr,
                                N, C, H, W, groups, eps,
                                y_sN, y_sC, y_sH, y_sW,
                                out_sN, out_sC, out_sH, out_sW):
    # grid = (N, C)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    group_size_c = C // groups
    group_g = pid_c // group_size_c

    # load per-(n, group) sum and sumsq
    base = pid_n * groups + group_g
    sum_val = tl.load(sums_ptr + base * 2 + 0)
    sumsq_val = tl.load(sums_ptr + base * 2 + 1)

    count = H * W
    mean = sum_val / count
    var = sumsq_val / count - mean * mean
    rstd = tl.rsqrt(var + eps)

    scale = tl.load(scale_ptr + pid_c)
    bias = tl.load(bias_ptr + pid_c)

    for h in range(0, H):
        for w in range(0, W):
            y_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            y_val = tl.load(y_ptr + y_offset)
            norm = (y_val - mean) * rstd
            z = norm * scale + bias  # affine
            # SiLU: z * sigmoid(z) in Triton
            sig = 1.0 / (1.0 + tl.exp(-z))
            out_val = z * sig
            out_offset = pid_n * out_sN + pid_c * out_sC + h * out_sH + w * out_sW
            tl.store(out_ptr + out_offset, out_val)


# Elementwise SiLU: out = x * sigmoid(x)
@triton.jit
def silu_triton(x_ptr, out_ptr, total_elems,
                num_warps: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid
    mask = offs < total_elems
    x_val = tl.load(x_ptr + offs, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    tl.store(out_ptr + offs, x_val * sig)


# Elementwise residual addition: out = y + x
@triton.jit
def add_residual_kernel(y_ptr, x_ptr, out_ptr,
                        total_elems,
                        num_warps: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid
    mask = offs < total_elems
    y_val = tl.load(y_ptr + offs, mask=mask, other=0.0)
    x_val = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, y_val + x_val)


def _triton_conv3x3_nchw(x32: torch.Tensor, w32: torch.Tensor) -> torch.Tensor:
    # x: (N, C_in, H, W), w: (C_out, C_in, 3, 3)
    assert x32.is_cuda and w32.is_cuda
    N, C_in, H, W = x32.shape
    C_out = w32.shape[0]
    y = torch.empty((N, C_out, H, W), device=x32.device, dtype=x32.dtype)

    x_sN, x_sC, x_sH, x_sW = x32.stride()
    w_sCo, w_sCi, w_sKh, w_sKw = w32.stride()
    y_sN, y_sC, y_sH, y_sW = y.stride()

    conv3x3_nchw_4d[(N, C_out, H, W)](
        x32, w32, y,
        N, C_in, H, W,
        x_sN, x_sC, x_sH, x_sW,
        w_sCo, w_sCi, w_sKh, w_sKw,
        y_sN, y_sC, y_sH, y_sW,
        NUM_CI=C_in, NUM_H=3, NUM_W=3,
        num_warps=1
    )
    return y


def _groupnorm_triton(y32: torch.Tensor, groups: int, eps: float, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, C, H, W = y32.shape
    assert C % groups == 0

    # reduce: sums[n, group, 2] = [sum, sumsq]
    sums = torch.empty((N, groups, 2), device=y32.device, dtype=y32.dtype)

    y_sN, y_sC, y_sH, y_sW = y32.stride()
    grid_reduce = (N, groups)
    groupnorm_reduce_sums[grid_reduce](
        y32, sums,
        N, C, H, W, groups,
        y_sN, y_sC, y_sH, y_sW,
        num_warps=1
    )

    out = torch.empty_like(y32)

    out_sN, out_sC, out_sH, out_sW = out.stride()
    grid_apply = (N, C)
    groupnorm_apply_affine_silu[grid_apply](
        y32, out, sums, weight, bias,
        N, C, H, W, groups, eps,
        y_sN, y_sC, y_sH, y_sW,
        out_sN, out_sC, out_sH, out_sW,
        num_warps=1
    )
    return out


def _triton_silu(x32: torch.Tensor) -> torch.Tensor:
    total = x32.numel()
    out = torch.empty_like(x32)
    silu_triton[(total,)](x32, out, total, num_warps=1)
    return out


def _triton_residual_add(y32: torch.Tensor, x32: torch.Tensor) -> torch.Tensor:
    total = y32.numel()
    out = torch.empty_like(y32)
    add_residual_kernel[(total,)](y32, x32, out, total, num_warps=1)
    return out


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        # Ensure contiguous and float32
        x32 = x.contiguous().to(torch.float32)
        conv1_weight32 = conv1_weight.contiguous().to(torch.float32)
        conv2_weight32 = conv2_weight.contiguous().to(torch.float32)
        norm1_weight32 = norm1_weight.contiguous().to(torch.float32)
        norm2_weight32 = norm2_weight.contiguous().to(torch.float32)
        norm1_bias32 = norm1_bias.contiguous().to(torch.float32)
        norm2_bias32 = norm2_bias.contiguous().to(torch.float32)

        # First conv: Triton
        y1 = _triton_conv3x3_nchw(x32, conv1_weight32)

        # GroupNorm 1 (num_groups=32): Triton
        y1_gn = _groupnorm_triton(y1, groups=32, eps=eps, weight=norm1_weight32, bias=norm1_bias32)

        # SiLU: Triton
        y1_silu = _triton_silu(y1_gn)

        # Second conv: Triton
        y2 = _triton_conv3x3_nchw(y1_silu, conv2_weight32)

        # GroupNorm 2 (num_groups=32): Triton
        y2_gn = _groupnorm_triton(y2, groups=32, eps=eps, weight=norm2_weight32, bias=norm2_bias32)

        # SiLU: Triton
        y2_silu = _triton_silu(y2_gn)

        # Residual add: Triton
        out = _triton_residual_add(y2_silu, x32)

        return out


def run(*args):
    return ModelNew()(*args)
