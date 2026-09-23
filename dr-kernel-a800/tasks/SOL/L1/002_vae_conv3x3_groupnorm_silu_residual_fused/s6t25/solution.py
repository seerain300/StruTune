import torch
import triton
import triton.language as tl


# Triton kernel: conv3x3 NCHW, stride=1, padding=1.
# Grid: (N, C_out, H, W). Each program computes one output element y[n, co, h, w].
@triton.jit
def conv3x3_nchw_4d(x_ptr, w_ptr, y_ptr,
                    N, C_in, H, W, C_out,
                    x_sN, x_sC, x_sH, x_sW,
                    w_sCo, w_sCi, w_sKh, w_sKw,
                    y_sN, y_sC, y_sH, y_sW,
                    NUM_CI: tl.constexpr, NUM_H: tl.constexpr, NUM_W: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 neighborhood
    for ci in range(NUM_CI):
        for kh in range(NUM_H):
            for kw in range(NUM_W):
                h_in = pid_h + kh
                w_in = pid_w + kw
                h_in_valid = (h_in >= 0) & (h_in < H)
                w_in_valid = (w_in >= 0) & (w_in < W)
                if h_in_valid and w_in_valid:
                    x_offset = pid_n * x_sN + ci * x_sC + h_in * x_sH + w_in * x_sW
                    x_val = tl.load(x_ptr + x_offset)
                else:
                    x_val = tl.zeros((), dtype=tl.float32)

                w_offset = pid_co * w_sCo + ci * w_sCi + kh * w_sKh + kw * w_sKw
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    y_offset = pid_n * y_sN + pid_co * y_sC + pid_h * y_sH + pid_w * y_sW
    tl.store(y_ptr + y_offset, acc)


# Triton kernel: GroupNorm reduction per (n, group) to compute sum and sumsq across channels in group and all spatial positions (H*W).
# Grid: (N, groups). groups is passed as constexpr (e.g., 32).
@triton.jit
def groupnorm_reduce_sums(y_ptr, sums_ptr,
                           N, C, H, W,
                           groups: tl.constexpr,
                           y_sN, y_sC, y_sH, y_sW):
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
    tl.store(sums_ptr + base * 2 + 0, sum_val)  # sum
    tl.store(sums_ptr + base * 2 + 1, sumsq_val)  # sumsq


# Triton kernel: GroupNorm apply + per-channel affine + SiLU for each (n, c) across H*W.
# Grid: (N, C). groups is constexpr (e.g., 32).
@triton.jit
def groupnorm_apply_affine_silu(y_ptr, sums_ptr, scale_ptr, bias_ptr, out_ptr,
                                N, C, H, W,
                                groups: tl.constexpr,
                                y_sN, y_sC, y_sH, y_sW,
                                out_sN, out_sC, out_sH, out_sW,
                                eps: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    group_size_c = C // groups
    g = pid_c // group_size_c  # group index for this channel

    # Load group stats for (n, g)
    base = pid_n * groups + g
    sum_val = tl.load(sums_ptr + base * 2 + 0)
    sumsq_val = tl.load(sums_ptr + base * 2 + 1)

    M = H * W
    mean = sum_val / M
    var = sumsq_val / M - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Per-channel affine parameters
    scale = tl.load(scale_ptr + pid_c)  # norm_weight[c]
    bias = tl.load(bias_ptr + pid_c)    # norm_bias[c]

    # Compute normalized + affine + SiLU: SiLU(z) = z * sigmoid(z)
    for h in range(0, H):
        for w in range(0, W):
            y_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            val = tl.load(y_ptr + y_offset)
            norm = (val - mean) * rstd
            z = norm * scale + bias
            sig = 1.0 / (1.0 + tl.exp(-z))
            out_val = z * sig
            out_offset = pid_n * out_sN + pid_c * out_sC + h * out_sH + w * out_sW
            tl.store(out_ptr + out_offset, out_val)


# Triton kernel: elementwise residual add over all elements (N, C, H, W) flattened.
@triton.jit
def add_residual_kernel(y_ptr, x_ptr, out_ptr, total_elems: tl.constexpr):
    pid = tl.program_id(0)
    if pid < total_elems:
        out_val = tl.load(y_ptr + pid)
        x_val = tl.load(x_ptr + pid)
        res = out_val + x_val
        tl.store(out_ptr + pid, res)


def _conv3x3_triton(x32, w32):
    """
    x: (N, C_in, H, W), w: (C_out, C_in, 3, 3)
    returns y: (N, C_out, H, W)
    """
    N, C_in, H, W = x32.shape
    C_out = w32.shape[0]

    y = torch.empty((N, C_out, H, W), dtype=torch.float32, device=x32.device)

    x_sN, x_sC, x_sH, x_sW = x32.stride()
    w_sCo, w_sCi, w_sKh, w_sKw = w32.stride()
    y_sN, y_sC, y_sH, y_sW = y.stride()

    grid = (N, C_out, H, W)
    conv3x3_nchw_4d[grid](
        x32, w32, y,
        N, C_in, H, W, C_out,
        x_sN, x_sC, x_sH, x_sW,
        w_sCo, w_sCi, w_sKh, w_sKw,
        y_sN, y_sC, y_sH, y_sW,
        NUM_CI=C_in, NUM_H=3, NUM_W=3,
        num_warps=1
    )
    return y


def _groupnorm_apply_silu_triton(y32, scale32, bias32, N, C, H, W, groups: int, eps: float):
    """
    Apply GroupNorm with num_groups=groups, per-channel affine (scale, bias), then SiLU.
    y32: input (N, C, H, W), scale32/bias32: (C,)
    returns out: (N, C, H, W)
    """
    # 1) Reduction to compute per-(n, group) sum and sumsq
    sums = torch.empty((N, groups, 2), dtype=torch.float32, device=y32.device)
    y_sN, y_sC, y_sH, y_sW = y32.stride()
    grid_reduce = (N, groups)
    groupnorm_reduce_sums[grid_reduce](
        y32, sums, N, C, H, W,
        groups,  # constexpr
        y_sN, y_sC, y_sH, y_sW,
        num_warps=1
    )

    # 2) Apply normalization + affine + SiLU
    out = torch.empty_like(y32)
    out_sN, out_sC, out_sH, out_sW = out.stride()
    grid_apply = (N, C)
    groupnorm_apply_affine_silu[grid_apply](
        y32, sums, scale32, bias32, out,
        N, C, H, W,
        groups,  # constexpr
        y_sN, y_sC, y_sH, y_sW,
        out_sN, out_sC, out_sH, out_sW,
        eps,  # constexpr for this kernel instance
        num_warps=1
    )
    return out


def _add_residual_triton(y, x):
    """
    y: (N, C, H, W) flattened output, x: residual (N, C, H, W) flattened.
    Residual add via Triton kernel.
    """
    total = y.numel()
    out = torch.empty_like(y)
    grid = (total,)
    add_residual_kernel[grid](y, x, out, total)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        # Ensure float32 and contiguous for predictable behavior
        x32 = x.contiguous().to(torch.float32)

        # First conv: (N, C, H, W) -> (N, C, H, W)
        y1 = _conv3x3_triton(x32, conv1_weight)
        # GroupNorm + SiLU (num_groups=32), per-channel affine
        y1 = _groupnorm_apply_silu_triton(y1, norm1_weight, norm1_bias, y1.shape[0], y1.shape[1], y1.shape[2], y1.shape[3], groups=32, eps=eps)
        # Second conv
        y2 = _conv3x3_triton(y1, conv2_weight)
        # GroupNorm + SiLU
        y2 = _groupnorm_apply_silu_triton(y2, norm2_weight, norm2_bias, y2.shape[0], y2.shape[1], y2.shape[2], y2.shape[3], groups=32, eps=eps)
        # Residual add
        out = _add_residual_triton(y2, x32)
        return out


def run(*args):
    return ModelNew()(*args)
