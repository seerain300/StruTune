import torch
import triton
import triton.language as tl


# 1D flattened conv3x3: each program computes one output element y[n, co, h, w].
# x: (N, C_in, H, W), w: (C_out, C_in, 3, 3)
# y: (N, C_out, H, W)
@triton.jit
def conv3x3_1d_flatten(x_ptr, w_ptr, y_ptr,
                       N, C_in, H, W, C_out,
                       x_sN, x_sC, x_sH, x_sW,
                       w_sCo, w_sCi, w_sKh, w_sKw,
                       y_sN, y_sC, y_sH, y_sW,
                       total_out):
    pid = tl.program_id(0)

    # Map linear index to (n, co, h, w)
    HW = H * W
    COHW = C_out * HW
    n = pid // COHW
    rem = pid % COHW
    co = rem // HW
    rem2 = rem % HW
    h = rem2 // W
    w = rem2 % W

    acc = tl.zeros((), dtype=tl.float32)

    for ci in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                h_in = h + kh - 1  # 1 padding
                w_in = w + kw - 1
                h_in_valid = (h_in >= 0) & (h_in < H)
                w_in_valid = (w_in >= 0) & (w_in < W)
                if h_in_valid and w_in_valid:
                    x_offset = n * x_sN + ci * x_sC + h_in * x_sH + w_in * x_sW
                    x_val = tl.load(x_ptr + x_offset)
                else:
                    x_val = tl.zeros((), dtype=tl.float32)

                w_offset = co * w_sCo + ci * w_sCi + kh * w_sKh + kw * w_sKw
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    y_offset = n * y_sN + co * y_sC + h * y_sH + w * y_sW
    tl.store(y_ptr + y_offset, acc)


# Triton kernel: reduce per-(n, group) sum and sumsq across channels in group and all spatial positions (H*W).
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


# Triton kernel: apply GroupNorm per channel across H*W, with per-channel affine and SiLU.
# Grid: (N, C). groups is constexpr (e.g., 32).
@triton.jit
def groupnorm_apply_affine_silu(y_ptr, sums_ptr, scale_ptr, bias_ptr, out_ptr,
                                N, C, H, W,
                                groups: tl.constexpr,
                                y_sN, y_sC, y_sH, y_sW,
                                out_sN, out_sC, out_sH, out_sW):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    group_size_c = C // groups
    group_g = pid_c // group_size_c

    sum_val = tl.load(sums_ptr + (pid_n * groups + group_g) * 2 + 0)
    sumsq_val = tl.load(sums_ptr + (pid_n * groups + group_g) * 2 + 1)
    hw = H * W
    mean = sum_val / hw
    var = sumsq_val / hw - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)  # epsilon for numerical stability

    # y: normalized + affine
    for h in range(0, H):
        for w in range(0, W):
            y_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            val = tl.load(y_ptr + y_offset) - mean
            val = val * rstd
            scale = tl.load(scale_ptr + pid_c)
            bias = tl.load(bias_ptr + pid_c)
            y_norm = val * scale + bias
            # SiLU activation: x * sigmoid(x)
            sig = 1.0 / (1.0 + tl.exp(-y_norm))
            out_val = y_norm * sig
            out_offset = pid_n * out_sN + pid_c * out_sC + h * out_sH + w * out_sW
            tl.store(out_ptr + out_offset, out_val)


# 1D Triton kernel: elementwise add for residual connection over total elements.
@triton.jit
def add_residual_1d(y_ptr, x_ptr, out_ptr, total_elems):
    pid = tl.program_id(0)
    idx = pid
    y_val = tl.load(y_ptr + idx)
    x_val = tl.load(x_ptr + idx)
    tl.store(out_ptr + idx, y_val + x_val)


def conv3x3_nchw_triton_1d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Compute y = conv3x3(x, w, stride=1, padding=1) in NCHW using Triton 1D flattened kernel.
    x: (N, C_in, H, W), w: (C_out, C_in, 3, 3)
    returns y: (N, C_out, H, W), float32
    """
    assert x.is_cuda and w.is_cuda
    N, C_in, H, W = x.shape
    C_out = w.shape[0]
    x_c = x.contiguous().to(torch.float32)
    w_c = w.contiguous().to(torch.float32)
    y = torch.empty((N, C_out, H, W), device=x.device, dtype=torch.float32)

    x_sN, x_sC, x_sH, x_sW = x_c.stride()
    w_sCo, w_sCi, w_sKh, w_sKw = w_c.stride()
    y_sN, y_sC, y_sH, y_sW = y.stride()

    total_out = N * C_out * H * W
    grid = (total_out,)
    conv3x3_1d_flatten[grid](
        x_c, w_c, y,
        N, C_in, H, W, C_out,
        x_sN, x_sC, x_sH, x_sW,
        w_sCo, w_sCi, w_sKh, w_sKw,
        y_sN, y_sC, y_sH, y_sW,
        total_out,
        num_warps=1
    )
    return y


def groupnorm_triton(y: torch.Tensor, groups: int, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Apply GroupNorm with num_groups=groups and per-channel affine (weight, bias) in Triton.
    y: (N, C, H, W), weight: (C,), bias: (C,)
    returns out: (N, C, H, W), float32
    """
    assert y.is_cuda and weight.is_cuda and bias.is_cuda
    N, C, H, W = y.shape
    assert C % groups == 0, "C must be divisible by groups"
    y_c = y.contiguous().to(torch.float32)
    weight_c = weight.contiguous().to(torch.float32)
    bias_c = bias.contiguous().to(torch.float32)

    # Buffer to store per-(n, group) sums: shape (N, groups, 2)
    sums = torch.empty((N, groups, 2), device=y.device, dtype=torch.float32)

    y_sN, y_sC, y_sH, y_sW = y_c.stride()
    grid_reduce = (N, groups)
    groupnorm_reduce_sums[grid_reduce](
        y_c, sums, N, C, H, W,
        groups,
        y_sN, y_sC, y_sH, y_sW,
        num_warps=1
    )

    out = torch.empty_like(y_c)

    out_sN, out_sC, out_sH, out_sW = out.stride()
    grid_apply = (N, C)
    groupnorm_apply_affine_silu[grid_apply](
        y_c, sums, weight_c, bias_c, out,
        N, C, H, W,
        groups,
        y_sN, y_sC, y_sH, y_sW,
        out_sN, out_sC, out_sH, out_sW,
        num_warps=1
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        # First conv3x3 via Triton 1D flattened kernel
        y1 = conv3x3_nchw_triton_1d(x, conv1_weight)  # (N, C, H, W)
        # GroupNorm + SiLU
        y1 = groupnorm_triton(y1, 32, norm1_weight, norm1_bias, eps)
        # Second conv3x3 via Triton 1D flattened kernel
        y2 = conv3x3_nchw_triton_1d(y1, conv2_weight)  # (N, C, H, W)
        # GroupNorm + SiLU
        y2 = groupnorm_triton(y2, 32, norm2_weight, norm2_bias, eps)
        # Residual add: y2 + x using 1D elementwise Triton kernel
        total = y2.numel()
        out = torch.empty_like(y2)
        add_residual_1d[(total,)](y2, x, out, total, num_warps=1)
        return out


def run(*args):
    return ModelNew()(*args)
