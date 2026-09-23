import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_4d(x_ptr, w_ptr, y_ptr,
                    N, C_in, C_out, H, W,
                    x_sN, x_sC, x_sH, x_sW,
                    w_sCo, w_sCi, w_sKh, w_sKw,
                    y_sN, y_sC, y_sH, y_sW,
                    NUM_CI: tl.constexpr):
    # One program computes one output element (n, co, h, w)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 neighborhood, with padding masks
    for ci in range(NUM_CI):
        for kh in range(3):
            for kw in range(3):
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

    # Load per-(n, group) stats
    base = pid_n * groups + group_g
    sum_val = tl.load(sums_ptr + base * 2 + 0)
    sumsq_val = tl.load(sums_ptr + base * 2 + 1)

    count = H * W
    mean = sum_val / count
    var = sumsq_val / count - mean * mean
    rstd = tl.rsqrt(var + eps)

    scale = tl.load(scale_ptr + pid_c)
    bias = tl.load(bias_ptr + pid_c)

    # Normalize and apply affine + SiLU for all spatial positions of this (n, c)
    for h in range(0, H):
        for w in range(0, W):
            y_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            val = tl.load(y_ptr + y_offset)
            norm = (val - mean) * rstd
            af = norm * scale + bias
            silu = af * (1.0 / (1.0 + tl.exp(-af)))
            out_offset = pid_n * out_sN + pid_c * out_sC + h * out_sH + w * out_sW
            tl.store(out_ptr + out_offset, silu)


@triton.jit
def add_residual(y_ptr, res_ptr, out_ptr,
                 N, C, H, W,
                 y_sN, y_sC, y_sH, y_sW,
                 res_sN, res_sC, res_sH, res_sW,
                 out_sN, out_sC, out_sH, out_sW):
    total = N * C * H * W
    for i in range(0, total):
        n = i // (C * H * W)
        rem = i % (C * H * W)
        c = rem // (H * W)
        rem2 = rem % (H * W)
        h = rem2 // W
        w = rem2 % W
        y_offset = n * y_sN + c * y_sC + h * y_sH + w * y_sW
        res_offset = n * res_sN + c * res_sC + h * res_sH + w * res_sW
        out_offset = n * out_sN + c * out_sC + h * out_sH + w * out_sW
        val = tl.load(y_ptr + y_offset) + tl.load(res_ptr + res_offset)
        tl.store(out_ptr + out_offset, val)


def _run_model_triton(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
    # Ensure float32 and contiguous
    x32 = x.contiguous().to(torch.float32)
    conv1_w32 = conv1_weight.contiguous().to(torch.float32)
    conv2_w32 = conv2_weight.contiguous().to(torch.float32)
    n, c_in, h, w = x32.shape
    c_out = conv1_weight.shape[0]

    # First conv: y1 = conv3x3(x)
    y1 = torch.empty((n, c_out, h, w), device=x32.device, dtype=torch.float32)
    grid = (n, c_out, h, w)
    conv3x3_nchw_4d[grid](
        x32, conv1_w32, y1,
        n, c_in, c_out, h, w,
        x32.stride(0), x32.stride(1), x32.stride(2), x32.stride(3),
        conv1_w32.stride(0), conv1_w32.stride(1), conv1_w32.stride(2), conv1_w32.stride(3),
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        NUM_CI=c_in,
        num_warps=4
    )

    # GroupNorm on y1
    num_groups = 32
    group_size_c = c_out // num_groups
    y1_sum = torch.empty((n, num_groups, 2), device=x32.device, dtype=torch.float32)
    groupnorm_reduce_sums[(n, num_groups)](
        y1,
        y1_sum,
        n, c_out, h, w, num_groups,
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        num_warps=1
    )
    y1_norm = torch.empty_like(y1)
    groupnorm_apply_affine_silu[(n, c_out)](
        y1, y1_norm, y1_sum, norm1_weight.contiguous().to(torch.float32), norm1_bias.contiguous().to(torch.float32),
        n, c_out, h, w, num_groups, eps,
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
        num_warps=4
    )

    # Second conv: y2 = conv3x3(y1_norm)
    y2 = torch.empty((n, c_out, h, w), device=x32.device, dtype=torch.float32)
    conv3x3_nchw_4d[(n, c_out, h, w)](
        y1_norm, conv2_w32, y2,
        n, c_out, c_out, h, w,
        y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
        conv2_w32.stride(0), conv2_w32.stride(1), conv2_w32.stride(2), conv2_w32.stride(3),
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        NUM_CI=c_out,
        num_warps=4
    )

    # GroupNorm on y2
    y2_sum = torch.empty((n, num_groups, 2), device=x32.device, dtype=torch.float32)
    groupnorm_reduce_sums[(n, num_groups)](
        y2,
        y2_sum,
        n, c_out, h, w, num_groups,
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        num_warps=1
    )
    y2_norm = torch.empty_like(y2)
    groupnorm_apply_affine_silu[(n, c_out)](
        y2, y2_norm, y2_sum, norm2_weight.contiguous().to(torch.float32), norm2_bias.contiguous().to(torch.float32),
        n, c_out, h, w, num_groups, eps,
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
        num_warps=4
    )

    # Residual add: out = y2_norm + x
    out = torch.empty_like(x32)
    total = n * c_out * h * w
    add_residual[(total,)](
        y2_norm, x32, out,
        n, c_out, h, w,
        y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
        x32.stride(0), x32.stride(1), x32.stride(2), x32.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        num_warps=1
    )

    # Cast back to original dtype if needed
    if x.dtype != torch.float32:
        out = out.to(x.dtype)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        # All computation via Triton; no torch ops.
        return _run_model_triton(
            x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps
        )


def run(*args):
    return ModelNew()(*args)
