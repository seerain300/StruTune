import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_4d(x_ptr, w_ptr, y_ptr,
                    N, C_in, H, W,
                    x_sN, x_sC, x_sH, x_sW,
                    w_sCo, w_sCi, w_sKh, w_sKw,
                    y_sN, y_sC, y_sH, y_sW,
                    BLOCK_CIN: tl.constexpr):
    # One program computes one output element y[n, co, h, w]
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    for ci in range(0, BLOCK_CIN):
        for kh in range(0, 3):
            for kw in range(0, 3):
                h_in = pid_h + kh
                w_in = pid_w + kw
                in_h = (h_in >= 0) & (h_in < H)
                in_w = (w_in >= 0) & (w_in < W)
                if in_h and in_w:
                    x_offset = pid_n * x_sN + ci * x_sC + h_in * x_sH + w_in * x_sW
                    x_val = tl.load(x_ptr + x_offset)
                else:
                    x_val = 0.0

                w_offset = pid_co * w_sCo + ci * w_sCi + kh * w_sKh + kw * w_sKw
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    y_offset = pid_n * y_sN + pid_co * y_sC + pid_h * y_sH + pid_w * y_sW
    tl.store(y_ptr + y_offset, acc)


@triton.jit
def groupnorm_reduce_sums(y_ptr, sums_ptr,
                           N, C, H, W, groups,
                           y_sN, y_sC, y_sH, y_sW):
    # Grid over (N, groups). Compute per-(n, group) sum and sumsq.
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
def groupnorm_apply_affine_silu(y_ptr, sums_ptr, scale_ptr, bias_ptr, out_ptr,
                                N, C, H, W, groups,
                                y_sN, y_sC, y_sH, y_sW,
                                out_sN, out_sC, out_sH, out_sW):
    # Grid over (N, C). Compute normalized + affine + SiLU for each channel across H*W
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    group_size_c = C // groups
    g = pid_c // group_size_c

    # Load sum and sumsq for this (n, group)
    base = pid_n * groups + g
    sum_val = tl.load(sums_ptr + base * 2 + 0)
    sumsq_val = tl.load(sums_ptr + base * 2 + 1)
    m = H * W
    mean = sum_val / m
    var = sumsq_val / m - mean * mean
    rstd = tl.math.rsqrt(var + 1e-5)

    gamma = tl.load(scale_ptr + pid_c)  # norm weight (per-channel)
    beta = tl.load(bias_ptr + pid_c)    # norm bias (per-channel)

    for h in range(0, H):
        for w in range(0, W):
            y_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            y_val = tl.load(y_ptr + y_offset)
            norm = (y_val - mean) * rstd
            out_val = norm * gamma + beta
            # SiLU: x * sigmoid(x)
            sig = 1.0 / (1.0 + tl.exp(-out_val))
            out_val = out_val * sig

            out_offset = pid_n * out_sN + pid_c * out_sC + h * out_sH + w * out_sW
            tl.store(out_ptr + out_offset, out_val)


@triton.jit
def add_residual_kernel(y_ptr, x_ptr, out_ptr, total_elems: tl.constexpr):
    # Elementwise add over a 1D range
    pid = tl.program_id(0)
    if pid < total_elems:
        y_val = tl.load(y_ptr + pid)
        x_val = tl.load(x_ptr + pid)
        tl.store(out_ptr + pid, y_val + x_val)


def _triton_conv3x3(x, w, out):
    # x: (N, C_in, H, W), w: (C_out, C_in, 3, 3), out: (N, C_out, H, W)
    assert x.is_cuda and w.is_cuda and out.is_cuda
    N, C_in, H, W = x.shape
    C_out = w.shape[0]

    x32 = x.contiguous().to(torch.float32)
    w32 = w.contiguous().to(torch.float32)
    out32 = out.contiguous().to(torch.float32)

    grid = (N, C_out, H, W)
    conv3x3_nchw_4d[grid](
        x32, w32, out32,
        N, C_in, H, W,
        x32.stride(0), x32.stride(1), x32.stride(2), x32.stride(3),
        w32.stride(0), w32.stride(1), w32.stride(2), w32.stride(3),
        out32.stride(0), out32.stride(1), out32.stride(2), out32.stride(3),
        BLOCK_CIN=C_in,
        num_warps=1,
    )
    return out32


class ModelNew(torch.nn.Module):
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        # x: (N, C_in, H, W), conv weights: (C_out, C_in, 3, 3)
        N, C_in, H, W = x.shape

        # First path: Conv3x3 -> GroupNorm -> SiLU
        y1 = torch.empty((N, C_in, H, W), device=x.device, dtype=torch.float32)
        y1 = _triton_conv3x3(x, conv1_weight, y1)

        groups = 32
        assert C_in % groups == 0, "Channel count must be divisible by num_groups (32)"
        sums1 = torch.empty((N, groups, 2), device=x.device, dtype=torch.float32)

        # GroupNorm reduction on y1
        groupnorm_reduce_sums[(N, groups)](
            y1, sums1,
            N, C_in, H, W, groups,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=1,
        )

        out1 = torch.empty_like(y1)
        # GroupNorm apply + affine + SiLU
        groupnorm_apply_affine_silu[(N, C_in)](
            y1, sums1, norm1_weight, norm1_bias, out1,
            N, C_in, H, W, groups,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
            num_warps=1,
        )

        # Second path: Conv3x3 -> GroupNorm -> SiLU
        y2 = torch.empty((N, C_in, H, W), device=x.device, dtype=torch.float32)
        y2 = _triton_conv3x3(out1, conv2_weight, y2)

        sums2 = torch.empty((N, groups, 2), device=x.device, dtype=torch.float32)
        groupnorm_reduce_sums[(N, groups)](
            y2, sums2,
            N, C_in, H, W, groups,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=1,
        )

        out2 = torch.empty_like(y2)
        groupnorm_apply_affine_silu[(N, C_in)](
            y2, sums2, norm2_weight, norm2_bias, out2,
            N, C_in, H, W, groups,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
            num_warps=1,
        )

        # Residual add: out2 + x (cast x to float32 for add)
        out = torch.empty_like(out2)
        total_elems = N * C_in * H * W
        add_residual_kernel[(total_elems,)](
            out2, x.contiguous().to(torch.float32), out,
            total_elems,
            num_warps=1,
        )
        return out


def run(*args):
    return ModelNew()(*args)
