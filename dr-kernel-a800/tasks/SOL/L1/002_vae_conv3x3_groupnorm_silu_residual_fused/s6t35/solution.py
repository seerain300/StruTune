import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_4d(x_ptr, w_ptr, y_ptr,
                    N, C_in, H, W,
                    x_sN, x_sC, x_sH, x_sW,
                    w_sCo, w_sCi, w_sKh, w_sKw,
                    y_sN, y_sC, y_sH, y_sW,
                    C_out: tl.constexpr):
    # Grid: (N, C_out, H, W)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Sum over input channels and 3x3 neighborhood
    for ci in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                h_in = pid_h + kh
                w_in = pid_w + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                if in_bounds:
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
def add_residual(y_ptr, x_ptr, out_ptr,
                 total_elems):
    # Elementwise addition: out = y + x
    pid = tl.program_id(0)
    idx = pid  # 1D grid
    if idx < total_elems:
        val = tl.load(y_ptr + idx)
        other = tl.load(x_ptr + idx)
        tl.store(out_ptr + idx, val + other)


@triton.jit
def groupnorm_reduce_sums(y_ptr, sums_ptr,
                           N, C, H, W,
                           y_sN, y_sC, y_sH, y_sW,
                           groups: tl.constexpr):
    # Grid: (N, groups). Compute per-(n, group) sum and sumsq across channels in group and all H*W.
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
def groupnorm_apply_affine_silu(y_ptr, out_ptr,
                                sums_ptr,
                                N, C, H, W,
                                y_sN, y_sC, y_sH, y_sW,
                                groups: tl.constexpr,
                                scale_ptr, bias_ptr,
                                eps):
    # Grid: (N, C). For each (n, c), iterate all H*W positions in its group, normalize, apply affine, SiLU.
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    group_size_c = C // groups
    g = pid_c // group_size_c

    sum_val = tl.load(sums_ptr + g * 2 + 0)
    sumsq_val = tl.load(sums_ptr + g * 2 + 1)
    mean = sum_val / (H * W)
    var = sumsq_val / (H * W) - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Per-channel affine parameters
    scale = tl.load(scale_ptr + pid_c)
    bias = tl.load(bias_ptr + pid_c)

    for h in range(0, H):
        for w in range(0, W):
            y_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            y_val = tl.load(y_ptr + y_offset)
            norm = (y_val - mean) * rstd
            # SiLU: norm * sigmoid(norm) = norm / (1 + exp(-norm))
            sig = 1.0 / (1.0 + tl.exp(-norm))
            silu = norm * sig
            z = silu * scale + bias
            tl.store(out_ptr + y_offset, z)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        # Ensure contiguity and dtype for Triton
        x32 = x.contiguous().to(torch.float32)
        N, C, H, W = x32.shape
        C_in = C  # input channels
        C_out = C_in  # output channels (both convs)

        # Allocate outputs
        y1 = torch.empty((N, C_out, H, W), dtype=torch.float32, device=x32.device)
        y2 = torch.empty((N, C_out, H, W), dtype=torch.float32, device=x32.device)
        out = torch.empty((N, C_out, H, W), dtype=torch.float32, device=x32.device)

        # Launch conv1: y1 = conv3x3(x)
        x_sN, x_sC, x_sH, x_sW = x32.stride()
        w1_sCo, w1_sCi, w1_sKh, w1_sKw = conv1_weight.stride()
        y1_sN, y1_sC, y1_sH, y1_sW = y1.stride()
        grid_conv1 = (N, C_out, H, W)
        conv3x3_nchw_4d[grid_conv1](
            x32, conv1_weight, y1,
            N, C_in, H, W,
            x_sN, x_sC, x_sH, x_sW,
            w1_sCo, w1_sCi, w1_sKh, w1_sKw,
            y1_sN, y1_sC, y1_sH, y1_sW,
            C_out=C_out,  # constexpr
            num_warps=1,
        )

        # GroupNorm 1: reduce
        sums1 = torch.empty((N, 32, 2), dtype=torch.float32, device=x32.device)
        y1_sN, y1_sC, y1_sH, y1_sW = y1.stride()
        grid_reduce1 = (N, 32)
        groupnorm_reduce_sums[grid_reduce1](
            y1, sums1,
            N, C_out, H, W,
            y1_sN, y1_sC, y1_sH, y1_sW,
            groups=32,
            num_warps=1,
        )

        # GroupNorm 1 apply affine + SiLU
        y1_norm = torch.empty_like(y1)
        grid_norm1 = (N, C_out)
        groupnorm_apply_affine_silu[grid_norm1](
            y1, y1_norm,
            sums1,
            N, C_out, H, W,
            y1_sN, y1_sC, y1_sH, y1_sW,
            groups=32,
            scale_ptr=norm1_weight, bias_ptr=norm1_bias,
            eps=eps,
            num_warps=1,
        )

        # Conv2: y2 = conv3x3(y1_norm)
        y1_norm_sN, y1_norm_sC, y1_norm_sH, y1_norm_sW = y1_norm.stride()
        w2_sCo, w2_sCi, w2_sKh, w2_sKw = conv2_weight.stride()
        y2_sN, y2_sC, y2_sH, y2_sW = y2.stride()
        grid_conv2 = (N, C_out, H, W)
        conv3x3_nchw_4d[grid_conv2](
            y1_norm, conv2_weight, y2,
            N, C_out, H, W,
            y1_norm_sN, y1_norm_sC, y1_norm_sH, y1_norm_sW,
            w2_sCo, w2_sCi, w2_sKh, w2_sKw,
            y2_sN, y2_sC, y2_sH, y2_sW,
            C_out=C_out,
            num_warps=1,
        )

        # GroupNorm 2: reduce
        sums2 = torch.empty((N, 32, 2), dtype=torch.float32, device=x32.device)
        y2_sN, y2_sC, y2_sH, y2_sW = y2.stride()
        grid_reduce2 = (N, 32)
        groupnorm_reduce_sums[grid_reduce2](
            y2, sums2,
            N, C_out, H, W,
            y2_sN, y2_sC, y2_sH, y2_sW,
            groups=32,
            num_warps=1,
        )

        # GroupNorm 2 apply affine + SiLU
        y2_norm = torch.empty_like(y2)
        grid_norm2 = (N, C_out)
        groupnorm_apply_affine_silu[grid_norm2](
            y2, y2_norm,
            sums2,
            N, C_out, H, W,
            y2_sN, y2_sC, y2_sH, y2_sW,
            groups=32,
            scale_ptr=norm2_weight, bias_ptr=norm2_bias,
            eps=eps,
            num_warps=1,
        )

        # Residual add: out = y2_norm + x32
        total_elems = N * C_out * H * W
        add_residual[(total_elems,)](
            y2_norm, x32, out,
            total_elems,
            num_warps=1,
        )

        return out


def run(*args):
    return ModelNew()(*args)
