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
    # One program per output element: (n, co, h, w)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 neighborhood
    for ci in range(NUM_CI):
        for kh in range(3):
            for kw in range(3):
                h_in = pid_h + kh
                w_in = pid_w + kw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                x_offset = pid_n * x_sN + ci * x_sC + h_in * x_sH + w_in * x_sW
                w_offset = pid_co * w_sCo + ci * w_sCi + kh * w_sKh + kw * w_sKw
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    y_offset = pid_n * y_sN + pid_co * y_sC + pid_h * y_sH + pid_w * y_sW
    tl.store(y_ptr + y_offset, acc)


@triton.jit
def groupnorm_reduce_sums(y_ptr, sums_ptr,
                           N, C, H, W, groups,
                           y_sN, y_sC, y_sH, y_sW):
    # Grid: (N, groups). Compute per-(n, group) sum and sumsq across channels in the group and all spatial positions.
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_size_c = C // groups
    start_c = pid_g * group_size_c

    sum_val = 0.0
    sumsq_val = 0.0

    # Loop over channels in this group
    for ci in range(start_c, start_c + group_size_c):
        # Loop over all spatial positions
        for h in range(0, H):
            for w in range(0, W):
                y_offset = pid_n * y_sN + ci * y_sC + h * y_sH + w * y_sW
                val = tl.load(y_ptr + y_offset)
                sum_val += val
                sumsq_val += val * val

    base = pid_n * groups + pid_g
    tl.store(sums_ptr + base * 2 + 0, sum_val)
    tl.store(sums_ptr + base * 2 + 1, sumsq_val)


@triton.jit
def groupnorm_apply_affine_silu(y_ptr, out_ptr, sums_ptr, scale_ptr, bias_ptr,
                                N, C, H, W, groups,
                                y_sN, y_sC, y_sH, y_sW):
    # Grid: (N, C). Each program handles one channel across all H*W, uses precomputed sums to normalize and apply affine + SiLU.
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    group_size_c = C // groups
    g = pid_c // group_size_c

    # Load per-(n, group) sums
    base = pid_n * groups + g
    sum_val = tl.load(sums_ptr + base * 2 + 0)
    sumsq_val = tl.load(sums_ptr + base * 2 + 1)
    hw = H * W
    mean = sum_val / hw
    var = sumsq_val / hw - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)  # eps equivalent

    # Per-channel affine params
    scale = tl.load(scale_ptr + pid_c)
    bias = tl.load(bias_ptr + pid_c)

    # Loop over spatial positions and apply normalized affine + SiLU
    for h in range(0, H):
        for w in range(0, W):
            y_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            val = tl.load(y_ptr + y_offset)
            norm = (val - mean) * rstd
            z = norm * scale + bias
            # SiLU: z * sigmoid(z) = z / (1 + exp(-z))
            sig = 1.0 / (1.0 + tl.exp(-z))
            out_val = z * sig
            out_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            tl.store(out_ptr + out_offset, out_val)


@triton.jit
def add_residual_kernel(y_ptr, x_ptr, out_ptr, total_elems):
    # 1D grid over total elements; elementwise add
    pid = tl.program_id(0)
    if pid < total_elems:
        val_y = tl.load(y_ptr + pid)
        val_x = tl.load(x_ptr + pid)
        tl.store(out_ptr + pid, val_y + val_x)


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
        # Ensure all tensors are contiguous and float32
        N, C, H, W = x.shape
        C_in = C
        # First conv: y1
        x32 = x.contiguous().to(torch.float32)
        w1 = conv1_weight.contiguous().to(torch.float32)
        y1 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)

        grid_conv = (N, C, H, W)
        conv3x3_nchw_4d[grid_conv](
            x32, w1, y1,
            N, C_in, C, H, W,
            x32.stride(0), x32.stride(1), x32.stride(2), x32.stride(3),
            w1.stride(0), w1.stride(1), w1.stride(2), w1.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            NUM_CI=C_in,  # compile-time constant channels
            num_warps=1
        )

        # GroupNorm1 (num_groups=32) + SiLU
        num_groups = 32
        groups = num_groups
        group_size_c = C // groups

        # Reduce sums and sumsq per (n, group)
        sums = torch.empty((N, groups, 2), device=x.device, dtype=torch.float32)
        grid_reduce = (N, groups)
        groupnorm_reduce_sums[grid_reduce](
            y1, sums,
            N, C, H, W, groups,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=1
        )

        # Apply affine + SiLU
        y1_out = torch.empty_like(y1)
        grid_apply = (N, C)
        groupnorm_apply_affine_silu[grid_apply](
            y1, y1_out, sums, norm1_weight, norm1_bias,
            N, C, H, W, groups,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=1
        )

        # Second conv: y2
        w2 = conv2_weight.contiguous().to(torch.float32)
        y2 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)

        conv3x3_nchw_4d[grid_conv](
            y1_out, w2, y2,
            N, C, C, H, W,
            y1_out.stride(0), y1_out.stride(1), y1_out.stride(2), y1_out.stride(3),
            w2.stride(0), w2.stride(1), w2.stride(2), w2.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            NUM_CI=C,  # output channels of second conv
            num_warps=1
        )

        # GroupNorm2 + SiLU
        sums2 = torch.empty((N, groups, 2), device=x.device, dtype=torch.float32)
        groupnorm_reduce_sums[grid_reduce](
            y2, sums2,
            N, C, H, W, groups,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=1
        )

        y2_out = torch.empty_like(y2)
        groupnorm_apply_affine_silu[grid_apply](
            y2, y2_out, sums2, norm2_weight, norm2_bias,
            N, C, H, W, groups,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=1
        )

        # Residual add: y2_out + x
        total_elems = N * C * H * W
        out = torch.empty_like(y2_out)
        add_residual_kernel[(total_elems,)](
            y2_out, x32, out, total_elems, num_warps=1
        )

        return out


def run(*args):
    return ModelNew()(*args)
