import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 Conv (N, C_in, H, W) -> (N, C_out, H, W), stride=1, padding=1
@triton.jit
def conv3x3_nchw_4d(x_ptr, w_ptr, y_ptr,
                    N, C_in, C_out, H, W,
                    x_sN, x_sC, x_sH, x_sW,
                    w_sCo, w_sCi, w_sKh, w_sKw,
                    y_sN, y_sC, y_sH, y_sW,
                    NUM_CI: tl.constexpr, NUM_H: tl.constexpr, NUM_W: tl.constexpr):
    # program ids: one program computes y[n, co, h, w]
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # sum over input channels and 3x3 neighborhood
    for ci in range(0, NUM_CI):
        for kh in range(0, NUM_H):
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


# Triton kernel: GroupNorm reduction per (n, group) -> sums[sum, sumsq]
@triton.jit
def groupnorm_reduce_sums(y_ptr, sums_ptr,
                           N, C, H, W, groups,
                           y_sN, y_sC, y_sH, y_sW):
    # grid: (N, groups)
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


# Triton kernel: GroupNorm apply + affine + SiLU for each (n, c)
@triton.jit
def groupnorm_apply_affine_silu(y_ptr, sums_ptr, scale_ptr, bias_ptr, out_ptr,
                                N, C, H, W, groups,
                                y_sN, y_sC, y_sH, y_sW,
                                scale_s, bias_s):
    # Grid: (N, C)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    group_size_c = C // groups
    g = pid_c // group_size_c  # which group this channel belongs to

    # load sums for this (n, group)
    base = pid_n * groups + g
    sum_val = tl.load(sums_ptr + base * 2 + 0)
    sumsq_val = tl.load(sums_ptr + base * 2 + 1)

    M = H * W
    mean = sum_val / M
    var = sumsq_val / M - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    # per-channel affine parameters
    scale = tl.load(scale_ptr + pid_c * scale_s)
    bias = tl.load(bias_ptr + pid_c * bias_s)

    # loop over all spatial positions
    for h in range(0, H):
        for w in range(0, W):
            y_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            val = tl.load(y_ptr + y_offset)
            norm = (val - mean) * rstd
            norm = norm * scale + bias
            # SiLU: x * sigmoid(x)
            sig = 1.0 / (1.0 + tl.exp(-norm))
            out = norm * sig
            tl.store(out_ptr + y_offset, out)


# Triton kernel: elementwise add: out = y + x (assumes contiguous and float32)
@triton.jit
def add_residual_kernel(y_ptr, x_ptr, out_ptr,
                         total_elems,
                         num_warps=1):
    pid = tl.program_id(0)
    offset = pid
    y_val = tl.load(y_ptr + offset)
    x_val = tl.load(x_ptr + offset)
    tl.store(out_ptr + offset, y_val + x_val)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor):
        # Ensure inputs/weights are contiguous and float32
        x32 = x.contiguous().to(torch.float32)

        # First conv: y1 = conv3x3(x, conv1_weight)
        N, C_in, H, W = x32.shape
        C_out1 = conv1_weight.shape[0]
        y1 = torch.empty((N, C_out1, H, W), dtype=torch.float32, device=x32.device)
        conv3x3_nchw_4d[(N, C_out1, H, W)](
            x32, conv1_weight, y1,
            N, C_in, C_out1, H, W,
            x32.stride(0), x32.stride(1), x32.stride(2), x32.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            NUM_CI=C_in, NUM_H=3, NUM_W=3,
            num_warps=1
        )

        # GroupNorm and SiLU for y1 (num_groups=32)
        groups = 32
        group_size_c1 = C_out1 // groups
        sums1 = torch.empty((N, groups, 2), dtype=torch.float32, device=x32.device)
        groupnorm_reduce_sums[(N, groups)](
            y1, sums1,
            N, C_out1, H, W, groups,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=1
        )
        y1_out = torch.empty_like(y1)
        groupnorm_apply_affine_silu[(N, C_out1)](
            y1, sums1, norm1_weight, norm1_bias, y1_out,
            N, C_out1, H, W, groups,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            norm1_weight.stride(0), norm1_bias.stride(0),
            num_warps=1
        )

        # Second conv: y2 = conv3x3(y1_out, conv2_weight)
        N2, C_in2, H2, W2 = y1_out.shape
        C_out2 = conv2_weight.shape[0]
        y2 = torch.empty((N, C_out2, H, W), dtype=torch.float32, device=x32.device)
        conv3x3_nchw_4d[(N, C_out2, H, W)](
            y1_out, conv2_weight, y2,
            N, C_in2, C_out2, H, W,
            y1_out.stride(0), y1_out.stride(1), y1_out.stride(2), y1_out.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            NUM_CI=C_in2, NUM_H=3, NUM_W=3,
            num_warps=1
        )

        # GroupNorm and SiLU for y2 (num_groups=32)
        groups2 = 32
        group_size_c2 = C_out2 // groups2
        sums2 = torch.empty((N, groups2, 2), dtype=torch.float32, device=x32.device)
        groupnorm_reduce_sums[(N, groups2)](
            y2, sums2,
            N, C_out2, H, W, groups2,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=1
        )
        y2_out = torch.empty_like(y2)
        groupnorm_apply_affine_silu[(N, C_out2)](
            y2, sums2, norm2_weight, norm2_bias, y2_out,
            N, C_out2, H, W, groups2,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            norm2_weight.stride(0), norm2_bias.stride(0),
            num_warps=1
        )

        # Residual add: out = y2_out + x
        out = torch.empty_like(y2_out)
        total_elems = N * C_out2 * H * W
        add_residual_kernel[(total_elems,)](
            y2_out, x32, out,
            total_elems,
            num_warps=1
        )

        return out


def run(*args):
    return ModelNew()(*args)
