import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_4d(x_ptr, w_ptr, y_ptr,
                    N, C_in, H, W,
                    x_sN, x_sC, x_sH, x_sW,
                    w_sCo, w_sCi, w_sKh, w_sKw,
                    y_sN, y_sC, y_sH, y_sW,
                    NUM_CI: tl.constexpr, NUM_H: tl.constexpr, NUM_W: tl.constexpr,
                    num_warps: tl.constexpr):
    # Grid: (N, C_out, H, W)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 neighborhood with padding masks
    for ci in range(0, NUM_CI):
        for kh in range(0, NUM_H):
            for kw in range(0, NUM_W):
                h_in = pid_h + kh - 1  # padding=1
                w_in = pid_w + kw - 1
                h_in_valid = (h_in >= 0) & (h_in < H)
                w_in_valid = (w_in >= 0) & (w_in < W)
                if h_in_valid and w_in_valid:
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
    # Grid: (N, groups)
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
    # Grid: (N, C)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    # Compute per-group stats
    group_size_c = C // groups
    pid_g = pid_c // group_size_c

    base = pid_n * groups + pid_g
    sum_val = tl.load(sums_ptr + base * 2 + 0)
    sumsq_val = tl.load(sums_ptr + base * 2 + 1)

    HW = H * W
    mean = sum_val / HW
    var = sumsq_val / HW - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Affine parameters
    scale = tl.load(scale_ptr + pid_c)
    bias = tl.load(bias_ptr + pid_c)

    # Apply normalization and SiLU
    for h in range(0, H):
        for w in range(0, W):
            y_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            y_val = tl.load(y_ptr + y_offset)
            norm = (y_val - mean) * rstd
            z = norm * scale + bias
            # SiLU: z * sigmoid(z) where sigmoid(z) = 1 / (1 + exp(-z))
            sig = 1.0 / (1.0 + tl.exp(-z))
            out_val = z * sig
            out_offset = pid_n * out_sN + pid_c * out_sC + h * out_sH + w * out_sW
            tl.store(out_ptr + out_offset, out_val)


@triton.jit
def add_residual_kernel(y_ptr, x_ptr, out_ptr, total_elems: tl.constexpr):
    pid = tl.program_id(0)
    if pid < total_elems:
        val = tl.load(y_ptr + pid) + tl.load(x_ptr + pid)
        tl.store(out_ptr + pid, val)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        # Ensure CUDA and contiguity; compute in float32
        assert x.is_cuda, "Input tensor must be on CUDA device"
        N, C_in, H, W = x.shape
        C = C_in

        x32 = x.contiguous().to(torch.float32)
        conv1_w32 = conv1_weight.contiguous().to(torch.float32)  # (C, C, 3, 3)
        conv2_w32 = conv2_weight.contiguous().to(torch.float32)  # (C, C, 3, 3)
        norm1_w32 = norm1_weight.contiguous().to(torch.float32)  # (C,)
        norm1_b32 = norm1_bias.contiguous().to(torch.float32)    # (C,)
        norm2_w32 = norm2_weight.contiguous().to(torch.float32)  # (C,)
        norm2_b32 = norm2_bias.contiguous().to(torch.float32)    # (C,)

        groups = 32
        assert C % groups == 0, "num_groups must divide C"

        # 1) Conv1: y1
        y1 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)
        grid1 = (N, C, H, W)
        conv3x3_nchw_4d[grid1](
            x32, conv1_w32, y1,
            N, C, H, W,
            x32.stride(0), x32.stride(1), x32.stride(2), x32.stride(3),
            conv1_w32.stride(0), conv1_w32.stride(1), conv1_w32.stride(2), conv1_w32.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            NUM_CI=C, NUM_H=3, NUM_W=3, num_warps=1
        )

        # 2) GroupNorm1 + SiLU
        sums1 = torch.empty((N, groups, 2), device=x.device, dtype=torch.float32)
        grid_reduce1 = (N, groups)
        groupnorm_reduce_sums[grid_reduce1](
            y1, sums1,
            N, C, H, W, groups,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=1
        )
        y1_out = torch.empty_like(y1)
        grid_apply1 = (N, C)
        groupnorm_apply_affine_silu[grid_apply1](
            y1, y1_out, sums1, norm1_w32, norm1_b32,
            N, C, H, W, groups, eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_out.stride(0), y1_out.stride(1), y1_out.stride(2), y1_out.stride(3),
            num_warps=1
        )

        # 3) Conv2: y2
        y2 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)
        grid2 = (N, C, H, W)
        conv3x3_nchw_4d[grid2](
            y1_out, conv2_w32, y2,
            N, C, H, W,
            y1_out.stride(0), y1_out.stride(1), y1_out.stride(2), y1_out.stride(3),
            conv2_w32.stride(0), conv2_w32.stride(1), conv2_w32.stride(2), conv2_w32.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            NUM_CI=C, NUM_H=3, NUM_W=3, num_warps=1
        )

        # 4) GroupNorm2 + SiLU
        sums2 = torch.empty((N, groups, 2), device=x.device, dtype=torch.float32)
        grid_reduce2 = (N, groups)
        groupnorm_reduce_sums[grid_reduce2](
            y2, sums2,
            N, C, H, W, groups,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=1
        )
        y2_out = torch.empty_like(y2)
        grid_apply2 = (N, C)
        groupnorm_apply_affine_silu[grid_apply2](
            y2, y2_out, sums2, norm2_w32, norm2_b32,
            N, C, H, W, groups, eps,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_out.stride(0), y2_out.stride(1), y2_out.stride(2), y2_out.stride(3),
            num_warps=1
        )

        # 5) Add residual x
        total_elems = N * C * H * W
        out = torch.empty_like(y2_out)
        add_residual_kernel[(total_elems,)](
            y2_out, x32, out,
            total_elems, num_warps=1
        )

        return out


def run(*args):
    return ModelNew()(*args)
