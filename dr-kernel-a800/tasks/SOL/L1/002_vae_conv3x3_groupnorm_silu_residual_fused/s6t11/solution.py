import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw(x_ptr, w_ptr, y_ptr,
                 N, C_in, C_out, H, W,
                 x_sN, x_sC, x_sH, x_sW,
                 w_sCo, w_sCi, w_sKh, w_sKw,
                 y_sN, y_sC, y_sH, y_sW,
                 num_warps: tl.constexpr):
    # Each program computes one output element (n, co, h, w)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 neighborhood
    for ci in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                h_in = pid_h + kh
                w_in = pid_w + kw
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
    # Grid over (N, groups). Compute per-(n, group) sum and sumsq over all channels in group and all spatial.
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


@triton.jit
def groupnorm_apply_affine_silu(y_ptr, out_ptr, norm_weight_ptr, norm_bias_ptr, sums_ptr,
                                N, C, H, W, groups, eps,
                                y_sN, y_sC, y_sH, y_sW,
                                out_sN, out_sC, out_sH, out_sW):
    # Grid over (N * C,), each program handles one channel c for one batch n across all H*W
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    group_size_c = C // groups
    g = c // group_size_c

    total = H * W
    mean = tl.load(sums_ptr + (n * groups + g) * 2 + 0) / total
    sumsq = tl.load(sums_ptr + (n * groups + g) * 2 + 1)
    var = sumsq / total - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    scale = tl.load(norm_weight_ptr + c)
    bias = tl.load(norm_bias_ptr + c)

    for h in range(0, H):
        for w in range(0, W):
            y_offset = n * y_sN + c * y_sC + h * y_sH + w * y_sW
            val = tl.load(y_ptr + y_offset)
            norm = (val - mean) * rstd
            y_affine = norm * scale + bias
            # SiLU
            silu = y_affine * (1.0 / (1.0 + tl.exp(-y_affine)))
            out_offset = n * out_sN + c * out_sC + h * out_sH + w * out_sW
            tl.store(out_ptr + out_offset, silu)


@triton.jit
def add_residual(y_ptr, x_ptr, out_ptr, total_elements: tl.constexpr):
    pid = tl.program_id(0)
    if pid < total_elements:
        val_y = tl.load(y_ptr + pid)
        val_x = tl.load(x_ptr + pid)
        tl.store(out_ptr + pid, val_y + val_x)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        conv1_weight: torch.Tensor,
        norm1_weight: torch.Tensor,
        norm1_bias: torch.Tensor,
        conv2_weight: torch.Tensor,
        norm2_weight: torch.Tensor,
        norm2_bias: torch.Tensor,
        eps: float,
    ):
        # Ensure CUDA and contiguity; use float32 for stability
        assert x.is_cuda, "Input tensor must be on CUDA device"
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Weight tensors must be on CUDA device"
        assert norm1_weight.is_cuda and norm1_bias.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "Norm tensors must be on CUDA device"

        x32 = x.contiguous().to(torch.float32)
        conv1_w32 = conv1_weight.contiguous().to(torch.float32)
        conv2_w32 = conv2_weight.contiguous().to(torch.float32)
        norm1_w32 = norm1_weight.contiguous().to(torch.float32)
        norm1_b32 = norm1_bias.contiguous().to(torch.float32)
        norm2_w32 = norm2_weight.contiguous().to(torch.float32)
        norm2_b32 = norm2_bias.contiguous().to(torch.float32)

        N, C_in, H, W = x32.shape
        C = C_in  # Each conv uses same C_in=C_out

        # Conv1: y1 = conv3x3(x, conv1_weight)
        y1 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)
        grid1 = (N, C, H, W)
        conv3x3_nchw[grid1](
            x32, conv1_w32, y1,
            N, C_in, C, H, W,
            x32.stride(0), x32.stride(1), x32.stride(2), x32.stride(3),
            conv1_w32.stride(0), conv1_w32.stride(1), conv1_w32.stride(2), conv1_w32.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=1
        )

        # GroupNorm1 reduction
        groups = 32
        assert (C % groups) == 0, "C must be divisible by num_groups=32"
        sums1 = torch.empty((N, groups, 2), device=x.device, dtype=torch.float32)  # [sum, sumsq] per (n, group)
        grid_reduce1 = (N, groups)
        groupnorm_reduce_sums[grid_reduce1](
            y1, sums1,
            N, C, H, W, groups,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=1
        )

        # GroupNorm1 + SiLU
        y1_act = torch.empty_like(y1)
        grid_act1 = (N * C,)
        groupnorm_apply_affine_silu[grid_act1](
            y1, y1_act, norm1_w32, norm1_b32, sums1,
            N, C, H, W, groups, eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_act.stride(0), y1_act.stride(1), y1_act.stride(2), y1_act.stride(3),
            num_warps=1
        )

        # Conv2: y2 = conv3x3(y1_act, conv2_weight)
        y2 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)
        grid2 = (N, C, H, W)
        conv3x3_nchw[grid2](
            y1_act, conv2_w32, y2,
            N, C, C, H, W,
            y1_act.stride(0), y1_act.stride(1), y1_act.stride(2), y1_act.stride(3),
            conv2_w32.stride(0), conv2_w32.stride(1), conv2_w32.stride(2), conv2_w32.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=1
        )

        # GroupNorm2 reduction
        sums2 = torch.empty((N, groups, 2), device=x.device, dtype=torch.float32)
        grid_reduce2 = (N, groups)
        groupnorm_reduce_sums[grid_reduce2](
            y2, sums2,
            N, C, H, W, groups,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=1
        )

        # GroupNorm2 + SiLU
        y2_act = torch.empty_like(y2)
        grid_act2 = (N * C,)
        groupnorm_apply_affine_silu[grid_act2](
            y2, y2_act, norm2_w32, norm2_b32, sums2,
            N, C, H, W, groups, eps,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_act.stride(0), y2_act.stride(1), y2_act.stride(2), y2_act.stride(3),
            num_warps=1
        )

        # Residual add: out = y2_act + x32
        total = N * C * H * W
        out = torch.empty_like(y2_act)
        add_residual[total](y2_act, x32, out, total_elements=total, num_warps=1)

        return out


def run(*args):
    return ModelNew()(*args)
