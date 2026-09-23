import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw(x_ptr, w_ptr, y_ptr,
                 N, C_out, C_in, H, W,
                 x_sN, x_sC, x_sH, x_sW,
                 w_sCo, w_sCi, w_sKh, w_sKw,
                 y_sN, y_sC, y_sH, y_sW,
                 NUM_CI: tl.constexpr, NUM_H: tl.constexpr, NUM_W: tl.constexpr):
    # Grid = (N, C_out, H, W)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 neighborhood with padding mask
    for ci in range(0, NUM_CI):
        for kh in range(0, 3):
            for kw in range(0, 3):
                h_in = pid_h + kh
                w_in = pid_w + kw
                in_range = (h_in >= 0) & (h_in < NUM_H) & (w_in >= 0) & (w_in < NUM_W)
                if in_range:
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
                           y_sN, y_sC, y_sH, y_sW,
                           NUM_C: tl.constexpr, NUM_H: tl.constexpr, NUM_W: tl.constexpr):
    # Grid = (N, groups). Compute per-(n, group) sum and sumsq over channels in group and all spatial.
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_size_c = NUM_C // groups
    start_c = pid_g * group_size_c

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    for c in range(start_c, start_c + group_size_c):
        for h in range(0, NUM_H):
            for w in range(0, NUM_W):
                y_offset = pid_n * y_sN + c * y_sC + h * y_sH + w * y_sW
                val = tl.load(y_ptr + y_offset)
                sum_val += val
                sumsq_val += val * val

    base = pid_n * groups + pid_g
    # sums_ptr has shape (N, groups, 2), store [sum, sumsq] at [n, g, :]
    tl.store(sums_ptr + base * 2 + 0, sum_val)
    tl.store(sums_ptr + base * 2 + 1, sumsq_val)


@triton.jit
def groupnorm_apply_affine_silu(y_ptr, sums_ptr, norm_w_ptr, norm_b_ptr, out_ptr,
                                N, C, H, W, groups, eps,
                                y_sN, y_sC, y_sH, y_sW,
                                out_sN, out_sC, out_sH, out_sW,
                                NUM_C: tl.constexpr, NUM_H: tl.constexpr, NUM_W: tl.constexpr):
    # Grid = (N, C). For each (n, c), normalize over group and spatial, apply affine + SiLU.
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    group_size_c = NUM_C // groups
    g = pid_c // group_size_c  # group index for this channel

    base = pid_n * groups + g
    sum_val = tl.load(sums_ptr + base * 2 + 0)
    sumsq_val = tl.load(sums_ptr + base * 2 + 1)

    M = NUM_H * NUM_W
    mean = sum_val / M
    var = sumsq_val / M - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Per-channel affine parameters
    gamma = tl.load(norm_w_ptr + pid_c)
    beta = tl.load(norm_b_ptr + pid_c)

    for h in range(0, NUM_H):
        for w in range(0, NUM_W):
            y_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            val = tl.load(y_ptr + y_offset)
            norm_val = (val - mean) * rstd
            norm_val = norm_val * gamma + beta
            # SiLU: x * sigmoid(x)
            sig = 1.0 / (1.0 + tl.exp(-norm_val))
            out_val = norm_val * sig

            out_offset = pid_n * out_sN + pid_c * out_sC + h * out_sH + w * out_sW
            tl.store(out_ptr + out_offset, out_val)


@triton.jit
def add_residual_kernel(y_ptr, x_ptr, out_ptr,
                         total_elems: tl.constexpr):
    # Elementwise addition over flat array
    pid = tl.program_id(0)
    if pid < total_elems:
        y_val = tl.load(y_ptr + pid)
        x_val = tl.load(x_ptr + pid)
        tl.store(out_ptr + pid, y_val + x_val)


class ModelNew(torch.nn.Module):
    def forward(self,
                x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        # Ensure CUDA and contiguous; compute in float32 for stability
        assert x.is_cuda, "Input tensor must be on CUDA"
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Weight tensors must be on CUDA"
        assert norm1_weight.is_cuda and norm1_bias.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "Norm tensors must be on CUDA"

        N, C_in, H, W = x.shape
        C = C_in  # conv uses same in/out channels
        groups = 32
        assert (C % groups) == 0, "C must be divisible by num_groups=32"

        x32 = x.contiguous().to(torch.float32)
        conv1_w32 = conv1_weight.contiguous().to(torch.float32)
        conv2_w32 = conv2_weight.contiguous().to(torch.float32)
        norm1_w32 = norm1_weight.contiguous().to(torch.float32)
        norm1_b32 = norm1_bias.contiguous().to(torch.float32)
        norm2_w32 = norm2_weight.contiguous().to(torch.float32)
        norm2_b32 = norm2_bias.contiguous().to(torch.float32)

        # Conv 1: y1
        y1 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)
        grid1 = (N, C, H, W)
        conv3x3_nchw[grid1](
            x32, conv1_w32, y1,
            N, C, C_in, H, W,
            x32.stride(0), x32.stride(1), x32.stride(2), x32.stride(3),
            conv1_w32.stride(0), conv1_w32.stride(1), conv1_w32.stride(2), conv1_w32.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            C_in, H, W,
            num_warps=1
        )

        # GroupNorm1 reduction: sums[n, g, 2] for g in [0, groups)
        sums1 = torch.empty((N, groups, 2), device=x.device, dtype=torch.float32)
        grid_reduce1 = (N, groups)
        groupnorm_reduce_sums[grid_reduce1](
            y1, sums1,
            N, C, H, W, groups,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            C, H, W,
            num_warps=1
        )

        # Apply GroupNorm1 + SiLU1
        y1_out = torch.empty_like(y1)
        grid_apply1 = (N, C)
        groupnorm_apply_affine_silu[grid_apply1](
            y1, sums1, norm1_w32, norm1_b32, y1_out,
            N, C, H, W, groups, eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_out.stride(0), y1_out.stride(1), y1_out.stride(2), y1_out.stride(3),
            C, H, W,
            num_warps=1
        )

        # Conv 2: y2
        y2 = torch.empty((N, C, H, W), device=x.device, dtype=torch.float32)
        grid2 = (N, C, H, W)
        conv3x3_nchw[grid2](
            y1_out, conv2_w32, y2,
            N, C, C_in, H, W,
            y1_out.stride(0), y1_out.stride(1), y1_out.stride(2), y1_out.stride(3),
            conv2_w32.stride(0), conv2_w32.stride(1), conv2_w32.stride(2), conv2_w32.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            C_in, H, W,
            num_warps=1
        )

        # GroupNorm2 reduction
        sums2 = torch.empty((N, groups, 2), device=x.device, dtype=torch.float32)
        grid_reduce2 = (N, groups)
        groupnorm_reduce_sums[grid_reduce2](
            y2, sums2,
            N, C, H, W, groups,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            C, H, W,
            num_warps=1
        )

        # Apply GroupNorm2 + SiLU2
        y2_out = torch.empty_like(y2)
        grid_apply2 = (N, C)
        groupnorm_apply_affine_silu[grid_apply2](
            y2, sums2, norm2_w32, norm2_b32, y2_out,
            N, C, H, W, groups, eps,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_out.stride(0), y2_out.stride(1), y2_out.stride(2), y2_out.stride(3),
            C, H, W,
            num_warps=1
        )

        # Residual add: out = y2_out + x
        total_elems = N * C * H * W
        out = torch.empty_like(y2_out)
        add_residual_kernel[(total_elems,)](
            y2_out, x32, out,
            total_elems,
            num_warps=1
        )

        return out


def run(*args):
    return ModelNew()(*args)
