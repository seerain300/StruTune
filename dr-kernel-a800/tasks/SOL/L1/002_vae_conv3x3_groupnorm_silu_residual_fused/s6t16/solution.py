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
    # One program computes one output element y[n, co, h, w]
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
                in_h = (h_in >= 0) & (h_in < H)
                in_w = (w_in >= 0) & (w_in < W)
                if in_h & in_w:
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
def groupnorm_apply_affine_silu(y_ptr, out_ptr,
                                scale_ptr, bias_ptr,
                                N, C, H, W,
                                sums_ptr,
                                y_sN, y_sC, y_sH, y_sW,
                                groups, eps):
    # Grid over (N, C). For each (n, c), compute group index, normalize, affine, SiLU.
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    group_size_c = C // groups
    group_id = pid_c // group_size_c

    base = pid_n * groups + group_id
    sum_val = tl.load(sums_ptr + base * 2 + 0)
    sumsq_val = tl.load(sums_ptr + base * 2 + 1)

    group_size = group_size_c
    M = H * W
    mean = sum_val / M
    var = sumsq_val / M - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    scale = tl.load(scale_ptr + pid_c)
    bias = tl.load(bias_ptr + pid_c)

    for h in range(0, H):
        for w in range(0, W):
            y_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            val = tl.load(y_ptr + y_offset)
            norm = (val - mean) * rstd
            af = norm * scale + bias
            # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
            sig = 1.0 / (1.0 + tl.exp(-af))
            out_val = af * sig
            tl.store(out_ptr + y_offset, out_val)


@triton.jit
def add_residual_kernel(y_ptr, x_ptr, out_ptr, total_elems):
    pid = tl.program_id(0)
    val_y = tl.load(y_ptr + pid)
    val_x = tl.load(x_ptr + pid)
    tl.store(out_ptr + pid, val_y + val_x)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        """
        Fused residual block:
          Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        All computation is done in Triton kernels; no torch ops in forward.
        """
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Weights must be on CUDA."

        # Ensure contiguity and dtype
        x32 = x.contiguous().to(torch.float32)
        N, C_in, H, W = x32.shape
        N1, C1, K_h1, K_w1 = conv1_weight.shape
        N2, C_out, K_h2, K_w2 = conv2_weight.shape

        # First conv: y1 = conv3x3(x)
        y1 = torch.empty((N, C1, H, W), dtype=torch.float32, device=x32.device)
        grid = (N, C1, H, W)
        conv3x3_nchw_4d[grid](
            x32, conv1_weight, y1,
            N, C_in, C1, H, W,
            x32.stride(0), x32.stride(1), x32.stride(2), x32.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            NUM_CI=C_in
        )

        # GroupNorm 1 (num_groups=32) + SiLU
        groups = 32
        y1_sN, y1_sC, y1_sH, y1_sW = y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3)
        sums1 = torch.empty((N, groups, 2), dtype=torch.float32, device=x32.device)
        grid_reduce1 = (N, groups)
        groupnorm_reduce_sums[grid_reduce1](
            y1, sums1,
            N, C1, H, W, groups,
            y1_sN, y1_sC, y1_sH, y1_sW
        )

        y1_norm = torch.empty_like(y1)
        grid_apply1 = (N, C1)
        groupnorm_apply_affine_silu[grid_apply1](
            y1, y1_norm,
            norm1_weight, norm1_bias,
            N, C1, H, W,
            sums1,
            y1_sN, y1_sC, y1_sH, y1_sW,
            groups, eps
        )

        # Second conv: y2 = conv3x3(y1_norm)
        y2 = torch.empty((N, C_out, H, W), dtype=torch.float32, device=x32.device)
        grid = (N, C_out, H, W)
        conv3x3_nchw_4d[grid](
            y1_norm, conv2_weight, y2,
            N, C1, C_out, H, W,
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            NUM_CI=C1
        )

        # GroupNorm 2 (num_groups=32) + SiLU
        y2_sN, y2_sC, y2_sH, y2_sW = y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3)
        sums2 = torch.empty((N, groups, 2), dtype=torch.float32, device=x32.device)
        grid_reduce2 = (N, groups)
        groupnorm_reduce_sums[grid_reduce2](
            y2, sums2,
            N, C_out, H, W, groups,
            y2_sN, y2_sC, y2_sH, y2_sW
        )

        y2_norm = torch.empty_like(y2)
        grid_apply2 = (N, C_out)
        groupnorm_apply_affine_silu[grid_apply2](
            y2, y2_norm,
            norm2_weight, norm2_bias,
            N, C_out, H, W,
            sums2,
            y2_sN, y2_sC, y2_sH, y2_sW,
            groups, eps
        )

        # Residual add: out = y2_norm + x32
        total_elems = N * C_out * H * W
        out = torch.empty_like(y2_norm)
        grid_add = (total_elems,)
        add_residual_kernel[grid_add](
            y2_norm, x32, out, total_elems
        )

        return out


def run(*args):
    return ModelNew()(*args)
