import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_4d(x_ptr, w_ptr, y_ptr,
                    N, C_in, H, W,
                    x_sN, x_sC, x_sH, x_sW,
                    w_sCo, w_sCi, w_sKh, w_sKw,
                    y_sN, y_sC, y_sH, y_sW,
                    C_out: tl.constexpr,
                    NUM_CI: tl.constexpr):
    # Grid: (N, C_out, H, W)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 neighborhood with padding mask
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
    # Grid: (N, groups)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_size_c = C // groups
    start_c = pid_g * group_size_c

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Loop over channels in this group and all spatial positions
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
                                N, C, H, W,
                                y_sN, y_sC, y_sH, y_sW,
                                out_sN, out_sC, out_sH, out_sW):
    # Grid: (N, C)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    group_size_c = C // 32  # num_groups is fixed at 32
    g = pid_c // group_size_c

    sum_val = tl.load(sums_ptr + (pid_n * 32 + g) * 2 + 0)
    sumsq_val = tl.load(sums_ptr + (pid_n * 32 + g) * 2 + 1)
    numel = H * W
    mean = sum_val / numel
    var = sumsq_val / numel - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    # Affine parameters per channel
    scale = tl.load(scale_ptr + pid_c)
    bias = tl.load(bias_ptr + pid_c)

    for h in range(0, H):
        for w in range(0, W):
            y_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            y_val = tl.load(y_ptr + y_offset)
            norm = (y_val - mean) * rstd
            z = norm * scale + bias
            # SiLU: z * sigmoid(z) = z / (1 + exp(-z))
            sig = 1.0 / (1.0 + tl.exp(-z))
            out_val = z * sig
            out_offset = pid_n * out_sN + pid_c * out_sC + h * out_sH + w * out_sW
            tl.store(out_ptr + out_offset, out_val)


@triton.jit
def add_residual_kernel(y_ptr, x_ptr, out_ptr, total_elems):
    # Elementwise add over a flat array
    pid = tl.program_id(0)
    val = tl.load(y_ptr + pid)
    val += tl.load(x_ptr + pid)
    tl.store(out_ptr + pid, val)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        """
        Triton-optimized fused residual block:
        Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        All computations performed by Triton kernels.
        """
        # Ensure contiguity and float32 for stability
        x32 = x.contiguous().to(torch.float32)
        C = x32.shape[1]
        H = x32.shape[2]
        W = x32.shape[3]
        N = x32.shape[0]

        # Prepare weight tensors as float32 and contiguous
        conv1_w = conv1_weight.contiguous().to(torch.float32)
        conv2_w = conv2_weight.contiguous().to(torch.float32)
        norm1_scale = norm1_weight.contiguous().to(torch.float32)
        norm1_bias = norm1_bias.contiguous().to(torch.float32)
        norm2_scale = norm2_weight.contiguous().to(torch.float32)
        norm2_bias = norm2_bias.contiguous().to(torch.float32)

        # Output buffers for convs
        y1_out = torch.empty((N, C, H, W), dtype=torch.float32, device=x32.device)
        y2_out = torch.empty((N, C, H, W), dtype=torch.float32, device=x32.device)

        # Launch conv1: F.conv2d(x, conv1_weight) -> y1_out
        grid1 = (N, C, H, W)
        conv3x3_nchw_4d[grid1](
            x32, conv1_w, y1_out,
            N, C, H, W,
            x32.stride(0), x32.stride(1), x32.stride(2), x32.stride(3),
            conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2), conv1_w.stride(3),
            y1_out.stride(0), y1_out.stride(1), y1_out.stride(2), y1_out.stride(3),
            C_out=C, NUM_CI=conv1_w.shape[1],
            num_warps=1
        )

        # GroupNorm and SiLU for y1_out
        groups = 32
        group_size_c = C // groups
        sums1 = torch.empty((N, groups, 2), dtype=torch.float32, device=x32.device)
        # Reduce: sums1[n, g, 0]=sum, sums1[n, g, 1]=sumsq
        grid_reduce1 = (N, groups)
        groupnorm_reduce_sums[grid_reduce1](
            y1_out, sums1,
            N, C, H, W, groups,
            y1_out.stride(0), y1_out.stride(1), y1_out.stride(2), y1_out.stride(3),
            num_warps=1
        )

        y1_norm = torch.empty_like(y1_out)
        grid_apply1 = (N, C)
        groupnorm_apply_affine_silu[grid_apply1](
            y1_out, sums1, norm1_scale, norm1_bias, y1_norm,
            N, C, H, W,
            y1_out.stride(0), y1_out.stride(1), y1_out.stride(2), y1_out.stride(3),
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            num_warps=1
        )

        # Launch conv2: F.conv2d(y1_norm, conv2_weight) -> y2_out
        grid2 = (N, C, H, W)
        conv3x3_nchw_4d[grid2](
            y1_norm, conv2_w, y2_out,
            N, C, H, W,
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2), conv2_w.stride(3),
            y2_out.stride(0), y2_out.stride(1), y2_out.stride(2), y2_out.stride(3),
            C_out=C, NUM_CI=conv2_w.shape[1],
            num_warps=1
        )

        # GroupNorm and SiLU for y2_out
        sums2 = torch.empty((N, groups, 2), dtype=torch.float32, device=x32.device)
        grid_reduce2 = (N, groups)
        groupnorm_reduce_sums[grid_reduce2](
            y2_out, sums2,
            N, C, H, W, groups,
            y2_out.stride(0), y2_out.stride(1), y2_out.stride(2), y2_out.stride(3),
            num_warps=1
        )

        y2_norm = torch.empty_like(y2_out)
        grid_apply2 = (N, C)
        groupnorm_apply_affine_silu[grid_apply2](
            y2_out, sums2, norm2_scale, norm2_bias, y2_norm,
            N, C, H, W,
            y2_out.stride(0), y2_out.stride(1), y2_out.stride(2), y2_out.stride(3),
            y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
            num_warps=1
        )

        # Residual add: y2_norm + x
        total_elems = N * C * H * W
        out = torch.empty_like(y2_norm)
        add_residual_kernel[(total_elems,)](
            y2_norm, x32, out,
            total_elems,
            num_warps=1
        )

        return out


def run(*args):
    return ModelNew()(*args)
