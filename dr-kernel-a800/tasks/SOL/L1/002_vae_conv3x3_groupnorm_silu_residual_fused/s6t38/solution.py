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

    # Loop over input channels and 3x3 neighborhood with padding masks
    for ci in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                h_in = pid_h + kh
                w_in = pid_w + kw
                h_in_valid = (h_in >= 0) & (h_in < H)
                w_in_valid = (w_in >= 0) & (w_in < W)

                x_offset = pid_n * x_sN + ci * x_sC + h_in * x_sH + w_in * x_sW
                x_val = tl.load(x_ptr + x_offset, mask=h_in_valid & w_in_valid, other=0.0)

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

    group_size = C // groups
    start_c = pid_g * group_size

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    for c in range(start_c, start_c + group_size):
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
                                N, C, H, W, eps,
                                sums_ptr,
                                y_sN, y_sC, y_sH, y_sW,
                                out_sN, out_sC, out_sH, out_sW,
                                norm_weight_ptr, norm_bias_ptr):
    # Grid over (N, C). For each (n, c), loop over H*W and apply GroupNorm + affine + SiLU.
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    group_size = C // 32  # num_groups is 32 in the original code
    g = pid_c // group_size

    # Load sum and sumsq for this (n, group)
    sum_val = tl.load(sums_ptr + (pid_n * 32 + g) * 2 + 0)
    sumsq_val = tl.load(sums_ptr + (pid_n * 32 + g) * 2 + 1)

    M = H * W
    mean = sum_val / M
    var = sumsq_val / M - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    gamma = tl.load(norm_weight_ptr + pid_c)  # per-channel scale
    beta = tl.load(norm_bias_ptr + pid_c)     # per-channel bias

    for h in range(0, H):
        for w in range(0, W):
            y_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            val = tl.load(y_ptr + y_offset)
            norm = (val - mean) * rstd
            activated = norm * gamma + beta
            # SiLU: x * sigmoid(x) where sigmoid(x) = 1 / (1 + exp(-x))
            sig = 1.0 / (1.0 + tl.exp(-activated))
            silu = activated * sig
            out_offset = pid_n * out_sN + pid_c * out_sC + h * out_sH + w * out_sW
            tl.store(out_ptr + out_offset, silu)


@triton.jit
def add_residual(out_ptr, x_ptr, y_ptr,
                 total_elems: tl.constexpr):
    # Elementwise add: y = out + x
    pid = tl.program_id(0)
    # Each program handles one element; total_elems is passed as constexpr so grid = (total_elems,)
    out_val = tl.load(out_ptr + pid)
    x_val = tl.load(x_ptr + pid)
    y_val = out_val + x_val
    tl.store(y_ptr + pid, y_val)


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
        # Ensure float32 and contiguous for Triton
        x32 = x.contiguous().to(torch.float32)
        conv1_w = conv1_weight.contiguous().to(torch.float32)
        conv2_w = conv2_weight.contiguous().to(torch.float32)

        N, C_in, H, W = x32.shape
        C_out = conv1_w.shape[0]  # equals C_in for standard Conv3x3

        # First conv: y1
        y1 = torch.empty((N, C_out, H, W), device=x32.device, dtype=torch.float32)
        conv3x3_nchw_4d[(N, C_out, H, W)](
            x32, conv1_w, y1,
            N, C_in, H, W,
            x32.stride(0), x32.stride(1), x32.stride(2), x32.stride(3),
            conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2), conv1_w.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            C_out=C_out,
            num_warps=1
        )

        # GroupNorm and SiLU after conv1
        y1_gn_sums = torch.empty((N, 32, 2), device=x32.device, dtype=torch.float32)
        groupnorm_reduce_sums[(N, 32)](
            y1,
            y1_gn_sums,
            N, C_out, H, W, 32,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=1
        )

        y1_after_gn = torch.empty_like(y1)
        groupnorm_apply_affine_silu[(N, C_out)](
            y1, y1_after_gn,
            N, C_out, H, W, eps,
            y1_gn_sums,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_after_gn.stride(0), y1_after_gn.stride(1), y1_after_gn.stride(2), y1_after_gn.stride(3),
            norm1_weight.contiguous().to(torch.float32), norm1_bias.contiguous().to(torch.float32),
            num_warps=1
        )

        # Second conv: y2
        y2 = torch.empty((N, C_out, H, W), device=x32.device, dtype=torch.float32)
        conv3x3_nchw_4d[(N, C_out, H, W)](
            y1_after_gn, conv2_w, y2,
            N, C_out, H, W,
            y1_after_gn.stride(0), y1_after_gn.stride(1), y1_after_gn.stride(2), y1_after_gn.stride(3),
            conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2), conv2_w.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            C_out=C_out,
            num_warps=1
        )

        # GroupNorm and SiLU after conv2
        y2_gn_sums = torch.empty((N, 32, 2), device=x32.device, dtype=torch.float32)
        groupnorm_reduce_sums[(N, 32)](
            y2,
            y2_gn_sums,
            N, C_out, H, W, 32,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=1
        )

        y2_after_gn = torch.empty_like(y2)
        groupnorm_apply_affine_silu[(N, C_out)](
            y2, y2_after_gn,
            N, C_out, H, W, eps,
            y2_gn_sums,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_after_gn.stride(0), y2_after_gn.stride(1), y2_after_gn.stride(2), y2_after_gn.stride(3),
            norm2_weight.contiguous().to(torch.float32), norm2_bias.contiguous().to(torch.float32),
            num_warps=1
        )

        # Residual add: out = y2_after_gn + x32
        total_elems = N * C_out * H * W
        out = torch.empty_like(y2_after_gn)
        add_residual[(total_elems,)](
            y2_after_gn, x32, out,
            total_elems,
            num_warps=1
        )

        return out


def run(*args):
    return ModelNew()(*args)
