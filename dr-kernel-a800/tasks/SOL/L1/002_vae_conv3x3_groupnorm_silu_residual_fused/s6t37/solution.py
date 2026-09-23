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
    # Grid: (N, C_out, H, W) -> one program per output element
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
    # Grid: (N, groups) -> one program per (n, group)
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
def groupnorm_apply_affine_silu(y_ptr, norm_weight_ptr, norm_bias_ptr, out_ptr,
                                N, C, H, W, eps,
                                y_sN, y_sC, y_sH, y_sW,
                                groups):
    # Grid: (N, C) -> one program per (n, c)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    group_size_c = C // groups
    g = pid_c // group_size_c

    # Load per-(n,g) stats
    sum_val = tl.load(sums_ptr + (pid_n * groups + g) * 2 + 0)
    sumsq_val = tl.load(sums_ptr + (pid_n * groups + g) * 2 + 1)

    hw = H * W
    mean = sum_val / hw
    var = sumsq_val / hw - mean * mean
    rstd = tl.rsqrt(var + eps)

    # Apply normalization and affine, then SiLU, across H*W
    for h in range(0, H):
        for w in range(0, W):
            y_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            val = tl.load(y_ptr + y_offset)
            norm = (val - mean) * rstd

            scale = tl.load(norm_weight_ptr + pid_c)
            bias = tl.load(norm_bias_ptr + pid_c)
            z = norm * scale + bias

            # SiLU: z * sigmoid(z)
            sig = 1.0 / (1.0 + tl.exp(-z))
            out = z * sig

            out_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            tl.store(out_ptr + out_offset, out)


@triton.jit
def add_residual_kernel(out_ptr, x_ptr, total_elems: tl.constexpr):
    # Simple elementwise addition: out[i] = out[i] + x[i]
    pid = tl.program_id(0)
    idx = pid  # linear index
    out_val = tl.load(out_ptr + idx)
    x_val = tl.load(x_ptr + idx)
    tl.store(out_ptr + idx, out_val + x_val)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        # Ensure float32 and contiguous
        x32 = x.contiguous().to(torch.float32)
        N = x32.shape[0]
        C_in = x32.shape[1]
        H = x32.shape[2]
        W = x32.shape[3]

        # First conv: y1 = conv3x3(x32, conv1_weight)
        C_out1 = conv1_weight.shape[0]
        y1 = torch.empty((N, C_out1, H, W), dtype=torch.float32, device=x32.device)

        conv3x3_nchw_4d[(N, C_out1, H, W)](
            x32, conv1_weight,
            y1,
            N, C_in, H, W,
            x32.stride(0), x32.stride(1), x32.stride(2), x32.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            C_out=C_out1,
            num_warps=1
        )

        # GroupNorm and SiLU for y1 (num_groups=32)
        groups1 = 32
        C1 = C_out1
        sums1 = torch.empty((N, groups1, 2), dtype=torch.float32, device=x32.device)

        groupnorm_reduce_sums[(N, groups1)](
            y1,
            sums1,
            N, C1, H, W, groups1,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=1
        )

        y1_norm_silu = torch.empty_like(y1)
        groupnorm_apply_affine_silu[(N, C1)](
            y1, norm1_weight, norm1_bias, y1_norm_silu,
            N, C1, H, W, eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            groups1,
            num_warps=1
        )

        # Second conv: y2 = conv3x3(y1_norm_silu, conv2_weight)
        C_out2 = conv2_weight.shape[0]
        y2 = torch.empty((N, C_out2, H, W), dtype=torch.float32, device=x32.device)

        conv3x3_nchw_4d[(N, C_out2, H, W)](
            y1_norm_silu, conv2_weight,
            y2,
            N, C1, H, W,
            y1_norm_silu.stride(0), y1_norm_silu.stride(1), y1_norm_silu.stride(2), y1_norm_silu.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            C_out=C_out2,
            num_warps=1
        )

        # GroupNorm and SiLU for y2 (num_groups=32)
        groups2 = 32
        C2 = C_out2
        sums2 = torch.empty((N, groups2, 2), dtype=torch.float32, device=x32.device)

        groupnorm_reduce_sums[(N, groups2)](
            y2,
            sums2,
            N, C2, H, W, groups2,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=1
        )

        y2_norm_silu = torch.empty_like(y2)
        groupnorm_apply_affine_silu[(N, C2)](
            y2, norm2_weight, norm2_bias, y2_norm_silu,
            N, C2, H, W, eps,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            groups2,
            num_warps=1
        )

        # Residual add: out = y2_norm_silu + x32
        total_elems = N * C2 * H * W
        out = torch.empty_like(y2_norm_silu)
        add_residual_kernel[(total_elems,)](
            y2_norm_silu, x32,
            total_elems,
            num_warps=1
        )

        return out


def run(*args):
    return ModelNew()(*args)
