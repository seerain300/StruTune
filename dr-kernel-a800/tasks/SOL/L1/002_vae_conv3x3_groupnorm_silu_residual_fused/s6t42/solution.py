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
    # Grid: (N, C_out, H, W) — one program per output element
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
    # Grid: (N, groups) — one program per (n, group)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_size_c = C // groups
    start_c = pid_g * group_size_c

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    for ci in range(start_c, start_c + group_size_c):
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
def groupnorm_apply_affine_silu(y_ptr, sums_ptr, scale_ptr, bias_ptr, out_ptr,
                                N, C, H, W, groups,
                                y_sN, y_sC, y_sH, y_sW,
                                out_sN, out_sC, out_sH, out_sW):
    # Grid: (N, C) — one program per (n, c)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    group_size_c = C // groups
    g = pid_c // group_size_c

    sum_val = tl.load(sums_ptr + (pid_n * groups + g) * 2 + 0)
    sumsq_val = tl.load(sums_ptr + (pid_n * groups + g) * 2 + 1)
    hw = H * W
    mean = sum_val / hw
    var = sumsq_val / hw - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)  # eps for numerical stability

    # Process all spatial positions for channel pid_c
    for h in range(0, H):
        for w in range(0, W):
            y_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            val = tl.load(y_ptr + y_offset)
            norm = (val - mean) * rstd
            scale = tl.load(scale_ptr + pid_c)
            bias = tl.load(bias_ptr + pid_c)
            lin = norm * scale + bias
            # SiLU: x * sigmoid(x)
            sig = 1.0 / (1.0 + tl.exp(-lin))
            out = lin * sig
            out_offset = pid_n * out_sN + pid_c * out_sC + h * out_sH + w * out_sW
            tl.store(out_ptr + out_offset, out)


@triton.jit
def add_residual(y_ptr, x_ptr, out_ptr,
                 total_elems,
                 num_warps: tl.constexpr):
    # Elementwise add over total_elems
    pid = tl.program_id(0)
    offset = pid  # assuming grid size == total_elems; one element per program
    val_y = tl.load(y_ptr + offset)
    val_x = tl.load(x_ptr + offset)
    tl.store(out_ptr + offset, val_y + val_x)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        # Ensure contiguous and float32 for stable Triton computation
        x32 = x.contiguous().to(torch.float32)

        # First conv: (N, C, H, W) -> (N, C, H, W)
        N, C_in, H, W = x32.shape
        C_out1 = conv1_weight.shape[0]
        y1 = torch.empty((N, C_out1, H, W), device=x32.device, dtype=torch.float32)
        conv3x3_nchw_4d[(N, C_out1, H, W)](
            x32, conv1_weight, y1,
            N, C_in, H, W,
            x32.stride(0), x32.stride(1), x32.stride(2), x32.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            C_out=C_out1,
            NUM_CI=C_in,
            num_warps=1
        )

        # GroupNorm 1: num_groups=32, per-channel affine, SiLU
        groups = 32
        C = y1.shape[1]
        assert C % groups == 0, "Channels must be divisible by num_groups"
        sums1 = torch.empty((N, groups, 2), device=y1.device, dtype=torch.float32)
        groupnorm_reduce_sums[(N, groups)](
            y1, sums1,
            N, C, H, W, groups,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=1
        )
        y1_norm = torch.empty_like(y1)
        groupnorm_apply_affine_silu[(N, C)](
            y1, sums1, norm1_weight, norm1_bias, y1_norm,
            N, C, H, W, groups,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            num_warps=1
        )

        # Second conv: (N, C, H, W) -> (N, C, H, W)
        C_out2 = conv2_weight.shape[0]  # output channels of second conv
        assert C_out2 == C, "Second conv's output channels must match first conv's channels for this reference"
        y2 = torch.empty((N, C, H, W), device=y1_norm.device, dtype=torch.float32)
        conv3x3_nchw_4d[(N, C, H, W)](
            y1_norm, conv2_weight, y2,
            N, C, H, W,
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            C_out=C,
            NUM_CI=C,
            num_warps=1
        )

        # GroupNorm 2: num_groups=32, per-channel affine, SiLU
        sums2 = torch.empty((N, groups, 2), device=y2.device, dtype=torch.float32)
        groupnorm_reduce_sums[(N, groups)](
            y2, sums2,
            N, C, H, W, groups,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=1
        )
        y2_norm = torch.empty_like(y2)
        groupnorm_apply_affine_silu[(N, C)](
            y2, sums2, norm2_weight, norm2_bias, y2_norm,
            N, C, H, W, groups,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
            num_warps=1
        )

        # Residual add: out = y2_norm + x32
        total_elems = N * C * H * W
        out = torch.empty_like(y2_norm)
        add_residual[(total_elems,)](
            y2_norm, x32, out,
            total_elems,
            num_warps=1
        )

        return out


def run(*args):
    return ModelNew()(*args)
