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
    # Grid: (N, C_out, H, W) one program per output element (n, co, h, w)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Sum over input channels and 3x3 neighborhood with padding masks
    for ci in range(NUM_CI):
        for kh in range(3):
            for kw in range(3):
                h_in = pid_h + kh
                w_in = pid_w + kw
                h_valid = (h_in >= 0) & (h_in < H)
                w_valid = (w_in >= 0) & (w_in < W)
                if h_valid and w_valid:
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
    # Grid: (N, groups). Compute per-(n, group) sum and sumsq across channels in group and all spatial positions.
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
def groupnorm_apply_affine_silu(y_ptr, out_ptr, sums_ptr,
                                N, C, H, W, groups,
                                y_sN, y_sC, y_sH, y_sW,
                                out_sN, out_sC, out_sH, out_sW,
                                scale_ptr, bias_ptr, eps):
    # Grid: (N, C). For each (n, c), compute normalized, apply per-channel affine, then SiLU: x * sigmoid(x).
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    group_size_c = C // groups
    group_g = pid_c // group_size_c

    sum_ptr_base = pid_n * groups + group_g
    sum_val = tl.load(sums_ptr + sum_ptr_base * 2 + 0)
    sumsq_val = tl.load(sums_ptr + sum_ptr_base * 2 + 1)

    spatial = H * W
    mean = sum_val / spatial
    var = sumsq_val / spatial - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    scale = tl.load(scale_ptr + pid_c)
    bias = tl.load(bias_ptr + pid_c)

    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    for h in range(0, H):
        for w in range(0, W):
            y_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            val = tl.load(y_ptr + y_offset)
            norm = (val - mean) * rstd
            y_affine = norm * scale + bias
            out_val = y_affine * (1.0 / (1.0 + tl.exp(-y_affine)))
            out_offset = pid_n * out_sN + pid_c * out_sC + h * out_sH + w * out_sW
            tl.store(out_ptr + out_offset, out_val)


@triton.jit
def add_residual_kernel(y_ptr, x_ptr, out_ptr, N, C, H, W,
                        y_sN, y_sC, y_sH, y_sW,
                        x_sN, x_sC, x_sH, x_sW,
                        out_sN, out_sC, out_sH, out_sW):
    # Elementwise add: out = y + x
    total = N * C * H * W
    for i in range(total):
        n = i // (C * H * W)
        rem = i % (C * H * W)
        c = rem // (H * W)
        rem2 = rem % (H * W)
        h = rem2 // W
        w = rem2 % (W)
        y_offset = n * y_sN + c * y_sC + h * y_sH + w * y_sW
        x_offset = n * x_sN + c * x_sC + h * x_sH + w * x_sW
        y_val = tl.load(y_ptr + y_offset)
        x_val = tl.load(x_ptr + x_offset)
        tl.store(out_ptr + (n * out_sN + c * out_sC + h * out_sH + w * out_sW), y_val + x_val)


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
        """
        Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        Triton-only implementation; no torch ops for computation.
        """
        # Ensure contiguous and float32 for numerical stability
        x32 = x.contiguous().to(torch.float32)
        conv1_w32 = conv1_weight.contiguous().to(torch.float32)
        conv2_w32 = conv2_weight.contiguous().to(torch.float32)
        norm1_w32 = norm1_weight.contiguous().to(torch.float32)
        norm1_b32 = norm1_bias.contiguous().to(torch.float32)
        norm2_w32 = norm2_weight.contiguous().to(torch.float32)
        norm2_b32 = norm2_bias.contiguous().to(torch.float32)

        N, C, H, W = x32.shape
        # convs use C_in -> C_out=C for 3x3
        C_in = C
        C_out = C_in

        # First conv: y1 = conv3x3(x)
        y1 = torch.empty((N, C_in, H, W), device=x.device, dtype=torch.float32)
        conv3x3_nchw_4d[(N, C_in, H, W)](
            x32, conv1_w32, y1,
            N, C_in, C_in, H, W,
            x32.stride(0), x32.stride(1), x32.stride(2), x32.stride(3),
            conv1_w32.stride(0), conv1_w32.stride(1), conv1_w32.stride(2), conv1_w32.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            NUM_CI=C_in
        )

        # GroupNorm 1 (num_groups=32) and SiLU
        num_groups = 32
        groups1 = num_groups
        sums1 = torch.empty((N, groups1, 2), device=x.device, dtype=torch.float32)
        # Reduce: compute per-(n, group) sums and sumsq
        groupnorm_reduce_sums[(N, groups1)](
            y1, sums1,
            N, C_in, H, W, groups1,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3)
        )
        # Apply affine + SiLU
        y1_after = torch.empty_like(y1)
        groupnorm_apply_affine_silu[(N, C_in)](
            y1, y1_after, sums1,
            N, C_in, H, W, groups1,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_after.stride(0), y1_after.stride(1), y1_after.stride(2), y1_after.stride(3),
            norm1_w32, norm1_b32, eps
        )

        # Second conv: y2 = conv3x3(y1_after)
        y2_before = torch.empty((N, C_out, H, W), device=x.device, dtype=torch.float32)
        conv3x3_nchw_4d[(N, C_out, H, W)](
            y1_after, conv2_w32, y2_before,
            N, C_in, C_out, H, W,
            y1_after.stride(0), y1_after.stride(1), y1_after.stride(2), y1_after.stride(3),
            conv2_w32.stride(0), conv2_w32.stride(1), conv2_w32.stride(2), conv2_w32.stride(3),
            y2_before.stride(0), y2_before.stride(1), y2_before.stride(2), y2_before.stride(3),
            NUM_CI=C_in
        )

        # GroupNorm 2 (num_groups=32) and SiLU
        sums2 = torch.empty((N, groups1, 2), device=x.device, dtype=torch.float32)
        groupnorm_reduce_sums[(N, groups1)](
            y2_before, sums2,
            N, C_out, H, W, groups1,
            y2_before.stride(0), y2_before.stride(1), y2_before.stride(2), y2_before.stride(3)
        )
        y2_after = torch.empty_like(y2_before)
        groupnorm_apply_affine_silu[(N, C_out)](
            y2_before, y2_after, sums2,
            N, C_out, H, W, groups1,
            y2_before.stride(0), y2_before.stride(1), y2_before.stride(2), y2_before.stride(3),
            y2_after.stride(0), y2_after.stride(1), y2_after.stride(2), y2_after.stride(3),
            norm2_w32, norm2_b32, eps
        )

        # Residual add: out = y2_after + x
        out = torch.empty_like(y2_after)
        add_residual_kernel[(N * C_out * H * W,)](
            y2_after, x32, out,
            N, C_out, H, W,
            y2_after.stride(0), y2_after.stride(1), y2_after.stride(2), y2_after.stride(3),
            x32.stride(0), x32.stride(1), x32.stride(2), x32.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3)
        )

        return out


def run(*args):
    return ModelNew()(*args)
