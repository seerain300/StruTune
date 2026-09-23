import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_4d(x_ptr, w_ptr, y_ptr,
                    N, C_in, H, W,
                    x_sN, x_sC, x_sH, x_sW,
                    w_sCo, w_sCi, w_sKh, w_sKw,
                    y_sN, y_sC, y_sH, y_sW,
                    NUM_CI: tl.constexpr,
                    NUM_H: tl.constexpr,
                    NUM_W: tl.constexpr):
    # Grid: (N, C_out, H, W)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Sum over input channels and 3x3 neighborhood
    for ci in range(NUM_CI):
        for kh in range(NUM_H):
            for kw in range(NUM_W):
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
                                N, C, H, W,
                                norm_weight_ptr, norm_bias_ptr,
                                sums_ptr, eps,
                                y_sN, y_sC, y_sH, y_sW,
                                out_sN, out_sC, out_sH, out_sW,
                                GROUPS: tl.constexpr):
    # Grid: (N, C)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    group_size_c = C // GROUPS
    group_g = pid_c // group_size_c

    sum_val = tl.load(sums_ptr + (pid_n * GROUPS + group_g) * 2 + 0)
    sumsq_val = tl.load(sums_ptr + (pid_n * GROUPS + group_g) * 2 + 1)

    HxW = H * W
    mean = sum_val / HxW
    var = sumsq_val / HxW - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    scale = tl.load(norm_weight_ptr + pid_c)
    bias = tl.load(norm_bias_ptr + pid_c)

    for h in range(0, H):
        for w in range(0, W):
            y_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            val = tl.load(y_ptr + y_offset)
            norm = (val - mean) * rstd
            y_affine = norm * scale + bias
            # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
            silu = y_affine * (1.0 / (1.0 + tl.exp(-y_affine)))
            out_offset = pid_n * out_sN + pid_c * out_sC + h * out_sH + w * out_sW
            tl.store(out_ptr + out_offset, silu)


@triton.jit
def add_residual(y_ptr, x_ptr, out_ptr, total_elems):
    # Elementwise add: out = y + x
    pid = tl.program_id(0)
    if pid < total_elems:
        # Compute n, c, h, w from pid
        # Note: total_elems = N * C * H * W
        # We'll implement direct indexing by pid for simplicity
        # out_ptr is expected to be same shape as y_ptr, so we can compute offsets via modulo
        # However, since we have linear ids, we can't infer shape here; instead, we assume y and out are contiguous and same shape
        # We'll rely on host to pass pointers and we perform linear indexing as out[pid] = y[pid] + x[pid]
        # Triton expects pointer arithmetic, so we do:
        val = tl.load(y_ptr + pid) + tl.load(x_ptr + pid)
        tl.store(out_ptr + pid, val)


def run_triton_conv3x3(x32: torch.Tensor, w32: torch.Tensor) -> torch.Tensor:
    """
    Triton forward-only 3x3 conv NCHW, stride=1, padding=1.
    x32: (N, C_in, H, W) float32, contiguous
    w32: (C_out, C_in, 3, 3) float32, contiguous
    returns: y (N, C_out, H, W) float32
    """
    assert x32.is_cuda and w32.is_cuda
    assert x32.dtype == torch.float32 and w32.dtype == torch.float32
    x = x32.contiguous()
    w = w32.contiguous()
    N, C_in, H, W = x.shape
    C_out = w.shape[0]
    y = torch.empty((N, C_out, H, W), device=x.device, dtype=torch.float32)

    grid = (N, C_out, H, W)
    conv3x3_nchw_4d[grid](
        x, w, y,
        N, C_in, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        NUM_CI=C_in, NUM_H=3, NUM_W=3,
        num_warps=1
    )
    return y


def run_triton_groupnorm(y32: torch.Tensor, norm_weight: torch.Tensor, norm_bias: torch.Tensor, eps: float):
    """
    Triton GroupNorm with num_groups=32, per-channel affine. y32 shape (N, C, H, W).
    Returns normalized + affine + SiLU result.
    """
    assert y32.is_cuda and norm_weight.is_cuda and norm_bias.is_cuda
    assert y32.dtype == torch.float32 and norm_weight.dtype == torch.float32 and norm_bias.dtype == torch.float32
    y = y32.contiguous()
    N, C, H, W = y.shape
    groups = 32
    assert (C % groups) == 0, "C must be divisible by num_groups=32"
    # Buffer for sums per (n, group)
    sums = torch.empty((N, groups, 2), device=y.device, dtype=torch.float32)
    # Reduction
    grid_reduce = (N, groups)
    groupnorm_reduce_sums[grid_reduce](
        y,
        sums,
        N, C, H, W, groups,
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        num_warps=1
    )
    # Apply + SiLU
    out = torch.empty_like(y)
    grid_apply = (N, C)
    groupnorm_apply_affine_silu[grid_apply](
        y, out,
        N, C, H, W,
        norm_weight, norm_bias,
        sums, eps,
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        GROUPS=groups,
        num_warps=1
    )
    return out


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
        # Ensure CUDA and float32, contiguous
        x32 = x.contiguous().to(torch.float32)
        conv1_w32 = conv1_weight.contiguous().to(torch.float32)
        conv2_w32 = conv2_weight.contiguous().to(torch.float32)
        norm1_w32 = norm1_weight.contiguous().to(torch.float32)
        norm1_b32 = norm1_bias.contiguous().to(torch.float32)
        norm2_w32 = norm2_weight.contiguous().to(torch.float32)
        norm2_b32 = norm2_bias.contiguous().to(torch.float32)

        # Conv1: (N, C, H, W) -> (N, C, H, W)
        y1 = run_triton_conv3x3(x32, conv1_w32)
        # GroupNorm1 + SiLU
        y1 = run_triton_groupnorm(y1, norm1_w32, norm1_b32, eps)
        # Conv2
        y2 = run_triton_conv3x3(y1, conv2_w32)
        # GroupNorm2 + SiLU
        y2 = run_triton_groupnorm(y2, norm2_w32, norm2_b32, eps)
        # Residual add: y2 + x
        total = y2.numel()
        out = torch.empty_like(y2)
        add_residual[(total,)](
            y2, x32, out,
            total,
            num_warps=1
        )
        return out


def run(*args):
    return ModelNew()(*args)
