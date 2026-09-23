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
def groupnorm_apply_affine_silu(y_ptr, out_ptr, sums_ptr, norm_weight_ptr, norm_bias_ptr,
                                N, C, H, W, groups, eps,
                                y_sN, y_sC, y_sH, y_sW,
                                out_sN, out_sC, out_sH, out_sW):
    # Grid: (N, C)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    group_size_c = C // groups
    g = pid_c // group_size_c

    base = pid_n * groups + g
    sum_val = tl.load(sums_ptr + base * 2 + 0)
    sumsq_val = tl.load(sums_ptr + base * 2 + 1)
    num_pixels = H * W
    mean = sum_val / num_pixels
    var = sumsq_val / num_pixels - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    scale = tl.load(norm_weight_ptr + pid_c)
    bias = tl.load(norm_bias_ptr + pid_c)

    for h in range(0, H):
        for w in range(0, W):
            y_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            val = tl.load(y_ptr + y_offset)
            norm = (val - mean) * rstd
            out_val = norm * scale + bias
            sig = 1.0 / (1.0 + tl.exp(-out_val))
            out_val = out_val * sig

            out_offset = pid_n * out_sN + pid_c * out_sC + h * out_sH + w * out_sW
            tl.store(out_ptr + out_offset, out_val)


@triton.jit
def add_residual(out_ptr, y_ptr, x_ptr, N, C, H, W):
    # 1D grid over total elements. Each program handles one element.
    pid = tl.program_id(0)
    total = N * C * H * W
    if pid >= total:
        return

    # Compute indices via nested div/mod to map pid to (n, c, h, w)
    hw = H * W
    n = pid // (C * hw)
    rem = pid % (C * hw)
    c = rem // hw
    rem2 = rem % hw
    h = rem2 // W
    w = rem2 % W

    y_offset = n * (C * hw) + c * hw + h * W + w
    x_offset = n * (C * hw) + c * hw + h * W + w
    # out_ptr is a new tensor where we already have y2_out; we add x32 to it
    # We cannot directly index out_ptr using pid; instead, each program reads out_ptr[y_offset] and x_ptr[x_offset], adds, and stores back.
    # However, out_ptr is newly allocated; to avoid complex indexing, we simply assume out_ptr is the destination and compute offsets accordingly.
    # Simpler: we will allocate out_ptr as a flat buffer and compute addresses manually.

    # Better: allocate out_ptr as y2_out and perform in-place addition in forward. But Triton cannot easily update both; so we use a separate out tensor.
    # Since we can't index with pid, we'll instead launch with grid=(total,) and compute addresses via div/mod.

    # We'll implement the add by reading y2_out and x and writing y2_out + x into out_ptr. Each program reads and writes its element.
    # But Triton expects a pointer; we can't derive base addresses from pid unless we have a 2D/3D grid. Therefore, we instead compute per-channel per-HW grid or use torch for add. However, to comply, we will implement a correct 1D add kernel.
    # Derivation:
    # total = N * C * H * W. We map pid to (n, c, h, w) and then out_offset = n*y_sN + c*y_sC + h*y_sH + w*y_sW
    # x_offset = n*x_sN + c*x_sC + h*x_sH + w*x_sW. Since x is x32 with same shape, we can reuse y strides; but we need x strides.
    # For simplicity, we pass both out and x pointers and compute offsets with y strides and x strides separately. But we need to know x strides.
    # Since x is x32 contiguous NCHW, strides are straightforward: x_sN=C*H*W, x_sC=H*W, x_sH=W, x_sW=1.

    # Define x tensor: x32 is input; its strides are:
    x_sN = C * H * W
    x_sC = H * W
    x_sH = W
    x_sW = 1

    y_offset = n * (C * hw) + c * hw + h * W + w
    x_offset = n * x_sN + c * x_sC + h * x_sH + w

    y_val = tl.load(y_ptr + y_offset)
    x_val = tl.load(x_ptr + x_offset)
    tl.store(out_ptr + y_offset, y_val + x_val)


# ModelNew: perform all operations in Triton and return result
class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        # Ensure contiguous and float32
        N, C_in, H, W = x.shape
        x32 = x.contiguous().to(torch.float32)

        # conv1: y1 = conv3x3(x32, conv1_weight)
        C_in_w = conv1_weight.shape[0]  # output channels of first conv
        y1 = torch.empty((N, C_in_w, H, W), dtype=torch.float32, device=x.device)

        conv3x3_nchw_4d[(N, C_in_w, H, W)](
            x32, conv1_weight, y1,
            N, C_in, H, W,
            x32.stride(0), x32.stride(1), x32.stride(2), x32.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            C_out=C_in_w,
            num_warps=1
        )

        # GroupNorm and SiLU on y1 (num_groups=32)
        groups = 32
        assert C_in_w % groups == 0, "C must be divisible by num_groups=32"
        group_size_c = C_in_w // groups

        y1_stats = torch.empty((N, groups, 2), dtype=torch.float32, device=x.device)

        groupnorm_reduce_sums[(N, groups)](
            y1, y1_stats,
            N, C_in_w, H, W, groups,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=1
        )

        y1_out = torch.empty_like(y1)
        groupnorm_apply_affine_silu[(N, C_in_w)](
            y1, y1_out, y1_stats, norm1_weight, norm1_bias,
            N, C_in_w, H, W, groups, eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_out.stride(0), y1_out.stride(1), y1_out.stride(2), y1_out.stride(3),
            num_warps=1
        )

        # conv2: y2 = conv3x3(y1_out, conv2_weight)
        C_in_w2 = conv2_weight.shape[0]  # output channels of second conv
        y2 = torch.empty((N, C_in_w2, H, W), dtype=torch.float32, device=x.device)

        conv3x3_nchw_4d[(N, C_in_w2, H, W)](
            y1_out, conv2_weight, y2,
            N, C_in_w, H, W,
            y1_out.stride(0), y1_out.stride(1), y1_out.stride(2), y1_out.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            C_out=C_in_w2,
            num_warps=1
        )

        # GroupNorm and SiLU on y2
        groups2 = 32
        assert C_in_w2 % groups2 == 0, "C must be divisible by num_groups=32"
        group_size_c2 = C_in_w2 // groups2
        y2_stats = torch.empty((N, groups2, 2), dtype=torch.float32, device=x.device)

        groupnorm_reduce_sums[(N, groups2)](
            y2, y2_stats,
            N, C_in_w2, H, W, groups2,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=1
        )

        y2_out = torch.empty_like(y2)
        groupnorm_apply_affine_silu[(N, C_in_w2)](
            y2, y2_out, y2_stats, norm2_weight, norm2_bias,
            N, C_in_w2, H, W, groups2, eps,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_out.stride(0), y2_out.stride(1), y2_out.stride(2), y2_out.stride(3),
            num_warps=1
        )

        # Residual add: out = y2_out + x32
        total_elems = N * C_in_w2 * H * W
        out = torch.empty_like(y2_out)
        add_residual[(total_elems,)](
            out, y2_out, x32,
            N, C_in_w2, H, W,
            num_warps=1
        )

        return out


def run(*args):
    return ModelNew()(*args)
