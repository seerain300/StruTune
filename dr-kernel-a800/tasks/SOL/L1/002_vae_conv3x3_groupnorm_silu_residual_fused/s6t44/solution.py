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
def groupnorm_apply_affine_silu(y_ptr, sums_ptr, scale_ptr, bias_ptr, out_ptr,
                                N, C, H, W,
                                y_sN, y_sC, y_sH, y_sW):
    # Grid: (N, C)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    group_size_c = C // 32  # num_groups is fixed at 32 in the original model
    g = pid_c // group_size_c

    sum_val = tl.load(sums_ptr + (pid_n * 32 + g) * 2 + 0)
    sumsq_val = tl.load(sums_ptr + (pid_n * 32 + g) * 2 + 1)
    mean = sum_val / (H * W)
    var = sumsq_val / (H * W) - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    for h in range(0, H):
        for w in range(0, W):
            y_offset = pid_n * y_sN + pid_c * y_sC + h * y_sH + w * y_sW
            val = tl.load(y_ptr + y_offset)
            norm = (val - mean) * rstd
            scale = tl.load(scale_ptr + pid_c)
            bias = tl.load(bias_ptr + pid_c)
            activated = norm * scale + bias  # affine before SiLU
            # SiLU: x * sigmoid(x)
            sig = 1.0 / (1.0 + tl.exp(-activated))
            out_val = activated * sig
            tl.store(out_ptr + y_offset, out_val)


@triton.jit
def add_residual(y_ptr, res_ptr, out_ptr, total_elems):
    # 1D grid: elementwise add
    pid = tl.program_id(0)
    # Note: we assume out_ptr and y_ptr share the same element layout; total_elems should be N*C*H*W
    y_val = tl.load(y_ptr + pid)
    res_val = tl.load(res_ptr + pid)
    tl.store(out_ptr + pid, y_val + res_val)


def _ceil_div(a, b):
    return (a + b - 1) // b


class ModelNew(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.num_groups = 32

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        Args:
            x: Input tensor of shape (B, C, H, W), float32
            conv1_weight: First conv weights (C, C, 3, 3), float32
            norm1_weight: First GroupNorm scale (C,), float32
            norm1_bias: First GroupNorm bias (C,), float32
            conv2_weight: Second conv weights (C, C, 3, 3), float32
            norm2_weight: Second GroupNorm scale (C,), float32
            norm2_bias: Second GroupNorm bias (C,), float32
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels"
        B, C, H, W = x.shape

        # Ensure contiguous and float32
        x32 = x.contiguous().to(torch.float32)
        conv1_w = conv1_weight.contiguous().to(torch.float32)
        conv2_w = conv2_weight.contiguous().to(torch.float32)
        norm1_s = norm1_weight.contiguous().to(torch.float32)
        norm1_b = norm1_bias.contiguous().to(torch.float32)
        norm2_s = norm2_weight.contiguous().to(torch.float32)
        norm2_b = norm2_bias.contiguous().to(torch.float32)

        # First conv: y1_out = conv3x3(x32, conv1_w)
        y1_out = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)
        conv3x3_nchw_4d[(B, C, H, W)](
            x32, conv1_w,
            y1_out,
            B, C, H, W,
            x32.stride(0), x32.stride(1), x32.stride(2), x32.stride(3),
            conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2), conv1_w.stride(3),
            y1_out.stride(0), y1_out.stride(1), y1_out.stride(2), y1_out.stride(3),
            C_out=C,
            NUM_CI=C,
            num_warps=1,
        )

        # GroupNorm 1 + SiLU
        sums1 = torch.empty((B, self.num_groups, 2), dtype=torch.float32, device=x.device)
        groupnorm_reduce_sums[(B, self.num_groups)](
            y1_out, sums1,
            B, C, H, W, self.num_groups,
            y1_out.stride(0), y1_out.stride(1), y1_out.stride(2), y1_out.stride(3),
            num_warps=1,
        )

        y1_norm = torch.empty_like(y1_out)
        groupnorm_apply_affine_silu[(B, C)](
            y1_out, sums1, norm1_s, norm1_b, y1_norm,
            B, C, H, W,
            y1_out.stride(0), y1_out.stride(1), y1_out.stride(2), y1_out.stride(3),
            num_warps=1,
        )

        # Second conv: y2_out = conv3x3(y1_norm, conv2_w)
        y2_out = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)
        conv3x3_nchw_4d[(B, C, H, W)](
            y1_norm, conv2_w,
            y2_out,
            B, C, H, W,
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2), conv2_w.stride(3),
            y2_out.stride(0), y2_out.stride(1), y2_out.stride(2), y2_out.stride(3),
            C_out=C,
            NUM_CI=C,
            num_warps=1,
        )

        # GroupNorm 2 + SiLU
        sums2 = torch.empty((B, self.num_groups, 2), dtype=torch.float32, device=x.device)
        groupnorm_reduce_sums[(B, self.num_groups)](
            y2_out, sums2,
            B, C, H, W, self.num_groups,
            y2_out.stride(0), y2_out.stride(1), y2_out.stride(2), y2_out.stride(3),
            num_warps=1,
        )

        y2_norm = torch.empty_like(y2_out)
        groupnorm_apply_affine_silu[(B, C)](
            y2_out, sums2, norm2_s, norm2_b, y2_norm,
            B, C, H, W,
            y2_out.stride(0), y2_out.stride(1), y2_out.stride(2), y2_out.stride(3),
            num_warps=1,
        )

        # Residual add: y2_norm += x32
        total_elems = B * C * H * W
        final_out = torch.empty_like(y2_norm)
        add_residual[(total_elems,)](
            y2_norm, x32, final_out,
            total_elems,
            num_warps=1,
        )

        return final_out


def run(*args):
    return ModelNew()(*args)
