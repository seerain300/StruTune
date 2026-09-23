import torch
import triton
import triton.language as tl

# Triton kernel: 3x3 Conv (NCHW, stride=1, padding=1, no bias) with 4D grid (B, C_out, H_out, W_out)
@triton.jit
def conv3x3_nchw_nobias(x_ptr, w_ptr, y_ptr,
                         B, C_in, C_out, H, W, H_out, W_out,
                         x_stride_n, x_stride_c, x_stride_h, x_stride_w,
                         w_stride_co, w_stride_ci, w_stride_dh, w_stride_dw,
                         y_stride_n, y_stride_c, y_stride_h, y_stride_w,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 neighborhood with masked loads for padding
    for ci in range(0, C_in):
        for dh in range(0, 3):
            h_in = pid_h + dh - 1  # padding=1
            for dw in range(0, 3):
                w_in = pid_w + dw - 1
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                x_offset = pid_n * x_stride_n + ci * x_stride_c + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
                w_offset = pid_co * w_stride_co + ci * w_stride_ci + dh * w_stride_dh + dw * w_stride_dw
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    y_offset = pid_n * y_stride_n + pid_co * y_stride_c + pid_h * y_stride_h + pid_w * y_stride_w
    tl.store(y_ptr + y_offset, acc)


# Triton kernel: GroupNorm over channels (per-channel stats across spatial), num_groups fixed
# y_in: (B, C, H_out, W_out), y_out: same shape
@triton.jit
def groupnorm_triton_channels(y_in_ptr, weight_ptr, bias_ptr, y_out_ptr,
                               B, C, H_out, W_out,
                               y_in_stride_n, y_in_stride_c, y_in_stride_h, y_in_stride_w,
                               y_out_stride_n, y_out_stride_c, y_out_stride_h, y_out_stride_w,
                               num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    N = H_out * W_out  # number of spatial elements per channel

    # accumulate sum and sum of squares over spatial plane
    sum_val = 0.0
    sum_sq = 0.0
    for h in range(0, H_out):
        for w in range(0, W_out):
            ptr = pid_n * y_in_stride_n + pid_c * y_in_stride_c + h * y_in_stride_h + w * y_in_stride_w
            x = tl.load(y_in_ptr + ptr)
            sum_val += x
            sum_sq += x * x

    mean = sum_val / N
    var = sum_sq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    gamma = tl.load(weight_ptr + pid_c)
    beta = tl.load(bias_ptr + pid_c)

    for h in range(0, H_out):
        for w in range(0, W_out):
            ptr_in = pid_n * y_in_stride_n + pid_c * y_in_stride_c + h * y_in_stride_h + w * y_in_stride_w
            x = tl.load(y_in_ptr + ptr_in)
            norm = (x - mean) * rstd
            y = norm * gamma + beta
            ptr_out = pid_n * y_out_stride_n + pid_c * y_out_stride_c + h * y_out_stride_h + w * y_out_stride_w
            tl.store(y_out_ptr + ptr_out, y)


# Triton elementwise SiLU: y = x * sigmoid(x) with 1D grid
@triton.jit
def silu_triton_1d(x_ptr, y_ptr, N,
                   num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(y_ptr + idx, y)


# Triton elementwise residual add: y = y + x with 1D grid
@triton.jit
def add_residual_triton_1d(y_ptr, x_ptr, out_ptr, N,
                            num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    y = tl.load(y_ptr + idx, mask=mask, other=0.0)
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    out = y + x
    tl.store(out_ptr + idx, out)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps  # not used in this Triton-only forward

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        Fused residual block: Conv3x3 -> GroupNorm (num_groups=32) -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        Input x: (B, C, H, W), NCHW, contiguous, float32
        conv weights: (C_out, C_in, 3, 3)
        norm scale/bias: (C,)
        """
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        assert x.dtype == torch.float32, "This implementation expects float32 tensors."
        B, C, H, W = x.shape
        C_in = C  # input channels equal to output channels of the first conv
        C_out = C  # both convs have same C

        # Ensure inputs are contiguous
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        # conv1: output (B, C, H-2, W-2)
        H_out1 = H - 2
        W_out1 = W - 2
        y1 = torch.empty((B, C_out, H_out1, W_out1), device=x.device, dtype=torch.float32)

        # grid for conv1: (B, C, H_out1, W_out1)
        grid1 = (B, C_out, H_out1, W_out1)
        conv3x3_nchw_nobias[grid1](
            x, conv1_weight, y1,
            B, C_in, C_out, H, W, H_out1, W_out1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=4, num_stages=2,
        )

        # GroupNorm1 (num_groups=32), per-channel stats across spatial
        norm1_weight = norm1_weight.to(dtype=torch.float32, device=x.device).contiguous()
        norm1_bias = norm1_bias.to(dtype=torch.float32, device=x.device).contiguous()
        y1_norm = torch.empty_like(y1)
        grid_gn1 = (B, C_out)
        groupnorm_triton_channels[grid_gn1](
            y1, norm1_weight, norm1_bias, y1_norm,
            B, C_out, H_out1, W_out1,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            num_warps=1, num_stages=2,
        )

        # SiLU1
        y1_silu = torch.empty_like(y1_norm)
        N1 = y1_norm.numel()
        grid_silu1 = (triton.cdiv(N1, 256),)
        silu_triton_1d[grid_silu1](
            y1_norm, y1_silu, N1,
            num_warps=4, num_stages=2,
        )

        # conv2: output (B, C, H-4, W-4)
        H_out2 = H - 4
        W_out2 = W - 4
        assert H_out2 > 0 and W_out2 > 0, "Input size too small for conv2 with padding=1 and 3x3."
        y2 = torch.empty((B, C_out, H_out2, W_out2), device=x.device, dtype=torch.float32)

        grid2 = (B, C_out, H_out2, W_out2)
        conv3x3_nchw_nobias[grid2](
            y1_silu, conv2_weight, y2,
            B, C_out, C_out, (H - 2), (W - 2), H_out2, W_out2,
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=4, num_stages=2,
        )

        # GroupNorm2
        norm2_weight = norm2_weight.to(dtype=torch.float32, device=x.device).contiguous()
        norm2_bias = norm2_bias.to(dtype=torch.float32, device=x.device).contiguous()
        y2_norm = torch.empty_like(y2)
        grid_gn2 = (B, C_out)
        groupnorm_triton_channels[grid_gn2](
            y2, norm2_weight, norm2_bias, y2_norm,
            B, C_out, H_out2, W_out2,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
            num_warps=1, num_stages=2,
        )

        # SiLU2
        y2_silu = torch.empty_like(y2_norm)
        N2 = y2_norm.numel()
        grid_silu2 = (triton.cdiv(N2, 256),)
        silu_triton_1d[grid_silu2](
            y2_norm, y2_silu, N2,
            num_warps=4, num_stages=2,
        )

        # Residual addition: y_out = y2_silu + x
        # x: (B, C, H, W), y2_silu: (B, C, H-4, W-4). We add over overlapping elements only.
        out = torch.empty_like(y2_silu)
        Nadd = min(y2_silu.numel(), x.numel())
        grid_add = (triton.cdiv(Nadd, 256),)
        add_residual_triton_1d[grid_add](
            y2_silu, x, out, Nadd,
            num_warps=4, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
