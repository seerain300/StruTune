import torch
import triton
import triton.language as tl

# Triton kernel: 3x3 Conv (NCHW, stride=1, padding=1, no bias)
# Each program computes one output element y[n, co, h, w].
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

    acc = 0.0

    # Loop over input channels and 3x3 neighborhood (padding handled by masked loads)
    for ci in range(0, C_in):
        for dh in range(0, 3):
            h_in = pid_h + dh
            valid_h = (h_in >= 0) & (h_in < H_out)
            for dw in range(0, 3):
                w_in = pid_w + dw
                valid_w = (w_in >= 0) & (w_in < W_out)
                mask = valid_h & valid_w
                x_offset = pid_n * x_stride_n + ci * x_stride_c + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
                w_offset = pid_co * w_stride_co + ci * w_stride_ci + dh * w_stride_dh + dw * w_stride_dw
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    y_offset = pid_n * y_stride_n + pid_co * y_stride_c + pid_h * y_stride_h + pid_w * y_stride_w
    tl.store(y_ptr + y_offset, acc)


# Triton kernel: GroupNorm over channels (per-channel stats across spatial), num_groups fixed
# Grid: (B, C). Each program handles one sample and one channel.
@triton.jit
def groupnorm_triton_channels(y_in_ptr, weight_ptr, bias_ptr, y_out_ptr,
                               B, C, H_out, W_out,
                               y_in_stride_n, y_in_stride_c, y_in_stride_h, y_in_stride_w,
                               y_out_stride_n, y_out_stride_c, y_out_stride_h, y_out_stride_w,
                               num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    N = H_out * W_out  # number of spatial elements per channel

    # Accumulate sum and sum of squares over spatial plane
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

    # Apply normalization and affine
    for h in range(0, H_out):
        for w in range(0, W_out):
            ptr_in = pid_n * y_in_stride_n + pid_c * y_in_stride_c + h * y_in_stride_h + w * y_in_stride_w
            x = tl.load(y_in_ptr + ptr_in)
            norm = (x - mean) * rstd
            gamma = tl.load(weight_ptr + pid_c)
            beta = tl.load(bias_ptr + pid_c)
            y = norm * gamma + beta
            ptr_out = pid_n * y_out_stride_n + pid_c * y_out_stride_c + h * y_out_stride_h + w * y_out_stride_w
            tl.store(y_out_ptr + ptr_out, y)


# Triton elementwise SiLU: y = x * sigmoid(x)
@triton.jit
def silu_triton(x_ptr, y_ptr, N,
                 num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * num_warps + tl.arange(0, num_warps)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y)


# Triton elementwise residual add: y = x1 + x2
@triton.jit
def add_triton(x1_ptr, x2_ptr, y_ptr, N,
                num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * num_warps + tl.arange(0, num_warps)
    mask = offs < N
    a = tl.load(x1_ptr + offs, mask=mask, other=0.0)
    b = tl.load(x2_ptr + offs, mask=mask, other=0.0)
    y = a + b
    tl.store(y_ptr + offs, y)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        """
        Fused residual block: Conv3x3 -> GroupNorm (num_groups=32) -> SiLU -> Conv3x3 -> GroupNorm (num_groups=32) -> SiLU -> Add
        All ops are implemented in Triton and launched from forward.
        """
        assert x.is_cuda and x.dtype == torch.float32, "x must be CUDA float32 tensor"
        B, C, H, W = x.shape
        device = x.device

        # Ensure weights/biases are on correct device and dtype
        conv1_weight = conv1_weight.contiguous().to(device=device, dtype=torch.float32)
        conv2_weight = conv2_weight.contiguous().to(device=device, dtype=torch.float32)
        norm1_weight = norm1_weight.contiguous().to(device=device, dtype=torch.float32)
        norm2_weight = norm2_weight.contiguous().to(device=device, dtype=torch.float32)
        norm1_bias = norm1_bias.contiguous().to(device=device, dtype=torch.float32)
        norm2_bias = norm2_bias.contiguous().to(device=device, dtype=torch.float32)

        # First path: Conv3x3 -> GroupNorm (channels) -> SiLU
        H1 = H - 2
        W1 = W - 2
        y1 = torch.empty((B, C, H1, W1), device=device, dtype=torch.float32)

        grid_conv1 = (B, C, H1, W1)
        conv3x3_nchw_nobias[grid_conv1](
            x, conv1_weight, y1,
            B, C, C, H, W, H1, W1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=4, num_stages=2
        )

        # GroupNorm1: per-channel stats across spatial H1*W1
        y1_norm = torch.empty_like(y1)
        grid_gn1 = (B, C)
        groupnorm_triton_channels[grid_gn1](
            y1, norm1_weight, norm1_bias, y1_norm,
            B, C, H1, W1,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            num_warps=4, num_stages=2
        )

        # SiLU1
        y1_silu = torch.empty_like(y1_norm)
        N1 = B * C * H1 * W1
        silu_triton[(N1 + 1023) // 1024,](
            y1_norm, y1_silu, N1,
            num_warps=4, num_stages=2
        )

        # Second path: Conv3x3 -> GroupNorm (channels) -> SiLU
        H2 = H - 4
        W2 = W - 4
        y2 = torch.empty((B, C, H2, W2), device=device, dtype=torch.float32)

        grid_conv2 = (B, C, H2, W2)
        conv3x3_nchw_nobias[grid_conv2](
            y1_silu, conv2_weight, y2,
            B, C, C, H1, W1, H2, W2,
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=4, num_stages=2
        )

        # GroupNorm2: per-channel stats across spatial H2*W2
        y2_norm = torch.empty_like(y2)
        grid_gn2 = (B, C)
        groupnorm_triton_channels[grid_gn2](
            y2, norm2_weight, norm2_bias, y2_norm,
            B, C, H2, W2,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
            num_warps=4, num_stages=2
        )

        # SiLU2
        y2_silu = torch.empty_like(y2_norm)
        N2 = B * C * H2 * W2
        silu_triton[(N2 + 1023) // 1024,](
            y2_norm, y2_silu, N2,
            num_warps=4, num_stages=2
        )

        # Residual addition: y_out = y2_silu + x
        # y2_silu: (B, C, H-4, W-4), x: (B, C, H, W). Add elementwise over matching region.
        y_out = torch.empty((B, C, H, W), device=device, dtype=torch.float32)

        # Flatten for elementwise add, N_add equals number of elements in y2_silu
        x_flat = x.contiguous().view(-1)
        y2_silu_flat = y2_silu.contiguous().view(-1)
        y_out_flat = y_out.contiguous().view(-1)
        N_add = B * C * (H - 4) * (W - 4)
        add_triton[(N_add + 1023) // 1024,](
            x_flat, y2_silu_flat, y_out_flat, N_add,
            num_warps=4, num_stages=2
        )

        return y_out


def run(*args):
    return ModelNew()(*args)
