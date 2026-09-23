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
# grid: (B, C)
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

    # apply normalization and affine
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
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + idx, y, mask=mask)

# Triton elementwise residual add: y_out = y + x (broadcast x to y's spatial shape)
@triton.jit
def add_residual_triton(y_ptr, x_ptr, y_out_ptr,
                         B, C, H_out, W_out,
                         y_stride_n, y_stride_c, y_stride_h, y_stride_w,
                         x_stride_n, x_stride_c, x_stride_h, x_stride_w,
                         y_out_stride_n, y_out_stride_c, y_out_stride_h, y_out_stride_w,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    y_val = tl.load(y_ptr + pid_n * y_stride_n + pid_c * y_stride_c + pid_h * y_stride_h + pid_w * y_stride_w)
    x_val = tl.load(x_ptr + pid_n * x_stride_n + pid_c * x_stride_c + pid_h * x_stride_h + pid_w * x_stride_w)
    out = y_val + x_val
    tl.store(y_out_ptr + pid_n * y_out_stride_n + pid_c * y_out_stride_c + pid_h * y_out_stride_h + pid_w * y_out_stride_w, out)

def conv3x3_nchw_triton(x, weight, H_out, W_out):
    # x: (B, C, H, W), weight: (C_out, C, 3, 3)
    B, C_in, H, W = x.shape
    C_out = weight.shape[0]
    y = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=torch.float32)
    grid = (B, C_out, H_out, W_out)
    conv3x3_nchw_nobias[grid](
        x, weight, y,
        B, C_in, C_out, H, W, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        num_warps=4, num_stages=2,
    )
    return y

def groupnorm_channels_triton(y_in, weight, bias):
    # y_in: (B, C, H_out, W_out), weight/bias: (C,)
    B, C, H_out, W_out = y_in.shape
    y = torch.empty_like(y_in)
    grid = (B, C)
    groupnorm_triton_channels[grid](
        y_in, weight, bias, y,
        B, C, H_out, W_out,
        y_in.stride(0), y_in.stride(1), y_in.stride(2), y_in.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        num_warps=4, num_stages=2,
    )
    return y

def silu_triton_overall(x, N):
    # x: 1D flattened tensor
    y = torch.empty_like(x)
    grid = (triton.cdiv(N, 256),)
    silu_triton[grid](
        x, y, N,
        num_warps=4, num_stages=2,
    )
    return y

def add_residual_triton_forward(y, x):
    # y: (B, C, H_out2, W_out2), x: (B, C, H, W), add with broadcasting to y's spatial shape by zero-padding
    B, C, H_out2, W_out2 = y.shape
    # Zero-pad x to (H_out2, W_out2) for addition
    x_padded = torch.nn.functional.pad(x, (0, max(0, W_out2 - x.shape[3]), 0, max(0, H_out2 - x.shape[2])))
    # Now x_padded has shape (B, C, H_out2, W_out2) or broadcast-compatible with y
    y_out = torch.empty_like(y)
    grid = (B, C, H_out2, W_out2)
    add_residual_triton[grid](
        y, x_padded, y_out,
        B, C, H_out2, W_out2,
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        x_padded.stride(0), x_padded.stride(1), x_padded.stride(2), x_padded.stride(3),
        y_out.stride(0), y_out.stride(1), y_out.stride(2), y_out.stride(3),
        num_warps=4, num_stages=2,
    )
    return y_out

class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        # conv1: (B, C, H, W) -> (B, C, H-2, W-2)
        H, W = x.shape[2], x.shape[3]
        H_out1 = H - 2
        W_out1 = W - 2
        y1 = conv3x3_nchw_triton(x, conv1_weight, H_out1, W_out1)

        # GroupNorm1 (num_groups=32): per-channel stats across spatial plane
        y1 = groupnorm_channels_triton(y1, norm1_weight, norm1_bias)

        # SiLU1
        y1_flat = y1.reshape(-1)
        y1_silu = silu_triton_overall(y1_flat, y1_flat.numel())
        y1_silu = y1_silu.reshape(y1.shape)

        # conv2: (B, C, H-2, W-2) -> (B, C, H-4, W-4)
        H_out2 = H_out1 - 2
        W_out2 = W_out1 - 2
        y2 = conv3x3_nchw_triton(y1_silu, conv2_weight, H_out2, W_out2)

        # GroupNorm2 (num_groups=32): per-channel stats across spatial plane
        y2 = groupnorm_channels_triton(y2, norm2_weight, norm2_bias)

        # SiLU2
        y2_flat = y2.reshape(-1)
        y2_silu = silu_triton_overall(y2_flat, y2_flat.numel())
        y2_silu = y2_silu.reshape(y2.shape)

        # Residual addition: y = y + x (pad x to conv2 output spatial shape)
        y_out = add_residual_triton_forward(y2_silu, x)

        return y_out


def run(*args):
    return ModelNew()(*args)
