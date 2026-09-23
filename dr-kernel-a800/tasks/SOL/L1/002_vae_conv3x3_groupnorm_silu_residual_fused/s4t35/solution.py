import torch
import triton
import triton.language as tl

# Triton kernel: 3x3 Conv (NCHW, stride=1, padding=1, no bias)
# y[n, co, h, w] = sum_{ci, dh, dw} x[n, ci, h+dh, w+dw] * w[co, ci, dh, dw]
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
    # Loop over input channels
    for ci in range(0, C_in):
        # Loop over 3x3 neighborhood
        for dh in range(0, 3):
            h_in = pid_h + dh
            valid_h = h_in < H_out
            for dw in range(0, 3):
                w_in = pid_w + dw
                valid_w = w_in < W_out
                mask = valid_h & valid_w
                x_offset = pid_n * x_stride_n + ci * x_stride_c + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
                w_offset = pid_co * w_stride_co + ci * w_stride_ci + dh * w_stride_dh + dw * w_stride_dw
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    y_offset = pid_n * y_stride_n + pid_co * y_stride_c + pid_h * y_stride_h + pid_w * y_stride_w
    tl.store(y_ptr + y_offset, acc)

# Triton kernel: GroupNorm with num_groups fixed (32), per-channel stats across spatial plane
# grid: (B, num_groups) ; each program handles one group and all its channels
@triton.jit
def groupnorm_triton_fixed(y_in_ptr, weight_ptr, bias_ptr, y_out_ptr,
                            B, C, H_out, W_out, num_groups: tl.constexpr,
                            y_in_stride_n, y_in_stride_c, y_in_stride_h, y_in_stride_w,
                            y_out_stride_n, y_out_stride_c, y_out_stride_h, y_out_stride_w,
                            num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    channels_per_group = C // num_groups
    group_start = pid_g * channels_per_group

    # Compute per-channel stats for each channel in the group
    for ci in range(0, channels_per_group):
        c_idx = group_start + ci

        N = H_out * W_out  # number of spatial elements per channel
        sum_val = 0.0
        sum_sq = 0.0

        # Accumulate sum and sum of squares across spatial plane
        for h in range(0, H_out):
            for w in range(0, W_out):
                in_ptr = pid_n * y_in_stride_n + c_idx * y_in_stride_c + h * y_in_stride_h + w * y_in_stride_w
                x = tl.load(y_in_ptr + in_ptr)
                sum_val += x
                sum_sq += x * x

        mean = sum_val / N
        var = sum_sq / N - mean * mean
        rstd = 1.0 / tl.sqrt(var + 1e-5)

        gamma = tl.load(weight_ptr + c_idx)
        beta = tl.load(bias_ptr + c_idx)

        # Normalize and apply affine
        for h in range(0, H_out):
            for w in range(0, W_out):
                in_ptr = pid_n * y_in_stride_n + c_idx * y_in_stride_c + h * y_in_stride_h + w * y_in_stride_w
                x = tl.load(y_in_ptr + in_ptr)
                norm = (x - mean) * rstd
                y = norm * gamma + beta
                out_ptr = pid_n * y_out_stride_n + c_idx * y_out_stride_c + h * y_out_stride_h + w * y_out_stride_w
                tl.store(y_out_ptr + out_ptr, y)

# Triton elementwise SiLU: y = x * sigmoid(x)
@triton.jit
def silu_triton(x_ptr, y_ptr, N,
                 num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-x))
    y = x * s
    tl.store(y_ptr + idx, y)

# Triton elementwise residual addition: y = y + x (in-place to y)
@triton.jit
def add_residual_triton(y_ptr, x_ptr, N,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    y = tl.load(y_ptr + idx, mask=mask, other=0.0)
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    z = y + x
    tl.store(y_ptr + idx, z)

def _conv3x3_nobias_triton(x, weight, out=None):
    # x: (B, C_in, H, W), weight: (C_out, C_in, 3, 3), NCHW
    B, C_in, H, W = x.shape
    C_out = weight.shape[0]
    H_out = H - 2
    W_out = W - 2

    if out is None:
        out = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    x_strides = x.stride()
    w_strides = weight.stride()
    y_strides = out.stride()

    grid = (B, C_out, H_out, W_out)
    conv3x3_nchw_nobias[grid](x, weight, out,
                              B, C_in, C_out, H, W, H_out, W_out,
                              x_strides[0], x_strides[1], x_strides[2], x_strides[3],
                              w_strides[0], w_strides[1], w_strides[2], w_strides[3],
                              y_strides[0], y_strides[1], y_strides[2], y_strides[3],
                              num_warps=4, num_stages=2)
    return out

def _groupnorm_triton_fixed(y_in, weight, bias, y_out=None):
    # y_in: (B, C, H_out, W_out), num_groups=32, per-channel scale/bias
    B, C, H_out, W_out = y_in.shape
    num_groups = 32
    if C % num_groups != 0:
        raise ValueError(f"num_groups=32 must divide channels C={C}.")
    if y_out is None:
        y_out = torch.empty_like(y_in)

    y_in_strides = y_in.stride()
    y_out_strides = y_out.stride()

    grid = (B, num_groups)
    groupnorm_triton_fixed[grid](y_in, weight, bias, y_out,
                                 B, C, H_out, W_out, num_groups,
                                 y_in_strides[0], y_in_strides[1], y_in_strides[2], y_in_strides[3],
                                 y_out_strides[0], y_out_strides[1], y_out_strides[2], y_out_strides[3],
                                 num_warps=4, num_stages=2)
    return y_out

def _silu_triton(x, out=None):
    # x: any tensor, out: same shape
    N = x.numel()
    if out is None:
        out = torch.empty_like(x)

    x_contig = x.contiguous()
    out_contig = out.contiguous()

    grid = (triton.cdiv(N, 1024),)
    silu_triton[grid](x_contig, out_contig, N, num_warps=4, num_stages=2)

    if out_contig.shape != out.shape:
        out.copy_(out_contig.view(out.shape))
    else:
        out.copy_(out_contig)
    return out

def _add_residual_triton(y, x, out=None):
    # y: (B, C, H, W) , x: (B, C, H, W), out = y + x (in-place into y)
    N = y.numel()
    if out is None:
        out = y  # do in-place

    y_contig = y.contiguous()
    x_contig = x.contiguous()

    grid = (triton.cdiv(N, 1024),)
    add_residual_triton[grid](y_contig, x_contig, N, num_warps=4, num_stages=2)

    if out.data_ptr() != y.data_ptr():
        out.copy_(y_contig.view(y.shape))
    else:
        out = y_contig.view(y.shape)
    return out

class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        # Ensure inputs are contiguous
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        # Conv1
        y1 = _conv3x3_nobias_triton(x, conv1_weight)
        # GroupNorm1
        y1 = _groupnorm_triton_fixed(y1, norm1_weight, norm1_bias)
        # SiLU1
        y1 = _silu_triton(y1)
        # Conv2
        y2 = _conv3x3_nobias_triton(y1, conv2_weight)
        # GroupNorm2
        y2 = _groupnorm_triton_fixed(y2, norm2_weight, norm2_bias)
        # SiLU2
        y2 = _silu_triton(y2)
        # Residual add: y2 += x
        y2 = _add_residual_triton(y2, x)
        return y2


def run(*args):
    return ModelNew()(*args)
