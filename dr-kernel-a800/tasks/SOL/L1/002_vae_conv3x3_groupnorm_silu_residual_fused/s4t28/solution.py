import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 Conv (NCHW, stride=1, padding=1, no bias)
@triton.jit
def conv3x3_nchw_nobias(x_ptr, w_ptr, y_ptr,
                         B, C_in, C_out, H, W, H_out, W_out,
                         x_stride_n, x_stride_c, x_stride_h, x_stride_w,
                         w_stride_co, w_stride_ci, w_stride_dh, w_stride_dw,
                         y_stride_n, y_stride_c, y_stride_h, y_stride_w,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    # Grid: (B, C_out, H_out, W_out)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    acc = 0.0
    # Loop over input channels and 3x3 neighborhood
    for ci in range(0, C_in):
        for dh in [-1, 0, 1]:
            in_h = pid_h + dh
            if in_h < 0 or in_h >= H_out:
                continue
            for dw in [-1, 0, 1]:
                in_w = pid_w + dw
                if in_w < 0 or in_w >= W_out:
                    continue
                # Load input with padding (masked), handle negative indices
                x_off = pid_n * x_stride_n + ci * x_stride_c + in_h * x_stride_h + in_w * x_stride_w
                x_val = tl.load(x_ptr + x_off)  # x is float32
                # Load weight scalar
                w_off = pid_co * w_stride_co + ci * w_stride_ci + dh * w_stride_dh + dw * w_stride_dw
                w_val = tl.load(w_ptr + w_off)
                acc += x_val * w_val

    # Store output
    y_off = pid_n * y_stride_n + pid_co * y_stride_c + pid_h * y_stride_h + pid_w * y_stride_w
    tl.store(y_ptr + y_off, acc)


# Triton kernel: GroupNorm over channels (per-channel stats across spatial), num_groups fixed
# For this implementation, we treat GroupNorm as per-channel across the entire map (matches provided PyTorch code).
# grid: (B, C)
@triton.jit
def groupnorm_channels(y_in_ptr, weight_ptr, bias_ptr, y_out_ptr,
                        B, C, H_out, W_out,
                        y_in_stride_n, y_in_stride_c, y_in_stride_h, y_in_stride_w,
                        y_out_stride_n, y_out_stride_c, y_out_stride_h, y_out_stride_w,
                        num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    N = H_out * W_out  # number of spatial elements per channel

    # Compute sum and sum of squares across spatial plane
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

    # Apply normalization and affine (gamma, beta) per channel
    gamma = tl.load(weight_ptr + pid_c)
    beta = tl.load(bias_ptr + pid_c)

    for h in range(0, H_out):
        for w in range(0, W_out):
            in_ptr = pid_n * y_in_stride_n + pid_c * y_in_stride_c + h * y_in_stride_h + w * y_in_stride_w
            x = tl.load(y_in_ptr + in_ptr)
            norm = (x - mean) * rstd
            y = norm * gamma + beta
            out_ptr = pid_n * y_out_stride_n + pid_c * y_out_stride_c + h * y_out_stride_h + w * y_out_stride_w
            tl.store(y_out_ptr + out_ptr, y)


# Triton elementwise SiLU: y = x * sigmoid(x)
@triton.jit
def silu_triton(x_ptr, y_ptr, N, num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * num_warps + tl.arange(0, num_warps)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = x * (1.0 / (1.0 + tl.exp(-x)))  # sigmoid(x)
    tl.store(y_ptr + offs, y, mask=mask)


# Triton elementwise residual add: y = y + x (broadcast adds x to y)
@triton.jit
def add_residual_triton(y_ptr, x_ptr, out_ptr, N, num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * num_warps + tl.arange(0, num_warps)
    mask = offs < N
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, y + x, mask=mask)


def conv3x3_nchw_nobias_launch(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor):
    B, C_in, H, W = x.shape
    C_out, C_in_w, Hk, Wk = w.shape
    assert C_in == C_in_w and Hk == 3 and Wk == 3, "Weight shape must be (C_out, C_in, 3, 3)"
    H_out = H - 2
    W_out = W - 2
    # Ensure contiguous and float32
    x = x.contiguous().to(torch.float32)
    w = w.contiguous().to(torch.float32)
    y = y.contiguous().to(torch.float32)

    grid = (B, C_out, H_out, W_out)
    conv3x3_nchw_nobias[grid](
        x, w, y,
        B, C_in, C_out, H, W, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        num_warps=4, num_stages=2
    )
    return y, (H_out, W_out)


def groupnorm_channels_launch(y_in: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, y_out: torch.Tensor):
    B, C, H_out, W_out = y_in.shape
    grid = (B, C)
    groupnorm_channels[grid](
        y_in, weight, bias, y_out,
        B, C, H_out, W_out,
        y_in.stride(0), y_in.stride(1), y_in.stride(2), y_in.stride(3),
        y_out.stride(0), y_out.stride(1), y_out.stride(2), y_out.stride(3),
        num_warps=4, num_stages=2
    )
    return y_out


def silu_triton_launch(x: torch.Tensor, y: torch.Tensor):
    N = x.numel()
    grid = (triton.cdiv(N, 1024),)
    silu_triton[grid](x, y, N, num_warps=4, num_stages=2)
    return y


def add_residual_triton_launch(y: torch.Tensor, x: torch.Tensor, out: torch.Tensor):
    N = y.numel()
    grid = (triton.cdiv(N, 1024),)
    add_residual_triton[grid](y, x, out, N, num_warps=4, num_stages=2)
    return out


def run_triton(x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
               conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
    # Ensure CUDA tensors and float32
    x = x.contiguous().to(torch.float32)
    conv1_weight = conv1_weight.contiguous().to(torch.float32)
    norm1_weight = norm1_weight.contiguous().to(torch.float32)
    norm1_bias = norm1_bias.contiguous().to(torch.float32)
    conv2_weight = conv2_weight.contiguous().to(torch.float32)
    norm2_weight = norm2_weight.contiguous().to(torch.float32)
    norm2_bias = norm2_bias.contiguous().to(torch.float32)

    # 1) conv1
    y1 = torch.empty((x.shape[0], conv1_weight.shape[0], x.shape[2] - 2, x.shape[3] - 2), dtype=torch.float32, device=x.device)
    y1, (H1, W1) = conv3x3_nchw_nobias_launch(x, conv1_weight, y1)

    # 2) GroupNorm1 (per-channel across spatial)
    y1_norm = torch.empty_like(y1)
    y1_norm = groupnorm_channels_launch(y1, norm1_weight, norm1_bias, y1_norm)

    # 3) SiLU1
    y1_silu = torch.empty_like(y1_norm)
    y1_silu = silu_triton_launch(y1_norm, y1_silu)

    # 4) conv2
    y2 = torch.empty((y1_silu.shape[0], conv2_weight.shape[0], y1_silu.shape[2] - 2, y1_silu.shape[3] - 2), dtype=torch.float32, device=x.device)
    y2, (H2, W2) = conv3x3_nchw_nobias_launch(y1_silu, conv2_weight, y2)

    # 5) GroupNorm2 (per-channel across spatial)
    y2_norm = torch.empty_like(y2)
    y2_norm = groupnorm_channels_launch(y2, norm2_weight, norm2_bias, y2_norm)

    # 6) SiLU2
    y2_silu = torch.empty_like(y2_norm)
    y2_silu = silu_triton_launch(y2_norm, y2_silu)

    # 7) Residual add: y2_silu = y2_silu + x
    out = torch.empty_like(y2_silu)
    out = add_residual_triton_launch(y2_silu, x, out)

    return out


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        # Move tensors to CUDA if not already
        if not x.is_cuda:
            x = x.cuda()
        return run_triton(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)


def run(*args):
    return ModelNew()(*args)
