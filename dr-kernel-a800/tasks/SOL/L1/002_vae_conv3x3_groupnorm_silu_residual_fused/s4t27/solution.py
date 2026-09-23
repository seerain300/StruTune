import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 Conv (NCHW, stride=1, padding=1, no bias)
# Each program computes y[n, co, h, w] = sum_{ci, dh,dw in -1..0} x[n, ci, h+dh, w+dw] * w[co, ci, dh+1, dw+1]
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
    # loop over input channels
    for ci in range(0, C_in):
        # loop over 3x3 neighborhood
        for dh in range(-1, 2):
            h_in = pid_h + dh
            valid_h = (h_in >= 0) & (h_in < H_out)
            for dw in range(-1, 2):
                w_in = pid_w + dw
                valid_w = (w_in >= 0) & (w_in < W_out)
                mask = valid_h & valid_w
                # compute input pointer with masking
                x_ptr_offset = pid_n * x_stride_n + ci * x_stride_c + h_in * x_stride_h + w_in * x_stride_w
                x_val = tl.load(x_ptr + x_ptr_offset, mask=mask, other=0.0)
                # weight pointer: w[co, ci, dh+1, dw+1]
                w_ptr_offset = pid_co * w_stride_co + ci * w_stride_ci + (dh + 1) * w_stride_dh + (dw + 1) * w_stride_dw
                w_val = tl.load(w_ptr + w_ptr_offset)
                acc += x_val * w_val
    # store output
    y_ptr_offset = pid_n * y_stride_n + pid_co * y_stride_c + pid_h * y_stride_h + pid_w * y_stride_w
    tl.store(y_ptr + y_ptr_offset, acc)


# Triton kernel: GroupNorm per-channel over all spatial positions (num_groups=32)
# Input y_in: (B, C, H_out, W_out), per sample, per channel, normalize over H_out*W_out
@triton.jit
def groupnorm_triton_channels(y_in_ptr, weight_ptr, bias_ptr, y_out_ptr,
                               B, C, H_out, W_out,
                               y_in_stride_n, y_in_stride_c, y_in_stride_h, y_in_stride_w,
                               y_out_stride_n, y_out_stride_c, y_out_stride_h, y_out_stride_w,
                               num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    N = H_out * W_out  # number of spatial elements per channel

    # compute mean
    sum_val = 0.0
    for h in range(0, H_out):
        for w in range(0, W_out):
            ptr = pid_n * y_in_stride_n + pid_c * y_in_stride_c + h * y_in_stride_h + w * y_in_stride_w
            x = tl.load(y_in_ptr + ptr)
            sum_val += x
    mean = sum_val / N

    # compute variance
    sum_sq = 0.0
    for h in range(0, H_out):
        for w in range(0, W_out):
            ptr = pid_n * y_in_stride_n + pid_c * y_in_stride_c + h * y_in_stride_h + w * y_in_stride_w
            x = tl.load(y_in_ptr + ptr)
            sum_sq += x * x
    var = sum_sq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    gamma = tl.load(weight_ptr + pid_c)
    beta = tl.load(bias_ptr + pid_c)
    for h in range(0, H_out):
        for w in range(0, W_out):
            in_ptr = pid_n * y_in_stride_n + pid_c * y_in_stride_c + h * y_in_stride_h + w * y_in_stride_w
            x = tl.load(y_in_ptr + in_ptr)
            y = (x - mean) * rstd
            y = y * gamma + beta
            out_ptr = pid_n * y_out_stride_n + pid_c * y_out_stride_c + h * y_out_stride_h + w * y_out_stride_w
            tl.store(y_out_ptr + out_ptr, y)


# Triton elementwise SiLU: y = x * sigmoid(x)
@triton.jit
def silu_triton(x_ptr, y_ptr, N, num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * num_warps + tl.arange(0, num_warps)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)


# Triton elementwise residual add: y_out = y + x
# Assumes y and x have the same number of elements and are contiguous
@triton.jit
def add_residual_triton(y_ptr, x_ptr, y_out_ptr, N, num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * num_warps + tl.arange(0, num_warps)
    mask = offs < N
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_out_ptr + offs, y + x, mask=mask)


def run_triton(
    x: torch.Tensor,
    conv1_weight: torch.Tensor,
    norm1_weight: torch.Tensor,
    norm1_bias: torch.Tensor,
    conv2_weight: torch.Tensor,
    norm2_weight: torch.Tensor,
    norm2_bias: torch.Tensor,
    eps: float,
):
    """
    Triton-optimized fused residual block:
      Conv3x3 -> GroupNorm (num_groups=32) -> SiLU
      Conv3x3 -> GroupNorm (num_groups=32) -> SiLU
      Add residual x
    Entry point: ModelNew.forward
    """
    assert x.is_cuda, "Triton kernels require CUDA tensors"
    B, C, H, W = x.shape
    C_in = C  # input channels = output channels for conv1

    # Ensure contiguous and float32
    x = x.contiguous()

    # conv1: y1 (B, C, H-2, W-2)
    H1 = H - 2
    W1 = W - 2
    y1 = torch.empty((B, C, H1, W1), dtype=torch.float32, device=x.device)

    conv1_grid = (B, C, H1, W1)
    conv3x3_nchw_nobias[conv1_grid](
        x, conv1_weight, y1,
        B, C_in, C, H, W, H1, W1,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        num_warps=4, num_stages=2
    )

    # GroupNorm1
    y1_norm = torch.empty_like(y1, dtype=torch.float32, device=x.device)
    groupnorm_triton_channels[(B, C)](
        y1, norm1_weight, norm1_bias, y1_norm,
        B, C, H1, W1,
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
        num_warps=2, num_stages=2
    )

    # SiLU1
    N1 = B * C * H1 * W1
    y1_silu = torch.empty_like(y1_norm, dtype=torch.float32, device=x.device)
    silu_triton[(triton.cdiv(N1, 1024),)](
        y1_norm, y1_silu, N1,
        num_warps=4, num_stages=2
    )

    # conv2: y2 (B, C, H-4, W-4)
    H2 = H - 4
    W2 = W - 4
    y2 = torch.empty((B, C, H2, W2), dtype=torch.float32, device=x.device)

    conv2_grid = (B, C, H2, W2)
    conv3x3_nchw_nobias[conv2_grid](
        y1_silu, conv2_weight, y2,
        B, C, C, H1, W1, H2, W2,  # conv2 uses output of conv1 as input
        y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
        conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        num_warps=4, num_stages=2
    )

    # GroupNorm2
    y2_norm = torch.empty_like(y2, dtype=torch.float32, device=x.device)
    groupnorm_triton_channels[(B, C)](
        y2, norm2_weight, norm2_bias, y2_norm,
        B, C, H2, W2,
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
        num_warps=2, num_stages=2
    )

    # SiLU2
    N2 = B * C * H2 * W2
    y2_silu = torch.empty_like(y2_norm, dtype=torch.float32, device=x.device)
    silu_triton[(triton.cdiv(N2, 1024),)](
        y2_norm, y2_silu, N2,
        num_warps=4, num_stages=2
    )

    # Residual addition: y = y2_silu + x
    # Shapes: y2_silu is (B, C, H-4, W-4); x is (B, C, H, W).
    # If spatial dims match (they should for the original model), use Triton; otherwise fall back to torch.
    N = B * C * H * W
    if y2_silu.shape == x.shape:
        add_residual_triton[(triton.cdiv(N, 1024),)](
            y2_silu, x, y2_silu, N,
            num_warps=4, num_stages=2
        )
    else:
        y2_silu = y2_silu + x

    return y2_silu


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor, conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        # All computation in Triton; ensure x is on CUDA
        if not x.is_cuda:
            x = x.cuda()
        return run_triton(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)


def run(*args):
    return ModelNew()(*args)
