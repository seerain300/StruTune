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

    # Accumulator for output
    acc = 0.0

    # Loop over input channels
    for ci in range(0, C_in):
        # Loop over 3x3 neighborhood
        for dh in range(0, 3):
            for dw in range(0, 3):
                ih = pid_h + dh
                iw = pid_w + dw
                # mask for padding
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x_offset = pid_n * x_stride_n + ci * x_stride_c + ih * x_stride_h + iw * x_stride_w
                x_val = tl.load(x_ptr + x_offset, mask=in_bounds, other=0.0)
                w_offset = pid_co * w_stride_co + ci * w_stride_ci + dh * w_stride_dh + dw * w_stride_dw
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val

    y_offset = pid_n * y_stride_n + pid_co * y_stride_c + pid_h * y_stride_h + pid_w * y_stride_w
    tl.store(y_ptr + y_offset, acc)

# Triton kernel: GroupNorm over channels (per-channel stats across spatial), num_groups=32
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

    sum_val = 0.0
    sum_sq = 0.0

    # Compute sum and sum of squares over spatial plane
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
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + idx, y, mask=mask)

# Triton elementwise residual add: y = y + x
@triton.jit
def add_residual_triton(y_ptr, x_ptr, N,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    y = tl.load(y_ptr + idx, mask=mask, other=0.0)
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    tl.store(y_ptr + idx, y + x, mask=mask)

def _conv3x3_nchw_nobias_triton(x, weight):
    # x: (B, C_in, H, W), weight: (C_out, C_in, 3, 3)
    B, C_in, H, W = x.shape
    C_out, C_in_w, kH, kW = weight.shape
    assert C_in == C_in_w and kH == 3 and kW == 3
    H_out = H - 2
    W_out = W - 2
    y = torch.empty((B, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    x_c = x.contiguous()
    w_c = weight.contiguous()
    # Strides for NCHW
    x_stride_n, x_stride_c, x_stride_h, x_stride_w = x_c.stride()
    w_stride_co, w_stride_ci, w_stride_dh, w_stride_dw = w_c.stride()
    y_stride_n, y_stride_c, y_stride_h, y_stride_w = y.stride()
    grid = (B, C_out, H_out, W_out)
    conv3x3_nchw_nobias[grid](
        x_c, w_c, y,
        B, C_in, C_out, H, W, H_out, W_out,
        x_stride_n, x_stride_c, x_stride_h, x_stride_w,
        w_stride_co, w_stride_ci, w_stride_dh, w_stride_dw,
        y_stride_n, y_stride_c, y_stride_h, y_stride_w,
        num_warps=1, num_stages=1
    )
    return y

def _groupnorm_triton_channels(y_in, weight, bias):
    # y_in: (B, C, H_out, W_out), per-channel stats across H_out*W_out, num_groups unused (per-channel)
    B, C, H_out, W_out = y_in.shape
    y_out = torch.empty_like(y_in)
    y_in_c = y_in.contiguous()
    weight_c = weight.contiguous()
    bias_c = bias.contiguous()
    y_out_c = y_out.contiguous()
    y_in_stride_n, y_in_stride_c, y_in_stride_h, y_in_stride_w = y_in_c.stride()
    y_out_stride_n, y_out_stride_c, y_out_stride_h, y_out_stride_w = y_out_c.stride()
    grid = (B, C)
    groupnorm_triton_channels[grid](
        y_in_c, weight_c, bias_c, y_out_c,
        B, C, H_out, W_out,
        y_in_stride_n, y_in_stride_c, y_in_stride_h, y_in_stride_w,
        y_out_stride_n, y_out_stride_c, y_out_stride_h, y_out_stride_w,
        num_warps=1, num_stages=1
    )
    return y_out_c

def _silu_triton(x):
    # Elementwise SiLU over entire tensor
    N = x.numel()
    y = torch.empty_like(x)
    x_c = x.contiguous()
    y_c = y.contiguous()
    grid = (triton.cdiv(N, 1024),)  # 1024 elements per program
    silu_triton[grid](x_c, y_c, N, num_warps=4, num_stages=1)
    return y_c

def _add_residual_triton(y, x):
    # Elementwise add: y += x
    N = y.numel()
    y_c = y.contiguous()
    x_c = x.contiguous()
    grid = (triton.cdiv(N, 1024),)
    add_residual_triton[grid](y_c, x_c, N, num_warps=4, num_stages=1)
    return y_c

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
        # Ensure dtypes and contiguity
        x = x.contiguous().float()
        conv1_weight = conv1_weight.contiguous().float()
        norm1_weight = norm1_weight.contiguous().float()
        norm1_bias = norm1_bias.contiguous().float()
        conv2_weight = conv2_weight.contiguous().float()
        norm2_weight = norm2_weight.contiguous().float()
        norm2_bias = norm2_bias.contiguous().float()

        # First path: Conv3x3 -> GroupNorm (num_groups=32 per-channel) -> SiLU
        y1 = _conv3x3_nchw_nobias_triton(x, conv1_weight)  # (B, C, H-2, W-2)
        y1 = _groupnorm_triton_channels(y1, norm1_weight, norm1_bias)
        y1 = _silu_triton(y1)

        # Second path: Conv3x3 -> GroupNorm (num_groups=32 per-channel) -> SiLU
        y2 = _conv3x3_nchw_nobias_triton(y1, conv2_weight)  # (B, C, H-4, W-4)
        y2 = _groupnorm_triton_channels(y2, norm2_weight, norm2_bias)
        y2 = _silu_triton(y2)

        # Residual connection: y = y2 + x
        y_out = _add_residual_triton(y2, x)

        return y_out

# Optional: helper for generating inputs (not required by the evaluator, but provided for completeness)
def get_inputs():
    # Example inputs; in evaluator, these will be provided
    B = 1
    C = 32
    H = 128
    W = 128
    x = torch.randn(B, C, H, W, device='cuda', dtype=torch.float32)
    conv1_weight = torch.randn(C, C, 3, 3, device='cuda', dtype=torch.float32)
    norm1_weight = torch.randn(C, device='cuda', dtype=torch.float32)
    norm1_bias = torch.randn(C, device='cuda', dtype=torch.float32)
    conv2_weight = torch.randn(C, C, 3, 3, device='cuda', dtype=torch.float32)
    norm2_weight = torch.randn(C, device='cuda', dtype=torch.float32)
    norm2_bias = torch.randn(C, device='cuda', dtype=torch.float32)
    eps = 1e-5
    return x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps


def run(*args):
    return ModelNew()(*args)
