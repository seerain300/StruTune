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
    # program ids
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # accumulate
    acc = 0.0
    # loop over input channels
    for ci in range(0, C_in):
        # loop over 3x3 neighborhood
        for dh in range(0, 3):
            ih = pid_h * 3 + dh - 1  # (h*3 + dh) - 1 from padding
            valid_h = (ih >= 0) & (ih < H)
            for dw in range(0, 3):
                iw = pid_w * 3 + dw - 1  # (w*3 + dw) - 1 from padding
                valid_w = (iw >= 0) & (iw < W)
                # input pointer with masking
                ptr_in = pid_n * x_stride_n + ci * x_stride_c + ih * x_stride_h + iw * x_stride_w
                x_val = tl.load(x_ptr + ptr_in, mask=(valid_h & valid_w), other=0.0)
                # weight pointer: conv weights are (C_out, C_in, 3, 3)
                ptr_w = pid_co * w_stride_co + ci * w_stride_ci + dh * w_stride_dh + dw * w_stride_dw
                w_val = tl.load(w_ptr + ptr_w)
                acc += x_val * w_val

    # store output
    ptr_out = pid_n * y_stride_n + pid_co * y_stride_c + pid_h * y_stride_h + pid_w * y_stride_w
    tl.store(y_ptr + ptr_out, acc)


# Triton kernel: GroupNorm over per-channel (num_groups=32), apply affine scale/bias
# Each program handles one (n, c) pair and normalizes across all spatial positions of that channel.
@triton.jit
def groupnorm_channels(y_in_ptr, weight_ptr, bias_ptr, y_out_ptr,
                        B, C, H_out, W_out,
                        y_in_stride_n, y_in_stride_c, y_in_stride_h, y_in_stride_w,
                        y_out_stride_n, y_out_stride_c, y_out_stride_h, y_out_stride_w,
                        num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    N = H_out * W_out  # number of spatial elements per channel
    total_sum = 0.0
    total_sum_sq = 0.0

    # compute sum and sum of squares
    for h in range(0, H_out):
        for w in range(0, W_out):
            ptr = pid_n * y_in_stride_n + pid_c * y_in_stride_c + h * y_in_stride_h + w * y_in_stride_w
            x = tl.load(y_in_ptr + ptr)
            total_sum += x
            total_sum_sq += x * x

    mean = total_sum / N
    var = total_sum_sq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    gamma = tl.load(weight_ptr + pid_c)
    beta = tl.load(bias_ptr + pid_c)

    # normalize and write output
    for h in range(0, H_out):
        for w in range(0, W_out):
            ptr_in = pid_n * y_in_stride_n + pid_c * y_in_stride_c + h * y_in_stride_h + w * y_in_stride_w
            x = tl.load(y_in_ptr + ptr_in)
            norm = (x - mean) * rstd
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
def add_residual_triton(y_ptr, x_ptr, out_ptr, N,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    y = tl.load(y_ptr + idx, mask=mask, other=0.0)
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    out = y + x
    tl.store(out_ptr + idx, out, mask=mask)


def run_triton(x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor, conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
    # Ensure tensors are contiguous and on CUDA, dtype float32
    x = x.contiguous().to(dtype=torch.float32, device='cuda')
    conv1_weight = conv1_weight.contiguous().to(dtype=torch.float32, device='cuda')
    norm1_weight = norm1_weight.contiguous().to(dtype=torch.float32, device='cuda')
    norm1_bias = norm1_bias.contiguous().to(dtype=torch.float32, device='cuda')
    conv2_weight = conv2_weight.contiguous().to(dtype=torch.float32, device='cuda')
    norm2_weight = norm2_weight.contiguous().to(dtype=torch.float32, device='cuda')
    norm2_bias = norm2_bias.contiguous().to(dtype=torch.float32, device='cuda')

    B, C, H, W = x.shape
    C_w1, C_in, KH, KW = conv1_weight.shape  # C_w1 == C_out for conv1
    C_w2, C_w1_2, KH2, KW2 = conv2_weight.shape  # C_w2 == C for conv2

    # conv1
    H1 = H - 2  # output spatial size for padding=1, stride=1
    W1 = W - 2
    y1 = torch.empty((B, C_w1, H1, W1), device='cuda', dtype=torch.float32)

    grid1 = (B, C_w1, H1, W1)
    conv3x3_nchw_nobias[grid1](
        x, conv1_weight, y1,
        B, C_in, C_w1, H, W, H1, W1,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        num_warps=4, num_stages=2
    )

    # GroupNorm1, per-channel over spatial
    y1_norm = torch.empty_like(y1)
    grid_gn1 = (B, C_w1)
    groupnorm_channels[grid_gn1](
        y1, norm1_weight, norm1_bias, y1_norm,
        B, C_w1, H1, W1,
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
        num_warps=4, num_stages=2
    )

    # SiLU1
    y1_silu = torch.empty_like(y1_norm)
    N1 = B * C_w1 * H1 * W1
    grid_silu1 = (triton.cdiv(N1, 1024),)
    silu_triton[grid_silu1](
        y1_norm, y1_silu, N1,
        num_warps=4, num_stages=2
    )

    # conv2
    H2 = H - 4  # conv applied twice with padding=1 each
    W2 = W - 4
    y2 = torch.empty((B, C_w2, H2, W2), device='cuda', dtype=torch.float32)

    grid2 = (B, C_w2, H2, W2)
    conv3x3_nchw_nobias[grid2](
        y1_silu, conv2_weight, y2,
        B, C_w1_2, C_w2, H1, W1, H2, W2,
        y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
        conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        num_warps=4, num_stages=2
    )

    # GroupNorm2, per-channel over spatial
    y2_norm = torch.empty_like(y2)
    grid_gn2 = (B, C_w2)
    groupnorm_channels[grid_gn2](
        y2, norm2_weight, norm2_bias, y2_norm,
        B, C_w2, H2, W2,
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
        num_warps=4, num_stages=2
    )

    # SiLU2
    y2_silu = torch.empty_like(y2_norm)
    N2 = B * C_w2 * H2 * W2
    grid_silu2 = (triton.cdiv(N2, 1024),)
    silu_triton[grid_silu2](
        y2_norm, y2_silu, N2,
        num_warps=4, num_stages=2
    )

    # Residual add: y2_silu + x
    # y2_silu: (B, C_w2, H2, W2), x: (B, C, H, W), broadcasting adds last two dims:
    # We need shapes to match for elementwise add; given the original code path, C_w2 == C and H2=W2==H-4=W-4 in the residual addition step. For generality, ensure we broadcast safely.
    out = torch.empty((B, C_w2, H2, W2), device='cuda', dtype=torch.float32)
    Nres = N2
    grid_add = (triton.cdiv(Nres, 1024),)
    add_residual_triton[grid_add](
        y2_silu, x, out, Nres,
        num_warps=4, num_stages=2
    )

    return out


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor, conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        # All computation in Triton; ensure x is on CUDA and contiguous
        if not x.is_cuda:
            x = x.cuda()
        return run_triton(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)


def run(*args):
    return ModelNew()(*args)
