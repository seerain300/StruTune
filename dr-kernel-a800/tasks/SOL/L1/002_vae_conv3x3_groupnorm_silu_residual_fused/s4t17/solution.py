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
    pid_n = tl.program_id(0)  # batch
    pid_co = tl.program_id(1)  # output channel
    pid_h = tl.program_id(2)  # output height
    pid_w = tl.program_id(3)  # output width

    # accumulator
    acc = 0.0

    # loop over input channels and 3x3 neighborhood
    for ci in range(0, C_in):
        for dh in range(-1, 2):
            in_h = pid_h + dh
            # safe bounds check
            if (in_h >= 0) & (in_h < H):
                for dw in range(-1, 2):
                    in_w = pid_w + dw
                    if (in_w >= 0) & (in_w < W):
                        x_offset = pid_n * x_stride_n + ci * x_stride_c + in_h * x_stride_h + in_w * x_stride_w
                        # weight index: (co, ci, dh, dw) -> co*stride_co + ci*stride_ci + dh*stride_dh + dw*stride_dw
                        w_offset = pid_co * w_stride_co + ci * w_stride_ci + dh * w_stride_dh + dw * w_stride_dw
                        x_val = tl.load(x_ptr + x_offset)  # masked loads handled by bounds
                        w_val = tl.load(w_ptr + w_offset)
                        acc += x_val * w_val

    # store result
    y_offset = pid_n * y_stride_n + pid_co * y_stride_c + pid_h * y_stride_h + pid_w * y_stride_w
    tl.store(y_ptr + y_offset, acc)

# Triton kernel: GroupNorm per channel (per-channel stats across spatial), num_groups is not used here
# y_in shape: (B, C, H_out, W_out), y_out shape same
@triton.jit
def groupnorm_triton_channels(y_in_ptr, weight_ptr, bias_ptr, y_out_ptr,
                               B, C, H_out, W_out,
                               y_in_stride_n, y_in_stride_c, y_in_stride_h, y_in_stride_w,
                               y_out_stride_n, y_out_stride_c, y_out_stride_h, y_out_stride_w,
                               num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid_n = tl.program_id(0)  # batch
    pid_c = tl.program_id(1)  # channel

    N = H_out * W_out  # spatial elements per channel
    sum_val = 0.0
    sum_sq = 0.0

    # accumulate sum and sum of squares over spatial plane
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

    # normalize and affine per element
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

# Triton elementwise residual add: y = y + x (broadcast residual x over spatial of y)
@triton.jit
def add_residual_triton(x_ptr, y_ptr, N,
                         num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * num_warps + tl.arange(0, num_warps)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    y = tl.load(y_ptr + idx, mask=mask, other=0.0)
    y = y + x
    tl.store(y_ptr + idx, y, mask=mask)

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
        # x: (B, C, H, W), conv weights: (C, C, 3, 3), norm scale/bias: (C,)
        # Ensure CUDA tensors
        assert x.is_cuda, "Input must be on CUDA device for Triton"
        B, C, H, W = x.shape
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        # conv1: output (B, C, H-2, W-2)
        H_out1 = H - 2
        W_out1 = W - 2
        y1 = torch.empty((B, C, H_out1, W_out1), device=x.device, dtype=x.dtype)
        grid1 = (B, C, H_out1, W_out1)
        conv3x3_nchw_nobias[grid1](
            x, conv1_weight, y1,
            B, C, C, H, W, H_out1, W_out1,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            num_warps=4, num_stages=2
        )

        # GroupNorm1 per channel
        y1_norm = torch.empty_like(y1)
        grid_gn1 = (B, C)
        groupnorm_triton_channels[grid_gn1](
            y1, norm1_weight, norm1_bias, y1_norm,
            B, C, H_out1, W_out1,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            num_warps=4, num_stages=2
        )

        # SiLU1
        y1_silu = torch.empty_like(y1_norm)
        N1 = B * C * H_out1 * W_out1
        grid_silu1 = (triton.cdiv(N1, 1024),)
        silu_triton[grid_silu1](
            y1_norm, y1_silu, N1,
            num_warps=4, num_stages=2
        )

        # conv2: output (B, C, H-4, W-4)
        H_out2 = H - 4
        W_out2 = W - 4
        y2 = torch.empty((B, C, H_out2, W_out2), device=x.device, dtype=x.dtype)
        grid2 = (B, C, H_out2, W_out2)
        conv3x3_nchw_nobias[grid2](
            y1_silu, conv2_weight, y2,
            B, C, C, H_out1, W_out1, H_out2, W_out2,
            y1_silu.stride(0), y1_silu.stride(1), y1_silu.stride(2), y1_silu.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            num_warps=4, num_stages=2
        )

        # GroupNorm2 per channel
        y2_norm = torch.empty_like(y2)
        grid_gn2 = (B, C)
        groupnorm_triton_channels[grid_gn2](
            y2, norm2_weight, norm2_bias, y2_norm,
            B, C, H_out2, W_out2,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
            num_warps=4, num_stages=2
        )

        # SiLU2
        y2_silu = torch.empty_like(y2_norm)
        N2 = B * C * H_out2 * W_out2
        grid_silu2 = (triton.cdiv(N2, 1024),)
        silu_triton[grid_silu2](
            y2_norm, y2_silu, N2,
            num_warps=4, num_stages=2
        )

        # Residual addition: y = y2_silu + x
        N_out = N2
        grid_add = (triton.cdiv(N_out, 1024),)
        add_residual_triton[grid_add](
            x.reshape(-1), y2_silu.reshape(-1), N_out,
            num_warps=4, num_stages=2
        )

        return y2_silu


def run(*args):
    return ModelNew()(*args)
