import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_per_channel_kernel(
    x_ptr,         # *float32 input tensor (B, C_in, H, W)
    w_ptr,         # *float32 weight tensor (C_out, C_in, 3, 3)
    y_ptr,         # *float32 output tensor (B, C_out, H, W)
    N, C_in, H, W, C_out,
):
    # Each program computes the entire output for one batch sample n and one output channel c_out
    n = tl.program_id(0)
    c_out = tl.program_id(1)

    # accumulator for this (n, c_out)
    acc = 0.0  # scalar float32

    # loop over input channels and 3x3 taps, handling padding
    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                for oh in range(H):
                    ih = oh + kh - 1
                    valid_h = (ih >= 0) & (ih < H)
                    for ow in range(W):
                        iw = ow + kw - 1
                        valid_w = (iw >= 0) & (iw < W)
                        valid = valid_h & valid_w
                        # input linear index: (((n * C_in + cin) * H + ih) * W + iw)
                        in_index = (((n * C_in + cin) * H + ih) * W + iw)
                        x_val = tl.load(x_ptr + in_index, mask=valid, other=0.0)
                        # weight for (c_out, cin, kh, kw)
                        w_index = (c_out * (C_in * 9)) + (cin * 9) + (kh * 3 + kw)
                        w_val = tl.load(w_ptr + w_index)
                        acc += x_val * w_val

    # write acc to y[n, c_out, :, :]
    # y layout: ((n * C_out + c_out) * H + h) * W + w
    for oh in range(H):
        for ow in range(W):
            y_index = ((n * C_out + c_out) * H + oh) * W + ow
            tl.store(y_ptr + y_index, acc)


@triton.jit
def group_norm_kernel(
    x_ptr,          # *float32 input tensor (B, C, H, W)
    scale_ptr,      # *float32 GroupNorm weight (C,)
    bias_ptr,       # *float32 GroupNorm bias (C,)
    y_ptr,          # *float32 output tensor (B, C, H, W)
    N, C, H, W,
    num_groups: tl.constexpr,   # must be 32 in our case
    eps: tl.constexpr,
):
    # Each program handles one sample n and one group g
    n = tl.program_id(0)
    g = tl.program_id(1)

    channels_per_group = C // num_groups
    group_start = g * channels_per_group

    # Compute sum and sum of squares over group across all spatial
    sum_val = 0.0
    sum_sq = 0.0
    for ch in range(channels_per_group):
        c = group_start + ch
        for h in range(H):
            for w in range(W):
                x_index = ((n * C + c) * H + h) * W + w
                x_val = tl.load(x_ptr + x_index)
                sum_val += x_val
                sum_sq += x_val * x_val

    M = channels_per_group * H * W
    mean = sum_val / M
    var = sum_sq / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine, write to y
    for ch in range(channels_per_group):
        c = group_start + ch
        scale = tl.load(scale_ptr + c)
        beta = tl.load(bias_ptr + c)
        for h in range(H):
            for w in range(W):
                x_index = ((n * C + c) * H + h) * W + w
                x_val = tl.load(x_ptr + x_index)
                y_val = (x_val - mean) * inv_std * scale + beta
                y_index = ((n * C + c) * H + h) * W + w
                tl.store(y_ptr + y_index, y_val)


@triton.jit
def silu_kernel(
    x_ptr,          # *float32 input tensor (B, C, H, W)
    y_ptr,          # *float32 output tensor (B, C, H, W)
    N, C, H, W,
):
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    x_index = ((n * C + c) * H + h) * W + w
                    x_val = tl.load(x_ptr + x_index)
                    sig = 1.0 / (1.0 + tl.exp(-x_val))
                    y_val = x_val * sig
                    y_index = ((n * C + c) * H + h) * W + w
                    tl.store(y_ptr + y_index, y_val)


# The original code does not add residual x; conv2 takes conv1 result. So we omit residual kernel.


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        conv1_weight: torch.Tensor,
        norm1_weight: torch.Tensor,
        norm1_bias: torch.Tensor,
        conv2_weight: torch.Tensor,
        norm2_weight: torch.Tensor,
        norm2_bias: torch.Tensor,
        eps: float,
    ):
        # Ensure tensors are on CUDA for Triton
        assert x.is_cuda and conv1_weight.is_cuda and norm1_weight.is_cuda and norm1_bias.is_cuda \
               and conv2_weight.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "All tensors must be on CUDA"

        device = x.device
        # We will compute in float32 for numerical stability
        x32 = x.to(torch.float32).contiguous()
        conv1_w32 = conv1_weight.to(torch.float32).contiguous()
        conv2_w32 = conv2_weight.to(torch.float32).contiguous()
        norm1_w32 = norm1_weight.to(torch.float32).contiguous()
        norm1_b32 = norm1_bias.to(torch.float32).contiguous()
        norm2_w32 = norm2_weight.to(torch.float32).contiguous()
        norm2_b32 = norm2_bias.to(torch.float32).contiguous()

        N, C, H, W = x32.shape
        C1_out = conv1_w32.shape[0]
        C2_out = conv2_w32.shape[0]

        # 1) Conv1: y1 = conv3x3(x, conv1_weight)
        y1 = torch.empty((N, C1_out, H, W), device=device, dtype=torch.float32)
        grid_conv1 = (N, C1_out)
        conv3x3_stride1_pad1_per_channel_kernel[grid_conv1](
            x32, conv1_w32, y1, N, C, H, W, C1_out,
            num_warps=4, num_stages=2
        )

        # 2) GroupNorm1 (C1_out must be divisible by 32)
        assert C1_out % 32 == 0, "num_groups=32 requires C1_out to be divisible by 32"
        y2 = torch.empty_like(y1)
        grid_gn1 = (N, 32)
        group_norm_kernel[grid_gn1](
            y1, norm1_w32, norm1_b32, y2, N, C1_out, H, W,
            num_groups=32, eps=eps,
            num_warps=4, num_stages=2
        )

        # 3) SiLU1
        y3 = torch.empty_like(y2)
        grid_silu1 = (N, C1_out, H, W)
        silu_kernel[grid_silu1](
            y2, y3, N, C1_out, H, W,
            num_warps=4, num_stages=2
        )

        # 4) Conv2: y4 = conv3x3(y3, conv2_weight)
        y4 = torch.empty((N, C2_out, H, W), device=device, dtype=torch.float32)
        grid_conv2 = (N, C2_out)
        conv3x3_stride1_pad1_per_channel_kernel[grid_conv2](
            y3, conv2_w32, y4, N, C1_out, H, W, C2_out,
            num_warps=4, num_stages=2
        )

        # 5) GroupNorm2 (C2_out must be divisible by 32)
        assert C2_out % 32 == 0, "num_groups=32 requires C2_out to be divisible by 32"
        y5 = torch.empty_like(y4)
        grid_gn2 = (N, 32)
        group_norm_kernel[grid_gn2](
            y4, norm2_w32, norm2_b32, y5, N, C2_out, H, W,
            num_groups=32, eps=eps,
            num_warps=4, num_stages=2
        )

        # 6) SiLU2
        y6 = torch.empty_like(y5)
        grid_silu2 = (N, C2_out, H, W)
        silu_kernel[grid_silu2](
            y5, y6, N, C2_out, H, W,
            num_warps=4, num_stages=2
        )

        return y6


def run(*args):
    return ModelNew()(*args)
