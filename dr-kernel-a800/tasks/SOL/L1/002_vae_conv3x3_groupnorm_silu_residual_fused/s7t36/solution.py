import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_single_out_kernel(
    x_ptr,           # *float32, input tensor (B, C_in, H, W)
    w_ptr,           # *float32, weight tensor (C_out, C_in, 3, 3)
    y_ptr,           # *float32, output tensor (B, C_out, H_out, W_out)
    B, C_in, H, W, C_out, H_out, W_out,
):
    # One program computes one output element y[n, oc, oh, ow]
    n = tl.program_id(0)
    oc = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # Accumulator in float32
    acc = 0.0

    # Construct 3x3 input patch and dot with corresponding weights
    # For each input channel and each kernel tap
    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                ih = oh + kh - 1  # stride=1, pad=1
                iw = ow + kw - 1
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                # Linear index for input: (((n * C_in + cin) * H + ih) * W + iw)
                in_index = (((n * C_in + cin) * H + ih) * W + iw)
                x_val = tl.load(x_ptr + in_index, mask=valid, other=0.0)

                # Linear index for weight: (((oc * C_in + cin) * 9) + (kh * 3 + kw))
                w_index = (((oc * C_in + cin) * 9) + (kh * 3 + kw))
                w_val = tl.load(w_ptr + w_index)

                acc += x_val * w_val

    # Store to output: linear index y[n, oc, oh, ow]
    out_index = (((n * C_out + oc) * H_out + oh) * W_out + ow)
    tl.store(y_ptr + out_index, acc)


@triton.jit
def group_norm_affine_kernel(
    x_ptr,        # *float32, input tensor (B, C, H, W)
    gamma_ptr,    # *float32, scale (C,)
    beta_ptr,     # *float32, bias (C,)
    y_ptr,        # *float32, output tensor (B, C, H, W)
    N, C, H, W,
    num_groups: tl.constexpr,  # must be 32 here
    eps: tl.constexpr,
):
    # One program per (n, group)
    n = tl.program_id(0)
    g = tl.program_id(1)
    channels_per_group = C // num_groups
    group_start = g * channels_per_group

    # Accumulate sum and sumsq across group channels and all H*W
    sum_val = 0.0
    sumsq_val = 0.0

    # First pass: compute sum and sumsq
    for c in range(channels_per_group):
        c_idx = group_start + c
        for h in range(H):
            for w in range(W):
                x_index = (((n * C + c_idx) * H + h) * W + w)
                x_val = tl.load(x_ptr + x_index)
                sum_val += x_val
                sumsq_val += x_val * x_val

    numel_group = channels_per_group * H * W
    mean = sum_val / numel_group
    var = sumsq_val / numel_group - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, write to y
    for c in range(channels_per_group):
        c_idx = group_start + c
        gamma = tl.load(gamma_ptr + c_idx)
        beta = tl.load(beta_ptr + c_idx)
        for h in range(H):
            for w in range(W):
                x_index = (((n * C + c_idx) * H + h) * W + w)
                x_val = tl.load(x_ptr + x_index)
                y_val = (x_val - mean) * inv_std
                y_val = y_val * gamma + beta
                y_index = x_index  # y has same layout as x
                tl.store(y_ptr + y_index, y_val)


@triton.jit
def silu_kernel(
    x_ptr, y_ptr, N, C, H, W,
):
    # One program per element: n, c, h, w
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    x_index = (((n * C + c) * H + h) * W + w)
    x_val = tl.load(x_ptr + x_index)
    # SiLU: x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig
    tl.store(y_ptr + x_index, y_val)


@triton.jit
def add_residual_kernel(
    y_ptr, x_ptr, N, C, H, W,
):
    # One program per element: n, c, h, w
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    y_index = (((n * C + c) * H + h) * W + w)
    x_index = y_index  # assuming y and x have same shape and layout
    y_val = tl.load(y_ptr + y_index)
    x_val = tl.load(x_ptr + x_index)
    out = y_val + x_val
    tl.store(y_ptr + y_index, out)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Nothing to initialize; weights are passed at call time

    def forward(self,
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
        Triton-only fused residual block:
        Conv3x3 -> GroupNorm(num_groups=32) -> SiLU
        Conv3x3 -> GroupNorm(num_groups=32) -> SiLU
        Add residual
        """
        # Ensure CUDA tensors and float32 for numerical stability
        assert x.is_cuda, "Input tensor must be on CUDA device"
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Weight tensors must be on CUDA"
        assert norm1_weight.is_cuda and norm1_bias.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "GroupNorm params must be on CUDA"

        x = x.contiguous().float()
        B, C, H, W = x.shape
        assert C % 32 == 0, "num_groups=32 requires C % 32 == 0"

        C_in = C  # after first conv, C_in equals conv1_weight.shape[0], but here conv1_weight=(C, C, 3, 3)
        # We need to infer C_in for conv1: conv1_weight shape is (C, C, 3, 3) => C_in=C
        C1_out = C  # conv1_weight: (C_out, C_in, 3,3) => C_out=C
        C2_in = C1_out  # conv2 input channels equal C1_out
        C2_out = C  # conv2_weight: (C_out, C_in, 3,3) => C_out=C

        # Allocate output tensors
        y1 = torch.empty((B, C1_out, H, W), dtype=torch.float32, device=x.device)

        # Launch conv1: y1 = conv2d(x, conv1_weight)
        grid_conv1 = (B, C1_out, H, W)
        conv3x3_stride1_pad1_single_out_kernel[grid_conv1](
            x, conv1_weight, y1, B, C, H, W, C1_out, H, W, num_warps=1, num_stages=1
        )

        # GroupNorm1
        y1_norm = torch.empty_like(y1)
        group_norm_affine_kernel[(B, 32)](y1, norm1_weight, norm1_bias, y1_norm, B, C1_out, H, W,
                                           num_groups=32, eps=eps, num_warps=4, num_stages=2)

        # SiLU1
        y1_silu = torch.empty_like(y1_norm)
        grid_silu1 = (B, C1_out, H, W)
        silu_kernel[grid_silu1](y1_norm, y1_silu, B, C1_out, H, W, num_warps=1, num_stages=1)

        # Conv2
        y2 = torch.empty((B, C2_out, H, W), dtype=torch.float32, device=x.device)
        grid_conv2 = (B, C2_out, H, W)
        conv3x3_stride1_pad1_single_out_kernel[grid_conv2](
            y1_silu, conv2_weight, y2, B, C1_out, H, W, C2_out, H, W, num_warps=1, num_stages=1
        )

        # GroupNorm2
        y2_norm = torch.empty_like(y2)
        group_norm_affine_kernel[(B, 32)](y2, norm2_weight, norm2_bias, y2_norm, B, C2_out, H, W,
                                          num_groups=32, eps=eps, num_warps=4, num_stages=2)

        # SiLU2
        y2_silu = torch.empty_like(y2_norm)
        grid_silu2 = (B, C2_out, H, W)
        silu_kernel[grid_silu2](y2_norm, y2_silu, B, C2_out, H, W, num_warps=1, num_stages=1)

        # Residual add: y_out = y2_silu + x
        y_out = torch.empty_like(y2_silu)
        add_residual_kernel[grid_silu2](y2_silu, x, B, C2_out, H, W, num_warps=1, num_stages=1)

        return y_out


# Example usage (ensure tensors are on CUDA):
# model = ModelNew().cuda()
# x = torch.randn(B, C, H, W, device='cuda', dtype=torch.float32)
# conv1_weight = torch.randn(C, C, 3, 3, device='cuda', dtype=torch.float32)
# norm1_weight = torch.randn(C, device='cuda', dtype=torch.float32)
# norm1_bias = torch.randn(C, device='cuda', dtype=torch.float32)
# conv2_weight = torch.randn(C, C, 3, 3, device='cuda', dtype=torch.float32)
# norm2_weight = torch.randn(C, device='cuda', dtype=torch.float32)
# norm2_bias = torch.randn(C, device='cuda', dtype=torch.float32)
# eps = 1e-5
# out = model(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)


def run(*args):
    return ModelNew()(*args)
