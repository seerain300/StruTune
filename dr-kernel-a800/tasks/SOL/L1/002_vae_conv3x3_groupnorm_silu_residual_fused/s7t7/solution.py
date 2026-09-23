import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_innertile_kernel(
    x_ptr,          # *float32 input tensor (B, C_in, H, W)
    w_ptr,          # *float32 weight tensor (C_out, C_in, 3, 3)
    y_ptr,          # *float32 output tensor (B, C_out, H, W)
    N,              # int
    C_in,           # int
    H,              # int
    W,              # int
    C_out,          # int
):
    # Grid: (N, C_out, H, W)
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    acc = 0.0
    # Loop over input channels
    for cin in range(C_in):
        # Loop over 3x3 kernel taps
        for kh in range(3):
            for kw in range(3):
                ih = h + kh - 1
                iw = w + kw - 1
                # Valid region due to padding
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                # Linear index for input: (((n * C_in + cin) * H + ih) * W + iw)
                x_index = (((n * C_in + cin) * H + ih) * W + iw)
                x_val = tl.load(x_ptr + x_index, mask=valid, other=0.0)
                # Linear index for weight: (((c_out * C_in + cin) * 9) + (kh * 3 + kw))
                w_index = (((c_out * C_in + cin) * 9) + (kh * 3 + kw))
                w_val = tl.load(w_ptr + w_index)
                acc += x_val * w_val
    # Store output: y[n, c_out, h, w]
    y_index = (((n * C_out + c_out) * H + h) * W + w)
    tl.store(y_ptr + y_index, acc)


@triton.jit
def groupnorm_affine_kernel(
    x_ptr,          # *float32 input tensor (B, C, H, W)
    weight_ptr,     # *float32 GroupNorm scale (C,)
    bias_ptr,       # *float32 GroupNorm bias (C,)
    y_ptr,          # *float32 output tensor (B, C, H, W)
    N,              # int
    C,              # int
    H,              # int
    W,              # int
    num_groups,     # int (32)
    eps,            # float
):
    # Grid: (N, num_groups)
    n = tl.program_id(0)
    g = tl.program_id(1)
    channels_per_group = C // num_groups
    group_start = g * channels_per_group

    # First pass: compute sum and sumsq
    sum_val = 0.0
    sumsq_val = 0.0
    for cin in range(channels_per_group):
        c = group_start + cin
        for h in range(H):
            for w in range(W):
                index = (((n * C + c) * H + h) * W + w)
                x_val = tl.load(x_ptr + index)
                sum_val += x_val
                sumsq_val += x_val * x_val
    mean = sum_val / (channels_per_group * H * W)
    var = sumsq_val / (channels_per_group * H * W) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and affine
    for cin in range(channels_per_group):
        c = group_start + cin
        scale = tl.load(weight_ptr + c)
        beta = tl.load(bias_ptr + c)
        for h in range(H):
            for w in range(W):
                index = (((n * C + c) * H + h) * W + w)
                x_val = tl.load(x_ptr + index)
                y_val = (x_val - mean) * inv_std * scale + beta
                tl.store(y_ptr + index, y_val)


@triton.jit
def silu_kernel(
    x_ptr,          # *float32 input tensor (B, C, H, W)
    y_ptr,          # *float32 output tensor (B, C, H, W)
    N,              # int
    C,              # int
    H,              # int
    W,              # int
):
    # Grid: (N, C, H, W)
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    x_index = (((n * C + c) * H + h) * W + w)
    x_val = tl.load(x_ptr + x_index)
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig
    tl.store(y_ptr + x_index, y_val)


@triton.jit
def add_residual_kernel(
    y_ptr,          # *float32 first tensor (B, C, H, W)
    x_ptr,          # *float32 second tensor (B, C, H, W)
    out_ptr,        # *float32 output tensor (B, C, H, W)
    N,              # int
    C,              # int
    H,              # int
    W,              # int
):
    # Grid: (N, C, H, W)
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    index = (((n * C + c) * H + h) * W + w)
    y_val = tl.load(y_ptr + index)
    x_val = tl.load(x_ptr + index)
    out_val = y_val + x_val
    tl.store(out_ptr + index, out_val)


class ModelNew(torch.nn.Module):
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
        """
        Fused residual block implemented fully in Triton:
        Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add (x)
        """
        # Ensure CUDA tensors and float32
        assert x.is_cuda, "Input tensor must be on CUDA device for Triton kernels."
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Weights must be on CUDA device."
        x = x.contiguous().to(torch.float32)
        conv1_weight = conv1_weight.contiguous().to(torch.float32)
        conv2_weight = conv2_weight.contiguous().to(torch.float32)

        N, C, H, W = x.shape
        C1, C_in1, K_h1, K_w1 = conv1_weight.shape
        C2, C_in2, K_h2, K_w2 = conv2_weight.shape

        # Enforce GroupNorm divisibility requirement
        num_groups = 32
        assert C1 % num_groups == 0 and C2 % num_groups == 0, "num_groups=32 must divide channels for GroupNorm."

        # 1) Conv1: 3x3 stride=1, padding=1, bias=None
        y1 = torch.empty((N, C1, H, W), dtype=torch.float32, device=x.device)
        grid1 = (N, C1, H, W)
        conv3x3_stride1_pad1_innertile_kernel[grid1](
            x, conv1_weight, y1,
            N, C_in1, H, W, C1,
            num_warps=4, num_stages=2
        )

        # 2) GroupNorm1
        y1_gn = torch.empty_like(y1)
        groupnorm_affine_kernel[(N, num_groups)](
            y1, norm1_weight, norm1_bias, y1_gn,
            N, C1, H, W, num_groups, eps,
            num_warps=4, num_stages=2
        )

        # 3) SiLU1
        y1_silu = torch.empty_like(y1_gn)
        silu_kernel[(N, C1, H, W)](
            y1_gn, y1_silu,
            N, C1, H, W,
            num_warps=4, num_stages=2
        )

        # 4) Conv2: 3x3 stride=1, padding=1, bias=None
        y2 = torch.empty((N, C2, H, W), dtype=torch.float32, device=x.device)
        grid2 = (N, C2, H, W)
        conv3x3_stride1_pad1_innertile_kernel[grid2](
            y1_silu, conv2_weight, y2,
            N, C_in2, H, W, C2,
            num_warps=4, num_stages=2
        )

        # 5) GroupNorm2
        y2_gn = torch.empty_like(y2)
        groupnorm_affine_kernel[(N, num_groups)](
            y2, norm2_weight, norm2_bias, y2_gn,
            N, C2, H, W, num_groups, eps,
            num_warps=4, num_stages=2
        )

        # 6) SiLU2
        y2_silu = torch.empty_like(y2_gn)
        silu_kernel[(N, C2, H, W)](
            y2_gn, y2_silu,
            N, C2, H, W,
            num_warps=4, num_stages=2
        )

        # 7) Add residual x
        out = torch.empty_like(y2_silu)
        add_residual_kernel[(N, C2, H, W)](
            y2_silu, x, out,
            N, C2, H, W,
            num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
