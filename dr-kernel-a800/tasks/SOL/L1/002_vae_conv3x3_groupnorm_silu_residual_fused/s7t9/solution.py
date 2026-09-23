import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_innertile_kernel(
    x_ptr,            # *float32 input tensor (B, C_in, H, W)
    w_ptr,            # *float32 weight tensor (C_out, C_in, 3, 3)
    y_ptr,            # *float32 output tensor (B, C_out, H, W)
    B: tl.constexpr,  # int
    C_in,             # int
    C_out,            # int
    H,                # int
    W,                # int
):
    # Grid: (B, C_out, H, W) => each program handles one output element y[n, c_out, h, w]
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 taps
    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                ih = h + kh - 1  # stride=1, pad=1
                iw = w + kw - 1
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

                # Input linear index: (((n * C_in + cin) * H + ih) * W + iw)
                in_index = (((n * C_in + cin) * H + ih) * W + iw)

                # Weight scalar for (c_out, cin, kh, kw)
                w_index = (c_out * C_in + cin) * 9 + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_index)  # scalar
                # Load input or 0 if out-of-bounds
                x_val = tl.load(x_ptr + in_index, mask=valid, other=0.0)
                acc += x_val * w_val

    # Store result: y[n, c_out, h, w]
    y_index = (((n * C_out + c_out) * H + h) * W + w)
    tl.store(y_ptr + y_index, acc)


@triton.jit
def groupnorm_affine_kernel(
    x_ptr,            # *float32 input tensor (B, C, H, W)
    gamma_ptr,        # *float32 GroupNorm scale (C,)
    beta_ptr,         # *float32 GroupNorm bias (C,)
    y_ptr,            # *float32 output tensor (B, C, H, W)
    B,                # int
    C,                # int
    H,                # int
    W,                # int
    num_groups: tl.constexpr,  # 32
    eps,              # float
):
    # Grid: (B, num_groups)
    n = tl.program_id(0)
    g = tl.program_id(1)
    channels_per_group = C // num_groups
    group_start = g * channels_per_group

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # First pass: compute sum and sum of squares over group channels and all spatial positions
    for ch in range(channels_per_group):
        c = group_start + ch
        for h in range(H):
            for w in range(W):
                x_index = (((n * C + c) * H + h) * W + w)
                x_val = tl.load(x_ptr + x_index)
                sum_val += x_val
                sum_sq += x_val * x_val

    # Compute mean and variance
    M = channels_per_group * H * W
    mean = sum_val / M
    var = sum_sq / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for ch in range(channels_per_group):
        c = group_start + ch
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)
        for h in range(H):
            for w in range(W):
                x_index = (((n * C + c) * H + h) * W + w)
                x_val = tl.load(x_ptr + x_index)
                y_val = (x_val - mean) * inv_std
                y_val = y_val * gamma + beta
                y_index = x_index
                tl.store(y_ptr + y_index, y_val)


@triton.jit
def silu_kernel(
    x_ptr,            # *float32 input tensor (B, C, H, W)
    y_ptr,            # *float32 output tensor (B, C, H, W)
    B,                # int
    C,                # int
    H,                # int
    W,                # int
):
    # Grid: (B, C, H, W)
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
    y_ptr,            # *float32 first tensor (B, C, H, W)
    x_ptr,            # *float32 second tensor (B, C, H, W)
    out_ptr,          # *float32 output tensor (B, C, H, W)
    B,                # int
    C,                # int
    H,                # int
    W,                # int
):
    # Grid: (B, C, H, W)
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
        assert x.is_cuda, "Input tensor must be on CUDA device for Triton kernels."
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Weights must be on CUDA device."
        assert x.dtype == torch.float32, "Input and weights must be float32 for Triton kernels."
        assert conv1_weight.dtype == torch.float32 and conv2_weight.dtype == torch.float32

        N, C, H, W = x.shape
        C1, C_in1, K_h1, K_w1 = conv1_weight.shape
        C2, C_in2, K_h2, K_w2 = conv2_weight.shape
        num_groups = 32

        # Enforce GroupNorm divisibility requirement
        if (C1 % num_groups != 0) or (C2 % num_groups != 0):
            raise ValueError(f"num_groups=32 must divide channels for GroupNorm: got C1={C1}, C2={C2}")

        # 1) Conv1: 3x3 stride=1, padding=1, bias=None
        y1 = torch.empty((N, C1, H, W), dtype=torch.float32, device=x.device)
        grid1 = (N, C1, H, W)
        conv3x3_stride1_pad1_innertile_kernel[grid1](
            x, conv1_weight, y1,
            N, C_in1, C1, H, W,
            num_warps=4,
            num_stages=2,
        )

        # 2) GroupNorm1
        y1n = torch.empty_like(y1)
        grid_gn1 = (N, num_groups)
        groupnorm_affine_kernel[grid_gn1](
            y1, norm1_weight, norm1_bias, y1n,
            N, C1, H, W,
            num_groups=32, eps=eps,
            num_warps=4,
            num_stages=2,
        )

        # 3) SiLU1
        y1s = torch.empty_like(y1n)
        grid_silu1 = (N, C1, H, W)
        silu_kernel[grid_silu1](
            y1n, y1s,
            N, C1, H, W,
            num_warps=4,
            num_stages=2,
        )

        # 4) Conv2
        y2 = torch.empty((N, C2, H, W), dtype=torch.float32, device=x.device)
        grid2 = (N, C2, H, W)
        conv3x3_stride1_pad1_innertile_kernel[grid2](
            y1s, conv2_weight, y2,
            N, C1, C2, H, W,  # note: C1 is input channels for conv2 here (y1s), which is 32 in the original example
            num_warps=4,
            num_stages=2,
        )

        # Note: In the original code, conv2 input has channels=C1. In the provided example, conv2_weight has shape (C, C, 3, 3).
        # Since we don't have actual code to inspect, we assume conv2_weight is (C_out, C_in, 3, 3). The previous line computes y2 correctly.

        # 5) GroupNorm2
        y2n = torch.empty_like(y2)
        grid_gn2 = (N, num_groups)
        groupnorm_affine_kernel[grid_gn2](
            y2, norm2_weight, norm2_bias, y2n,
            N, C2, H, W,
            num_groups=32, eps=eps,
            num_warps=4,
            num_stages=2,
        )

        # 6) SiLU2
        y2s = torch.empty_like(y2n)
        grid_silu2 = (N, C2, H, W)
        silu_kernel[grid_silu2](
            y2n, y2s,
            N, C2, H, W,
            num_warps=4,
            num_stages=2,
        )

        # 7) Add residual x
        out = torch.empty_like(y2s)
        grid_add = (N, C2, H, W)
        add_residual_kernel[grid_add](
            y2s, x, out,
            N, C2, H, W,
            num_warps=4,
            num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
