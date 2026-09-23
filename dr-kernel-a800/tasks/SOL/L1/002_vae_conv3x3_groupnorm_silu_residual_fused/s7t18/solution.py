import torch
import triton
import triton.language as tl


# Triton kernel: conv2d 3x3 stride=1, padding=1, bias=None
# Computes a single output element y[n, c_out, h, w] by looping over input channels and 3x3 taps.
@triton.jit
def conv3x3_stride1_pad1_pixel_kernel(
    x_ptr,           # *float32 input tensor: (N, C_in, H, W)
    w_ptr,           # *float32 weights tensor: (C_out, C_in, 3, 3)
    y_ptr,           # *float32 output tensor: (N, C_out, H_out, W_out)
    N,               # int: batch size
    C_in,            # int: input channels
    H,               # int: input height
    W,               # int: input width
    C_out,           # int: output channels
    H_out,           # int: output height (same as H for stride=1, padding=1)
    W_out,           # int: output width (same as W for stride=1, padding=1)
):
    # Grid: (N, C_out, H_out, W_out)
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 taps
    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                ih = h + kh - 1   # stride=1, pad=1
                iw = w + kw - 1   # stride=1, pad=1
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                # Input linear index: (((n * C_in + cin) * H + ih) * W + iw)
                x_index = (((n * C_in + cin) * H + ih) * W + iw)
                x_val = tl.load(x_ptr + x_index, mask=valid, other=0.0)
                # Weight linear index: (((c_out * C_in + cin) * 9) + (kh * 3 + kw))
                w_index = (((c_out * C_in + cin) * 9) + (kh * 3 + kw))
                w_val = tl.load(w_ptr + w_index)
                acc += x_val * w_val

    # Output linear index: (((n * C_out + c_out) * H_out + h) * W_out + w)
    y_index = (((n * C_out + c_out) * H_out + h) * W_out + w)
    tl.store(y_ptr + y_index, acc)


# Triton kernel: GroupNorm with per-channel affine for a single (n, group)
# Assumes num_groups=32 and channels divisible by 32.
@triton.jit
def group_norm_affine_kernel(
    y_ptr,            # *float32 input tensor (B, C, H, W) to be normalized
    scale_ptr,        # *float32 per-channel scale (C,)
    bias_ptr,         # *float32 per-channel bias (C,)
    out_ptr,          # *float32 output tensor (B, C, H, W)
    N,                # int
    C,                # int
    H,                # int
    W,                # int
    num_groups: tl.constexpr,  # fixed 32
    eps,              # float
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    channels_per_group = C // num_groups
    group_start = g * channels_per_group

    # First pass: compute sum and sum of squares for the group across all H*W
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)
    for c in range(channels_per_group):
        c_idx = group_start + c
        # loop over all spatial positions
        for h in range(H):
            for w in range(W):
                y_index = (((n * C + c_idx) * H + h) * W + w)
                val = tl.load(y_ptr + y_index)
                sum_val += val
                sum_sq += val * val
    # Compute mean and variance
    M = channels_per_group * H * W
    mean = sum_val / M
    var = sum_sq / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, write to out
    for c in range(channels_per_group):
        c_idx = group_start + c
        gamma = tl.load(scale_ptr + c_idx)
        beta = tl.load(bias_ptr + c_idx)
        for h in range(H):
            for w in range(W):
                y_index = (((n * C + c_idx) * H + h) * W + w)
                val = tl.load(y_ptr + y_index)
                norm = (val - mean) * inv_std
                out_val = norm * gamma + beta
                out_index = (((n * C + c_idx) * H + h) * W + w)
                tl.store(out_ptr + out_index, out_val)


# Triton kernel: SiLU elementwise y = x * sigmoid(x)
@triton.jit
def silu_kernel(
    x_ptr,            # *float32 input tensor (B, C, H, W)
    y_ptr,            # *float32 output tensor (B, C, H, W)
    N, C, H, W,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    x_index = (((n * C + c) * H + h) * W + w)
    x_val = tl.load(x_ptr + x_index)
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig
    y_index = (((n * C + c) * H + h) * W + w)
    tl.store(y_ptr + y_index, y_val)


# Triton kernel: elementwise add residual y = y + x
@triton.jit
def add_residual_kernel(
    y_ptr,            # *float32 input tensor to add to (B, C, H, W)
    x_ptr,            # *float32 residual tensor (B, C, H, W)
    N, C, H, W,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    y_index = (((n * C + c) * H + h) * W + w)
    x_index = (((n * C + c) * H + h) * W + w)
    y_val = tl.load(y_ptr + y_index)
    x_val = tl.load(x_ptr + x_index)
    tl.store(y_ptr + y_index, y_val + x_val)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        """
        Fused residual block with Triton:
            Conv3x3 -> GroupNorm(num_groups=32) -> SiLU
            Conv3x3 -> GroupNorm(num_groups=32) -> SiLU
            Add residual x
        All heavy ops are Triton kernels. No torch ops in forward.
        """
        # Validate shapes
        B, C, H, W = x.shape
        C1, C_in1, KH1, KW1 = conv1_weight.shape
        C2, C_in2, KH2, KW2 = conv2_weight.shape
        assert KH1 == 3 and KW1 == 3 and KH2 == 3 and KW2 == 3, "Only 3x3 kernels supported"
        assert C1 == C and C2 == C, "Output channels must equal input channels"
        assert C % 32 == 0, "GroupNorm requires C divisible by 32"
        assert conv1_weight.dtype in (torch.float32, torch.float16), "Weights must be float16/float32"
        assert conv2_weight.dtype in (torch.float32, torch.float16), "Weights must be float16/float32"
        assert norm1_weight.shape == (C,), "norm1_weight must be per-channel (C,)"
        assert norm1_bias.shape == (C,), "norm1_bias must be per-channel (C,)"
        assert norm2_weight.shape == (C,), "norm2_weight must be per-channel (C,)"
        assert norm2_bias.shape == (C,), "norm2_bias must be per-channel (C,)"

        # Ensure tensors are contiguous float32 for Triton kernels
        x_f = x.contiguous().to(torch.float32)
        conv1_w_f = conv1_weight.contiguous().to(torch.float32)
        conv2_w_f = conv2_weight.contiguous().to(torch.float32)
        norm1_w_f = norm1_weight.contiguous().to(torch.float32)
        norm1_b_f = norm1_bias.contiguous().to(torch.float32)
        norm2_w_f = norm2_weight.contiguous().to(torch.float32)
        norm2_b_f = norm2_bias.contiguous().to(torch.float32)

        # 1) First conv: x -> out1 (N, C, H, W)
        out1 = torch.empty_like(x_f)
        grid_conv = (B, C, H, W)
        conv3x3_stride1_pad1_pixel_kernel[grid_conv](
            x_f, conv1_w_f, out1, B, C_in1, H, W, C, H, W,
            num_warps=4, num_stages=2,
        )

        # 2) GroupNorm1 on out1
        out1_norm = torch.empty_like(out1)
        grid_gn1 = (B, 32)
        group_norm_affine_kernel[grid_gn1](
            out1, norm1_w_f, norm1_b_f, out1_norm, B, C, H, W,
            num_groups=32, eps=float(eps),
            num_warps=4, num_stages=2,
        )

        # 3) SiLU1 on normalized out1
        out1_silu = torch.empty_like(out1_norm)
        grid_silu1 = (B, C, H, W)
        silu_kernel[grid_silu1](
            out1_norm, out1_silu, B, C, H, W,
            num_warps=4, num_stages=2,
        )

        # 4) Second conv: out1_silu -> out2 (N, C, H, W)
        out2 = torch.empty_like(x_f)
        grid_conv2 = (B, C, H, W)
        conv3x3_stride1_pad1_pixel_kernel[grid_conv2](
            out1_silu, conv2_w_f, out2, B, C_in2, H, W, C, H, W,
            num_warps=4, num_stages=2,
        )

        # 5) GroupNorm2 on out2
        out2_norm = torch.empty_like(out2)
        grid_gn2 = (B, 32)
        group_norm_affine_kernel[grid_gn2](
            out2, norm2_w_f, norm2_b_f, out2_norm, B, C, H, W,
            num_groups=32, eps=float(eps),
            num_warps=4, num_stages=2,
        )

        # 6) SiLU2 on normalized out2
        out2_silu = torch.empty_like(out2_norm)
        grid_silu2 = (B, C, H, W)
        silu_kernel[grid_silu2](
            out2_norm, out2_silu, B, C, H, W,
            num_warps=4, num_stages=2,
        )

        # 7) Add residual x
        y_out = torch.empty_like(out2_silu)
        grid_add = (B, C, H, W)
        add_residual_kernel[grid_add](
            out2_silu, x_f, B, C, H, W,
            num_warps=4, num_stages=2,
        )

        # Return in the original dtype
        return y_out.to(x.dtype)


def run(*args):
    return ModelNew()(*args)
