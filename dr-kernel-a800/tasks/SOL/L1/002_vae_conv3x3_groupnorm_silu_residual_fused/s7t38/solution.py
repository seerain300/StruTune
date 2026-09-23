import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_single_elem_kernel(
    x_ptr,           # *float32, shape (B, C_in, H, W)
    w_ptr,           # *float32, shape (C_out, C_in, 3, 3)
    y_ptr,           # *float32, shape (B, C_out, H, W)
    N, C_in, C_out, H, W,
):
    # program ids: n, c_out, oh, ow
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # Accumulate in float32
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 taps
    for cin in range(0, C_in):
        for kh in range(3):
            for kw in range(3):
                ih = oh + kh - 1  # stride=1, pad=1
                iw = ow + kw - 1
                in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                # Input linear index: (((n * C_in + cin) * H + ih) * W + iw)
                x_index = (((n * C_in + cin) * H + ih) * W + iw)
                x_val = tl.load(x_ptr + x_index, mask=in_bounds, other=0.0)
                # Weight scalar: w[c_out, cin, kh, kw]
                w_index = (((c_out * C_in) + cin) * 9 + (kh * 3 + kw))
                w_val = tl.load(w_ptr + w_index)
                acc += x_val * w_val

    # Output linear index: (((n * C_out) + c_out) * H * W + (oh * W + ow))
    y_index = (((n * C_out) + c_out) * H * W + (oh * W + ow))
    tl.store(y_ptr + y_index, acc)


@triton.jit
def group_norm_affine_kernel(
    x_ptr,           # *float32, input tensor (B, C, H, W)
    gamma_ptr,       # *float32, per-channel scale (C,)
    beta_ptr,        # *float32, per-channel bias (C,)
    y_ptr,           # *float32, output tensor (B, C, H, W)
    N, C, H, W,
    num_groups: tl.constexpr,  # we use 32
    eps: tl.constexpr,         # epsilon
):
    # one program per (n, group)
    n = tl.program_id(0)
    g = tl.program_id(1)
    channels_per_group = C // num_groups
    group_start = g * channels_per_group

    # Accumulate sum and sumsq across group's channels and all spatial positions
    total_sum = tl.zeros((), dtype=tl.float32)
    total_sumsq = tl.zeros((), dtype=tl.float32)

    for cin in range(channels_per_group):
        c = group_start + cin
        # Loop over H and W
        for h in range(H):
            for w in range(W):
                x_index = (((n * C + c) * H + h) * W + w)
                x_val = tl.load(x_ptr + x_index)
                total_sum += x_val
                total_sumsq += x_val * x_val

    # Compute mean and variance
    hw = H * W
    num_elems = channels_per_group * hw
    mean = total_sum / num_elems
    var = total_sumsq / num_elems - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine, write back
    for cin in range(channels_per_group):
        c = group_start + cin
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)
        for h in range(H):
            for w in range(W):
                x_index = (((n * C + c) * H + h) * W + w)
                x_val = tl.load(x_ptr + x_index)
                y_val = ((x_val - mean) * inv_std) * gamma + beta
                y_index = x_index  # same layout
                tl.store(y_ptr + y_index, y_val)


@triton.jit
def silu_kernel(
    x_ptr,           # *float32 input (B, C, H, W)
    y_ptr,           # *float32 output (B, C, H, W)
    N, C, H, W,
):
    # One program per element (N, C, H, W)
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    x_index = (((n * C + c) * H + h) * W + w)
    x_val = tl.load(x_ptr + x_index)
    # sigmoid
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig
    tl.store(y_ptr + x_index, y_val)


@triton.jit
def add_residual_kernel(
    y_ptr,           # *float32 (output)
    x_ptr,           # *float32 (input residual)
    N, C, H, W,
):
    # One program per element (N, C, H, W)
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    y_index = (((n * C + c) * H + h) * W + w)
    x_index = y_index
    y_val = tl.load(y_ptr + y_index)
    x_val = tl.load(x_ptr + x_index)
    tl.store(y_ptr + y_index, y_val + x_val)


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
        Triton-only implementation of the fused residual block:
        Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        All heavy lifting is done by Triton kernels. No torch ops in forward.
        """
        B, C, H, W = x.shape
        # Enforce GroupNorm constraint
        assert C % 32 == 0, "GroupNorm requires C % 32 == 0 for num_groups=32"

        # Ensure dtypes and contiguity
        x_f32 = x.contiguous().float()
        # Conv1: (B, C, H, W) -> (B, C, H, W)
        y1 = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)
        grid1 = (B, C, H, W)
        conv3x3_stride1_pad1_single_elem_kernel[grid1](
            x_f32, conv1_weight.float().contiguous(), y1,
            B, C, C, H, W,  # C_in == C
            num_warps=1, num_stages=1,
        )

        # GroupNorm1
        y1_norm = torch.empty_like(y1)
        group_norm_affine_kernel[(B, 32)](
            y1, norm1_weight.float().contiguous(), norm1_bias.float().contiguous(),
            y1_norm, B, C, H, W, num_groups=32, eps=eps,
            num_warps=4, num_stages=2,
        )

        # SiLU1
        y1_silu = torch.empty_like(y1_norm)
        silu_kernel[(B, C, H, W)](
            y1_norm, y1_silu, B, C, H, W,
            num_warps=4, num_stages=2,
        )

        # Conv2: (B, C, H, W) -> (B, C, H, W)
        y2 = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)
        grid2 = (B, C, H, W)
        conv3x3_stride1_pad1_single_elem_kernel[grid2](
            y1_silu, conv2_weight.float().contiguous(), y2,
            B, C, C, H, W,  # C_in == C
            num_warps=1, num_stages=1,
        )

        # GroupNorm2
        y2_norm = torch.empty_like(y2)
        group_norm_affine_kernel[(B, 32)](
            y2, norm2_weight.float().contiguous(), norm2_bias.float().contiguous(),
            y2_norm, B, C, H, W, num_groups=32, eps=eps,
            num_warps=4, num_stages=2,
        )

        # SiLU2
        y2_silu = torch.empty_like(y2_norm)
        silu_kernel[(B, C, H, W)](
            y2_norm, y2_silu, B, C, H, W,
            num_warps=4, num_stages=2,
        )

        # Add residual x (float32) to final
        y_out = torch.empty_like(y2_silu)
        add_residual_kernel[(B, C, H, W)](
            y2_silu, x_f32, B, C, H, W,
            num_warps=4, num_stages=2,
        )

        return y_out


def run(*args):
    return ModelNew()(*args)
