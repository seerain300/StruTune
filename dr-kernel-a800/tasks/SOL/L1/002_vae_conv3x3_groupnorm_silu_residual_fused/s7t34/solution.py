import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_kernel_two(
    x_ptr,          # *float32 input tensor (B, C_in, H, W)
    w_ptr,          # *float32 weights tensor (C_out, C_in, 3, 3)
    y_ptr,          # *float32 output tensor (B, C_out, H, W)
    N: tl.constexpr,  # batch size
    C_in: tl.constexpr,  # input channels
    C_out: tl.constexpr,  # output channels
    H: tl.constexpr,      # input height
    W: tl.constexpr,      # input width
    BLOCK_OC: tl.constexpr,
):
    # Grid: (N, ceil_div(C_out, BLOCK_OC))
    n = tl.program_id(0)
    oc_block_id = tl.program_id(1)
    oc_start = oc_block_id * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_out

    # Accumulator for the tile of output channels
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel taps
    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                # Output spatial dims are H and W (stride=1, padding=1)
                for h in range(H):
                    ih = h + kh - 1
                    valid_h = (ih >= 0) & (ih < H)
                    for wj in range(W):
                        iw = wj + kw - 1
                        valid = valid_h & (iw >= 0) & (iw < W)
                        # Input linear index: (((n * C_in + cin) * H + ih) * W + iw)
                        in_index = (((n * C_in + cin) * H + ih) * W + iw)
                        x_val = tl.load(x_ptr + in_index, mask=valid, other=0.0).to(tl.float32)
                        # Weight linear index: (((oc * C_in + cin) * 9) + (kh * 3 + kw))
                        for j in range(BLOCK_OC):
                            if oc_mask[oc_offsets[j]]:
                                w_index = (((oc_offsets[j] * C_in + cin) * 9) + (kh * 3 + kw))
                                w_val = tl.load(w_ptr + w_index).to(tl.float32)
                                acc[j] += x_val * w_val

    # Store results for all spatial positions (h, w)
    for j in range(BLOCK_OC):
        if oc_mask[oc_offsets[j]]:
            for h in range(H):
                for wj in range(W):
                    out_index = (((n * C_out + oc_offsets[j]) * H + h) * W + wj)
                    tl.store(y_ptr + out_index, acc[j])


@triton.jit
def conv3x3_stride1_pad1_kernel_two2(
    x_ptr,          # *float32 input tensor (B, C_in, H, W) — this is the activated tensor
    w_ptr,          # *float32 weights tensor (C_out, C_in, 3, 3)
    y_ptr,          # *float32 output tensor (B, C_out, H, W)
    N: tl.constexpr,  # batch size
    C_in: tl.constexpr,  # input channels
    C_out: tl.constexpr,  # output channels
    H: tl.constexpr,      # input height
    W: tl.constexpr,      # input width
    BLOCK_OC: tl.constexpr,
):
    # Same as conv1 kernel, just with different input (activated tensor)
    n = tl.program_id(0)
    oc_block_id = tl.program_id(1)
    oc_start = oc_block_id * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_out

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                for h in range(H):
                    ih = h + kh - 1
                    valid_h = (ih >= 0) & (ih < H)
                    for wj in range(W):
                        iw = wj + kw - 1
                        valid = valid_h & (iw >= 0) & (iw < W)
                        in_index = (((n * C_in + cin) * H + ih) * W + iw)
                        x_val = tl.load(x_ptr + in_index, mask=valid, other=0.0).to(tl.float32)
                        for j in range(BLOCK_OC):
                            if oc_mask[oc_offsets[j]]:
                                w_index = (((oc_offsets[j] * C_in + cin) * 9) + (kh * 3 + kw))
                                w_val = tl.load(w_ptr + w_index).to(tl.float32)
                                acc[j] += x_val * w_val

    for j in range(BLOCK_OC):
        if oc_mask[oc_offsets[j]]:
            for h in range(H):
                for wj in range(W):
                    out_index = (((n * C_out + oc_offsets[j]) * H + h) * W + wj)
                    tl.store(y_ptr + out_index, acc[j])


@triton.jit
def group_norm_affine_kernel(
    x_ptr,           # *float32 input tensor (B, C, H, W)
    gamma_ptr,       # *float32 scale (C,) — norm2_weight in original
    beta_ptr,        # *float32 bias (C,) — norm2_bias in original
    y_ptr,           # *float32 output tensor (B, C, H, W)
    N: tl.constexpr,  # batch size
    C: tl.constexpr,  # channels
    H: tl.constexpr,  # height
    W: tl.constexpr,  # width
    num_groups: tl.constexpr,  # fixed 32
    eps: tl.float32,           # epsilon
):
    # One program per (n, group)
    n = tl.program_id(0)
    g = tl.program_id(1)
    channels_per_group = C // num_groups
    group_start = g * channels_per_group
    group_end = group_start + channels_per_group

    # Compute sum and sum of squares across the group over all spatial positions
    sum_val = 0.0
    sum_sq = 0.0
    for c in range(group_start, group_end):
        for h in range(H):
            for w in range(W):
                index = (((n * C + c) * H + h) * W + w)
                x_val = tl.load(x_ptr + index)
                sum_val += x_val
                sum_sq += x_val * x_val

    # Compute mean and variance
    M = channels_per_group * H * W
    mean = sum_val / M
    var = sum_sq / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine: y = ((x - mean) * inv_std) * gamma + beta
    for c in range(group_start, group_end):
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)
        for h in range(H):
            for w in range(W):
                index = (((n * C + c) * H + h) * W + w)
                x_val = tl.load(x_ptr + index)
                norm = ((x_val - mean) * inv_std) * gamma + beta
                tl.store(y_ptr + index, norm)


@triton.jit
def silu_kernel(
    x_ptr,            # *float32 input tensor (B, C, H, W)
    y_ptr,            # *float32 output tensor (B, C, H, W)
    N: tl.constexpr,  # batch size
    C: tl.constexpr,  # channels
    H: tl.constexpr,  # height
    W: tl.constexpr,  # width
):
    # One program per (n, c, h, w)
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    index = (((n * C + c) * H + h) * W + w)
    x_val = tl.load(x_ptr + index)
    sig = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sig
    tl.store(y_ptr + index, y_val)


@triton.jit
def add_residual_kernel(
    x_ptr,            # *float32 input tensor (B, C, H, W), this is the final silu output
    res_ptr,          # *float32 input tensor (B, C, H, W), this is the original input x
    y_ptr,            # *float32 output tensor (B, C, H, W)
    N: tl.constexpr,  # batch size
    C: tl.constexpr,  # channels
    H: tl.constexpr,  # height
    W: tl.constexpr,  # width
):
    # One program per (n, c, h, w)
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    index = (((n * C + c) * H + h) * W + w)
    a = tl.load(x_ptr + index)
    b = tl.load(res_ptr + index)
    y = a + b
    tl.store(y_ptr + index, y)


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
        """
        Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        All operations implemented via Triton kernels; no torch ops in forward.
        """
        assert x.is_cuda, "Input tensor must be on CUDA device."
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Weights must be on CUDA device."
        assert norm1_weight.is_cuda and norm1_bias.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "Norm params must be on CUDA device."

        B, C, H, W = x.shape
        # Enforce GroupNorm constraint
        if (C % 32) != 0:
            raise RuntimeError(f"GroupNorm requires C divisible by num_groups (32). Got C={C}.")

        # Ensure contiguous
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        # 1) Conv1: x -> out1
        out1 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        BLOCK_OC = 32  # tile size for output channels
        grid_conv1 = (B, triton.cdiv(C, BLOCK_OC))
        conv3x3_stride1_pad1_kernel_two[
            grid_conv1
        ](
            x, conv1_weight, out1,
            N=B, C_in=C, C_out=C, H=H, W=W,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )

        # 2) GroupNorm1 (num_groups=32) on out1
        out1_norm = torch.empty_like(out1, dtype=torch.float32)
        grid_gn = (B, 32)  # one program per (n, group)
        group_norm_affine_kernel[grid_gn](
            out1, norm1_weight, norm1_bias, out1_norm,
            N=B, C=C, H=H, W=W, num_groups=32, eps=eps,
            num_warps=4,
            num_stages=2,
        )

        # 3) SiLU1 on normalized out1
        out1_silu = torch.empty_like(out1_norm, dtype=torch.float32)
        grid_silu = (B, C, H, W)
        silu_kernel[grid_silu](
            out1_norm, out1_silu,
            N=B, C=C, H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        # 4) Conv2: out1_silu -> out2
        out2 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid_conv2 = (B, triton.cdiv(C, BLOCK_OC))
        conv3x3_stride1_pad1_kernel_two2[grid_conv2](
            out1_silu, conv2_weight, out2,
            N=B, C_in=C, C_out=C, H=H, W=W,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )

        # 5) GroupNorm2 on out2
        out2_norm = torch.empty_like(out2, dtype=torch.float32)
        group_norm_affine_kernel[grid_gn](
            out2, norm2_weight, norm2_bias, out2_norm,
            N=B, C=C, H=H, W=W, num_groups=32, eps=eps,
            num_warps=4,
            num_stages=2,
        )

        # 6) SiLU2 on normalized out2
        out2_silu = torch.empty_like(out2_norm, dtype=torch.float32)
        silu_kernel[grid_silu](
            out2_norm, out2_silu,
            N=B, C=C, H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        # 7) Add residual x
        out = torch.empty_like(out2_silu, dtype=torch.float32)
        add_residual_kernel[grid_silu](
            out2_silu, x, out,
            N=B, C=C, H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
