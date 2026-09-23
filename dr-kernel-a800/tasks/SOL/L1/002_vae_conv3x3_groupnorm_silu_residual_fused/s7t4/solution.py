import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_kernel(
    x_ptr,          # *float32 input tensor (B, C_in, H, W)
    w_ptr,          # *float32 weight tensor (C_out, C_in, 3, 3)
    y_ptr,          # *float32 output tensor (B, C_out, H, W)
    N, C_in, H, W, C_out,
    BLOCK_OC: tl.constexpr,
):
    # Grid: (N, ceil_div(C_out, BLOCK_OC))
    n = tl.program_id(0)
    oc_block_id = tl.program_id(1)
    oc_start = oc_block_id * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_out

    # Loop over output spatial positions to compute full output for this tile
    for oh in range(H):
        for ow in range(W):
            # Accumulator for this (n, oc_tile) at fixed (oh, ow)
            acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)
            # Loop over input channels and 3x3 taps
            for cin in range(C_in):
                for kh in range(3):
                    for kw in range(3):
                        ih = oh + kh - 1
                        iw = ow + kw - 1
                        valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                        # Input linear index: (((n * C_in + cin) * H + ih) * W + iw)
                        x_index = (((n * C_in + cin) * H + ih) * W + iw)
                        x_val = tl.load(x_ptr + x_index, mask=valid, other=0.0)
                        # Weight linear index: (((oc * C_in + cin) * 9) + (kh * 3 + kw))
                        w_index_vec = (((oc_offsets * C_in + cin) * 9) + (kh * 3 + kw))
                        # Load weights for this oc tile (vector)
                        w_vals = tl.load(w_ptr + w_index_vec, mask=oc_mask, other=0.0)
                        # FMA accumulate
                        acc += w_vals * x_val
            # Store results for each oc in this tile at (n, oc, oh, ow)
            for j in range(BLOCK_OC):
                if oc_mask[oc_offsets[j]]:
                    y_index = (((n * C_out + oc_offsets[j]) * H + oh) * W + ow)
                    tl.store(y_ptr + y_index, acc[j])


@triton.jit
def group_norm_affine_kernel(
    y_in_ptr,       # *float32 input tensor after conv (B, C, H, W)
    scale_ptr,      # *float32 GroupNorm weight (C,)
    bias_ptr,       # *float32 GroupNorm bias (C,)
    y_out_ptr,      # *float32 output tensor (B, C, H, W)
    N, C, H, W,     # dims
    num_groups: tl.constexpr,  # e.g., 32
    eps: tl.float32,
):
    # Grid: (N, num_groups)
    n = tl.program_id(0)
    g = tl.program_id(1)

    channels_per_group = C // num_groups
    group_start_channel = g * channels_per_group
    group_elems = channels_per_group * H * W

    # First pass: compute mean and variance over the group
    total_sum = 0.0
    total_sumsq = 0.0
    for c_local in range(channels_per_group):
        c = group_start_channel + c_local
        base = (n * C + c) * H * W
        for oh in range(H):
            for ow in range(W):
                idx = oh * W + ow
                x = tl.load(y_in_ptr + base + idx)
                total_sum += x
                total_sumsq += x * x

    mean = total_sum / group_elems
    var = total_sumsq / group_elems - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine per channel in the group
    for c_local in range(channels_per_group):
        c = group_start_channel + c_local
        base_in = (n * C + c) * H * W
        scale = tl.load(scale_ptr + c)
        beta = tl.load(bias_ptr + c)
        base_out = (n * C + c) * H * W
        for oh in range(H):
            for ow in range(W):
                idx = oh * W + ow
                x = tl.load(y_in_ptr + base_in + idx)
                y = (x - mean) * inv_std
                y = y * scale + beta
                tl.store(y_out_ptr + base_out + idx, y)


@triton.jit
def silu_kernel(x_ptr, y_ptr, N, C, H, W):
    # Elementwise SiLU: y = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
    for n in range(N):
        for c in range(C):
            base = (n * C + c) * H * W
            for i in range(H * W):
                x = tl.load(x_ptr + base + i)
                s = 1.0 / (1.0 + tl.exp(-x))
                y = x * s
                tl.store(y_ptr + base + i, y)


@triton.jit
def add_residual_kernel(y_ptr, x_ptr, out_ptr, N, C, H, W):
    # out = y + x, elementwise
    for n in range(N):
        for c in range(C):
            base = (n * C + c) * H * W
            for i in range(H * W):
                a = tl.load(y_ptr + base + i)
                b = tl.load(x_ptr + base + i)
                c = a + b
                tl.store(out_ptr + base + i, c)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        """
        Fused residual block implemented fully in Triton:
        Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        All operations are Triton kernels; no torch ops in forward.
        """
        # Ensure inputs/weights on CUDA and contiguous
        device = x.device
        assert x.is_cuda and conv1_weight.is_cuda and norm1_weight.is_cuda and norm1_bias.is_cuda \
               and conv2_weight.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "All tensors must be on CUDA"
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()

        N, C, H, W = x.shape
        C_in = C  # conv input channels equals output channels of previous stage
        C_out = C  # same channel count

        # 1) Conv1 (Triton)
        y1 = torch.empty((N, C_out, H, W), dtype=torch.float32, device=device)
        BLOCK_OC = 64  # tuneable tile size for output channels
        grid_conv1 = (N, triton.cdiv(C_out, BLOCK_OC))
        conv3x3_stride1_pad1_kernel[grid_conv1](
            x, conv1_weight, y1,
            N, C_in, H, W, C_out,
            BLOCK_OC=BLOCK_OC,
            num_warps=4, num_stages=2
        )

        # 2) GroupNorm1 (Triton), requires C % 32 == 0
        assert C % 32 == 0, "GroupNorm: channels must be divisible by num_groups (32)"
        y2 = torch.empty_like(y1)
        group_norm_affine_kernel[(N, 32)](
            y1, norm1_weight, norm1_bias, y2,
            N, C, H, W,
            num_groups=32, eps=float(eps),
            num_warps=4, num_stages=2
        )

        # 3) SiLU1 (Triton)
        y3 = torch.empty_like(y2)
        silu_kernel[(1,)](y2, y3, N, C, H, W)

        # 4) Conv2 (Triton)
        y4 = torch.empty((N, C_out, H, W), dtype=torch.float32, device=device)
        grid_conv2 = (N, triton.cdiv(C_out, BLOCK_OC))
        conv3x3_stride1_pad1_kernel[grid_conv2](
            y3, conv2_weight, y4,
            N, C_in, H, W, C_out,
            BLOCK_OC=BLOCK_OC,
            num_warps=4, num_stages=2
        )

        # 5) GroupNorm2 (Triton)
        assert C % 32 == 0, "GroupNorm: channels must be divisible by num_groups (32)"
        y5 = torch.empty_like(y4)
        group_norm_affine_kernel[(N, 32)](
            y4, norm2_weight, norm2_bias, y5,
            N, C, H, W,
            num_groups=32, eps=float(eps),
            num_warps=4, num_stages=2
        )

        # 6) SiLU2 (Triton)
        y6 = torch.empty_like(y5)
        silu_kernel[(1,)](y5, y6, N, C, H, W)

        # 7) Add residual x (Triton)
        out = torch.empty_like(y6)
        add_residual_kernel[(1,)](y6, x, out, N, C, H, W)

        return out


def run(*args):
    return ModelNew()(*args)
