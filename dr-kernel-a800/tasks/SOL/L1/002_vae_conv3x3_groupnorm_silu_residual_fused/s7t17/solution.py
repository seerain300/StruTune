import torch
import triton
import triton.language as tl


# Triton kernel: conv2d 3x3 stride=1, padding=1, bias=None
# Each program handles one batch sample and a tile of output channels (BLOCK_OC).
@triton.jit
def conv3x3_stride1_pad1_oc_tile_kernel(
    x_ptr,           # *float32 input tensor: (B, C_in, H, W)
    w_ptr,           # *float32 weights tensor: (C_out, C_in, 3, 3)
    y_ptr,           # *float32 output tensor: (B, C_out, H, W)
    N,               # int: batch size
    C_in,            # int: input channels
    H,               # int: input height
    W,               # int: input width
    C_out,           # int: output channels
    H_out,           # int: output height (== H)
    W_out,           # int: output width (== W)
    BLOCK_OC: tl.constexpr,  # tile size for output channels
):
    # Grid: (N, ceil_div(C_out, BLOCK_OC))
    n = tl.program_id(0)
    oc_block_id = tl.program_id(1)
    oc_start = oc_block_id * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_out

    # Accumulator for output channels tile
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Loop over input channels and 3x3 taps
    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                for oh in range(H_out):
                    ih = oh + kh - 1  # stride=1, pad=1
                    for ow in range(W_out):
                        iw = ow + kw - 1  # stride=1, pad=1
                        valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                        # Input linear index: (((n * C_in + cin) * H + ih) * W + iw)
                        x_index = (((n * C_in + cin) * H + ih) * W + iw)
                        x_val = tl.load(x_ptr + x_index, mask=valid, other=0.0)

                        # Load weights for this oc tile: w[oc, cin, kh, kw]
                        # Compute linear index for weight vector of length BLOCK_OC
                        w_index_base = (oc_offsets * C_in + cin) * 9 + (kh * 3 + kw)
                        w_vals = tl.load(w_ptr + w_index_base, mask=oc_mask, other=0.0).to(tl.float32)
                        acc += x_val * w_vals

    # Store results y[n, oc, oh, ow] for all oh, ow (simple per-program write)
    for oh in range(H_out):
        for ow in range(W_out):
            for j in range(BLOCK_OC):
                if oc_mask[oc_offsets[j]]:
                    y_index = (((n * C_out + oc_offsets[j]) * H_out + oh) * W_out + ow)
                    tl.store(y_ptr + y_index, acc[j])


# Triton GroupNorm kernel: per (n, group), compute mean/var and normalize with affine
@triton.jit
def group_norm_affine_kernel(
    x_ptr,          # *float32 input tensor (B, C, H, W)
    weight_ptr,     # *float32 scale per channel (C,)
    bias_ptr,       # *float32 bias per channel (C,)
    y_ptr,          # *float32 output tensor (B, C, H, W)
    N,              # int
    C,              # int
    H,              # int
    W,              # int
    NUM_GROUPS: tl.constexpr,  # number of groups, typically 32
    EPS: tl.constexpr,          # epsilon
):
    # Grid: (N, NUM_GROUPS)
    n = tl.program_id(0)
    g = tl.program_id(1)

    channels_per_group = C // NUM_GROUPS
    group_start = g * channels_per_group

    # First pass: compute sum and sum of squares over group and all H*W
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for cin in range(channels_per_group):
        c = group_start + cin
        for oh in range(H):
            for ow in range(W):
                in_index = (((n * C + c) * H + oh) * W + ow)
                x_val = tl.load(x_ptr + in_index)
                sum_val += x_val
                sum_sq += x_val * x_val

    M = channels_per_group * H * W
    mean = sum_val / M
    var = sum_sq / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + EPS)

    # Second pass: normalize and apply affine, write to y
    for cin in range(channels_per_group):
        c = group_start + cin
        scale = tl.load(weight_ptr + c)
        bias = tl.load(bias_ptr + c)
        for oh in range(H):
            for ow in range(W):
                in_index = (((n * C + c) * H + oh) * W + ow)
                x_val = tl.load(x_ptr + in_index)
                y_val = (x_val - mean) * inv_std * scale + bias
                out_index = in_index  # same layout
                tl.store(y_ptr + out_index, y_val)


# Triton SiLU activation: y = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
@triton.jit
def silu_kernel(
    x_ptr,          # *float32 input tensor (B, C, H, W)
    y_ptr,          # *float32 output tensor (B, C, H, W)
    N,              # int
    C,              # int
    H,              # int
    W,              # int
):
    # Grid: (N, C, H, W) programs
    n = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    in_index = (((n * C + c) * H + oh) * W + ow)
    x_val = tl.load(x_ptr + in_index)
    sigmoid = 1.0 / (1.0 + tl.exp(-x_val))
    y_val = x_val * sigmoid
    tl.store(y_ptr + in_index, y_val)


# Triton residual add: y = y + x
@triton.jit
def add_residual_kernel(
    y_ptr,          # *float32 y tensor (B, C, H, W)
    x_ptr,          # *float32 x tensor (B, C, H, W), same shape
    N,              # int
    C,              # int
    H,              # int
    W,              # int
):
    # Grid: (N, C, H, W) programs
    n = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    index = (((n * C + c) * H + oh) * W + ow)
    y_val = tl.load(y_ptr + index)
    x_val = tl.load(x_ptr + index)
    y_val = y_val + x_val
    tl.store(y_ptr + index, y_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        conv1_weight: torch.Tensor,   # (C, C, 3, 3)
        norm1_weight: torch.Tensor,   # (C,)
        norm1_bias: torch.Tensor,     # (C,)
        conv2_weight: torch.Tensor,   # (C, C, 3, 3)
        norm2_weight: torch.Tensor,   # (C,)
        norm2_bias: torch.Tensor,     # (C,)
        eps: float,
    ):
        """
        Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        All computations are performed by Triton kernels. No torch ops are used in forward.
        """
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        N, C, H, W = x.shape
        # Ensure GroupNorm constraint: C must be divisible by num_groups (32)
        assert C % 32 == 0, f"Channels ({C}) must be divisible by num_groups (32) for GroupNorm."

        # 1) Conv1 (3x3, stride=1, padding=1, bias=None) -> out1
        out1 = torch.empty((N, C, H, W), dtype=torch.float32, device=x.device)
        BLOCK_OC = 32  # tile for output channels
        grid_conv1 = (N, triton.cdiv(C, BLOCK_OC))
        conv3x3_stride1_pad1_oc_tile_kernel[grid_conv1](
            x, conv1_weight, out1,
            N, C, H, W, C, H, W,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )

        # 2) GroupNorm1 (num_groups=32, affine)
        out1_norm = torch.empty_like(out1, dtype=torch.float32)
        grid_gn1 = (N, 32)
        group_norm_affine_kernel[grid_gn1](
            out1, norm1_weight, norm1_bias, out1_norm,
            N, C, H, W,
            NUM_GROUPS=32,
            EPS=eps,
            num_warps=4,
            num_stages=2,
        )

        # 3) SiLU1
        out1_silu = torch.empty_like(out1_norm, dtype=torch.float32)
        grid_silu1 = (N, C, H, W)
        silu_kernel[grid_silu1](
            out1_norm, out1_silu, N, C, H, W,
            num_warps=4,
            num_stages=2,
        )

        # Save residual
        residual = x

        # 4) Conv2 (3x3, stride=1, padding=1, bias=None) on out1_silu -> out2
        out2 = torch.empty((N, C, H, W), dtype=torch.float32, device=x.device)
        grid_conv2 = (N, triton.cdiv(C, BLOCK_OC))
        conv3x3_stride1_pad1_oc_tile_kernel[grid_conv2](
            out1_silu, conv2_weight, out2,
            N, C, H, W, C, H, W,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )

        # 5) GroupNorm2 (num_groups=32, affine)
        out2_norm = torch.empty_like(out2, dtype=torch.float32)
        grid_gn2 = (N, 32)
        group_norm_affine_kernel[grid_gn2](
            out2, norm2_weight, norm2_bias, out2_norm,
            N, C, H, W,
            NUM_GROUPS=32,
            EPS=eps,
            num_warps=4,
            num_stages=2,
        )

        # 6) SiLU2
        out2_silu = torch.empty_like(out2_norm, dtype=torch.float32)
        grid_silu2 = (N, C, H, W)
        silu_kernel[grid_silu2](
            out2_norm, out2_silu, N, C, H, W,
            num_warps=4,
            num_stages=2,
        )

        # 7) Residual add
        out_final = torch.empty_like(out2_silu, dtype=torch.float32)
        grid_add = (N, C, H, W)
        add_residual_kernel[grid_add](
            out2_silu, residual, N, C, H, W,
            num_warps=4,
            num_stages=2,
        )

        return out_final


def run(*args):
    return ModelNew()(*args)
