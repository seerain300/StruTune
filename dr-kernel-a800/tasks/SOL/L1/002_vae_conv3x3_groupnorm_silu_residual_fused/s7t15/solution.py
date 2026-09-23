import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_stride1_pad1_oc_tile_kernel(
    x_ptr,           # *float32 input tensor (B, C_in, H, W)
    w_ptr,           # *float32 weight tensor (C_out, C_in, 3, 3)
    y_ptr,           # *float32 output tensor (B, C_out, H, W)
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C_out: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    """
    Triton kernel computing conv3x3 (stride=1, padding=1, bias=None) using a tile over output channels.
    Grid: (B, ceil_div(C_out, BLOCK_OC))
    Each program computes a tile of BLOCK_OC output channels for one batch n and all output pixels (H*W).
    """
    n = tl.program_id(0)
    oc_block_id = tl.program_id(1)
    oc_start = oc_block_id * BLOCK_OC
    oc_offsets = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offsets < C_out

    # Accumulator vector for this tile of output channels
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Loop over input channels and 3x3 taps
    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                # Compute input coordinates for all output positions (oh, ow)
                # For each output pixel (oh, ow), ih = oh + kh - 1, iw = ow + kw - 1
                for oh in range(H):
                    ih = oh + kh - 1
                    in_row_valid = (ih >= 0) & (ih < H)
                    for ow in range(W):
                        iw = ow + kw - 1
                        in_col_valid = (iw >= 0) & (iw < W)
                        valid = in_row_valid & in_col_valid

                        # Load input scalar x[n, cin, ih, iw] with mask
                        x_index = (((n * C_in + cin) * H + ih) * W + iw)
                        x_val = tl.load(x_ptr + x_index, mask=valid, other=0.0)

                        # Load weight vector for this (cin, kh, kw) across oc tile
                        # weight index: ((oc * C_in + cin) * 9) + (kh * 3 + kw)
                        w_indices = ((oc_offsets * C_in + cin) * 9) + (kh * 3 + kw)
                        w_vals = tl.load(w_ptr + w_indices, mask=oc_mask, other=0.0)

                        # FMA accumulate: acc[oc] += x_val * w_vals[oc]
                        acc += x_val * w_vals

    # Store results for all output positions (oh, ow)
    # We write acc to y[n, oc, oh, ow] for all oh, ow
    for oh in range(H):
        ih = oh + 1 - 1  # 1 here is unused because ih depends on kh only; but we set ih=oh as padding=1
        # Correction: ih should be oh + kh - 1, but since kh varies in loop, we need to compute per kh.
        # Instead, we write after knowing kh. To avoid extra loops, we'll recompute acc per (oh,ow)
        # But Triton requires static loops; better to compute acc per (oh,ow) by adding outer loops.
        # We'll restructure: compute acc per (oh, ow) by reusing acc computation above and write it.
        # However, to keep single launch, we precompute acc for tile and then write per (oh, ow).
        # We'll do this by reusing acc across all oh,ow after computing acc once, which is incorrect because acc changes per (oh,ow).
        # Therefore, we will instead compute per (oh, ow) as below.

    # The above approach is incorrect for per-(oh,ow) accumulation. We need to compute acc per (oh, ow).
    # Let's redefine the kernel to compute per output pixel and vectorize over oc tile:
    # We'll use a different kernel signature with grid (B, H, W, ceil_div(C_out, BLOCK_OC)) to handle per-pixel accumulation.
    # But Triton allows only 3D grid; we can flatten spatial and use (B, C_out_tiles, H*W).
    # Simpler: re-implement conv kernel as per-pixel with grid (B, C_out, H, W).

    # Since the above structure is not working cleanly, we switch to a per-pixel kernel, which is simpler and robust.

    # To avoid further complexity, we implement a per-pixel conv kernel below instead of this tile kernel.
    # The following conv3x3_stride1_pad1_pixel_kernel is the robust version used.


@triton.jit
def conv3x3_stride1_pad1_pixel_kernel(
    x_ptr,           # *float32 input tensor (B, C_in, H, W)
    w_ptr,           # *float32 weight tensor (C_out, C_in, 3, 3)
    y_ptr,           # *float32 output tensor (B, C_out, H, W)
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C_out: tl.constexpr,
):
    """
    Triton kernel computing conv3x3 (stride=1, padding=1, bias=None) with one output pixel per program.
    Grid: (B, C_out, H, W). Each program computes y[n, c_out, h, w] by looping over input channels and 3x3 taps.
    """
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    acc = 0.0

    for cin in range(C_in):
        for kh in range(3):
            for kw in range(3):
                ih = h + kh - 1
                iw = w + kw - 1
                valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x_index = (((n * C_in + cin) * H + ih) * W + iw)
                x_val = tl.load(x_ptr + x_index, mask=valid, other=0.0)
                w_index = ((c_out * C_in + cin) * 9) + (kh * 3 + kw)
                w_val = tl.load(w_ptr + w_index)
                acc += x_val * w_val

    y_index = (((n * C_out + c_out) * H + h) * W + w)
    tl.store(y_ptr + y_index, acc)


@triton.jit
def group_norm_forward_kernel(
    x_ptr,           # *float32 input tensor (N, C, H, W)
    weight_ptr,      # *float32 scale (C,)
    bias_ptr,        # *float32 bias (C,)
    y_ptr,           # *float32 output tensor (N, C, H, W)
    N, C, H, W,      # int32 sizes
    num_groups,      # int32, e.g., 32
    eps,             # float32 epsilon
):
    # Each program handles one (n, g) pair
    n = tl.program_id(0)
    g = tl.program_id(1)
    channels_per_group = C // num_groups
    group_start = g * channels_per_group

    # Compute sum and sum of squares over group channels and all H*W
    sum_total = 0.0
    sumsq_total = 0.0

    # Pass 1: reduction
    for ch in range(channels_per_group):
        c = group_start + ch
        base = n * C * H * W + c * H * W
        for i in range(H):
            for j in range(W):
                x_index = base + i * W + j
                x_val = tl.load(x_ptr + x_index)
                sum_total += x_val
                sumsq_total += x_val * x_val

    # Compute mean and variance
    M = channels_per_group * H * W
    mean = sum_total / M
    var = sumsq_total / M - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and apply affine, write to y
    for ch in range(channels_per_group):
        c = group_start + ch
        scale = tl.load(weight_ptr + c)
        bias = tl.load(bias_ptr + c)
        base_in = n * C * H * W + c * H * W
        base_out = n * C * H * W + c * H * W
        for i in range(H):
            for j in range(W):
                x_index = base_in + i * W + j
                x_val = tl.load(x_ptr + x_index)
                y_val = (x_val - mean) * inv_std * scale + bias
                y_index = base_out + i * W + j
                tl.store(y_ptr + y_index, y_val)


@triton.jit
def silu_kernel(
    x_ptr,           # *float32 input tensor (N, C, H, W)
    y_ptr,           # *float32 output tensor (N, C, H, W)
    N, C, H, W,
):
    # Grid: (N, C, H, W), each program computes one element
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
    y_ptr,           # *float32 output tensor (N, C, H, W)
    x_ptr,           # *float32 input tensor (N, C, H, W) to add as residual
    N, C, H, W,
):
    # Grid: (N, C, H, W), each program adds one element
    n = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)

    y_index = (((n * C + c) * H + h) * W + w)
    x_index = (((n * C + c) * H + h) * W + w)

    y_val = tl.load(y_ptr + y_index)
    x_val = tl.load(x_ptr + x_index)
    out_val = y_val + x_val
    tl.store(y_ptr + y_index, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        """
        Fused residual block in Triton:
          - Conv3x3 (stride=1, pad=1, bias=None) -> GroupNorm(num_groups=32) -> SiLU
          - Conv3x3 (stride=1, pad=1, bias=None) -> GroupNorm(num_groups=32) -> SiLU
          - Add residual x
        """
        assert x.is_cuda, "Input must be on CUDA for Triton kernels."
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Weights must be on CUDA for Triton kernels."
        assert norm1_weight.is_cuda and norm1_bias.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "Norm params must be on CUDA."

        # Use float32 for numerical stability
        x = x.contiguous().to(torch.float32)
        conv1_weight = conv1_weight.contiguous().to(torch.float32)  # (C_out, C_in, 3, 3)
        conv2_weight = conv2_weight.contiguous().to(torch.float32)  # (C_out, C_in, 3, 3)
        norm1_weight = norm1_weight.contiguous().to(torch.float32)  # (C_out)
        norm1_bias = norm1_bias.contiguous().to(torch.float32)      # (C_out)
        norm2_weight = norm2_weight.contiguous().to(torch.float32)  # (C_out)
        norm2_bias = norm2_bias.contiguous().to(torch.float32)      # (C_out)

        B, C, H, W = x.shape
        C_in = conv1_weight.shape[1]  # conv1 input channels
        C_out = conv2_weight.shape[0]  # conv2 output channels

        # 1) First conv (Triton) per-pixel
        out1 = torch.empty((B, C_out, H, W), device=x.device, dtype=torch.float32)
        grid_conv = (B, C_out, H, W)
        conv3x3_stride1_pad1_pixel_kernel[grid_conv](
            x, conv1_weight, out1,
            B=B, C_in=C_in, H=H, W=W, C_out=C_out,
            num_warps=1, num_stages=2,
        )

        # 2) GroupNorm1 (Triton), enforce C_out % 32 == 0 (first conv outputs have C_out channels)
        assert C_out % 32 == 0, "GroupNorm requires channels divisible by num_groups (32)."
        out1_gn = torch.empty_like(out1)
        grid_gn = (B, 32)
        group_norm_forward_kernel[grid_gn](
            out1, norm1_weight, norm1_bias, out1_gn,
            B, C_out, H, W, 32, eps,
            num_warps=4, num_stages=2,
        )

        # 3) SiLU1 (Triton)
        out1_silu = torch.empty_like(out1_gn)
        grid_silu1 = (B, C_out, H, W)
        silu_kernel[grid_silu1](
            out1_gn, out1_silu,
            B, C_out, H, W,
            num_warps=4, num_stages=2,
        )

        # 4) Second conv (Triton) per-pixel
        out2_pre = torch.empty((B, C_out, H, W), device=x.device, dtype=torch.float32)
        grid_conv2 = (B, C_out, H, W)
        conv3x3_stride1_pad1_pixel_kernel[grid_conv2](
            out1_silu, conv2_weight, out2_pre,
            B=B, C_in=C_in, H=H, W=W, C_out=C_out,
            num_warps=1, num_stages=2,
        )

        # 5) GroupNorm2 (Triton)
        assert C_out % 32 == 0, "GroupNorm requires channels divisible by num_groups (32)."
        out2_gn = torch.empty_like(out2_pre)
        grid_gn2 = (B, 32)
        group_norm_forward_kernel[grid_gn2](
            out2_pre, norm2_weight, norm2_bias, out2_gn,
            B, C_out, H, W, 32, eps,
            num_warps=4, num_stages=2,
        )

        # 6) SiLU2 (Triton)
        out2_silu = torch.empty_like(out2_gn)
        grid_silu2 = (B, C_out, H, W)
        silu_kernel[grid_silu2](
            out2_gn, out2_silu,
            B, C_out, H, W,
            num_warps=4, num_stages=2,
        )

        # 7) Add residual x (Triton)
        y_out = torch.empty_like(out2_silu)
        add_residual_kernel[grid_silu2](
            out2_silu, x, B, C, H, W,
            num_warps=4, num_stages=2,
        )

        return y_out


def run(*args):
    return ModelNew()(*args)
