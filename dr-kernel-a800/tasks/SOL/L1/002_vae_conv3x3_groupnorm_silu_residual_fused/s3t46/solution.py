import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 conv forward (no bias), stride=1, padding=1
# x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), y: (B, C_out, H, W)
# Grid: (B, C_out, tiles over H*W). Each program handles one (b, oc tile) and one spatial tile.
@triton.jit
def conv3x3_triton(
    x_ptr, w_ptr, y_ptr,
    B, C_in, H, W, C_out,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK_OC: tl.constexpr,  # e.g., 32
    BLOCK_SP: tl.constexpr,  # e.g., 1024
):
    b = tl.program_id(0)
    oc_base = tl.program_id(1)
    tile_id = tl.program_id(2)

    # tile over spatial dimension
    start_sp = tile_id * BLOCK_SP
    sp = start_sp + tl.arange(0, BLOCK_SP)
    sp_mask = sp < (H * W)

    # map sp to (h, w)
    h = sp // W
    w = sp % W

    # vector of output channels for this tile
    oc_vec = oc_base + tl.arange(0, BLOCK_OC)
    oc_mask = oc_vec < C_out

    # initialize 2D accumulator for [BLOCK_OC, BLOCK_SP]
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # loop over input channels in tiles
    ic_base = 0
    while ic_base < C_in:
        ic_vec = ic_base + tl.arange(0, BLOCK_OC)
        ic_mask = ic_vec < C_in

        # For each input channel in this tile, accumulate contributions over 3x3 neighborhood
        for ic_i in range(0, BLOCK_OC):
            ic = ic_vec[ic_i]
            # If ic is out of bounds, skip
            valid_ic = ic < C_in

            # Accumulate over 3x3 neighborhood for all sp in the tile
            for kh in range(-1, 2):
                in_h = h + kh
                for kw in range(-1, 2):
                    in_w = w + kw

                    # load x values for this (b, ic, in_h, in_w) across sp
                    x_ptrs = x_ptr + b * x_stride_b + ic * x_stride_c + in_h * x_stride_h + in_w * x_stride_w
                    x_vals = tl.load(x_ptrs, mask=sp_mask & valid_ic, other=0.0)  # [BLOCK_SP]

                    # load w values for this (ic, oc_vec, kh, kw) across oc_vec
                    w_ptrs = w_ptr + ic * w_stride_cin + oc_vec * w_stride_cout + kh * w_stride_kh + kw * w_stride_kw
                    w_vals = tl.load(w_ptrs, mask=oc_mask & valid_ic, other=0.0)  # [BLOCK_OC]

                    # accumulate: acc += w_vals[:, None] * x_vals[None, :]
                    # When valid_ic is False, w_vals and x_vals are zeros due to masked load.
                    acc += w_vals[:, None] * x_vals[None, :]

        ic_base += BLOCK_OC

    # Store results for this (b, oc_tile, spatial_tile)
    # Pointer shape [BLOCK_OC, BLOCK_SP]
    y_ptrs = y_ptr + b * y_stride_b + oc_vec[:, None] * y_stride_c + h[None, :] * y_stride_h + w[None, :] * y_stride_w
    store_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(y_ptrs, acc, mask=store_mask)


# Triton kernel: GroupNorm + affine (scale, bias) + SiLU per (batch, group)
# x: (B, C, H, W), weight: (C,), bias: (C,), y: (B, C, H, W)
@triton.jit
def group_norm_affine_silu(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W, num_groups, eps,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    g = tl.program_id(1)

    group_channels = C // num_groups
    group_elements = group_channels * H * W

    # First pass: compute sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    for start in range(0, group_elements, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < group_elements

        ch = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr + b * x_stride_b + ch * x_stride_c + h * x_stride_h + w * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        sum_val += tl.sum(x_vals, axis=0)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    mean = sum_val / group_elements
    var = sum_sq / group_elements - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize, affine, SiLU
    for start in range(0, group_elements, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < group_elements

        ch = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr + b * x_stride_b + ch * x_stride_c + h * x_stride_h + w * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        norm_vals = (x_vals - mean) * inv_std

        scale = tl.load(weight_ptr + ch, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + ch, mask=mask, other=0.0)
        z = norm_vals * scale + bias

        s = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * s

        y_ptrs = y_ptr + b * y_stride_b + ch * y_stride_c + h * y_stride_h + w * y_stride_w
        tl.store(y_ptrs, y_vals, mask=mask)


# Triton kernel: elementwise residual add y = y + x
@triton.jit
def add_residual_kernel(out_ptr, y_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, a + b, mask=mask)


@torch.no_grad()
def run(
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
    Args:
        x: Input tensor of shape (B, C, H, W)
        conv1_weight: First conv weights (C, C, 3, 3)
        norm1_weight: First GroupNorm scale (C,)
        norm1_bias: First GroupNorm bias (C,)
        conv2_weight: Second conv weights (C, C, 3, 3)
        norm2_weight: Second GroupNorm scale (C,)
        norm2_bias: Second GroupNorm bias (C,)
        eps: Epsilon for GroupNorm numerical stability
    Returns:
        Output tensor of shape (B, C, H, W)
    """
    assert x.ndim == 4, "x must be (B, C, H, W)"
    assert conv1_weight.ndim == 4 and conv2_weight.ndim == 4, "conv weights must be (C_in, C_out, 3, 3)"
    assert norm1_weight.ndim == 1 and norm2_weight.ndim == 1 and norm1_bias.ndim == 1 and norm2_bias.ndim == 1, "norm params must be (C,)"
    assert x.is_cuda and conv1_weight.is_cuda and conv2_weight.is_cuda and norm1_weight.is_cuda and norm1_bias.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "All tensors must be CUDA"

    B, C, H, W = x.shape
    # Ensure contiguous
    x = x.contiguous()
    conv1_weight = conv1_weight.contiguous()
    conv2_weight = conv2_weight.contiguous()
    norm1_weight = norm1_weight.contiguous()
    norm1_bias = norm1_bias.contiguous()
    norm2_weight = norm2_weight.contiguous()
    norm2_bias = norm2_bias.contiguous()

    # Compute conv1
    y1 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)

    # Launch conv3x3_triton: grid over (B, C, tiles over H*W)
    BLOCK_OC = 32
    BLOCK_SP = 1024  # 1024 spatial tile; safe for common H*W
    tiles = (H * W + BLOCK_SP - 1) // BLOCK_SP
    grid_conv1 = (B, C, tiles)
    conv3x3_triton[grid_conv1](
        x, conv1_weight, y1,
        B, C, H, W, C,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
    )

    # GroupNorm + affine + SiLU for y1
    y1_norm = torch.empty_like(y1, dtype=torch.float32)
    num_groups = 32
    assert C % num_groups == 0, "C must be divisible by num_groups (32)"
    grid_gn1 = (B, num_groups)
    group_norm_affine_silu[grid_gn1](
        y1, norm1_weight, norm1_bias, y1_norm,
        B, C, H, W, num_groups, eps,
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
        BLOCK=1024,
    )

    # Conv2
    y2_pre = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
    conv3x3_triton[grid_conv1](
        y1_norm, conv2_weight, y2_pre,
        B, C, H, W, C,
        y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
        conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
        y2_pre.stride(0), y2_pre.stride(1), y2_pre.stride(2), y2_pre.stride(3),
        BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
    )

    # GroupNorm + affine + SiLU for y2
    y2 = torch.empty_like(y2_pre, dtype=torch.float32)
    grid_gn2 = (B, num_groups)
    group_norm_affine_silu[grid_gn2](
        y2_pre, norm2_weight, norm2_bias, y2,
        B, C, H, W, num_groups, eps,
        y2_pre.stride(0), y2_pre.stride(1), y2_pre.stride(2), y2_pre.stride(3),
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        BLOCK=1024,
    )

    # Residual add
    out = torch.empty_like(y2, dtype=torch.float32)
    N = out.numel()
    BLOCK_ADD = 4096
    grid_add = (triton.cdiv(N, BLOCK_ADD),)
    add_residual_kernel[grid_add](
        out, y2, x,
        N, BLOCK=BLOCK_ADD,
    )

    return out


class ModelNew(torch.nn.Module):
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        return run(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)


def run(*args):
    return ModelNew()(*args)
