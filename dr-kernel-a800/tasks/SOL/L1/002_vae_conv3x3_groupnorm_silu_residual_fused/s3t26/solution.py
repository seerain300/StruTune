import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 conv forward (no bias), stride=1, padding=1
# x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), y: (B, C_out, H, W)
# Grid: (B, C_out, tiles over spatial). Each program handles one (b, oc) and one spatial tile (BLOCK_H x BLOCK_W).
@triton.jit
def conv3x3_triton(
    x_ptr, w_ptr, y_ptr,
    B, C_in, H, W, C_out,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK_H: tl.constexpr,  # e.g., 16
    BLOCK_W: tl.constexpr,  # e.g., 16
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    tile_id = tl.program_id(2)

    # compute number of tiles along H and W
    num_tiles_h = (H + BLOCK_H - 1) // BLOCK_H
    num_tiles_w = (W + BLOCK_W - 1) // BLOCK_W
    tile_h = tile_id // num_tiles_w
    tile_w = tile_id % num_tiles_w

    h_start = tile_h * BLOCK_H
    w_start = tile_w * BLOCK_W

    h_vec = h_start + tl.arange(0, BLOCK_H)[:, None]  # [BLOCK_H, 1]
    w_vec = w_start + tl.arange(0, BLOCK_W)[None, :]  # [1, BLOCK_W]
    sp_mask = (h_vec < H) & (w_vec < W)              # [BLOCK_H, BLOCK_W]

    # accumulator for this tile
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # loop over input channels
    for ic in range(0, C_in):
        # loop over 3x3 neighborhood (padding=1, stride=1)
        for kh in range(-1, 2):
            for kw in range(-1, 2):
                in_h = h_vec + kh  # [BLOCK_H, 1]
                in_w = w_vec + kw  # [1, BLOCK_W]
                in_mask = sp_mask & (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W)

                # load input x[b, ic, in_h, in_w]
                x_ptrs = x_ptr \
                         + b * x_stride_b \
                         + ic * x_stride_c \
                         + in_h * x_stride_h \
                         + in_w * x_stride_w
                x_vals = tl.load(x_ptrs, mask=in_mask, other=0.0)  # [BLOCK_H, BLOCK_W]

                # load weight w[ic, oc, kh+1, kw+1]
                w_ptrs = w_ptr \
                         + ic * w_stride_cin \
                         + oc * w_stride_cout \
                         + (kh + 1) * w_stride_kh \
                         + (kw + 1) * w_stride_kw
                w_val = tl.load(w_ptrs)  # scalar

                # accumulate
                acc += x_vals * w_val

    # store to y[b, oc, h_vec, w_vec]
    y_ptrs = y_ptr \
             + b * y_stride_b \
             + oc * y_stride_c \
             + h_vec * y_stride_h \
             + w_vec * y_stride_w
    tl.store(y_ptrs, acc, mask=sp_mask)


# Triton kernel: GroupNorm forward + affine (scale, bias) + SiLU
# Assumes num_groups=32 and C % 32 == 0. Operates over (B, groups).
@triton.jit
def group_norm_affine_silu(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    B, C, H, W,
    num_groups: tl.constexpr,  # 32
    eps: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
):
    b = tl.program_id(0)
    g = tl.program_id(1)

    group_channels = C // num_groups
    group_elements = group_channels * H * W

    # First pass: compute sum and sum of squares over the group
    sum_val = 0.0
    sum_sq = 0.0
    for start in range(0, group_elements, 1024):
        idx = start + tl.arange(0, 1024)
        mask = idx < group_elements

        ch = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr \
                 + b * x_stride_b \
                 + ch * x_stride_c \
                 + h * x_stride_h \
                 + w * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        sum_val += tl.sum(x_vals, axis=0)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    mean = sum_val / group_elements
    var = sum_sq / group_elements - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize, affine, SiLU
    for start in range(0, group_elements, 1024):
        idx = start + tl.arange(0, 1024)
        mask = idx < group_elements

        ch = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr \
                 + b * x_stride_b \
                 + ch * x_stride_c \
                 + h * x_stride_h \
                 + w * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        norm_vals = (x_vals - mean) * inv_std

        scale = tl.load(weight_ptr + ch, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + ch, mask=mask, other=0.0)
        z = norm_vals * scale + bias

        s = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * s

        y_ptrs = y_ptr \
                 + b * y_stride_b \
                 + ch * y_stride_c \
                 + h * y_stride_h \
                 + w * y_stride_w
        tl.store(y_ptrs, y_vals, mask=mask)


# Triton kernel: elementwise residual add y = y + x (flattened)
@triton.jit
def add_residual_kernel(y_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    curr = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, curr + vals, mask=mask)


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
    """
    B, C, H, W = x.shape
    device = x.device
    dtype = x.dtype

    # Ensure tensors are on GPU and contiguous
    x = x.contiguous()
    conv1_weight = conv1_weight.contiguous()
    conv2_weight = conv2_weight.contiguous()
    norm1_weight = norm1_weight.contiguous()
    norm1_bias = norm1_bias.contiguous()
    norm2_weight = norm2_weight.contiguous()
    norm2_bias = norm2_bias.contiguous()

    # First conv
    y1 = torch.empty((B, C, H, W), device=device, dtype=dtype)
    tiles = (H + 15) // 16 * (W + 15) // 16
    grid_conv = (B, C, tiles)
    conv3x3_triton[grid_conv](
        x, conv1_weight, y1,
        B, C, H, W, C,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        16, 16,
        num_warps=4, num_stages=2
    )

    # First GroupNorm + affine + SiLU
    y1_norm = torch.empty_like(y1)
    grid_gn1 = (B, 32)
    group_norm_affine_silu[grid_gn1](
        y1, y1_norm, norm1_weight, norm1_bias,
        B, C, H, W,
        32, eps,
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
        num_warps=4, num_stages=2
    )

    # Second conv
    y2 = torch.empty((B, C, H, W), device=device, dtype=dtype)
    tiles2 = (H + 15) // 16 * (W + 15) // 16
    grid_conv2 = (B, C, tiles2)
    conv3x3_triton[grid_conv2](
        y1_norm, conv2_weight, y2,
        B, C, H, W, C,
        y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
        conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        16, 16,
        num_warps=4, num_stages=2
    )

    # Second GroupNorm + affine + SiLU
    y2_norm = torch.empty_like(y2)
    grid_gn2 = (B, 32)
    group_norm_affine_silu[grid_gn2](
        y2, y2_norm, norm2_weight, norm2_bias,
        B, C, H, W,
        32, eps,
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
        num_warps=4, num_stages=2
    )

    # Add residual: out = y2_norm + x
    N = B * C * H * W
    grid_add = (triton.cdiv(N, 1024),)
    add_residual_kernel[grid_add](
        y2_norm, x, N,
        1024,
        num_warps=4, num_stages=2
    )

    return y2_norm


class ModelNew(torch.nn.Module):
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        return run(
            x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps
        )


def run(*args):
    return ModelNew()(*args)
