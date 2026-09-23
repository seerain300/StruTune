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

    # Compute tile start indices
    h_start = (tile_id // (W // BLOCK_W)) * BLOCK_H
    w_start = (tile_id % (W // BLOCK_W)) * BLOCK_W

    h_vec = h_start + tl.arange(0, BLOCK_H)
    w_vec = w_start + tl.arange(0, BLOCK_W)

    # Masks for in-bounds h, w
    h_mask = h_vec < H
    w_mask = w_vec < W

    # Create 2D mesh for tile
    h_mat = h_vec[:, None]  # [BLOCK_H, 1]
    w_mat = w_vec[None, :]  # [1, BLOCK_W]
    mask_hw = h_mask[:, None] & w_mask[None, :]  # [BLOCK_H, BLOCK_W]

    # Accumulator
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    # Loop over input channels
    for ic in range(0, C_in):
        # Loop over 3x3 neighborhood (padding=1, stride=1)
        for kh in range(-1, 2):
            for kw in range(-1, 2):
                in_h = h_mat + kh  # [BLOCK_H, 1]
                in_w = w_mat + kw  # [1, BLOCK_W]
                in_mask = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W) & mask_hw

                # Load x[b, ic, in_h, in_w]
                x_ptrs = x_ptr \
                         + b * x_stride_b \
                         + ic * x_stride_c \
                         + in_h * x_stride_h \
                         + in_w * x_stride_w
                x_vals = tl.load(x_ptrs, mask=in_mask, other=0.0)  # [BLOCK_H, BLOCK_W]

                # Load w[ic, oc, kh+1, kw+1]
                w_ptrs = w_ptr \
                         + ic * w_stride_cin \
                         + oc * w_stride_cout \
                         + (kh + 1) * w_stride_kh \
                         + (kw + 1) * w_stride_kw
                w_val = tl.load(w_ptrs)

                # Accumulate
                acc += x_vals * w_val

    # Store to y[b, oc, h_vec, w_vec]
    y_ptrs = y_ptr \
             + b * y_stride_b \
             + oc * y_stride_c \
             + h_mat * y_stride_h \
             + w_mat * y_stride_w
    tl.store(y_ptrs, acc, mask=mask_hw)


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
    BLOCK: tl.constexpr,  # e.g., 1024
):
    b = tl.program_id(0)
    g = tl.program_id(1)
    group_channels = C // num_groups
    group_elements = group_channels * H * W

    # First pass: compute sum and sum of squares over the group
    sum_val = 0.0
    sum_sq = 0.0
    for start in range(0, group_elements, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < group_elements

        c_vec = g * group_channels + (idx // (H * W))
        sp_vec = idx % (H * W)
        h_vec = sp_vec // W
        w_vec = sp_vec % W

        x_ptrs = x_ptr \
                 + b * x_stride_b \
                 + c_vec * x_stride_c \
                 + h_vec * x_stride_h \
                 + w_vec * x_stride_w
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

        c_vec = g * group_channels + (idx // (H * W))
        sp_vec = idx % (H * W)
        h_vec = sp_vec // W
        w_vec = sp_vec % W

        x_ptrs = x_ptr \
                 + b * x_stride_b \
                 + c_vec * x_stride_c \
                 + h_vec * x_stride_h \
                 + w_vec * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        norm_vals = (x_vals - mean) * inv_std

        scale = tl.load(weight_ptr + c_vec, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + c_vec, mask=mask, other=0.0)
        z = norm_vals * scale + bias

        # SiLU: z * sigmoid(z)
        s = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * s

        y_ptrs = y_ptr \
                 + b * y_stride_b \
                 + c_vec * y_stride_c \
                 + h_vec * y_stride_h \
                 + w_vec * y_stride_w
        tl.store(y_ptrs, y_vals, mask=mask)


# Triton kernel: elementwise residual add y = y + x over flattened tensor
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
    # Ensure tensors are CUDA and contiguous
    assert x.is_cuda and conv1_weight.is_cuda and conv2_weight.is_cuda, "Tensors must be on CUDA."
    B, C, H, W = x.shape
    Cw1, Cw2 = conv1_weight.shape[0], conv2_weight.shape[0]
    assert Cw1 == C and Cw2 == C, "conv weights in/out channels must match input C."

    # 1) Conv1: y1 = conv2d(x, conv1_weight, stride=1, padding=1, no bias)
    y1 = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)
    grid1 = (B, C, (H * W + 15) // 16)  # tiles over spatial; BLOCK_H=16, BLOCK_W=16
    conv3x3_triton[grid1](
        x, conv1_weight, y1,
        B, C, H, W, C,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        16, 16,
        num_warps=4, num_stages=2
    )

    # 2) GroupNorm + affine (norm1_weight, norm1_bias) + SiLU on y1
    y1_norm = torch.empty_like(y1)
    grid_gn1 = (B, 32)
    group_norm_affine_silu[grid_gn1](
        y1, y1_norm, norm1_weight, norm1_bias,
        B, C, H, W,
        32, eps,
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
        1024,
        num_warps=4, num_stages=2
    )

    # 3) Conv2: y2 = conv2d(y1_norm, conv2_weight, stride=1, padding=1, no bias)
    y2 = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)
    grid2 = (B, C, (H * W + 15) // 16)
    conv3x3_triton[grid2](
        y1_norm, conv2_weight, y2,
        B, C, H, W, C,
        y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
        conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        16, 16,
        num_warps=4, num_stages=2
    )

    # 4) GroupNorm + affine (norm2_weight, norm2_bias) + SiLU on y2
    y2_norm = torch.empty_like(y2)
    grid_gn2 = (B, 32)
    group_norm_affine_silu[grid_gn2](
        y2, y2_norm, norm2_weight, norm2_bias,
        B, C, H, W,
        32, eps,
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
        1024,
        num_warps=4, num_stages=2
    )

    # 5) Residual add: out = y2_norm + x
    out = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)
    N = B * C * H * W
    grid_add = (triton.cdiv(N, 4096),)
    add_residual_kernel[grid_add](
        out, y2_norm, x, N, 4096,
        num_warps=4, num_stages=2
    )

    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect: x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps
        assert len(args) == 8, "ModelNew expects 8 arguments: x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps"
        x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps = args
        return run(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)


def run(*args):
    return ModelNew()(*args)
