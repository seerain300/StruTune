import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 conv forward (no bias), stride=1, padding=1
# x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), y: (B, C_out, H, W)
# Grid: (B, C_out, tiles over H*W). Each program handles one (b, oc) and one spatial tile.
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

    # accumulator for BLOCK_SP outputs
    acc = tl.zeros((BLOCK_SP,), dtype=tl.float32)

    # loop over output channels in blocks
    for oc_off in range(0, C_out, BLOCK_OC):
        oc_vec = oc_base + oc_off + tl.arange(0, BLOCK_OC)
        oc_mask = oc_vec < C_out

        # initialize partial accumulator for [BLOCK_OC, BLOCK_SP]
        acc_part = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

        # accumulate over input channels and 3x3 neighborhood
        for ic_base in range(0, C_in, BLOCK_OC):
            ic_vec = ic_base + tl.arange(0, BLOCK_OC)
            ic_mask = ic_vec < C_in

            # For each ic block, multiply by corresponding w[ic, oc, kh, kw] over 3x3
            for ic_i in range(0, BLOCK_OC):
                ic = ic_vec[ic_i]
                if ic_mask[ic_i]:
                    # Loop over the 3x3 neighborhood with padding=1
                    for kh in range(-1, 2):
                        for kw in range(-1, 2):
                            h_in = h + kh
                            w_in = w + kw
                            in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W) & sp_mask

                            x_ptrs = x_ptr \
                                     + b * x_stride_b \
                                     + ic * x_stride_c \
                                     + h_in * x_stride_h \
                                     + w_in * x_stride_w
                            x_vals = tl.load(x_ptrs, mask=in_bounds, other=0.0)  # [BLOCK_SP]

                            # Accumulate into each oc in the current block
                            for oc_i in range(0, BLOCK_OC):
                                oc = oc_vec[oc_i]
                                if oc_mask[oc_i]:
                                    w_ptrs = w_ptr \
                                             + ic * w_stride_cin \
                                             + oc * w_stride_cout \
                                             + kh * w_stride_kh \
                                             + kw * w_stride_kw
                                    w_val = tl.load(w_ptrs, mask=True, other=0.0)  # scalar
                                    acc_part[oc_i, :] += x_vals * w_val

        # Sum contributions from all ic blocks and add to acc
        for oc_i in range(0, BLOCK_OC):
            if oc_mask[oc_i]:
                acc += acc_part[oc_i, :]

    # Store results: one program handles a slice of oc of size BLOCK_OC; here we handle oc_base + 0
    # Note: We can only store a single oc slice per program since BLOCK_OC dimension was reduced to scalar acc.
    # Implement store for the first oc in the block; Triton allows dynamic indexing in store pointers.
    oc_i = oc_base  # first oc in block
    if oc_i < C_out:
        y_ptrs = y_ptr \
                 + b * y_stride_b \
                 + oc_i * y_stride_c \
                 + h * y_stride_h \
                 + w * y_stride_w
        tl.store(y_ptrs, acc, mask=sp_mask)


# Triton kernel: GroupNorm forward + affine (scale, bias) + SiLU
# Assumes num_groups=32 and C % 32 == 0.
@triton.jit
def group_norm_affine_silu(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W,
    num_groups: tl.constexpr,  # 32
    eps: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,  # reduction tile
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
    for start in range(0, group_elements, BLOCK):
        idx = start + tl.arange(0, BLOCK)
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


# Triton kernel: elementwise residual add y = y + x
@triton.jit
def add_residual_kernel(out_ptr, y_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y_vals + x_vals, mask=mask)


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
    assert x.is_cuda and conv1_weight.is_cuda and conv2_weight.is_cuda, "All tensors must be on CUDA"
    assert x.dtype == torch.float32 and conv1_weight.dtype == torch.float32 and conv2_weight.dtype == torch.float32, "Use float32 for correctness"
    B, C, H, W = x.shape

    # 1) First conv: (C, C, 3, 3) -> y1
    y1 = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)
    grid_conv = (B, C, triton.cdiv(H * W, 1024))
    conv3x3_triton[grid_conv](
        x, conv1_weight, y1,
        B, C, H, W, C,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        BLOCK_OC=32, BLOCK_SP=1024,
    )

    # 2) GroupNorm + affine + SiLU after conv1
    y1_norm = torch.empty_like(y1)
    grid_gn = (B, 32)
    group_norm_affine_silu[grid_gn](
        y1, norm1_weight, norm1_bias, y1_norm,
        B, C, H, W,
        32, eps,
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
        BLOCK=1024,
    )

    # 3) Second conv: (C, C, 3, 3) on y1_norm -> y2_pre
    y2_pre = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)
    grid_conv2 = (B, C, triton.cdiv(H * W, 1024))
    conv3x3_triton[grid_conv2](
        y1_norm, conv2_weight, y2_pre,
        B, C, H, W, C,
        y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
        conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
        y2_pre.stride(0), y2_pre.stride(1), y2_pre.stride(2), y2_pre.stride(3),
        BLOCK_OC=32, BLOCK_SP=1024,
    )

    # 4) GroupNorm + affine + SiLU after conv2
    y2_norm = torch.empty_like(y2_pre)
    grid_gn2 = (B, 32)
    group_norm_affine_silu[grid_gn2](
        y2_pre, norm2_weight, norm2_bias, y2_norm,
        B, C, H, W,
        32, eps,
        y2_pre.stride(0), y2_pre.stride(1), y2_pre.stride(2), y2_pre.stride(3),
        y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
        BLOCK=1024,
    )

    # 5) Residual addition: y2_norm = y2_norm + x
    out = torch.empty_like(y2_norm)
    N = B * C * H * W
    grid_add = (triton.cdiv(N, 4096),)
    add_residual_kernel[grid_add](
        out, y2_norm, x, N, 4096
    )

    return out


class ModelNew(torch.nn.Module):
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        # Use the Triton-based run
        return run(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)


def run(*args):
    return ModelNew()(*args)
