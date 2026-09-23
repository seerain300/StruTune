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

    # initialize accumulator for [BLOCK_SP]
    acc = tl.zeros((BLOCK_SP,), dtype=tl.float32)

    # iterate over output channels in the tile
    for oc_rel in range(0, BLOCK_OC):
        oc = oc_base + oc_rel
        oc_valid = oc < C_out

        # if oc is out of range, skip (grid may create tiles larger than C_out)
        if not oc_valid:
            continue

        # accumulate over input channels in blocks
        ic_base = 0
        while ic_base < C_in:
            ic_vec = ic_base + tl.arange(0, BLOCK_OC)
            ic_mask = ic_vec < C_in

            # For each (ic, oc), accumulate over 3x3 neighborhood
            # Note: we loop BLOCK_OC-sized vectors; only ic_rel < C_in contributes.
            for ic_rel in range(0, BLOCK_OC):
                ic = ic_vec[ic_rel]
                if ic_mask[ic_rel]:
                    # load weights for this (ic, oc, kh, kw)
                    # w layout: (C_in, C_out, 3, 3)
                    # We need w[ic, oc, kh, kw] for kh, kw in [-1, 0, 1]
                    # But since padding=1, indices are always in range.
                    # Compute scalar weights for each kh, kw
                    w00 = tl.load(w_ptr + ic * w_stride_cin + oc * w_stride_cout + 0 * w_stride_kh + 0 * w_stride_kw)
                    w01 = tl.load(w_ptr + ic * w_stride_cin + oc * w_stride_cout + 0 * w_stride_kh + 1 * w_stride_kw)
                    w02 = tl.load(w_ptr + ic * w_stride_cin + oc * w_stride_cout + 0 * w_stride_kh + 2 * w_stride_kw)
                    w10 = tl.load(w_ptr + ic * w_stride_cin + oc * w_stride_cout + 1 * w_stride_kh + 0 * w_stride_kw)
                    w11 = tl.load(w_ptr + ic * w_stride_cin + oc * w_stride_cout + 1 * w_stride_kh + 1 * w_stride_kw)
                    w12 = tl.load(w_ptr + ic * w_stride_cin + oc * w_stride_cout + 1 * w_stride_kh + 2 * w_stride_kw)
                    w20 = tl.load(w_ptr + ic * w_stride_cin + oc * w_stride_cout + 2 * w_stride_kh + 0 * w_stride_kw)
                    w21 = tl.load(w_ptr + ic * w_stride_cin + oc * w_stride_cout + 2 * w_stride_kh + 1 * w_stride_kw)
                    w22 = tl.load(w_ptr + ic * w_stride_cin + oc * w_stride_cout + 2 * w_stride_kh + 2 * w_stride_kw)

                    # For each kh, kw, compute input indices with padding=1:
                    # h_in = h + (kh - 1), w_in = w + (kw - 1)
                    # Masks ensure we don't read out of bounds.
                    # kh = -1 -> h_in = h - 1
                    h_in0 = h - 1
                    w_in0 = w - 1
                    mask0 = (h_in0 >= 0) & (h_in0 < H) & (w_in0 >= 0) & (w_in0 < W) & sp_mask
                    x00 = tl.load(x_ptr + b * x_stride_b + ic * x_stride_c + h_in0 * x_stride_h + w_in0 * x_stride_w, mask=mask0, other=0.0)
                    x01 = tl.load(x_ptr + b * x_stride_b + ic * x_stride_c + h_in0 * x_stride_h + (w + 1) * x_stride_w, mask=sp_mask, other=0.0)
                    x02 = tl.load(x_ptr + b * x_stride_b + ic * x_stride_c + h_in0 * x_stride_h + (w + 2) * x_stride_w, mask=sp_mask, other=0.0)

                    h_in1 = h
                    mask1 = (h_in1 >= 0) & (h_in1 < H) & sp_mask
                    x10 = tl.load(x_ptr + b * x_stride_b + ic * x_stride_c + h_in1 * x_stride_h + (w - 1) * x_stride_w, mask=mask1, other=0.0)
                    x11 = tl.load(x_ptr + b * x_stride_b + ic * x_stride_c + h_in1 * x_stride_h + w * x_stride_w, mask=sp_mask, other=0.0)
                    x12 = tl.load(x_ptr + b * x_stride_b + ic * x_stride_c + h_in1 * x_stride_h + (w + 1) * x_stride_w, mask=sp_mask, other=0.0)

                    h_in2 = h + 1
                    mask2 = (h_in2 >= 0) & (h_in2 < H) & sp_mask
                    x20 = tl.load(x_ptr + b * x_stride_b + ic * x_stride_c + h_in2 * x_stride_h + (w - 1) * x_stride_w, mask=mask2, other=0.0)
                    x21 = tl.load(x_ptr + b * x_stride_b + ic * x_stride_c + h_in2 * x_stride_h + w * x_stride_w, mask=sp_mask, other=0.0)
                    x22 = tl.load(x_ptr + b * x_stride_b + ic * x_stride_c + h_in2 * x_stride_h + (w + 1) * x_stride_w, mask=sp_mask, other=0.0)

                    # Accumulate contributions
                    acc += (x00 * w00 + x01 * w01 + x02 * w02 +
                            x10 * w10 + x11 * w11 + x12 * w12 +
                            x20 * w20 + x21 * w21 + x22 * w22)

            ic_base += BLOCK_OC  # advance input channel block

        # Store results for this oc
        y_ptrs = y_ptr + b * y_stride_b + oc * y_stride_c + h * y_stride_h + w * y_stride_w
        tl.store(y_ptrs, acc, mask=sp_mask)


# Triton kernel: GroupNorm forward + affine (scale, bias) + SiLU
# Assumes num_groups=32, and C % 32 == 0.
@triton.jit
def group_norm_affine_silu(
    x_ptr,           # *const float input (B, C, H, W)
    weight_ptr,      # *const float per-channel scale (C,)
    bias_ptr,        # *const float per-channel bias (C,)
    y_ptr,           # *float output (B, C, H, W)
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    num_groups: tl.constexpr,  # 32
    eps: tl.constexpr,
    # Strides for x and y (NCHW)
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,  # reduction/block size
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
    B, C, H, W = x.shape
    num_groups = 32

    # Ensure CUDA tensors and float32
    assert x.is_cuda, "Input tensor must be on CUDA for Triton kernels"
    # Allocate output buffers
    y1 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
    y2 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
    out = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)

    # Launch conv1
    grid_conv1 = (B, C, triton.cdiv(H * W, 1024))
    conv3x3_triton[grid_conv1](
        x, conv1_weight, y1,
        B, C, H, W, C,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        BLOCK_OC=32, BLOCK_SP=1024,
        num_warps=4,
    )

    # GroupNorm + affine + SiLU after conv1
    grid_gn1 = (B, num_groups)
    group_norm_affine_silu[grid_gn1](
        y1, norm1_weight, norm1_bias, y1,  # in-place for brevity
        B, C, H, W, num_groups, eps,
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
        BLOCK=1024,
        num_warps=4,
    )

    # SiLU in-place
    # We already fused SiLU in group_norm_affine_silu, y1 has SiLU applied

    # Residual path before conv2: out1 = y1
    out = y1

    # Launch conv2
    grid_conv2 = (B, C, triton.cdiv(H * W, 1024))
    conv3x3_triton[grid_conv2](
        out, conv2_weight, y2,
        B, C, H, W, C,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        BLOCK_OC=32, BLOCK_SP=1024,
        num_warps=4,
    )

    # GroupNorm + affine + SiLU after conv2
    grid_gn2 = (B, num_groups)
    group_norm_affine_silu[grid_gn2](
        y2, norm2_weight, norm2_bias, y2,  # in-place
        B, C, H, W, num_groups, eps,
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
        BLOCK=1024,
        num_warps=4,
    )

    # Residual add: y2 += x
    N = B * C * H * W
    grid_add = (triton.cdiv(N, 1024),)
    add_residual_kernel[grid_add](
        out, y2, x.contiguous(),
        N,
        BLOCK=1024,
        num_warps=4,
    )

    return out


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor, conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        # Ensure inputs are CUDA tensors (Triton requires CUDA)
        if not x.is_cuda:
            x = x.cuda()
        if not conv1_weight.is_cuda:
            conv1_weight = conv1_weight.cuda()
        if not norm1_weight.is_cuda:
            norm1_weight = norm1_weight.cuda()
        if not norm1_bias.is_cuda:
            norm1_bias = norm1_bias.cuda()
        if not conv2_weight.is_cuda:
            conv2_weight = conv2_weight.cuda()
        if not norm2_weight.is_cuda:
            norm2_weight = norm2_weight.cuda()
        if not norm2_bias.is_cuda:
            norm2_bias = norm2_bias.cuda()
        # Run Triton-based fused computation
        return run(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)


def run(*args):
    return ModelNew()(*args)
