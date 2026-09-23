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

    # Initialize accumulator for this tile: [BLOCK_SP]
    acc = tl.zeros((BLOCK_SP,), dtype=tl.float32)

    # Loop over input channels in blocks
    for ic_base in range(0, C_in, BLOCK_OC):
        ic_vec = ic_base + tl.arange(0, BLOCK_OC)
        ic_mask = ic_vec < C_in

        # For each of the 3x3 neighborhood
        for kh in range(3):
            ih = h + kh - 1  # padding=1
            for kw in range(3):
                iw = w + kw - 1  # padding=1

                # Build pointers for x[b, ic, ih, iw] with mask
                x_ptrs = x_ptr \
                         + b * x_stride_b \
                         + ic_vec[:, None] * x_stride_c \
                         + ih * x_stride_h \
                         + iw * x_stride_w
                x_mask = ic_mask[:, None] & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_IC, BLOCK_SP]

                # Load w[ic, oc, kh, kw] for this oc across BLOCK_OC input channels
                w_ptrs = w_ptr \
                         + ic_vec * w_stride_cin \
                         + oc_base * w_stride_cout \
                         + kh * w_stride_kh \
                         + kw * w_stride_kw
                w_vals = tl.load(w_ptrs, mask=ic_mask, other=0.0)  # [BLOCK_IC]

                # Accumulate: sum over ic dimension
                acc += tl.sum(x_vals * w_vals[:, None], axis=0)

    # Store results for this tile
    y_ptrs = y_ptr \
             + b * y_stride_b \
             + oc_base * y_stride_c \
             + h * y_stride_h \
             + w * y_stride_w
    tl.store(y_ptrs, acc, mask=sp_mask)


# Triton kernel: GroupNorm + affine (scale, bias) + SiLU over per-sample, per-group
# x: (B, C, H, W), weight: (C,), bias: (C,), y: (B, C, H, W)
# Grid: (B, num_groups). Assumes num_groups=32 and C % 32 == 0.
@triton.jit
def group_norm_affine_silu(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W, num_groups: tl.constexpr,  # 32
    eps: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,  # reduction tile size
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

        # SiLU: z * sigmoid(z) = z * (1 / (1 + exp(-z)))
        s = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * s

        y_ptrs = y_ptr \
                 + b * y_stride_b \
                 + ch * y_stride_c \
                 + h * y_stride_h \
                 + w * y_stride_w
        tl.store(y_ptrs, y_vals, mask=mask)


# Triton kernel: elementwise residual add y = y + x, over flattened N elements
@triton.jit
def add_residual_kernel(out_ptr, y_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y_vals + x_vals, mask=mask)


def _run_fused_block_triton(
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
    Triton implementation of:
      out = SiLU(GroupNorm(Conv3x3(x, conv1_weight), num_groups=32, weight=norm1_weight, bias=norm1_bias, eps=eps))
        + SiLU(GroupNorm(Conv3x3(out, conv2_weight), num_groups=32, weight=norm2_weight, bias=norm2_bias, eps=eps))
        + x
    Assumes:
      - x: (B, C, H, W), NCHW
      - conv weights: (C_in, C_out, 3, 3), NCHW
      - num_groups=32 and C % 32 == 0
    """
    # Ensure CUDA and float32 for math
    assert x.is_cuda, "Inputs must be CUDA tensors for Triton."
    # We'll compute in float32 for stability; cast if needed
    x_f = x.contiguous().to(torch.float32)
    conv1_weight_f = conv1_weight.contiguous().to(torch.float32)
    norm1_weight_f = norm1_weight.contiguous().to(torch.float32)
    norm1_bias_f = norm1_bias.contiguous().to(torch.float32)
    conv2_weight_f = conv2_weight.contiguous().to(torch.float32)
    norm2_weight_f = norm2_weight.contiguous().to(torch.float32)
    norm2_bias_f = norm2_bias.contiguous().to(torch.float32)

    B, C, H, W = x_f.shape

    # Output after first conv
    out1 = torch.empty((B, C, H, W), device=x_f.device, dtype=torch.float32)

    # Launch conv1 Triton kernel: grid over (B, C, tiles over H*W)
    BLOCK_OC = 32
    BLOCK_SP = 1024
    tiles_hw = (H * W + BLOCK_SP - 1) // BLOCK_SP
    grid1 = (B, C, tiles_hw)
    conv3x3_triton[grid1](
        x_f, conv1_weight_f, out1,
        B, C, H, W, C,
        x_f.stride(0), x_f.stride(1), x_f.stride(2), x_f.stride(3),
        conv1_weight_f.stride(0), conv1_weight_f.stride(1), conv1_weight_f.stride(2), conv1_weight_f.stride(3),
        out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
        num_stages=2,
    )

    # GroupNorm + SiLU after conv1
    out1_after = torch.empty_like(out1)
    grid_gn1 = (B, 32)
    group_norm_affine_silu[grid_gn1](
        out1, norm1_weight_f, norm1_bias_f, out1_after,
        B, C, H, W, 32, eps,
        out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        out1_after.stride(0), out1_after.stride(1), out1_after.stride(2), out1_after.stride(3),
        BLOCK=1024,
        num_warps=4,
        num_stages=2,
    )

    # Conv2
    out2_pre = torch.empty_like(out1_after)
    conv3x3_triton[grid1](
        out1_after, conv2_weight_f, out2_pre,
        B, C, H, W, C,
        out1_after.stride(0), out1_after.stride(1), out1_after.stride(2), out1_after.stride(3),
        conv2_weight_f.stride(0), conv2_weight_f.stride(1), conv2_weight_f.stride(2), conv2_weight_f.stride(3),
        out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
        num_stages=2,
    )

    # GroupNorm + SiLU after conv2
    out2_after = torch.empty_like(out2_pre)
    grid_gn2 = (B, 32)
    group_norm_affine_silu[grid_gn2](
        out2_pre, norm2_weight_f, norm2_bias_f, out2_after,
        B, C, H, W, 32, eps,
        out2_pre.stride(0), out2_pre.stride(1), out2_pre.stride(2), out2_pre.stride(3),
        out2_after.stride(0), out2_after.stride(1), out2_after.stride(2), out2_after.stride(3),
        BLOCK=1024,
        num_warps=4,
        num_stages=2,
    )

    # Residual add
    final_out = torch.empty((B, C, H, W), device=x_f.device, dtype=torch.float32)
    N = B * C * H * W
    add_residual_kernel[(N + 1023) // 1024,](
        final_out, out2_after, x_f, N,
        BLOCK=1024,
        num_warps=4,
        num_stages=2,
    )

    # Return final_out as float32 (matches original default dtype)
    return final_out


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor, conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        return _run_fused_block_triton(x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps)


def run(*args):
    return ModelNew()(*args)
