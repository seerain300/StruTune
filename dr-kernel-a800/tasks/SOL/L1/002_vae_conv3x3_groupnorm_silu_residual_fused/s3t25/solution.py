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
    BLOCK_SP: tl.constexpr,  # e.g., 1024
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    tile_id = tl.program_id(2)

    # spatial vector for this tile
    start_sp = tile_id * BLOCK_SP
    sp = start_sp + tl.arange(0, BLOCK_SP)
    sp_mask = sp < (H * W)
    h = sp // W
    w = sp % W

    # Accumulate over all input channels
    acc = tl.zeros((BLOCK_SP,), dtype=tl.float32)

    for ic in range(0, C_in):
        # 3x3 neighborhood
        for kh in range(-1, 2):
            for kw in range(-1, 2):
                in_h = h + kh
                in_w = w + kw
                in_mask = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W) & sp_mask

                x_ptrs = x_ptr \
                         + b * x_stride_b \
                         + ic * x_stride_c \
                         + in_h * x_stride_h \
                         + in_w * x_stride_w
                x_vals = tl.load(x_ptrs, mask=in_mask, other=0.0)

                w_ptrs = w_ptr \
                         + ic * w_stride_cin \
                         + oc * w_stride_cout \
                         + (kh + 1) * w_stride_kh \
                         + (kw + 1) * w_stride_kw
                w_val = tl.load(w_ptrs)

                acc += x_vals * w_val

    # Store results y[b, oc, h, w]
    y_ptrs = y_ptr \
             + b * y_stride_b \
             + oc * y_stride_c \
             + h * y_stride_h \
             + w * y_stride_w
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
    BLOCK: tl.constexpr,  # reduction block
):
    b = tl.program_id(0)
    g = tl.program_id(1)

    group_channels = C // num_groups
    group_elements = group_channels * H * W

    # First pass: sum and sum of squares
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


# Triton kernel: elementwise residual add out = y + x
@triton.jit
def add_residual_kernel(out_ptr, y_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y_vals + x_vals, mask=mask)


# Helper to run conv3x3 Triton kernel for a given weight and output tensor
def triton_conv3x3(x, weight, out):
    B, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = weight.shape
    assert C_in == C_in_w and KH == 3 and KW == 3 and C_out == C_in, "Weight shape must be (C_in, C_out, 3, 3) and match input channels"

    x = x.contiguous()
    weight = weight.contiguous()
    out = out.contiguous()

    x_stride_b, x_stride_c, x_stride_h, x_stride_w = x.stride()
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw = weight.stride()
    y_stride_b, y_stride_c, y_stride_h, y_stride_w = out.stride()

    BLOCK_SP = 1024
    tiles = (H * W + BLOCK_SP - 1) // BLOCK_SP
    grid = (B, C_out, tiles)

    conv3x3_triton[grid](
        x, weight, out,
        B, C_in, H, W, C_out,
        x_stride_b, x_stride_c, x_stride_h, x_stride_w,
        w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
        y_stride_b, y_stride_c, y_stride_h, y_stride_w,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
    )


def group_norm_affine_silu_forward(x, weight, bias, out, eps=1e-5):
    B, C, H, W = x.shape
    num_groups = 32
    assert C % num_groups == 0, "num_groups must divide C"
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = out.contiguous()

    x_stride_b, x_stride_c, x_stride_h, x_stride_w = x.stride()
    y_stride_b, y_stride_c, y_stride_h, y_stride_w = out.stride()

    grid = (B, num_groups)
    BLOCK = 1024
    group_norm_affine_silu[grid](
        x, out, weight, bias,
        B, C, H, W,
        num_groups, eps,
        x_stride_b, x_stride_c, x_stride_h, x_stride_w,
        y_stride_b, y_stride_c, y_stride_h, y_stride_w,
        BLOCK=BLOCK,
        num_warps=4,
    )


def add_residual(y, x, out):
    N = y.numel()
    out = out.contiguous()
    grid = (triton.cdiv(N, 1024),)
    add_residual_kernel[grid](out, y, x, N, BLOCK=1024, num_warps=4)


class ModelNew(torch.nn.Module):
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps=1e-5):
        # Ensure tensors are on CUDA
        assert x.is_cuda and conv1_weight.is_cuda and norm1_weight.is_cuda and norm1_bias.is_cuda \
               and conv2_weight.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "All tensors must be on CUDA"

        # conv1: y1 = conv3x3(x, conv1_weight)
        y1 = torch.empty_like(x)
        triton_conv3x3(x, conv1_weight, y1)

        # GroupNorm + SiLU after conv1
        y1_out = torch.empty_like(y1)
        group_norm_affine_silu_forward(y1, norm1_weight, norm1_bias, y1_out, eps=eps)

        # conv2: y2 = conv3x3(y1_out, conv2_weight)
        y2 = torch.empty_like(y1_out)
        triton_conv3x3(y1_out, conv2_weight, y2)

        # GroupNorm + SiLU after conv2
        y2_out = torch.empty_like(y2)
        group_norm_affine_silu_forward(y2, norm2_weight, norm2_bias, y2_out, eps=eps)

        # Residual add: out = y2_out + x
        out = torch.empty_like(x)
        add_residual(y2_out, x, out)

        return out


def run(*args):
    return ModelNew()(*args)
