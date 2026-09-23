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

    # initialize accumulator for [BLOCK_SP]
    acc = tl.zeros((BLOCK_SP,), dtype=tl.float32)

    # loop over output channels in blocks
    for oc_start in range(0, C_out, BLOCK_OC):
        oc_vec = oc_start + tl.arange(0, BLOCK_OC)
        oc_mask = oc_vec < C_out

        # loop over input channels in chunks
        for ic in range(0, C_in):
            # Accumulate contributions for each 3x3 neighborhood
            for kh in range(-1, 2):
                in_h = h + kh
                in_h_mask = (in_h >= 0) & (in_h < H)
                for kw in range(-1, 2):
                    in_w = w + kw
                    in_w_mask = (in_w >= 0) & (in_w < W)
                    mask = sp_mask & in_h_mask & in_w_mask

                    # x[b, ic, in_h, in_w]
                    x_ptrs = x_ptr \
                             + b * x_stride_b \
                             + ic * x_stride_c \
                             + in_h * x_stride_h \
                             + in_w * x_stride_w
                    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)  # [BLOCK_SP]

                    # w[ic, oc_vec, kh+1, kw+1]
                    w_ptrs = w_ptr \
                             + ic * w_stride_cin \
                             + oc_vec * w_stride_cout \
                             + (kh + 1) * w_stride_kh \
                             + (kw + 1) * w_stride_kw
                    oc_mask_vec = oc_mask
                    w_vals = tl.load(w_ptrs, mask=oc_mask_vec, other=0.0)  # [BLOCK_OC]

                    # outer product accumulate: [BLOCK_OC, 1] * [1, BLOCK_SP]
                    acc += w_vals[:, None] * x_vals[None, :]

    # Store results
    y_ptrs = y_ptr \
              + b * y_stride_b \
              + oc_vec * y_stride_c \
              + h * y_stride_h \
              + w * y_stride_w
    store_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(y_ptrs, acc, mask=store_mask)


# Triton kernel: GroupNorm + affine (scale, bias) + SiLU, per (batch, group)
# x: (B, C, H, W), weight: (C,), bias: (C,), y: (B, C, H, W)
@triton.jit
def group_norm_affine_silu(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W, num_groups, eps,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,  # reduction block size
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


# Triton kernel: elementwise residual add y = y + x over flattened N elements
@triton.jit
def add_residual_kernel(out_ptr, y_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out_vals = y_vals + x_vals
    tl.store(out_ptr + offsets, out_vals, mask=mask)


# ModelNew: Triton-based forward, must launch all defined kernels
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias,
                conv2_weight, norm2_weight, norm2_bias, eps):
        B, C, H, W = x.shape
        device = x.device

        # Ensure tensors are on the same device; cast to float32 for Triton math
        x_c = x.to(device=device, dtype=torch.float32)
        conv1_w = conv1_weight.to(device=device, dtype=torch.float32)
        conv2_w = conv2_weight.to(device=device, dtype=torch.float32)
        norm1_scale = norm1_weight.to(device=device, dtype=torch.float32)
        norm1_bias = norm1_bias.to(device=device, dtype=torch.float32)
        norm2_scale = norm2_weight.to(device=device, dtype=torch.float32)
        norm2_bias = norm2_bias.to(device=device, dtype=torch.float32)

        # First conv
        y1 = torch.empty((B, C, H, W), device=device, dtype=torch.float32)

        BLOCK_OC = 32
        BLOCK_SP = 1024
        tiles = (H * W + BLOCK_SP - 1) // BLOCK_SP
        grid_conv = (B, C, tiles)

        conv3x3_triton[grid_conv](
            x_c, conv1_w, y1,
            B, C, H, W, C,
            x_c.stride(0), x_c.stride(1), x_c.stride(2), x_c.stride(3),
            conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2), conv1_w.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
        )

        # GroupNorm1 + affine + SiLU
        y1_out = torch.empty_like(y1)
        grid_gn1 = (B, 32)
        group_norm_affine_silu[grid_gn1](
            y1, norm1_scale, norm1_bias, y1_out,
            B, C, H, W, 32, eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_out.stride(0), y1_out.stride(1), y1_out.stride(2), y1_out.stride(3),
            BLOCK=1024,
        )

        # Second conv
        y2 = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        conv3x3_triton[grid_conv](
            y1_out, conv2_w, y2,
            B, C, H, W, C,
            y1_out.stride(0), y1_out.stride(1), y1_out.stride(2), y1_out.stride(3),
            conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2), conv2_w.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
        )

        # GroupNorm2 + affine + SiLU
        y2_out = torch.empty_like(y2)
        grid_gn2 = (B, 32)
        group_norm_affine_silu[grid_gn2](
            y2, norm2_scale, norm2_bias, y2_out,
            B, C, H, W, 32, eps,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_out.stride(0), y2_out.stride(1), y2_out.stride(2), y2_out.stride(3),
            BLOCK=1024,
        )

        # Residual add: y2_out + x_c
        N = B * C * H * W
        out_flat = torch.empty(N, device=device, dtype=torch.float32)
        add_residual_kernel[(N + 1024 - 1) // 1024, ](
            out_flat, y2_out.reshape(-1), x_c.reshape(-1), N,
            BLOCK=1024,
        )
        out = out_flat.reshape(B, C, H, W)

        return out


def run(*args):
    return ModelNew()(*args)
