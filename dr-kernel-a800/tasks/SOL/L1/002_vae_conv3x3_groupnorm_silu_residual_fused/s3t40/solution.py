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

    # loop over output channels in blocks
    oc_vec = oc_base + tl.arange(0, BLOCK_OC)
    oc_mask = oc_vec < C_out

    # loop over input channels in blocks
    for ic_base in range(0, C_in, 8):  # BLOCK_IC = 8; loop structure handles any C_in
        ic_vec = ic_base + tl.arange(0, 8)
        ic_mask = ic_vec < C_in

        # for each ic in the block, accumulate into acc[oc, sp]
        for ic in tl.static_range(8):  # unroll 8
            # if ic >= C_in, skip (ic_mask handles this)
            # construct pointers for x (neighborhood) and w, then multiply and accumulate
            # x neighborhood pointers for this ic: (h+kh, w+kw)
            # kh, kw in -1..1 -> map to 0..2 indices for w
            # Note: we load x with masks; invalid loads return 0.0
            for kh in range(-1, 2):
                ih = h + kh
                ih = tl.max(ih, 0)
                ih = tl.min(ih, H - 1)
                for kw in range(-1, 2):
                    iw = w + kw
                    iw = tl.max(iw, 0)
                    iw = tl.min(iw, W - 1)

                    x_ptrs = x_ptr \
                             + b * x_stride_b \
                             + ic_vec * x_stride_c \
                             + ih * x_stride_h \
                             + iw * x_stride_w
                    x_mask = ic_mask & sp_mask
                    x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_SP]

                    w_ptrs = w_ptr \
                             + ic_vec * w_stride_cin \
                             + (oc_vec + oc_base) * w_stride_cout \
                             + (kh + 1) * w_stride_kh \
                             + (kw + 1) * w_stride_kw
                    w_mask = oc_mask & ic_mask
                    w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_OC]

                    # outer product and accumulate
                    # acc[oc, sp] += sum over ic contribution
                    # Use broadcasting: w_vals[:, None] * x_vals[None, :]
                    contrib = w_vals[:, None] * x_vals[None, :]
                    acc += tl.sum(contrib, axis=0)  # sum across ic dimension

    # store results
    y_ptrs = y_ptr \
             + b * y_stride_b \
             + (oc_vec + oc_base) * y_stride_c \
             + h * y_stride_h \
             + w * y_stride_w
    tl.store(y_ptrs, acc, mask=oc_mask & sp_mask)


# Triton kernel: GroupNorm + affine (scale, bias) + SiLU, per (batch, group)
@triton.jit
def group_norm_affine_silu(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W, num_groups: tl.constexpr, eps: tl.constexpr,
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
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y_vals + x_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor):
        # Ensure contiguity and dtypes
        x = x.contiguous()
        # conv weights: (C_in, C_out, 3, 3)
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()

        B, C, H, W = x.shape
        # conv1
        y1 = torch.empty_like(x)
        tiles_hw = (H * W + 1023) // 1024  # BLOCK_SP=1024
        grid1 = (B, C, tiles_hw)
        conv3x3_triton[grid1](
            x, conv1_weight, y1,
            B, C, H, W, C,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_OC=32, BLOCK_SP=1024,
        )
        # GroupNorm + affine + SiLU for conv1
        y1_grouped = torch.empty_like(y1)
        grid_gn1 = (B, self.num_groups)
        group_norm_affine_silu[grid_gn1](
            y1, norm1_weight, norm1_bias, y1_grouped,
            B, C, H, W, self.num_groups, self.eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_grouped.stride(0), y1_grouped.stride(1), y1_grouped.stride(2), y1_grouped.stride(3),
            BLOCK=1024,
        )
        # conv2 on the result of conv1
        y2_pre = torch.empty_like(y1_grouped)
        tiles_hw2 = (H * W + 1023) // 1024  # same spatial tiling
        grid2 = (B, C, tiles_hw2)
        conv3x3_triton[grid2](
            y1_grouped, conv2_weight, y2_pre,
            B, C, H, W, C,
            y1_grouped.stride(0), y1_grouped.stride(1), y1_grouped.stride(2), y1_grouped.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2_pre.stride(0), y2_pre.stride(1), y2_pre.stride(2), y2_pre.stride(3),
            BLOCK_OC=32, BLOCK_SP=1024,
        )
        # GroupNorm + affine + SiLU for conv2
        y2_grouped = torch.empty_like(y2_pre)
        grid_gn2 = (B, self.num_groups)
        group_norm_affine_silu[grid_gn2](
            y2_pre, norm2_weight, norm2_bias, y2_grouped,
            B, C, H, W, self.num_groups, self.eps,
            y2_pre.stride(0), y2_pre.stride(1), y2_pre.stride(2), y2_pre.stride(3),
            y2_grouped.stride(0), y2_grouped.stride(1), y2_grouped.stride(2), y2_grouped.stride(3),
            BLOCK=1024,
        )
        # Residual add: y2 + x
        N = B * C * H * W
        out = torch.empty(N, dtype=torch.float32, device=x.device)
        grid_add = ((N + 1023) // 1024,)
        add_residual_kernel[grid_add](
            out, y2_grouped, x, N, 1024
        )
        # Reshape back to (B, C, H, W)
        out = out.view(B, C, H, W)
        return out


def run(*args):
    return ModelNew()(*args)
