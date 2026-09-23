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

    # initialize accumulator for [BLOCK_OC, BLOCK_SP]
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # loop over input channels in blocks
    for ic_base in range(0, C_in, BLOCK_OC):
        oc_vec = oc_base + tl.arange(0, BLOCK_OC)
        oc_mask = oc_vec < C_out

        # For each 3x3 neighborhood
        for kh in range(-1, 2):
            for kw in range(-1, 2):
                in_h = h + kh
                in_w = w + kw

                # valid positions within input
                valid_hw = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W) & sp_mask

                # load input vector x[b, ic, in_h, in_w] for all sp in this tile
                # We will loop over ic in this block
                for ic in range(0, BLOCK_OC):
                    ic_idx = ic_base + ic
                    ic_valid = ic_idx < C_in
                    # x_ptrs: b, ic_idx, in_h, in_w
                    x_ptrs = x_ptr \
                              + b * x_stride_b \
                              + ic_idx * x_stride_c \
                              + in_h * x_stride_h \
                              + in_w * x_stride_w
                    x_mask = valid_hw & ic_valid
                    x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_SP]

                    # load weight vector w[ic_idx, oc_vec, kh, kw]
                    w_ptrs = w_ptr \
                              + ic_idx * w_stride_cin \
                              + (oc_base + ic) * w_stride_cout \
                              + kh * w_stride_kh \
                              + kw * w_stride_kw
                    w_mask = oc_mask & ic_valid
                    w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_OC]

                    # outer product accumulate
                    acc += w_vals[:, None] * x_vals[None, :]

    # store results
    y_ptrs = y_ptr \
              + b * y_stride_b \
              + oc_vec[:, None] * y_stride_c \
              + h[None, :] * y_stride_h \
              + w[None, :] * y_stride_w
    oc_sp_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(y_ptrs, acc, mask=oc_sp_mask)


# Triton kernel: GroupNorm + affine (scale, bias) + SiLU per (batch, group)
# y: output (B, C, H, W), x: input (B, C, H, W), weight: (C,), bias: (C,), eps: float
# Grid: (B, num_groups). Assumes C % num_groups == 0, here num_groups=32.
@triton.jit
def group_norm_affine_silu(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    num_groups: tl.constexpr,  # 32
    eps: tl.constexpr,
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
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        B, C, H, W = x.shape

        # Ensure tensors are on CUDA and contiguous
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        # Allocate outputs for convs
        # conv1
        y1 = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)
        # conv2
        y2 = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)

        # Launch conv1 Triton
        BLOCK_OC = 32
        BLOCK_SP = 1024
        tiles = (H * W + BLOCK_SP - 1) // BLOCK_SP
        grid_conv = (B, C, tiles)
        conv3x3_triton[grid_conv](
            x, conv1_weight, y1,
            B, C, H, W, C,  # C_in = C_out = C for conv1
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
            num_warps=4, num_stages=2
        )

        # GroupNorm + affine + SiLU for conv1
        y1_gn = torch.empty_like(y1, dtype=torch.float32, device=x.device)
        grid_gn = (B, 32)
        group_norm_affine_silu[grid_gn](
            y1, norm1_weight, norm1_bias, y1_gn,
            B, C, H, W,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_gn.stride(0), y1_gn.stride(1), y1_gn.stride(2), y1_gn.stride(3),
            num_groups=32, eps=eps,
            BLOCK=1024,
            num_warps=4, num_stages=2
        )

        # conv2
        conv3x3_triton[grid_conv](
            y1_gn, conv2_weight, y2,
            B, C, H, W, C,  # C_in = C_out = C for conv2
            y1_gn.stride(0), y1_gn.stride(1), y1_gn.stride(2), y1_gn.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
            num_warps=4, num_stages=2
        )

        # GroupNorm + affine + SiLU for conv2
        y2_gn = torch.empty_like(y2, dtype=torch.float32, device=x.device)
        group_norm_affine_silu[grid_gn](
            y2, norm2_weight, norm2_bias, y2_gn,
            B, C, H, W,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_gn.stride(0), y2_gn.stride(1), y2_gn.stride(2), y2_gn.stride(3),
            num_groups=32, eps=eps,
            BLOCK=1024,
            num_warps=4, num_stages=2
        )

        # Final residual add: out = y2_gn + x
        out = torch.empty_like(y2, dtype=torch.float32, device=x.device)
        N = out.numel()
        grid_add = (triton.cdiv(N, 1024),)
        add_residual_kernel[grid_add](
            out, y2_gn, x,
            N,
            BLOCK=1024,
            num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
