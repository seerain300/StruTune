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
    BLOCK_SP: tl.constexpr,  # e.g., 256
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
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

    # loop over input channels in blocks
    for ic_base in range(0, C_in, BLOCK_OC):
        # iterate over the BLOCK_OC input channels in this block
        for k in range(0, BLOCK_OC):
            ic = ic_base + k
            valid_ic = ic < C_in

            # 3x3 neighborhood with padding=1
            for kh in range(3):
                for kw in range(3):
                    ih = h + (kh - 1)
                    iw = w + (kw - 1)

                    x_ptrs = x_ptr \
                             + b * x_stride_b \
                             + ic * x_stride_c \
                             + ih * x_stride_h \
                             + iw * x_stride_w
                    mask = sp_mask & valid_ic
                    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

                    w_ptrs = w_ptr \
                             + ic * w_stride_cin \
                             + oc * w_stride_cout \
                             + kh * w_stride_kh \
                             + kw * w_stride_kw
                    w_val = tl.load(w_ptrs, mask=valid_ic, other=0.0)

                    acc += x_vals * w_val

    # store results to y
    y_ptrs = y_ptr \
             + b * y_stride_b \
             + oc * y_stride_c \
             + h * y_stride_h \
             + w * y_stride_w
    tl.store(y_ptrs, acc, mask=sp_mask)


# Triton kernel: GroupNorm + affine (scale, bias) + SiLU
# Input y: (B, C, H, W), Output out: (B, C, H, W)
# weight_ptr, bias_ptr: (C,)
# Grid: (B, num_groups). Each program handles one (b, group).
@triton.jit
def group_norm_affine_silu_triton(
    y_ptr, weight_ptr, bias_ptr, out_ptr,
    B, C, H, W,
    num_groups: tl.constexpr,  # 32
    eps: tl.constexpr,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    out_stride_b, out_stride_c, out_stride_h, out_stride_w,
    BLOCK: tl.constexpr,  # reduction/block size for group elements
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

        y_ptrs = y_ptr \
                 + b * y_stride_b \
                 + ch * y_stride_c \
                 + h * y_stride_h \
                 + w * y_stride_w
        y_vals = tl.load(y_ptrs, mask=mask, other=0.0)

        sum_val += tl.sum(y_vals, axis=0)
        sum_sq += tl.sum(y_vals * y_vals, axis=0)

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

        y_ptrs = y_ptr \
                 + b * y_stride_b \
                 + ch * y_stride_c \
                 + h * y_stride_h \
                 + w * y_stride_w
        y_vals = tl.load(y_ptrs, mask=mask, other=0.0)

        norm_vals = (y_vals - mean) * inv_std

        scale = tl.load(weight_ptr + ch, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + ch, mask=mask, other=0.0)
        z = norm_vals * scale + bias

        s = 1.0 / (1.0 + tl.exp(-z))  # sigmoid
        y_act = z * s

        out_ptrs = out_ptr \
                   + b * out_stride_b \
                   + ch * out_stride_c \
                   + h * out_stride_h \
                   + w * out_stride_w
        tl.store(out_ptrs, y_act, mask=mask)


# Triton kernel: elementwise residual add out = out + x
@triton.jit
def add_residual_triton(out_ptr, y_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    a = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, a + b, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        B, C, H, W = x.shape
        device = x.device
        # Ensure contiguity
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        # First conv: y1 = conv3x3(x, conv1_weight)
        y1 = torch.empty((B, C, H, W), device=device, dtype=x.dtype)
        tiles = (H * W + 255) // 256  # number of spatial tiles
        grid = (B, C, tiles)
        conv3x3_triton[grid](
            x, conv1_weight, y1,
            B, C, H, W, C,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_OC=32, BLOCK_SP=256
        )

        # First GroupNorm + affine + SiLU
        out1 = torch.empty_like(y1)
        grid_gn = (B, 32)
        group_norm_affine_silu_triton[grid_gn](
            y1, norm1_weight, norm1_bias, out1,
            B, C, H, W,
            32, self.eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
            BLOCK=1024
        )

        # Second conv: y2 = conv3x3(out1, conv2_weight)
        y2 = torch.empty((B, C, H, W), device=device, dtype=x.dtype)
        grid = (B, C, tiles)
        conv3x3_triton[grid](
            out1, conv2_weight, y2,
            B, C, H, W, C,
            out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_OC=32, BLOCK_SP=256
        )

        # Second GroupNorm + affine + SiLU
        out2 = torch.empty_like(y2)
        group_norm_affine_silu_triton[grid_gn](
            y2, norm2_weight, norm2_bias, out2,
            B, C, H, W,
            32, self.eps,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
            BLOCK=1024
        )

        # Residual add: out = out2 + x
        out = torch.empty_like(x)
        N = B * C * H * W
        grid_add = (triton.cdiv(N, 1024),)
        add_residual_triton[grid_add](
            out, out2, x, N, BLOCK=1024
        )

        return out


def run(*args):
    return ModelNew()(*args)
