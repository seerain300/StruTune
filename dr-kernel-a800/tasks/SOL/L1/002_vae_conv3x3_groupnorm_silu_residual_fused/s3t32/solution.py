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
    BLOCK_SP: tl.constexpr,  # e.g., 256
    BLOCK_IC: tl.constexpr,  # e.g., 32
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
    for ic_base in range(0, C_in, BLOCK_IC):
        # iterate over the BLOCK_IC input channels in this block
        for k in range(0, BLOCK_IC):
            ic = ic_base + k
            valid_ic = ic < C_in  # scalar boolean

            # 3x3 neighborhood with padding=1
            for kh in range(3):
                for kw in range(3):
                    ih = h + (kh - 1)
                    iw = w + (kw - 1)

                    # guard for valid x indices due to padding
                    in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & sp_mask & valid_ic

                    x_ptrs = x_ptr \
                             + b * x_stride_b \
                             + ic * x_stride_c \
                             + ih * x_stride_h \
                             + iw * x_stride_w

                    # load x with mask; use 0 for out-of-bounds
                    x_val = tl.load(x_ptrs, mask=in_bounds, other=0.0)

                    # load corresponding weight w[ic, oc, kh, kw] as scalar
                    w_ptrs = w_ptr \
                             + ic * w_stride_cin \
                             + oc * w_stride_cout \
                             + kh * w_stride_kh \
                             + kw * w_stride_kw
                    w_val = tl.load(w_ptrs)  # scalar

                    acc += x_val * w_val

    # store results
    y_ptrs = y_ptr \
             + b * y_stride_b \
             + oc * y_stride_c \
             + h * y_stride_h \
             + w * y_stride_w
    tl.store(y_ptrs, acc, mask=sp_mask)


# Triton kernel: GroupNorm forward + affine (scale, bias) + SiLU
# Assumes num_groups=32 and C % 32 == 0.
# x: (B, C, H, W), weight: (C,), bias: (C,), y: (B, C, H, W)
@triton.jit
def group_norm_affine_silu_triton(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W, num_groups, eps,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,  # reduction chunk
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

        # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
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
def add_residual_triton(out_ptr, y_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out_vals = x_vals + y_vals
    tl.store(out_ptr + offsets, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        conv1_weight: torch.Tensor,
        norm1_weight: torch.Tensor,
        norm1_bias: torch.Tensor,
        conv2_weight: torch.Tensor,
        norm2_weight: torch.Tensor,
        norm2_bias: torch.Tensor,
        eps: float,
    ):
        # Ensure CUDA tensors and contiguity
        assert x.is_cuda, "Input tensor must be on CUDA device"
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Weight tensors must be on CUDA device"
        assert norm1_weight.is_cuda and norm2_weight.is_cuda and norm1_bias.is_cuda and norm2_bias.is_cuda, "Norm params must be on CUDA device"
        B, C, H, W = x.shape
        C1_in = conv1_weight.shape[0]
        C1_out = conv1_weight.shape[1]
        C2_in = conv2_weight.shape[0]
        C2_out = conv2_weight.shape[1]
        assert C1_in == C and C1_out == C, "conv1_weight must be (C, C, 3, 3)"
        assert C2_in == C1_out and C2_out == C, "conv2_weight must be (C, C, 3, 3)"
        assert C % 32 == 0, "C must be divisible by num_groups=32"

        # 1) First conv3x3 (no bias), stride=1, padding=1
        y1 = torch.empty((B, C1_out, H, W), device=x.device, dtype=x.dtype)
        tiles_sp = triton.cdiv(H * W, 256)
        grid1 = (B, C1_out, tiles_sp)
        conv3x3_triton[grid1](
            x, conv1_weight, y1,
            B, C, H, W, C1_out,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_SP=256, BLOCK_IC=32,
        )

        # 2) GroupNorm (num_groups=32) + affine + SiLU
        y1n = torch.empty((B, C1_out, H, W), device=x.device, dtype=x.dtype)
        grid_gn1 = (B, 32)
        group_norm_affine_silu_triton[grid_gn1](
            y1, norm1_weight, norm1_bias, y1n,
            B, C1_out, H, W, 32, eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1n.stride(0), y1n.stride(1), y1n.stride(2), y1n.stride(3),
            BLOCK=1024,
        )

        # 3) Second conv3x3 (no bias), stride=1, padding=1
        y2 = torch.empty((B, C2_out, H, W), device=x.device, dtype=x.dtype)
        tiles_sp2 = triton.cdiv(H * W, 256)
        grid2 = (B, C2_out, tiles_sp2)
        conv3x3_triton[grid2](
            y1n, conv2_weight, y2,
            B, C1_out, H, W, C2_out,
            y1n.stride(0), y1n.stride(1), y1n.stride(2), y1n.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_SP=256, BLOCK_IC=32,
        )

        # 4) GroupNorm (num_groups=32) + affine + SiLU
        y2n = torch.empty((B, C2_out, H, W), device=x.device, dtype=x.dtype)
        grid_gn2 = (B, 32)
        group_norm_affine_silu_triton[grid_gn2](
            y2, norm2_weight, norm2_bias, y2n,
            B, C2_out, H, W, 32, eps,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2n.stride(0), y2n.stride(1), y2n.stride(2), y2n.stride(3),
            BLOCK=1024,
        )

        # 5) Add residual x
        out = torch.empty((B, C2_out, H, W), device=x.device, dtype=x.dtype)
        N_out = out.numel()
        grid_add = (triton.cdiv(N_out, 1024),)
        add_residual_triton[grid_add](out, y2n, x, N_out, 1024)

        return out


def run(*args):
    return ModelNew()(*args)
