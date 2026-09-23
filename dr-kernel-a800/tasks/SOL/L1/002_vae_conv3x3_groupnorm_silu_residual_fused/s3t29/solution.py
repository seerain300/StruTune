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

    # spatial tile
    start_sp = tile_id * BLOCK_SP
    sp = start_sp + tl.arange(0, BLOCK_SP)
    sp_mask = sp < (H * W)

    # map sp to (h, w)
    h = sp // W
    w = sp % W

    # accumulator for this tile
    acc = tl.zeros((BLOCK_SP,), dtype=tl.float32)

    # loop over input channels
    for ic in range(0, C_in):
        # 3x3 neighborhood with padding=1 (kernel indices 0..2 map to -1..1 offsets)
        for kh in range(-1, 2):
            for kw in range(-1, 2):
                in_h = h + kh
                in_w = w + kw
                in_bounds = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W) & sp_mask

                # Load input x[b, ic, in_h, in_w]
                x_ptrs = x_ptr \
                         + b * x_stride_b \
                         + ic * x_stride_c \
                         + in_h * x_stride_h \
                         + in_w * x_stride_w
                x_vals = tl.load(x_ptrs, mask=in_bounds, other=0.0)

                # Load weight w[ic, oc, kh+1, kw+1]
                w_ptrs = w_ptr \
                         + ic * w_stride_cin \
                         + oc * w_stride_cout \
                         + (kh + 1) * w_stride_kh \
                         + (kw + 1) * w_stride_kw
                w_val = tl.load(w_ptrs)  # scalar per oc

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
    BLOCK: tl.constexpr,  # reduction tile, e.g., 1024
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

        c = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr \
                 + b * x_stride_b \
                 + c * x_stride_c \
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

        c = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr \
                 + b * x_stride_b \
                 + c * x_stride_c \
                 + h * x_stride_h \
                 + w * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        norm_vals = (x_vals - mean) * inv_std

        scale = tl.load(weight_ptr + c, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
        z = norm_vals * scale + bias

        # SiLU: z * sigmoid(z)
        s = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * s

        y_ptrs = y_ptr \
                 + b * y_stride_b \
                 + c * y_stride_c \
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
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        All computation is performed by Triton kernels.
        """
        # Ensure CUDA and contiguous
        device = x.device
        assert device.type == 'cuda', "Inputs must be on CUDA for Triton kernels"
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        B, C, H, W = x.shape

        # 1) First conv
        y1 = torch.empty((B, C, H, W), device=device, dtype=x.dtype)
        grid1 = (B, C, (H * W + 1023) // 1024)
        conv3x3_triton[grid1](
            x, conv1_weight, y1,
            B, C, H, W, C,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_SP=1024,
            num_warps=4, num_stages=2
        )

        # 2) GroupNorm + affine (norm1_weight, norm1_bias) + SiLU on y1
        y1_norm = torch.empty_like(y1)
        grid_gn1 = (B, 32)
        group_norm_affine_silu[grid_gn1](
            y1, y1_norm, norm1_weight, norm1_bias,
            B, C, H, W,
            32, self.eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            BLOCK=1024,
            num_warps=4, num_stages=2
        )

        # 3) Second conv on normalized output
        y2 = torch.empty((B, C, H, W), device=device, dtype=x.dtype)
        grid2 = (B, C, (H * W + 1023) // 1024)
        conv3x3_triton[grid2](
            y1_norm, conv2_weight, y2,
            B, C, H, W, C,
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_SP=1024,
            num_warps=4, num_stages=2
        )

        # 4) GroupNorm + affine (norm2_weight, norm2_bias) + SiLU on y2
        y2_norm = torch.empty_like(y2)
        grid_gn2 = (B, 32)
        group_norm_affine_silu[grid_gn2](
            y2, y2_norm, norm2_weight, norm2_bias,
            B, C, H, W,
            32, self.eps,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
            BLOCK=1024,
            num_warps=4, num_stages=2
        )

        # 5) Residual add: out = y2_norm + x
        out = torch.empty((B, C, H, W), device=device, dtype=x.dtype)
        N = B * C * H * W
        grid_add = ( (N + 1023) // 1024, )
        add_residual_kernel[grid_add](
            out, y2_norm, x, N, BLOCK=1024,
            num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
