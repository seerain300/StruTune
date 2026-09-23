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

    # tile over spatial dimension
    start_sp = tile_id * BLOCK_SP
    sp = start_sp + tl.arange(0, BLOCK_SP)
    sp_mask = sp < (H * W)

    # map sp to (h, w)
    h = sp // W
    w = sp % W

    # initialize accumulator for [BLOCK_SP]
    acc = tl.zeros((BLOCK_SP,), dtype=tl.float32)

    # loop over input channels
    for ic in range(0, C_in):
        # base x pointer for this (b, ic)
        x_base = x_ptr + b * x_stride_b + ic * x_stride_c

        # iterate over 3x3 neighborhood with padding=1
        for kh in range(3):
            h_in = h - 1 + kh  # -1, 0, 1 with padding
            h_in_mask = (h_in >= 0) & (h_in < H)
            for kw in range(3):
                w_in = w - 1 + kw
                w_in_mask = (w_in >= 0) & (w_in < W)

                # combined spatial mask
                spatial_mask = sp_mask & h_in_mask & w_in_mask

                # load x for this (ic, h_in, w_in) per lane
                x_ptrs = x_base + h_in * x_stride_h + w_in * x_stride_w
                x_vals = tl.load(x_ptrs, mask=spatial_mask, other=0.0)

                # load weight scalar w[ic, oc, kh, kw]
                w_val = tl.load(
                    w_ptr + ic * w_stride_cin + oc * w_stride_cout + kh * w_stride_kh + kw * w_stride_kw
                )

                # accumulate
                acc += x_vals * w_val

    # store results
    y_ptrs = y_ptr + b * y_stride_b + oc * y_stride_c + h * y_stride_h + w * y_stride_w
    tl.store(y_ptrs, acc, mask=sp_mask)


# Triton kernel: GroupNorm (num_groups fixed) + affine (scale, bias) + SiLU
# x: (B, C, H, W), weight: (C,), bias: (C,), y: (B, C, H, W)
# Grid: (B, num_groups)
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

        x_ptrs = x_ptr + b * x_stride_b + ch * x_stride_c + h * x_stride_h + w * x_stride_w
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

        x_ptrs = x_ptr + b * x_stride_b + ch * x_stride_c + h * x_stride_h + w * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        norm_vals = (x_vals - mean) * inv_std

        scale = tl.load(weight_ptr + ch, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + ch, mask=mask, other=0.0)
        z = norm_vals * scale + bias

        # SiLU: z * sigmoid(z)
        s = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * s

        y_ptrs = y_ptr + b * y_stride_b + ch * y_stride_c + h * y_stride_h + w * y_stride_w
        tl.store(y_ptrs, y_vals, mask=mask)


# Triton kernel: elementwise residual add out = out + x
@triton.jit
def add_residual_kernel(out_ptr, y_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y_vals + x_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups=32, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        # Ensure tensors are on CUDA and same dtype for simplicity
        assert x.is_cuda, "Input tensor must be on CUDA for Triton kernels."
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Conv weights must be on CUDA."
        assert norm1_weight.is_cuda and norm1_bias.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "Norm params must be on CUDA."

        B, C, H, W = x.shape

        # 1) conv1: y1 = conv3x3(x)
        y1 = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)
        BLOCK_SP = 1024
        tiles = triton.cdiv(H * W, BLOCK_SP)
        grid1 = (B, C, tiles)
        conv3x3_triton[grid1](
            x, conv1_weight, y1,
            B, C, H, W, C,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_SP=BLOCK_SP, num_warps=4, num_stages=2
        )

        # 2) GroupNorm + affine + SiLU for y1
        y1_norm = torch.empty_like(y1)
        group_norm_affine_silu[(B, self.num_groups)](
            y1, norm1_weight, norm1_bias, y1_norm,
            B, C, H, W, self.num_groups, self.eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            BLOCK=1024, num_warps=4, num_stages=2
        )

        # 3) conv2: y2 = conv3x3(y1_norm)
        y2 = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)
        conv3x3_triton[(B, C, tiles)](
            y1_norm, conv2_weight, y2,
            B, C, H, W, C,
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_SP=BLOCK_SP, num_warps=4, num_stages=2
        )

        # 4) GroupNorm + affine + SiLU for y2
        y2_norm = torch.empty_like(y2)
        group_norm_affine_silu[(B, self.num_groups)](
            y2, norm2_weight, norm2_bias, y2_norm,
            B, C, H, W, self.num_groups, self.eps,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
            BLOCK=1024, num_warps=4, num_stages=2
        )

        # 5) Final residual: out = y2_norm + x
        out = torch.empty_like(x)
        N = x.numel()
        add_residual_kernel[(triton.cdiv(N, 1024),)](
            out, y2_norm, x, N, BLOCK=1024, num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
