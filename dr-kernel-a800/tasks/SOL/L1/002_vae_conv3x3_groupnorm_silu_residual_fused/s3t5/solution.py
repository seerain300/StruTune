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

    # loop over input channels in blocks
    ic_base = 0
    while ic_base < C_in:
        ic_vec = ic_base + tl.arange(0, BLOCK_OC)
        ic_mask = ic_vec < C_in

        # for each kh, kw in 3x3 neighborhood
        for kh in range(3):
            for kw in range(3):
                hin = h + kh - 1
                win = w + kw - 1

                # compute x pointers and load with mask
                x_ptrs = x_ptr \
                         + b * x_stride_b \
                         + ic_vec[:, None] * x_stride_c \
                         + hin[None, :] * x_stride_h \
                         + win[None, :] * x_stride_w

                x_mask = ic_mask[:, None] & sp_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_OC, BLOCK_SP]

                # load weight for this (ic block, oc, kh, kw)
                w_ptrs = w_ptr \
                         + ic_vec * w_stride_cin \
                         + oc_base * w_stride_cout \
                         + kh * w_stride_kh \
                         + kw * w_stride_kw

                oc_mask = oc_base < C_out  # scalar since oc_base is single
                w_vals = tl.load(w_ptrs, mask=ic_mask, other=0.0)  # [BLOCK_OC]
                # reduce over ic dimension
                acc += tl.sum(x_vals * w_vals[:, None], axis=0)

        ic_base += BLOCK_OC

    # store results for this (b, oc_base, spatial tile)
    y_ptrs = y_ptr \
             + b * y_stride_b \
             + oc_base * y_stride_c \
             + h * y_stride_h \
             + w * y_stride_w

    oc_ok = oc_base < C_out
    store_mask = oc_ok & sp_mask
    tl.store(y_ptrs, acc, mask=store_mask)


# Triton kernel: GroupNorm + affine (scale, bias) + SiLU per (batch, group)
@triton.jit
def group_norm_affine_silu(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W,
    num_groups,  # 32
    eps,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,  # reduction tile size
):
    b = tl.program_id(0)
    g = tl.program_id(1)

    group_channels = C // num_groups
    group_elements = group_channels * H * W

    # First pass: compute sum and sum of squares
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

        # SiLU: z * sigmoid(z)
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
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    r = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y + r, mask=mask)


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
        # Ensure CUDA
        assert x.is_cuda, "Input tensor must be on CUDA for Triton kernels."
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Conv weights must be on CUDA."
        assert norm1_weight.is_cuda and norm1_bias.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "Norm params must be on CUDA."

        B, C, H, W = x.shape
        C1, C_in1, KH1, KW1 = conv1_weight.shape
        C2, C_in2, KH2, KW2 = conv2_weight.shape
        assert C_in1 == C and C_in2 == C, "conv1_weight: (C_in, C, 3, 3) must have C_in=C, and conv2_weight: (C_in, C, 3, 3) must have C_in=C"

        # Output tensors for convs
        y1 = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)
        y2 = torch.empty((B, C, H, W), device=x.device, dtype=x.dtype)

        # Launch conv1: (B, C, H, W)
        tiles_sp = (H * W + 1023) // 1024  # BLOCK_SP = 1024
        grid1 = (B, C, tiles_sp)
        conv3x3_triton[grid1](
            x, conv1_weight, y1,
            B, C, H, W, C,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_OC=32, BLOCK_SP=1024,
            num_warps=4,
        )

        # GroupNorm + affine + SiLU for conv1 output
        y1_out = torch.empty_like(y1)
        grid_gn1 = (B, self.num_groups)
        group_norm_affine_silu[grid_gn1](
            y1, norm1_weight, norm1_bias, y1_out,
            B, C, H, W,
            self.num_groups, self.eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_out.stride(0), y1_out.stride(1), y1_out.stride(2), y1_out.stride(3),
            BLOCK=1024,
            num_warps=4,
        )

        # Launch conv2: (B, C, H, W) conv over y1_out
        tiles_sp2 = (H * W + 1023) // 1024
        grid2 = (B, C, tiles_sp2)
        conv3x3_triton[grid2](
            y1_out, conv2_weight, y2,
            B, C, H, W, C,
            y1_out.stride(0), y1_out.stride(1), y1_out.stride(2), y1_out.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_OC=32, BLOCK_SP=1024,
            num_warps=4,
        )

        # GroupNorm + affine + SiLU for conv2 output
        y2_out = torch.empty_like(y2)
        grid_gn2 = (B, self.num_groups)
        group_norm_affine_silu[grid_gn2](
            y2, norm2_weight, norm2_bias, y2_out,
            B, C, H, W,
            self.num_groups, self.eps,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_out.stride(0), y2_out.stride(1), y2_out.stride(2), y2_out.stride(3),
            BLOCK=1024,
            num_warps=4,
        )

        # Final residual add: out = y2_out + x
        out = torch.empty_like(y2_out)
        N = out.numel()
        BLOCK = 1024
        grid_add = ((N + BLOCK - 1) // BLOCK,)
        add_residual_kernel[grid_add](
            out, y2_out, x,
            N,
            BLOCK,
            num_warps=4,
        )

        return out


def run(*args):
    return ModelNew()(*args)
