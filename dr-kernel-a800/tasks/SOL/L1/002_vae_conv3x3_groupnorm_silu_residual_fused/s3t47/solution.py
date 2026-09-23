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

    # tile over spatial dimension (linearized H*W)
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
        ic_vec = ic_base + tl.arange(0, BLOCK_OC)
        ic_mask = ic_vec < C_in

        # iterate over the 3x3 neighborhood
        for kh in range(3):
            for kw in range(3):
                h_idx = h + kh
                w_idx = w + kw
                # load input x for this ic block and spatial tile
                x_ptrs = x_ptr \
                         + b * x_stride_b \
                         + ic_vec[:, None] * x_stride_c \
                         + h_idx[None, :] * x_stride_h \
                         + w_idx[None, :] * x_stride_w
                x_mask = ic_mask[:, None] & sp_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_OC, BLOCK_SP]

                # load weights for this ic block, oc tile, and (kh, kw)
                oc_vec = oc_base + tl.arange(0, BLOCK_OC)
                w_ptrs = w_ptr \
                         + ic_vec[:, None] * w_stride_cin \
                         + oc_vec[None, :] * w_stride_cout \
                         + kh * w_stride_kh \
                         + kw * w_stride_kw
                w_mask = ic_mask[:, None] & (oc_vec[None, :] < C_out)
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_OC, BLOCK_SP]

                # accumulate
                acc += w_vals * x_vals

    # store results
    y_ptrs = y_ptr \
             + b * y_stride_b \
             + (oc_base + tl.arange(0, BLOCK_OC))[:, None] * y_stride_c \
             + h[None, :] * y_stride_h \
             + w[None, :] * y_stride_w
    oc_mask = (oc_base + tl.arange(0, BLOCK_OC)) < C_out
    store_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(y_ptrs, acc, mask=store_mask)


# Triton kernel: GroupNorm (num_groups=32) + affine (scale, bias) + SiLU
@triton.jit
def group_norm_affine_silu(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W,
    num_groups: tl.constexpr,  # 32
    eps: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,  # reduction/block size
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
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y + x, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

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
        # Ensure CUDA and float32 for Triton
        assert x.is_cuda, "Input must be on CUDA for Triton kernels"
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Conv weights must be CUDA tensors"
        assert norm1_weight.is_cuda and norm2_weight.is_cuda and norm1_bias.is_cuda and norm2_bias.is_cuda, "Norm params must be CUDA tensors"

        x = x.contiguous().to(torch.float32)
        conv1_weight = conv1_weight.contiguous().to(torch.float32)
        conv2_weight = conv2_weight.contiguous().to(torch.float32)
        norm1_weight = norm1_weight.contiguous().to(torch.float32)
        norm1_bias = norm1_bias.contiguous().to(torch.float32)
        norm2_weight = norm2_weight.contiguous().to(torch.float32)
        norm2_bias = norm2_bias.contiguous().to(torch.float32)

        B, C, H, W = x.shape
        C_in = C
        C_out = C  # output channels equal input channels (as in original code)

        # Triton meta-parameters
        BLOCK_OC = 32
        BLOCK_SP = 1024
        tiles = triton.cdiv(H * W, BLOCK_SP)

        # First path: Conv3x3 -> GroupNorm -> SiLU
        y1 = torch.empty((B, C_out, H, W), device=x.device, dtype=torch.float32)
        grid1 = (B, C_out, tiles)
        conv3x3_triton[grid1](
            x, conv1_weight, y1,
            B, C_in, H, W, C_out,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
        )
        # GroupNorm + SiLU
        y1 = y1.contiguous()
        grid_gn = (B, 32)
        y1_out = torch.empty_like(y1)
        group_norm_affine_silu[grid_gn](
            y1, norm1_weight, norm1_bias, y1_out,
            B, C_out, H, W,
            num_groups=32, eps=eps,
            x_stride_b=y1.stride(0), x_stride_c=y1.stride(1), x_stride_h=y1.stride(2), x_stride_w=y1.stride(3),
            y_stride_b=y1_out.stride(0), y_stride_c=y1_out.stride(1), y_stride_h=y1_out.stride(2), y_stride_w=y1_out.stride(3),
            BLOCK=1024,
        )

        # Second path: Conv3x3 -> GroupNorm -> SiLU
        y2 = torch.empty((B, C_out, H, W), device=x.device, dtype=torch.float32)
        grid2 = (B, C_out, tiles)
        conv3x3_triton[grid2](
            y1_out, conv2_weight, y2,
            B, C_in, H, W, C_out,
            y1_out.stride(0), y1_out.stride(1), y1_out.stride(2), y1_out.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
        )
        # GroupNorm + SiLU
        y2 = y2.contiguous()
        y2_out = torch.empty_like(y2)
        group_norm_affine_silu[grid_gn](
            y2, norm2_weight, norm2_bias, y2_out,
            B, C_out, H, W,
            num_groups=32, eps=eps,
            x_stride_b=y2.stride(0), x_stride_c=y2.stride(1), x_stride_h=y2.stride(2), x_stride_w=y2.stride(3),
            y_stride_b=y2_out.stride(0), y_stride_c=y2_out.stride(1), y_stride_h=y2_out.stride(2), y_stride_w=y2_out.stride(3),
            BLOCK=1024,
        )

        # Residual add: out = y2_out + x
        N = B * C * H * W
        out = torch.empty_like(y2_out)
        add_residual_kernel[(triton.cdiv(N, 1024),)](
            out, y2_out, x,
            N, BLOCK=1024,
        )

        return out


def run(*args):
    return ModelNew()(*args)
