import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 conv forward (no bias), stride=1, padding=1
# x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), y: (B, C_out, H, W)
# Grid: (B, C_out, tiles over H*W), tiles = ceil(H*W / BLOCK_SP).
@triton.jit
def conv3x3_triton_kernel(
    x_ptr, w_ptr, y_ptr,
    B, C_in, H, W, C_out,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK_OC: tl.constexpr,  # e.g., 32
    BLOCK_SP: tl.constexpr,  # e.g., 256
):
    b = tl.program_id(0)
    oc_base = tl.program_id(1)
    tile_id = tl.program_id(2)

    # Tile over spatial dimension
    start_sp = tile_id * BLOCK_SP
    sp = start_sp + tl.arange(0, BLOCK_SP)
    sp_mask = sp < (H * W)

    # Map sp to (h, w)
    h = sp // W
    w = sp % W

    # Initialize accumulator [BLOCK_OC, BLOCK_SP]
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Loop over input channels in blocks
    ic_base = 0
    while ic_base < C_in:
        ic_vec = ic_base + tl.arange(0, BLOCK_OC)
        ic_mask = ic_vec < C_in

        # Accumulate over 3x3 neighborhood
        # kh, kw in -1..1; with padding=1, input indices are valid
        for kh in range(-1, 2):
            for kw in range(-1, 2):
                # Load weights for all output channels in this block
                w_ptrs = w_ptr \
                         + ic_vec * w_stride_cin \
                         + oc_base * w_stride_cout \
                         + kh * w_stride_kh \
                         + kw * w_stride_kw
                # If oc_base >= C_out, mask prevents load; here oc_base < C_out guaranteed by grid
                w_vals = tl.load(w_ptrs, mask=ic_mask, other=0.0)  # [BLOCK_OC]

                # Load input x for all channels in this block and spatial positions
                x_ptrs = x_ptr \
                         + b * x_stride_b \
                         + ic_vec[:, None] * x_stride_c \
                         + (h[None, :] + kh) * x_stride_h \
                         + (w[None, :] + kw) * x_stride_w
                x_mask = ic_mask[:, None] & sp_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_IC, BLOCK_SP]

                # Accumulate: [BLOCK_OC] * [BLOCK_IC, BLOCK_SP] -> broadcast along SP
                # Convert w_vals to [BLOCK_OC, 1] and multiply; sum over IC axis
                # Note: we need to build [BLOCK_OC, BLOCK_SP] contribution: sum over ic_vec
                # Strategy: for each ic in ic_vec, add w_vals[ic] * x_vals[ic, :]
                # We can do it in a loop over BLOCK_IC.
                for i in range(BLOCK_OC):
                    ic_i = ic_vec[i]
                    w_i = w_vals[i]
                    x_row = x_vals[i, :]  # [BLOCK_SP]
                    acc[i, :] += w_i * x_row

        ic_base += BLOCK_OC

    # Store results for this tile
    y_ptrs = y_ptr \
             + b * y_stride_b \
             + (oc_base + tl.arange(0, BLOCK_OC))[:, None] * y_stride_c \
             + h[None, :] * y_stride_h \
             + w[None, :] * y_stride_w
    oc_mask = (oc_base + tl.arange(0, BLOCK_OC)) < C_out
    store_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(y_ptrs, acc, mask=store_mask)


# Triton kernel: GroupNorm + affine (scale, bias) + SiLU per (batch, group)
# x: (B, C, H, W), y: (B, C, H, W), weight: (C,), bias: (C,)
# Grid: (B, num_groups). Each program handles one (b, g).
@triton.jit
def group_norm_affine_silu_kernel(
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

        c_vec = g * group_channels + (idx // (H * W))
        sp_vec = idx % (H * W)
        h_vec = sp_vec // W
        w_vec = sp_vec % W

        x_ptrs = x_ptr \
                 + b * x_stride_b \
                 + c_vec * x_stride_c \
                 + h_vec * x_stride_h \
                 + w_vec * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        sum_val += tl.sum(x_vals, axis=0)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    mean = sum_val / group_elements
    var = sum_sq / group_elements - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize, affine, SiLU, store
    for start in range(0, group_elements, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < group_elements

        c_vec = g * group_channels + (idx // (H * W))
        sp_vec = idx % (H * W)
        h_vec = sp_vec // W
        w_vec = sp_vec % W

        x_ptrs = x_ptr \
                 + b * x_stride_b \
                 + c_vec * x_stride_c \
                 + h_vec * x_stride_h \
                 + w_vec * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        norm_vals = (x_vals - mean) * inv_std

        scale = tl.load(weight_ptr + c_vec, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + c_vec, mask=mask, other=0.0)

        z = norm_vals * scale + bias
        s = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * s

        y_ptrs = y_ptr \
                 + b * y_stride_b \
                 + c_vec * y_stride_c \
                 + h_vec * y_stride_h \
                 + w_vec * y_stride_w
        tl.store(y_ptrs, y_vals, mask=mask)


# Triton kernel: elementwise residual add y = y + x
@triton.jit
def add_residual_kernel(out_ptr, y_ptr, x_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y + x, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
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
    ):
        """
        Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        x: (B, C, H, W), conv weights: (C, C, 3, 3), norm params: (C,)
        """
        assert x.is_cuda and conv1_weight.is_cuda and conv2_weight.is_cuda, "Tensors must be CUDA for Triton kernels."
        B, C, H, W = x.shape
        # Ensure contiguity
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        # Output buffers
        y1 = torch.empty_like(x)  # conv1 output
        y2 = torch.empty_like(x)  # conv2 output
        y1_grouped = torch.empty_like(x)  # GroupNorm(conv1) + affine + SiLU
        y2_grouped = torch.empty_like(x)  # GroupNorm(conv2) + affine + SiLU

        # Launch conv1: grid over (B, C, tiles over H*W)
        tiles1 = (H * W + 255) // 256  # BLOCK_SP = 256
        grid_conv1 = (B, C, tiles1)
        conv3x3_triton_kernel[grid_conv1](
            x, conv1_weight, y1,
            B, C, H, W, C,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_OC=32, BLOCK_SP=256,
        )

        # GroupNorm + affine + SiLU for conv1
        grid_gn1 = (B, self.num_groups)
        group_norm_affine_silu_kernel[grid_gn1](
            y1, norm1_weight, norm1_bias, y1_grouped,
            B, C, H, W, self.num_groups, self.eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_grouped.stride(0), y1_grouped.stride(1), y1_grouped.stride(2), y1_grouped.stride(3),
            BLOCK=1024,
        )

        # conv2 on y1_grouped
        tiles2 = (H * W + 255) // 256
        grid_conv2 = (B, C, tiles2)
        conv3x3_triton_kernel[grid_conv2](
            y1_grouped, conv2_weight, y2,
            B, C, H, W, C,
            y1_grouped.stride(0), y1_grouped.stride(1), y1_grouped.stride(2), y1_grouped.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_OC=32, BLOCK_SP=256,
        )

        # GroupNorm + affine + SiLU for conv2
        grid_gn2 = (B, self.num_groups)
        group_norm_affine_silu_kernel[grid_gn2](
            y2, norm2_weight, norm2_bias, y2_grouped,
            B, C, H, W, self.num_groups, self.eps,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_grouped.stride(0), y2_grouped.stride(1), y2_grouped.stride(2), y2_grouped.stride(3),
            BLOCK=1024,
        )

        # Residual add: y2_grouped + x
        N = B * C * H * W
        out = torch.empty_like(x)
        grid_add = (triton.cdiv(N, 1024),)
        add_residual_kernel[grid_add](
            out, y2_grouped, x, N, 1024
        )

        return out


def run(*args):
    return ModelNew()(*args)
