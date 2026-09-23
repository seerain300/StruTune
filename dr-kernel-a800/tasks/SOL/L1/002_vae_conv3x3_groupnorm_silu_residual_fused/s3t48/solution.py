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
    BLOCK_SP: tl.constexpr,  # e.g., 256
):
    b = tl.program_id(0)
    oc_base = tl.program_id(1)
    tile_id = tl.program_id(2)

    # Compute spatial indices for this tile
    start_sp = tile_id * BLOCK_SP
    sp = start_sp + tl.arange(0, BLOCK_SP)
    sp_mask = sp < (H * W)

    # Map sp to (h, w)
    h = sp // W
    w = sp % W

    # Initialize accumulator for [BLOCK_OC, BLOCK_SP]
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Loop over input channels in blocks of BLOCK_OC (but since we need per-ic weights, we use a smaller BLOCK_IC)
    # We'll iterate ic_base from 0 to C_in in steps of 1 to ensure all input channels are covered. Using BLOCK_IC=1 is safe.
    # For better performance, one can increase BLOCK_IC, but correctness comes first.
    for ic in range(0, C_in):
        # Output channels for this program
        oc_vec = oc_base + tl.arange(0, BLOCK_OC)
        oc_mask = oc_vec < C_out

        # Accumulate contributions from the 3x3 neighborhood
        # kh, kw in [-1, 1], map to input indices h+kh, w+kw; only if valid
        # We load w[ic, oc, kh, kw] for oc_vec and spatial positions w
        for kh in range(-1, 2):
            for kw in range(-1, 2):
                in_h = h + kh
                in_w = w + kw

                # Valid spatial positions (padding=1 so always valid for interior, but keep masks)
                valid_sp = (in_h >= 0) & (in_h < H) & (in_w >= 0) & (in_w < W) & sp_mask

                # Load x[b, ic, in_h, in_w] vector over spatial positions
                x_ptrs = x_ptr \
                         + b * x_stride_b \
                         + ic * x_stride_c \
                         + in_h * x_stride_h \
                         + in_w * x_stride_w
                x_vals = tl.load(x_ptrs, mask=valid_sp, other=0.0)  # [BLOCK_SP]

                # Load w[ic, oc, kh, kw] vector over output channels
                w_ptrs = w_ptr \
                         + ic * w_stride_cin \
                         + oc_vec * w_stride_cout \
                         + kh * w_stride_kh \
                         + kw * w_stride_kw
                w_mask = oc_mask
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_OC]

                # Accumulate: acc += w_vals[:, None] * x_vals[None, :]
                acc += w_vals[:, None] * x_vals[None, :]

    # Store results to y[b, oc_vec, h, w] for all spatial positions in this tile
    # We can't write a 2D tile directly; instead, we store for each oc in oc_vec
    for oc_off in range(0, BLOCK_OC):
        oc = oc_base + oc_off
        oc_mask_scalar = oc < C_out
        y_ptrs = y_ptr \
                 + b * y_stride_b \
                 + oc * y_stride_c \
                 + h * y_stride_h \
                 + w * y_stride_w
        # Select the oc-th row from acc and store with sp_mask
        row = acc[oc_off, :]
        tl.store(y_ptrs, row, mask=sp_mask & oc_mask_scalar)


# Triton kernel: GroupNorm + affine (scale, bias) + SiLU per (batch, group)
# x: input (B, C, H, W), weight: per-channel scale (C,), bias: per-channel bias (C,), y: output (B, C, H, W)
# Grid: (B, num_groups). Assumes num_groups=32 and C % 32 == 0.
@triton.jit
def group_norm_affine_silu(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W, num_groups,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,  # reduction tile size
    eps: tl.constexpr,
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


# Triton kernel: elementwise residual add y = y + x over flattened tensor
@triton.jit
def add_residual_kernel(out_ptr, y_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y_vals + x_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                 conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        super().__init__()
        # Store weights/biases
        self.conv1_weight = conv1_weight
        self.norm1_weight = norm1_weight
        self.norm1_bias = norm1_bias
        self.conv2_weight = conv2_weight
        self.norm2_weight = norm2_weight
        self.norm2_bias = norm2_bias
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA and dtype float32
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        x = x.contiguous().float()
        B, C, H, W = x.shape
        device = x.device

        # 1) First conv -> GroupNorm -> SiLU
        y1 = torch.empty((B, C, H, W), device=device, dtype=torch.float32)

        # Launch conv3x3 for conv1
        BLOCK_OC = 32
        BLOCK_SP = 256
        tiles = (H * W + BLOCK_SP - 1) // BLOCK_SP
        grid_conv = (B, C, tiles)
        conv3x3_triton[grid_conv](
            x, self.conv1_weight, y1,
            B, C, H, W, C,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            self.conv1_weight.stride(0), self.conv1_weight.stride(1), self.conv1_weight.stride(2), self.conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
        )

        # GroupNorm 1 + affine + SiLU
        y1_norm = torch.empty_like(y1)
        grid_gn1 = (B, 32)
        group_norm_affine_silu[grid_gn1](
            y1, self.norm1_weight, self.norm1_bias, y1_norm,
            B, C, H, W, 32,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            BLOCK=1024, eps=self.eps,
        )

        # SiLU (already applied in group_norm_affine_silu)

        # 2) Second conv -> GroupNorm -> SiLU
        y2_pre = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        # Launch conv3x3 for conv2 on y1_norm
        conv3x3_triton[grid_conv](
            y1_norm, self.conv2_weight, y2_pre,
            B, C, H, W, C,
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            self.conv2_weight.stride(0), self.conv2_weight.stride(1), self.conv2_weight.stride(2), self.conv2_weight.stride(3),
            y2_pre.stride(0), y2_pre.stride(1), y2_pre.stride(2), y2_pre.stride(3),
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
        )

        # GroupNorm 2 + affine + SiLU
        y2 = torch.empty_like(y2_pre)
        grid_gn2 = (B, 32)
        group_norm_affine_silu[grid_gn2](
            y2_pre, self.norm2_weight, self.norm2_bias, y2,
            B, C, H, W, 32,
            y2_pre.stride(0), y2_pre.stride(1), y2_pre.stride(2), y2_pre.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK=1024, eps=self.eps,
        )

        # 3) Add residual x
        out = torch.empty_like(y2)
        N = B * C * H * W
        grid_add = ((N + 1024 - 1) // 1024,)
        add_residual_kernel[grid_add](out, y2, x, N, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
