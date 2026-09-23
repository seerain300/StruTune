import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 conv forward (no bias), stride=1, padding=1
# x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), y: (B, C_out, H, W)
# Grid: (B, C_out, H_out, tiles over W_out). Each program handles one (b, oc, h_out) and a vector of w_out.
@triton.jit
def conv3x3_triton(
    x_ptr, w_ptr, y_ptr,
    B, C_in, H, W, C_out, H_out, W_out,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK_OC: tl.constexpr,  # e.g., 32
    BLOCK_W: tl.constexpr,   # e.g., 128
):
    b = tl.program_id(0)
    oc_base = tl.program_id(1)
    h_out = tl.program_id(2)
    tile_w_id = tl.program_id(3)

    # Vector of width positions for this tile
    w_vec = tile_w_id * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_vec < W_out

    # Initialize accumulator for output channels block
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Loop over input channels in blocks
    for ic_base in range(0, C_in, BLOCK_OC):
        oc_vec = oc_base + tl.arange(0, BLOCK_OC)
        oc_mask = oc_vec < C_out

        # Accumulate over 3x3 neighborhood
        for kh in range(-1, 2):
            for kw in range(-1, 2):
                # Compute input h, w for this output row and width vector
                h_in = h_out + kh
                w_in_vec = w_vec + kw

                # Loop over input channels for this block
                for ic in range(0, BLOCK_OC):
                    ic_full = ic_base + ic
                    ic_valid = ic_full < C_in

                    # Compute weights for this (ic_full, oc_vec, kh, kw)
                    w_ptrs = w_ptr \
                             + ic_full * w_stride_cin \
                             + oc_vec * w_stride_cout \
                             + kh * w_stride_kh \
                             + kw * w_stride_kw
                    # Load weights for this kh, kw across oc block
                    w_vals = tl.load(w_ptrs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    # Compute input pointers for this (b, ic_full, h_in, w_in_vec)
                    x_ptrs = x_ptr \
                             + b * x_stride_b \
                             + ic_full * x_stride_c \
                             + h_in * x_stride_h \
                             + w_in_vec * x_stride_w
                    x_vals = tl.load(x_ptrs, mask=w_mask, other=0.0)  # [BLOCK_W]

                    # Accumulate: acc += w_vals[:, None] * x_vals[None, :]
                    acc += w_vals * x_vals

        # Store accumulated results for this block of output channels
        y_ptrs = y_ptr \
                 + b * y_stride_b \
                 + oc_vec * y_stride_c \
                 + h_out * y_stride_h \
                 + w_vec * y_stride_w
        tl.store(y_ptrs, acc, mask=oc_mask & w_mask)


# Triton kernel: GroupNorm (num_groups=32) + affine (scale, bias) + SiLU
# x: (B, C, H, W), weight: (C,), bias: (C,), y: (B, C, H, W)
# Grid: (B, num_groups). Each program handles one (batch, group).
@triton.jit
def group_norm_affine_silu(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W, num_groups: tl.constexpr,  # 32
    eps: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,  # reduction tile
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
        c_vec = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W
        x_ptrs = x_ptr + b * x_stride_b + c_vec * x_stride_c + h * x_stride_h + w * x_stride_w
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
        c_vec = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr + b * x_stride_b + c_vec * x_stride_c + h * x_stride_h + w * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        norm_vals = (x_vals - mean) * inv_std

        scale = tl.load(weight_ptr + c_vec, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + c_vec, mask=mask, other=0.0)
        z = norm_vals * scale + bias

        s = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * s

        y_ptrs = y_ptr + b * y_stride_b + c_vec * y_stride_c + h * y_stride_h + w * y_stride_w
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


def _run_group_norm_affine_silu(x, weight, bias, num_groups: int = 32, eps: float = 1e-5):
    # x: (B, C, H, W) CUDA contiguous
    B, C, H, W = x.shape
    y = torch.empty_like(x)
    grid = (B, num_groups)
    group_norm_affine_silu[grid](
        x, weight, bias, y,
        B, C, H, W, num_groups, eps,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK=1024,
        num_warps=4, num_stages=2,
    )
    return y


def _conv3x3_triton(x, w):
    # x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), no bias, stride=1, padding=1
    B, C_in, H, W = x.shape
    C_out = w.shape[1]
    y = torch.empty((B, C_out, H, W), device=x.device, dtype=x.dtype)
    # Choose blocks
    BLOCK_OC = 32
    # Tiles over width: choose 128; if W < 128, last tile masked
    BLOCK_W = 128
    tiles_w = (W + BLOCK_W - 1) // BLOCK_W
    H_out = H
    W_out = W
    grid = (B, C_out, H_out, tiles_w)
    conv3x3_triton[grid](
        x, w, y,
        B, C_in, H, W, C_out, H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        BLOCK_OC=BLOCK_OC, BLOCK_W=BLOCK_W,
        num_warps=4, num_stages=2,
    )
    return y


class ModelNew(torch.nn.Module):
    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps: float):
        # Ensure tensors are on CUDA and contiguous
        assert x.is_cuda, "Input must be on CUDA for Triton kernels"
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        # First path: Conv3x3 -> GroupNorm -> SiLU
        out = _conv3x3_triton(x, conv1_weight)
        out = _run_group_norm_affine_silu(out, norm1_weight, norm1_bias, num_groups=32, eps=eps)
        out = F.silu(out)  # SiLU is fine here

        # Second path: Conv3x3 -> GroupNorm -> SiLU
        out = _conv3x3_triton(out, conv2_weight)
        out = _run_group_norm_affine_silu(out, norm2_weight, norm2_bias, num_groups=32, eps=eps)
        out = F.silu(out)

        # Residual connection
        out = out + x

        return out


def run(*args):
    return ModelNew()(*args)
