import math
import torch
import triton
import triton.language as tl

# Conv3x3 NCHW, stride=1, padding=1, no bias. Computes one output element per program.
@triton.jit
def conv3x3_nchw_fp32(
    x_ptr, w_ptr, y_ptr,
    B, Cin, Cout, H, W,
    x_stride_n, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
    y_stride_n, y_stride_c, y_stride_h, y_stride_w,
    BLOCK_IN: tl.constexpr, BLOCK_HW: tl.constexpr
):
    # Grid: (B, Cout, tiles over H_out*W_out)
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    tile_id = tl.program_id(2)

    H_out = H + 2  # padding=1
    W_out = W + 2

    HW_out = H_out * W_out
    hw_start = tile_id * BLOCK_HW
    offs_hw = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW_out

    h_out_vec = offs_hw // W_out
    w_out_vec = offs_hw % W_out

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Loop over input channels in chunks
    for cin_base in range(0, Cin, BLOCK_IN):
        cin_offsets = cin_base + tl.arange(0, BLOCK_IN)
        mask_cin = cin_offsets < Cin

        # For each input channel chunk, loop over 3x3 kernel
        for kh in range(3):
            h_in = h_out_vec + 1 - kh  # padding=1
            # Valid mask for rows
            mask_h = (h_in >= 0) & (h_in < H)
            for kw in range(3):
                w_in = w_out_vec + 1 - kw
                # Valid mask for cols
                mask_w = (w_in >= 0) & (w_in < W)

                # Combine masks
                valid = mask_hw & mask_h & mask_w & mask_cin

                # Load x: x[n, cin_offsets, h_in, w_in]
                x_ptrs = x_ptr + n * x_stride_n \
                           + cin_offsets[:, None] * x_stride_c \
                           + h_in[None, :] * x_stride_h \
                           + w_in[None, :] * x_stride_w
                # other=0.0 for masked lanes
                x_vals = tl.load(x_ptrs, mask=valid, other=0.0)

                # Load w: w[cin_offsets, c_out, kh, kw]
                w_ptrs = w_ptr + cin_offsets * w_stride_cin \
                           + c_out * w_stride_cout \
                           + kh * w_stride_kh \
                           + kw * w_stride_kw
                w_vals = tl.load(w_ptrs, mask=mask_cin, other=0.0)  # [BLOCK_IN]

                # Accumulate outer product: [BLOCK_IN, BLOCK_HW] * [BLOCK_HW] -> [BLOCK_IN, BLOCK_HW]
                # We want [BLOCK_HW] = sum over input channels of w_vals[:, None] * x_vals
                # So first multiply per-channel contribution into acc_hw per valid kw,kh.
                # Compute contribution for each cin in chunk:
                for i in range(BLOCK_IN):
                    if mask_cin[i]:
                        contrib = w_vals[i] * x_vals[i, :]
                        acc += contrib

    # Store acc to y[n, c_out, h_out_vec, w_out_vec]
    y_ptrs = y_ptr + n * y_stride_n + c_out * y_stride_c \
               + h_out_vec * y_stride_h + w_out_vec * y_stride_w
    tl.store(y_ptrs, acc, mask=mask_hw)


# GroupNorm with affine, per (n, group). Assumes input is (B, C, H_out, W_out) contiguous.
@triton.jit
def groupnorm_affine_kernel(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H_out, W_out, num_groups, eps,
    BLOCK_HW: tl.constexpr
):
    pid = tl.program_id(0)  # one program per (n, group)
    n = pid // num_groups
    group = pid % num_groups

    group_channels = C // num_groups
    group_start_c = group * group_channels

    # Compute sum and sum of squares over channels in group and all spatial positions
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for c in range(0, group_channels):
        c_idx = group_start_c + c
        HW = H_out * W_out
        # Pass 1: accumulation
        for hw_base in range(0, HW, BLOCK_HW):
            offs = hw_base + tl.arange(0, BLOCK_HW)
            mask = offs < HW
            h = offs // W_out
            w = offs % W_out
            x_ptrs = x_ptr + n * (C * H_out * W_out) + c_idx * (H_out * W_out) + h * W_out + w
            x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
            sum_val += tl.sum(x_vals, axis=0)
            sum_sq += tl.sum(x_vals * x_vals, axis=0)

    mean = sum_val / (group_channels * HW)
    var = sum_sq / (group_channels * HW) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and apply affine
    for c in range(0, group_channels):
        c_idx = group_start_c + c
        for hw_base in range(0, HW, BLOCK_HW):
            offs = hw_base + tl.arange(0, BLOCK_HW)
            mask = offs < HW
            h = offs // W_out
            w = offs % W_out

            x_ptrs = x_ptr + n * (C * H_out * W_out) + c_idx * (H_out * W_out) + h * W_out + w
            y_ptrs = y_ptr + n * (C * H_out * W_out) + c_idx * (H_out * W_out) + h * W_out + w

            x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
            norm = (x_vals - mean) * inv_std
            scale = tl.load(weight_ptr + c_idx, mask=True, other=1.0)
            bias = tl.load(bias_ptr + c_idx, mask=True, other=0.0)
            y_vals = norm * scale + bias
            tl.store(y_ptrs, y_vals, mask=mask)


# SiLU elementwise: y = x * sigmoid(x)
@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)


# Elementwise residual add: y = x1 + x2 (float32), assumes same shape
@triton.jit
def add_residual_kernel(x1_ptr, x2_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    a = tl.load(x1_ptr + offs, mask=mask, other=0.0)
    b = tl.load(x2_ptr + offs, mask=mask, other=0.0)
    y = a + b
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups=32):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        # Ensure device/dtype
        device = x.device
        dtype = torch.float32
        B, C, H, W = x.shape

        # Cast inputs and weights to fp32 and make contiguous
        x0 = x.contiguous().to(dtype)

        # conv1
        H_out = H + 2  # padding=1
        W_out = W + 2
        conv1_out = torch.empty((B, C, H_out, W_out), device=device, dtype=dtype)
        conv1_w = conv1_weight.contiguous().to(dtype)

        grid_conv1 = (B, C, triton.cdiv(H_out * W_out, 1024))
        conv3x3_nchw_fp32[grid_conv1](
            x0, conv1_w, conv1_out,
            B, C, C, H, W, H_out, W_out,
            x0.stride(0), x0.stride(1), x0.stride(2), x0.stride(3),
            conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2), conv1_w.stride(3),
            conv1_out.stride(0), conv1_out.stride(1), conv1_out.stride(2), conv1_out.stride(3),
            BLOCK_IN=64, BLOCK_HW=1024,
            num_warps=4, num_stages=2
        )

        # GroupNorm 1
        gn1_out = torch.empty_like(conv1_out)
        grid_gn1 = (B * self.num_groups,)
        groupnorm_affine_kernel[grid_gn1](
            conv1_out, norm1_weight.to(dtype), norm1_bias.to(dtype), gn1_out,
            B, C, H_out, W_out, self.num_groups, eps,
            BLOCK_HW=1024,
            num_warps=4, num_stages=2
        )

        # SiLU 1
        silu1_out = torch.empty_like(gn1_out)
        n_elements1 = gn1_out.numel()
        grid_silu1 = (triton.cdiv(n_elements1, 1024),)
        silu_kernel[grid_silu1](gn1_out, silu1_out, n_elements1, 1024,
                                num_warps=4, num_stages=2)

        # conv2: input is silu1_out, output has same spatial size as conv1_out
        conv2_out = torch.empty((B, C, H_out, W_out), device=device, dtype=dtype)
        conv2_w = conv2_weight.contiguous().to(dtype)

        grid_conv2 = (B, C, triton.cdiv(H_out * W_out, 1024))
        conv3x3_nchw_fp32[grid_conv2](
            silu1_out, conv2_w, conv2_out,
            B, C, C, H_out, W_out, H_out, W_out,
            silu1_out.stride(0), silu1_out.stride(1), silu1_out.stride(2), silu1_out.stride(3),
            conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2), conv2_w.stride(3),
            conv2_out.stride(0), conv2_out.stride(1), conv2_out.stride(2), conv2_out.stride(3),
            BLOCK_IN=64, BLOCK_HW=1024,
            num_warps=4, num_stages=2
        )

        # GroupNorm 2
        gn2_out = torch.empty_like(conv2_out)
        grid_gn2 = (B * self.num_groups,)
        groupnorm_affine_kernel[grid_gn2](
            conv2_out, norm2_weight.to(dtype), norm2_bias.to(dtype), gn2_out,
            B, C, H_out, W_out, self.num_groups, eps,
            BLOCK_HW=1024,
            num_warps=4, num_stages=2
        )

        # SiLU 2
        silu2_out = torch.empty_like(gn2_out)
        n_elements2 = gn2_out.numel()
        grid_silu2 = (triton.cdiv(n_elements2, 1024),)
        silu_kernel[grid_silu2](gn2_out, silu2_out, n_elements2, 1024,
                                num_warps=4, num_stages=2)

        # Residual addition: add original x (cast to fp32 and padded to H_out/W_out) to final output
        # Create residual tensor with same shape as silu2_out
        # Since we need x to match conv2_out shape (H_out, W_out), pad x to (H_out, W_out)
        x_padded = torch.nn.functional.pad(x0, (1, 1, 1, 1))  # padding (left,right) and (top,bottom) for H,W
        x_res = x_padded  # already (B, C, H_out, W_out) via padding
        add_out = torch.empty_like(silu2_out)

        n_elements_add = silu2_out.numel()
        grid_add = (triton.cdiv(n_elements_add, 1024),)
        add_residual_kernel[grid_add](silu2_out, x_res, add_out, n_elements_add, 1024,
                                      num_warps=4, num_stages=2)

        return add_out


def run(*args):
    return ModelNew()(*args)
