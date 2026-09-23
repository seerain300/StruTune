import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_fp32(
    x_ptr, w_ptr, y_ptr,
    B, C_in, H, W, C_out,
    H_out, W_out,
    BLOCK_IN: tl.constexpr
):
    # One program computes one output element y[n, c_out, h_out, w_out]
    total = B * C_out * H_out * W_out
    pid = tl.program_id(0)
    n = pid // (C_out * H_out * W_out)
    rem = pid % (C_out * H_out * W_out)
    c_out = rem // (H_out * W_out)
    rem2 = rem % (H_out * W_out)
    h_out = rem2 // W_out
    w_out = rem2 % W_out

    # Initialize accumulator
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels in chunks
    for c_start in range(0, C_in, BLOCK_IN):
        c_offsets = c_start + tl.arange(0, BLOCK_IN)
        mask_c = c_offsets < C_in

        # Accumulate over 3x3 window with padding masks
        # Note: ih, iw computed from output indices; handle padding via masks
        for kh in range(3):
            ih = h_out + kh - 1  # -1 because output spatial dims are H_out=W_out=H=W due to padding
            # ih may be out-of-range; we'll mask loads
            for kw in range(3):
                iw = w_out + kw - 1

                # Build pointer offsets for input x and weights
                # x[n, c_in, ih, iw] contiguous NCHW
                x_base = n * C_in * H * W
                # Loop over BLOCK_IN channels
                for ci in range(BLOCK_IN):
                    c_in = c_start + ci
                    valid_c = c_in < C_in
                    valid_i = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & valid_c
                    # Load input if valid; else 0.0
                    x_offset = x_base + c_in * H * W + ih * W + iw
                    x_val = tl.load(x_ptr + x_offset, mask=valid_i, other=0.0)

                    # Load weight w[c_out, c_in, kh, kw] as scalar
                    # Weight is (C_out, C_in, 3, 3), contiguous
                    w_base = 0
                    w_offset = c_out * (C_in * 9) + c_in * 9 + kh * 3 + kw
                    w_val = tl.load(w_ptr + w_base + w_offset)

                    # Accumulate
                    acc += x_val * w_val

    # Store result
    y_offset = (n * C_out + c_out) * H_out * W_out + h_out * W_out + w_out
    tl.store(y_ptr + y_offset, acc)


@triton.jit
def conv3x3_nchw_fp32_2(
    x_ptr, w_ptr, y_ptr,
    B, C_in, H, W, C_out,
    H_out, W_out,
    BLOCK_IN: tl.constexpr
):
    total = B * C_out * H_out * W_out
    pid = tl.program_id(0)
    n = pid // (C_out * H_out * W_out)
    rem = pid % (C_out * H_out * W_out)
    c_out = rem // (H_out * W_out)
    rem2 = rem % (H_out * W_out)
    h_out = rem2 // W_out
    w_out = rem2 % W_out

    acc = tl.zeros((), dtype=tl.float32)

    for c_start in range(0, C_in, BLOCK_IN):
        c_offsets = c_start + tl.arange(0, BLOCK_IN)
        mask_c = c_offsets < C_in

        for kh in range(3):
            ih = h_out + kh - 1
            for kw in range(3):
                iw = w_out + kw - 1

                for ci in range(BLOCK_IN):
                    c_in = c_start + ci
                    valid_c = c_in < C_in
                    valid_i = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & valid_c

                    x_base = n * C_in * H * W
                    x_offset = x_base + c_in * H * W + ih * W + iw
                    x_val = tl.load(x_ptr + x_offset, mask=valid_i, other=0.0)

                    w_base = 0
                    w_offset = c_out * (C_in * 9) + c_in * 9 + kh * 3 + kw
                    w_val = tl.load(w_ptr + w_base + w_offset)

                    acc += x_val * w_val

    y_offset = (n * C_out + c_out) * H_out * W_out + h_out * W_out + w_out
    tl.store(y_ptr + y_offset, acc)


@triton.jit
def groupnorm_affine_kernel(
    x_ptr, y_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    B, C, H, W, NUM_GROUPS,
    H_out, W_out,  # H_out==H, W_out==W preserved
    BLOCK_HW: tl.constexpr
):
    # One program per (n, group)
    pid = tl.program_id(0)
    n = pid // NUM_GROUPS
    g = pid % NUM_GROUPS

    group_channels = C // NUM_GROUPS
    group_start = g * group_channels

    # Accumulate sum and sum of squares over group
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    total_hw = H * W

    # Pass 1: reduction
    for c in range(0, group_channels):
        c_idx = group_start + c
        for start in range(0, total_hw, BLOCK_HW):
            idx = start + tl.arange(0, BLOCK_HW)
            mask = idx < total_hw
            # Map idx to (h, w)
            h = idx // W
            w = idx % W
            x_offset = (n * C + c_idx) * H * W + h * W + w
            x_vec = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
            sum_val += tl.sum(x_vec, axis=0)
            sum_sq += tl.sum(x_vec * x_vec, axis=0)

    hw = H * W
    mean = sum_val / (group_channels * hw)
    var = sum_sq / (group_channels * hw) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + 1e-5)  # epsilon-like for numerical stability

    # Store mean/var for this (n, group)
    mean_offset = n * NUM_GROUPS + g
    tl.store(mean_ptr + mean_offset, mean)
    tl.store(var_ptr + mean_offset, var)

    # Pass 2: normalize and affine
    for c in range(0, group_channels):
        c_idx = group_start + c
        for start in range(0, total_hw, BLOCK_HW):
            idx = start + tl.arange(0, BLOCK_HW)
            mask = idx < total_hw
            h = idx // W
            w = idx % W

            x_offset = (n * C + c_idx) * H * W + h * W + w
            x_vec = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
            norm = (x_vec - mean) * inv_std

            # Load per-channel scale and bias
            weight_c = tl.load(weight_ptr + c_idx)
            bias_c = tl.load(bias_ptr + c_idx)

            y_vec = norm * weight_c + bias_c
            tl.store(y_ptr + x_offset, y_vec, mask=mask)


@triton.jit
def silu_kernel(x_ptr, y_ptr, total: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def add_residual_kernel(a_ptr, b_ptr, y_ptr, total: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    y = a + b
    tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        # Ensure float32 and contiguous for Triton kernels
        device = x.device
        dtype = torch.float32

        x = x.to(dtype).contiguous()

        # First conv: y1 = conv3x3(x, conv1_weight)
        B, C_in, H, W = x.shape
        C_out = conv1_weight.shape[0]
        H_out, W_out = H, W  # stride=1, padding=1 preserves dims
        y1 = torch.empty((B, C_out, H_out, W_out), device=device, dtype=dtype)

        total = B * C_out * H_out * W_out
        grid_conv1 = (total,)
        conv3x3_nchw_fp32[grid_conv1](
            x, conv1_weight.contiguous().to(dtype),
            y1,
            B, C_in, H, W, C_out,
            H_out, W_out,
            BLOCK_IN=8,
            num_warps=4,
        )

        # GroupNorm 1 with affine
        y1_contig = y1.contiguous()
        num_groups = 32
        assert C_out % num_groups == 0, "C_out must be divisible by num_groups"
        mean1 = torch.empty((B * num_groups,), device=device, dtype=dtype)
        var1 = torch.empty((B * num_groups,), device=device, dtype=dtype)
        y1_norm = torch.empty_like(y1_contig, device=device, dtype=dtype)
        grid_gn1 = (B * num_groups,)
        groupnorm_affine_kernel[grid_gn1](
            y1_contig, y1_norm, mean1, var1, norm1_weight.to(dtype).contiguous(), norm1_bias.to(dtype).contiguous(),
            B, C_out, H, W, num_groups,
            H_out, W_out,
            BLOCK_HW=1024,
            num_warps=4,
        )

        # SiLU on normalized output
        y1_silu = torch.empty_like(y1_norm, device=device, dtype=dtype)
        total1 = B * C_out * H * W
        grid_silu = (triton.cdiv(total1, 1024),)
        silu_kernel[grid_silu](
            y1_norm, y1_silu, total1, 1024,
            num_warps=4,
        )

        # Second conv: y2 = conv3x3(y1_silu, conv2_weight)
        y2 = torch.empty((B, C_out, H_out, W_out), device=device, dtype=dtype)
        total2 = B * C_out * H_out * W_out
        grid_conv2 = (total2,)
        conv3x3_nchw_fp32_2[grid_conv2](
            y1_silu, conv2_weight.contiguous().to(dtype),
            y2,
            B, C_out, H, W, C_out,
            H_out, W_out,
            BLOCK_IN=8,
            num_warps=4,
        )

        # GroupNorm 2 with affine
        y2_contig = y2.contiguous()
        mean2 = torch.empty((B * num_groups,), device=device, dtype=dtype)
        var2 = torch.empty((B * num_groups,), device=device, dtype=dtype)
        y2_norm = torch.empty_like(y2_contig, device=device, dtype=dtype)
        grid_gn2 = (B * num_groups,)
        groupnorm_affine_kernel[grid_gn2](
            y2_contig, y2_norm, mean2, var2, norm2_weight.to(dtype).contiguous(), norm2_bias.to(dtype).contiguous(),
            B, C_out, H, W, num_groups,
            H_out, W_out,
            BLOCK_HW=1024,
            num_warps=4,
        )

        # SiLU on normalized output
        y2_silu = torch.empty_like(y2_norm, device=device, dtype=dtype)
        total2_s = B * C_out * H * W
        grid_silu2 = (triton.cdiv(total2_s, 1024),)
        silu_kernel[grid_silu2](
            y2_norm, y2_silu, total2_s, 1024,
            num_warps=4,
        )

        # Residual addition (original input x added to final output)
        # Shapes must match; convs preserve spatial dims (H, W), output has same (B, C, H, W)
        final = torch.empty_like(y2_silu, device=device, dtype=dtype)
        total_add = B * C_out * H * W
        grid_add = (triton.cdiv(total_add, 1024),)
        add_residual_kernel[grid_add](
            y2_silu, x, final, total_add, 1024,
            num_warps=4,
        )

        return final


def run(*args):
    return ModelNew()(*args)
