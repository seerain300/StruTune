import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_fp32(
    x_ptr,         # *const float
    w_ptr,         # *const float
    y_ptr,         # *float
    B: tl.constexpr, C_in: tl.constexpr, H_in: tl.constexpr, W_in: tl.constexpr,
    C_out: tl.constexpr,  # note: w is (C_in, C_out, 3, 3), so C_in != C_out
    BLOCK_IN: tl.constexpr,
):
    # one program per output element y[n, c_out, h_out, w_out]
    pid = tl.program_id(0)
    HW = H_in * W_in

    n = pid // (C_out * HW)
    rem = pid % (C_out * HW)
    c_out = rem // HW
    rem2 = rem % HW
    h_out = rem2 // W_in
    w_out = rem2 % W_in

    # accumulate in float32
    acc = 0.0

    # iterate over input channels in chunks
    for cin_base in range(0, C_in, BLOCK_IN):
        cin_vec = cin_base + tl.arange(0, BLOCK_IN)
        mask_c = cin_vec < C_in

        # compute contributions from 3x3 window
        for kh in range(0, 3):
            h_in = h_out + kh - 1  # -1 due to conv index (h_out - kh + 1)
            valid_h = (h_in >= 0) & (h_in < H_in)
            for kw in range(0, 3):
                w_in = w_out + kw - 1  # -1 due to conv index (w_out - kw + 1)
                valid_w = (w_in >= 0) & (w_in < W_in)

                # base offset for input tensor x[n, cin, h_in, w_in]
                base = n * (C_in * H_in * W_in) + h_in * (W_in * C_in) + w_in * C_in

                # load input vector for current chunk of cin with mask
                x_vals = tl.load(x_ptr + base + cin_vec, mask=mask_c & valid_h & valid_w, other=0.0)

                # load corresponding weights w[cin, c_out, kh, kw], flattened as w_ptr indexed by (cin * C_out * 9 + c_out * 9 + kh * 3 + kw)
                # weight layout: w_ptr is (C_in, C_out, 3, 3) contiguous -> index = cin * C_out*9 + c_out * 9 + kh*3 + kw
                weight_index = cin_vec * (C_out * 9) + c_out * 9 + kh * 3 + kw
                w_vals = tl.load(w_ptr + weight_index, mask=mask_c, other=0.0)

                # outer product accumulate
                acc += tl.sum(x_vals[:, None] * w_vals[None, :], axis=0)

    # store result to y[n, c_out, h_out, w_out]
    y_index = n * (C_out * H_in * W_in) + c_out * (H_in * W_in) + h_out * W_in + w_out
    tl.store(y_ptr + y_index, acc)


@triton.jit
def groupnorm_affine_nchw_fp32(
    y_ptr,         # *const float (input for GN)
    w_ptr,         # *const float (norm weight, shape [C])
    b_ptr,         # *const float (norm bias, shape [C])
    y_out_ptr,     # *float (output)
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    groups: tl.constexpr, eps: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    # one program per (n, group)
    pid = tl.program_id(0)
    n = pid // groups
    g = pid % groups

    # channels and elements per group
    channels_per_group = C // groups
    HW = H * W

    # compute sum and sumsq over group
    sum_val = 0.0
    sumsq_val = 0.0

    # pass 1: compute mean and variance
    for ch in range(0, channels_per_group):
        c = g * channels_per_group + ch
        for hw_base in range(0, HW, BLOCK_HW):
            idx_vec = hw_base + tl.arange(0, BLOCK_HW)
            mask_hw = idx_vec < HW
            h = idx_vec // W
            w = idx_vec % W
            y_index = n * (C * H * W) + c * (H * W) + h * W + w
            x = tl.load(y_ptr + y_index, mask=mask_hw, other=0.0)
            sum_val += tl.sum(x, axis=0)
            sumsq_val += tl.sum(x * x, axis=0)

    m = channels_per_group * HW
    mean = sum_val / m
    var = sumsq_val / m - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # pass 2: normalize and apply affine
    for ch in range(0, channels_per_group):
        c = g * channels_per_group + ch
        gamma = tl.load(w_ptr + c)
        beta = tl.load(b_ptr + c)
        for hw_base in range(0, HW, BLOCK_HW):
            idx_vec = hw_base + tl.arange(0, BLOCK_HW)
            mask_hw = idx_vec < HW
            h = idx_vec // W
            w = idx_vec % W
            y_index_in = n * (C * H * W) + c * (H * W) + h * W + w
            x = tl.load(y_ptr + y_index_in, mask=mask_hw, other=0.0)
            norm = (x - mean) * inv_std
            y_norm = norm * gamma + beta
            y_index_out = n * (C * H * W) + c * (H * W) + h * W + w
            tl.store(y_out_ptr + y_index_out, y_norm, mask=mask_hw)


@triton.jit
def silu_fp32(x_ptr, y_ptr, total_elems: tl.constexpr, BLOCK: tl.constexpr):
    # elementwise: y = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
    for base in range(0, total_elems, BLOCK):
        idx = base + tl.arange(0, BLOCK)
        mask = idx < total_elems
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        s = 1.0 / (1.0 + tl.exp(-x))
        y = x * s
        tl.store(y_ptr + idx, y, mask=mask)


@triton.jit
def add_residual_fp32(x_ptr, y_ptr, out_ptr, total_elems: tl.constexpr, BLOCK: tl.constexpr):
    # elementwise: out = y + x
    for base in range(0, total_elems, BLOCK):
        idx = base + tl.arange(0, BLOCK)
        mask = idx < total_elems
        a = tl.load(x_ptr + idx, mask=mask, other=0.0)
        b = tl.load(y_ptr + idx, mask=mask, other=0.0)
        c = a + b
        tl.store(out_ptr + idx, c, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps):
        # Ensure float32 contiguous
        device = x.device
        B, C, H, W = x.shape
        groups = 32

        # First path: Conv3x3 -> GroupNorm -> SiLU
        # conv1
        y1 = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        grid_conv1 = (B * C * H * W,)
        conv3x3_nchw_fp32[grid_conv1](
            x, conv1_weight.to(torch.float32).contiguous(), y1,
            B, C, H, W, C,  # C_in=C, C_out=C
            BLOCK_IN=32,
            num_warps=4
        )
        # GroupNorm1 (C=32*C/groups, so channels_per_group=C//32)
        y1_out = torch.empty_like(y1, device=device, dtype=torch.float32)
        grid_gn1 = (B * groups,)
        groupnorm_affine_nchw_fp32[grid_gn1](
            y1, norm1_weight.to(torch.float32).contiguous(), norm1_bias.to(torch.float32).contiguous(), y1_out,
            B, C, H, W, groups, eps,
            BLOCK_HW=1024,
            num_warps=4
        )
        # SiLU1
        y1_silu = torch.empty_like(y1_out, device=device, dtype=torch.float32)
        total1 = y1_out.numel()
        grid_silu1 = (triton.cdiv(total1, 1024),)
        silu_fp32[grid_silu1](y1_out, y1_silu, total1, 1024, num_warps=4)

        # Second path: Conv3x3 -> GroupNorm -> SiLU
        # conv2
        y2 = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        grid_conv2 = (B * C * H * W,)
        conv3x3_nchw_fp32[grid_conv2](
            y1_silu, conv2_weight.to(torch.float32).contiguous(), y2,
            B, C, H, W, C,  # C_in=C, C_out=C
            BLOCK_IN=32,
            num_warps=4
        )
        # GroupNorm2
        y2_out = torch.empty_like(y2, device=device, dtype=torch.float32)
        grid_gn2 = (B * groups,)
        groupnorm_affine_nchw_fp32[grid_gn2](
            y2, norm2_weight.to(torch.float32).contiguous(), norm2_bias.to(torch.float32).contiguous(), y2_out,
            B, C, H, W, groups, eps,
            BLOCK_HW=1024,
            num_warps=4
        )
        # SiLU2
        y2_silu = torch.empty_like(y2_out, device=device, dtype=torch.float32)
        total2 = y2_out.numel()
        grid_silu2 = (triton.cdiv(total2, 1024),)
        silu_fp32[grid_silu2](y2_out, y2_silu, total2, 1024, num_warps=4)

        # Final residual addition: out = y2_silu + x
        out = torch.empty_like(y2_silu, device=device, dtype=torch.float32)
        total_final = y2_silu.numel()
        grid_add = (triton.cdiv(total_final, 1024),)
        add_residual_fp32[grid_add](y2_silu, x.to(torch.float32).contiguous(), out, total_final, 1024, num_warps=4)

        return out


def run(*args):
    return ModelNew()(*args)
