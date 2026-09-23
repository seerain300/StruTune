import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_fp32(
    x_ptr,         # *const float
    w_ptr,         # *const float
    y_ptr,         # *float
    B: tl.constexpr, C_in: tl.constexpr, H_in: tl.constexpr, W_in: tl.constexpr,
    C_out: tl.constexpr, K: tl.constexpr,  # K = 3
    BLOCK_IN: tl.constexpr,
):
    # program id maps to (n, c_out, h_out, w_out)
    pid = tl.program_id(0)
    HW = H_in * W_in
    N = B
    C_OUT = C_out

    n = pid // (C_OUT * HW)
    rem = pid % (C_OUT * HW)
    c_out = rem // HW
    rem2 = rem % HW
    h_out = rem2 // W_in
    w_out = rem2 % W_in

    # initialize accumulator
    acc = 0.0

    # loop over input channels in chunks
    for cin_base in range(0, C_in, BLOCK_IN):
        # compute input spatial base
        # For padding=1 and stride=1: h_in = h_out + kh, w_in = w_out + kw
        # We need to loop over the 3x3 window.
        for kh in range(0, K):
            for kw in range(0, K):
                h_in = h_out + kh - 1  # shift by -1 because window centers at (kh,kw)
                w_in = w_out + kw - 1
                # check bounds for input spatial
                valid_hw = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
                # loop over input channels for this chunk
                for ci in range(0, BLOCK_IN):
                    cin = cin_base + ci
                    valid_ci = cin < C_in
                    # load input x[n, cin, h_in, w_in] with mask
                    x_off = (((n * C_in + cin) * H_in + h_in) * W_in + w_in)
                    x_val = tl.load(x_ptr + x_off, mask=valid_ci & valid_hw, other=0.0)
                    # load weight w[cin, c_out, kh, kw] (note: original conv has weight (C_out, C_in, K, K))
                    w_off = (((c_out * C_in + cin) * (K * K) + (kh * K + kw)))
                    w_val = tl.load(w_ptr + w_off)
                    # accumulate
                    acc += x_val * w_val

    # store output y[n, c_out, h_out, w_out]
    y_off = (((n * C_out + c_out) * H_in + h_out) * W_in + w_out)
    tl.store(y_ptr + y_off, acc)


@triton.jit
def groupnorm_affine_nchw_fp32(
    x_ptr,         # *const float
    weight_ptr,    # *const float (C,)
    bias_ptr,      # *const float (C,)
    y_ptr,         # *float
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    GROUPS: tl.constexpr, EPS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # one program per (n, group)
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS
    channels_per_group = C // GROUPS
    start_c = g * channels_per_group

    # compute sum and sum of squares over group
    total = H * W
    sum_val = 0.0
    sum_sq = 0.0
    for c_local in range(0, channels_per_group):
        c = start_c + c_local
        for hw_base in range(0, total, BLOCK_HW):
            hw_idx = hw_base + tl.arange(0, BLOCK_HW)
            mask = hw_idx < total
            h = hw_idx // W
            w = hw_idx % W
            x_off = (((n * C + c) * H + h) * W + w)
            x_val = tl.load(x_ptr + x_off, mask=mask, other=0.0)
            sum_val += tl.sum(x_val, axis=0)
            sum_sq += tl.sum(x_val * x_val, axis=0)

    mean = sum_val / (channels_per_group * total)
    var = sum_sq / (channels_per_group * total) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + EPS)

    # normalize and apply affine, write back
    for c_local in range(0, channels_per_group):
        c = start_c + c_local
        for hw_base in range(0, total, BLOCK_HW):
            hw_idx = hw_base + tl.arange(0, BLOCK_HW)
            mask = hw_idx < total
            h = hw_idx // W
            w = hw_idx % W
            x_off = (((n * C + c) * H + h) * W + w)
            x_val = tl.load(x_ptr + x_off, mask=mask, other=0.0)
            norm = (x_val - mean) * inv_std
            gamma = tl.load(weight_ptr + c)
            beta = tl.load(bias_ptr + c)
            y_val = norm * gamma + beta
            y_off = (((n * C + c) * H + h) * W + w)
            tl.store(y_ptr + y_off, y_val, mask=mask)


@triton.jit
def silu_kernel_fp32(x_ptr, y_ptr, total: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < total
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + idx, y, mask=mask)


@triton.jit
def add_residual_kernel_fp32(x_ptr, y_ptr, out_ptr, total: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    mask = idx < total
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    y = tl.load(y_ptr + idx, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + idx, out, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        conv1_weight: torch.Tensor,   # (C, C, 3, 3)
        norm1_weight: torch.Tensor,   # (C,)
        norm1_bias: torch.Tensor,     # (C,)
        conv2_weight: torch.Tensor,   # (C, C, 3, 3)
        norm2_weight: torch.Tensor,   # (C,)
        norm2_bias: torch.Tensor,     # (C,)
        eps: float,
    ):
        # Cast to float32 and contiguous for Triton kernels
        x = x.contiguous().to(torch.float32)
        device = x.device

        # Dimensions
        B, C, H, W = x.shape

        # First conv: y1 = conv3x3(x, conv1_weight, stride=1, padding=1)
        y1 = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        grid_conv1 = (B * C * H * W,)
        conv3x3_nchw_fp32[grid_conv1](
            x, conv1_weight.contiguous().to(torch.float32), y1,
            B, C, H, W, C, 3, 32,
            num_warps=4
        )

        # First GroupNorm with affine
        y2 = torch.empty_like(y1, device=device, dtype=torch.float32)
        groups = 32
        grid_gn1 = (B * groups,)
        groupnorm_affine_nchw_fp32[grid_gn1](
            y1, norm1_weight.contiguous().to(torch.float32), norm1_bias.contiguous().to(torch.float32), y2,
            B, C, H, W, groups, eps, 1024,
            num_warps=4
        )

        # SiLU 1
        y3 = torch.empty_like(y2, device=device, dtype=torch.float32)
        total1 = y2.numel()
        grid_silu1 = (triton.cdiv(total1, 1024),)
        silu_kernel_fp32[grid_silu1](y2, y3, total1, 1024, num_warps=4)

        # Second conv: y4 = conv3x3(y3, conv2_weight, stride=1, padding=1)
        y4 = torch.empty((B, C, H, W), device=device, dtype=torch.float32)
        grid_conv2 = (B * C * H * W,)
        conv3x3_nchw_fp32[grid_conv2](
            y3, conv2_weight.contiguous().to(torch.float32), y4,
            B, C, H, W, C, 3, 32,
            num_warps=4
        )

        # Second GroupNorm with affine
        y5 = torch.empty_like(y4, device=device, dtype=torch.float32)
        grid_gn2 = (B * groups,)
        groupnorm_affine_nchw_fp32[grid_gn2](
            y4, norm2_weight.contiguous().to(torch.float32), norm2_bias.contiguous().to(torch.float32), y5,
            B, C, H, W, groups, eps, 1024,
            num_warps=4
        )

        # SiLU 2
        y6 = torch.empty_like(y5, device=device, dtype=torch.float32)
        total2 = y5.numel()
        grid_silu2 = (triton.cdiv(total2, 1024),)
        silu_kernel_fp32[grid_silu2](y5, y6, total2, 1024, num_warps=4)

        # Final residual addition: out = y6 + x
        out_final = torch.empty_like(y6, device=device, dtype=torch.float32)
        total_final = y6.numel()
        grid_add = (triton.cdiv(total_final, 1024),)
        add_residual_kernel_fp32[grid_add](y6, x, out_final, total_final, 1024, num_warps=4)

        return out_final


# Example helper to generate inputs (optional for local testing)
def get_inputs():
    # You can adapt these shapes to match the evaluation workload axes.
    B = 1
    C = 32
    H = 128
    W = 128
    x = torch.randn(B, C, H, W, device='cuda', dtype=torch.float32)
    conv1_weight = torch.randn(C, C, 3, 3, device='cuda', dtype=torch.float32)
    norm1_weight = torch.randn(C, device='cuda', dtype=torch.float32)
    norm1_bias = torch.randn(C, device='cuda', dtype=torch.float32)
    conv2_weight = torch.randn(C, C, 3, 3, device='cuda', dtype=torch.float32)
    norm2_weight = torch.randn(C, device='cuda', dtype=torch.float32)
    norm2_bias = torch.randn(C, device='cuda', dtype=torch.float32)
    eps = 1e-5
    return x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias, eps


def run(*args):
    return ModelNew()(*args)
