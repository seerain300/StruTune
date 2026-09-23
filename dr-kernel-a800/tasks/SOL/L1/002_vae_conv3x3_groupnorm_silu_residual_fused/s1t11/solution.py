import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_fp32(
    x_ptr,         # *const float
    w_ptr,         # *const float
    y_ptr,         # *float
    B, C_in, H_in, W_in,
    C_out, K,
    BLOCK_IN: tl.constexpr,
):
    # program id maps to (n, c_out, h_out, w_out)
    pid = tl.program_id(0)
    HW = H_in * W_in
    N = B

    n = pid // (C_out * HW)
    rem = pid % (C_out * HW)
    c_out = rem // HW
    rem2 = rem % HW
    h_out = rem2 // W_in
    w_out = rem2 % W_in

    # initialize accumulator
    acc = 0.0

    # loop over input channels in chunks
    for cin_base in range(0, C_in, BLOCK_IN):
        offs = tl.arange(0, BLOCK_IN)
        cin = cin_base + offs
        mask_c = cin < C_in

        acc_chunk = tl.zeros([BLOCK_IN], dtype=tl.float32)

        # loop over 3x3 kernel window with padding=1
        for kh in range(0, 3):
            for kw in range(0, 3):
                h_in = h_out + kh - 1
                w_in = w_out + kw - 1

                mask_hw = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)

                x_off = n * (C_in * H_in * W_in) + cin * (H_in * W_in) + h_in * W_in + w_in
                x_mask = mask_c & mask_hw

                x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                w_off = cin * (K * K * C_out) + kh * (K * C_out) + kw * C_out + c_out
                w_vals = tl.load(w_ptr + w_off, mask=mask_c, other=0.0)

                acc_chunk += x_vals * w_vals

        acc += tl.sum(acc_chunk, axis=0)

    y_off = n * (C_out * H_in * W_in) + c_out * (H_in * W_in) + h_out * W_in + w_out
    tl.store(y_ptr + y_off, acc)


@triton.jit
def group_norm_affine_fp32(
    x_ptr,          # *const float
    weight_ptr,     # *const float (C,)
    bias_ptr,       # *const float (C,)
    y_ptr,          # *float
    B, C, H, W,
    num_groups,
    eps: tl.float32,
    BLOCK_HW: tl.constexpr,
):
    # one program per (n, group)
    pid = tl.program_id(0)
    n = pid // num_groups
    g = pid % num_groups

    CPG = C // num_groups

    sum_val = 0.0
    sum_sq = 0.0

    for c_start in range(0, CPG):
        c = g * CPG + c_start
        for hw_start in range(0, H * W, BLOCK_HW):
            offs = hw_start + tl.arange(0, BLOCK_HW)
            mask = offs < (H * W)
            h = offs // W
            w = offs % W

            x_off = n * (C * H * W) + c * (H * W) + h * W + w
            x_vals = tl.load(x_ptr + x_off, mask=mask, other=0.0)

            sum_val += tl.sum(x_vals, axis=0)
            sum_sq += tl.sum(x_vals * x_vals, axis=0)

    m = CPG * (H * W)
    mean = sum_val / m
    var = sum_sq / m - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    for c_start in range(0, CPG):
        c = g * CPG + c_start
        for hw_start in range(0, H * W, BLOCK_HW):
            offs = hw_start + tl.arange(0, BLOCK_HW)
            mask = offs < (H * W)
            h = offs // W
            w = offs % W

            x_off = n * (C * H * W) + c * (H * W) + h * W + w
            x_vals = tl.load(x_ptr + x_off, mask=mask, other=0.0)

            w_val = tl.load(weight_ptr + c, mask=True, other=1.0)
            b_val = tl.load(bias_ptr + c, mask=True, other=0.0)

            y_vals = ((x_vals - mean) * inv_std) * w_val + b_val

            y_off = n * (C * H * W) + c * (H * W) + h * W + w
            tl.store(y_ptr + y_off, y_vals)


@triton.jit
def silu_kernel_fp32(x_ptr, y_ptr, total: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y)


@triton.jit
def add_residual_kernel_fp32(a_ptr, b_ptr, out_ptr, total: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    out = a + b
    tl.store(out_ptr + offs, out)


def _run_triton_res_block(x0: torch.Tensor,
                           conv1_weight: torch.Tensor,
                           norm1_weight: torch.Tensor,
                           norm1_bias: torch.Tensor,
                           conv2_weight: torch.Tensor,
                           norm2_weight: torch.Tensor,
                           norm2_bias: torch.Tensor,
                           eps: float) -> torch.Tensor:
    """
    Triton-only fused residual block:
      conv1 (3x3) -> GroupNorm(32) -> SiLU -> conv2 (3x3) -> GroupNorm(32) -> SiLU -> Add(x0)
    All computation done by Triton kernels. x0 is added elementwise to final output via Triton.
    """
    # Ensure dtype and contiguity for Triton
    x = x0.contiguous().to(torch.float32)
    B, C, H, W = x.shape

    # First conv: conv3x3 stride=1, padding=1 (no bias), NCHW
    C_in = C
    C_out1 = C
    K = 3
    H1 = H
    W1 = W
    x1 = torch.empty((B, C_out1, H1, W1), device=x.device, dtype=torch.float32)

    grid_conv1 = (B * C_out1 * H1 * W1,)
    conv3x3_nchw_fp32[grid_conv1](
        x, conv1_weight.to(torch.float32).contiguous(), x1,
        B, C_in, H, W,
        C_out1, K,
        BLOCK_IN=32,
        num_warps=4
    )

    # First GroupNorm: num_groups=32, affine, eps=eps
    y1 = torch.empty_like(x1, device=x.device, dtype=torch.float32)
    num_groups = 32
    grid_gn1 = (B * num_groups,)
    group_norm_affine_fp32[grid_gn1](
        x1, norm1_weight.to(torch.float32).contiguous(), norm1_bias.to(torch.float32).contiguous(), y1,
        B, C_out1, H1, W1,
        num_groups,
        eps,
        BLOCK_HW=256,
        num_warps=4
    )

    # SiLU 1
    y1_silu = torch.empty_like(y1, device=x.device, dtype=torch.float32)
    total1 = y1.numel()
    grid_silu1 = (triton.cdiv(total1, 1024),)
    silu_kernel_fp32[grid_silu1](y1, y1_silu, total1, 1024, num_warps=4)

    # Second conv: conv3x3 stride=1, padding=1 (no bias), NCHW
    C_in2 = C_out1
    C_out2 = C_in2
    H2 = H1
    W2 = W1
    x2 = torch.empty((B, C_out2, H2, W2), device=x.device, dtype=torch.float32)

    grid_conv2 = (B * C_out2 * H2 * W2,)
    conv3x3_nchw_fp32[grid_conv2](
        y1_silu, conv2_weight.to(torch.float32).contiguous(), x2,
        B, C_in2, H2, W2,
        C_out2, K,
        BLOCK_IN=32,
        num_warps=4
    )

    # Second GroupNorm: num_groups=32, affine, eps=eps
    y2 = torch.empty_like(x2, device=x.device, dtype=torch.float32)
    grid_gn2 = (B * num_groups,)
    group_norm_affine_fp32[grid_gn2](
        x2, norm2_weight.to(torch.float32).contiguous(), norm2_bias.to(torch.float32).contiguous(), y2,
        B, C_out2, H2, W2,
        num_groups,
        eps,
        BLOCK_HW=256,
        num_warps=4
    )

    # SiLU 2
    y2_silu = torch.empty_like(y2, device=x.device, dtype=torch.float32)
    total2 = y2.numel()
    grid_silu2 = (triton.cdiv(total2, 1024),)
    silu_kernel_fp32[grid_silu2](y2, y2_silu, total2, 1024, num_warps=4)

    # Residual addition: y2_silu + x via Triton
    out_final = torch.empty_like(y2_silu, device=x.device, dtype=torch.float32)
    total_final = y2_silu.numel()
    grid_add = (triton.cdiv(total_final, 1024),)
    add_residual_kernel_fp32[grid_add](y2_silu, x, out_final, total_final, 1024, num_warps=4)

    return out_final


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        # Triton-only forward: all computation in kernels
        return _run_triton_res_block(
            x,
            conv1_weight, norm1_weight, norm1_bias,
            conv2_weight, norm2_weight, norm2_bias,
            eps
        )


def run(*args):
    return ModelNew()(*args)
