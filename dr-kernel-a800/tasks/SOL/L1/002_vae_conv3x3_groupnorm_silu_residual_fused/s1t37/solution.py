import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_fp32(x_ptr, w_ptr, y_ptr,
                      B, C_in, H_in, W_in, C_out, H_out, W_out,
                      BLOCK_IN: tl.constexpr):
    # One program computes one output element y[n, c_out, h_out, w_out]
    pid = tl.program_id(axis=0)
    total = B * C_out * H_out * W_out
    if pid >= total:
        return

    n = pid // (C_out * H_out * W_out)
    tmp = pid % (C_out * H_out * W_out)
    c_out = tmp // (H_out * W_out)
    tmp2 = tmp % (H_out * W_out)
    h_out = tmp2 // W_out
    w_out = tmp2 % W_out

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels in chunks
    for c_in_start in range(0, C_in, BLOCK_IN):
        cin_idx = c_in_start + tl.arange(0, BLOCK_IN)
        for kh in range(3):
            h_in = h_out * 3 + kh - 1  # -1..+2, masks handle padding
            for kw in range(3):
                w_in = w_out * 3 + kw - 1
                mask_h = (h_in >= 0) & (h_in < H_in)
                mask_w = (w_in >= 0) & (w_in < W_in)
                for ci in range(BLOCK_IN):
                    cin = c_in_start + ci
                    valid_c = cin < C_in
                    x_off = ((n * C_in + cin) * H_in + h_in) * W_in + w_in
                    x_mask = mask_h & mask_w & valid_c
                    x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)
                    # weights layout: [C_out, C_in, 3, 3], contiguous
                    w_off = ((c_out * C_in + cin) * 9 + (kh * 3 + kw))
                    w_val = tl.load(w_ptr + w_off)
                    acc += x_val * w_val

    y_off = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out
    tl.store(y_ptr + y_off, acc)


@triton.jit
def groupnorm_affine_kernel(x_ptr, y_ptr, weight_ptr, bias_ptr,
                            B, C, H, W, num_groups,
                            eps, BLOCK_HW: tl.constexpr):
    # One program per (n, group)
    pid = tl.program_id(axis=0)
    if pid >= B * num_groups:
        return

    n = pid // num_groups
    group = pid % num_groups

    channels_per_group = C // num_groups
    group_c_start = group * channels_per_group

    # Pass 1: compute mean and variance over group channels and all spatial positions
    sum_total = tl.zeros((), dtype=tl.float32)
    sumsq_total = tl.zeros((), dtype=tl.float32)

    for c in range(channels_per_group):
        c_idx = group_c_start + c
        HW = H * W
        for hw_start in range(0, HW, BLOCK_HW):
            hw_idx = hw_start + tl.arange(0, BLOCK_HW)
            mask_hw = hw_idx < HW
            h = hw_idx // W
            w = hw_idx % W
            x_off = n * C * H * W + c_idx * H * W + h * W + w
            x_vec = tl.load(x_ptr + x_off, mask=mask_hw, other=0.0)
            sum_total += tl.sum(x_vec, axis=0)
            sumsq_total += tl.sum(x_vec * x_vec, axis=0)

    mean = sum_total / (channels_per_group * H * W)
    var = sumsq_total / (channels_per_group * H * W) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and apply affine, then store
    for c in range(channels_per_group):
        c_idx = group_c_start + c
        HW = H * W
        for hw_start in range(0, HW, BLOCK_HW):
            hw_idx = hw_start + tl.arange(0, BLOCK_HW)
            mask_hw = hw_idx < HW
            h = hw_idx // W
            w = hw_idx % W
            x_off = n * C * H * W + c_idx * H * W + h * W + w
            x_vec = tl.load(x_ptr + x_off, mask=mask_hw, other=0.0)

            scale = tl.load(weight_ptr + c_idx)
            bias = tl.load(bias_ptr + c_idx)

            y_vec = (x_vec - mean) * inv_std
            y_vec = y_vec * scale + bias

            y_off = n * C * H * W + c_idx * H * W + h * W + w
            tl.store(y_ptr + y_off, y_vec, mask=mask_hw)


@triton.jit
def silu_kernel(x_ptr, y_ptr, total_elems, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    y = x * (1.0 / (1.0 + tl.exp(-x)))
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def add_residual_kernel(a_ptr, b_ptr, out_ptr, total_elems, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a + b, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor):
        """
        Triton-optimized fused residual block:
          Conv3x3 (NCHW, stride=1, pad=1, no bias) -> GroupNorm(num_groups=32, affine) -> SiLU
          Conv3x3 (NCHW, stride=1, pad=1, no bias) -> GroupNorm(num_groups=32, affine) -> SiLU
          Add original input x as residual.
        """
        device = x.device
        dtype = torch.float32

        # First conv: conv1
        x0 = x.to(dtype).contiguous()
        B, C, H, W = x0.shape
        C_in1 = C
        C_out1 = conv1_weight.shape[0]
        H1 = H
        W1 = W

        y1 = torch.empty((B, C_out1, H1, W1), device=device, dtype=dtype)

        total_out1 = B * C_out1 * H1 * W1
        grid1 = (triton.cdiv(total_out1, 1),)
        conv3x3_nchw_fp32[grid1](
            x0, conv1_weight.to(dtype).contiguous(), y1,
            B, C_in1, H1, W1, C_out1, H1, W1,
            BLOCK_IN=64
        )

        # GroupNorm 1
        y1_gn = torch.empty((B, C_out1, H1, W1), device=device, dtype=dtype)
        groupnorm_affine_kernel[(B * 32,)](
            y1, y1_gn, norm1_weight.to(dtype).contiguous(), norm1_bias.to(dtype).contiguous(),
            B, C_out1, H1, W1, 32, self.eps, BLOCK_HW=256
        )

        # SiLU 1
        y1_silu = torch.empty((B, C_out1, H1, W1), device=device, dtype=dtype)
        total1 = y1_silu.numel()
        silu_block = 1024
        silu_kernel[(triton.cdiv(total1, silu_block),)](y1_gn, y1_silu, total1, BLOCK=silu_block)

        # Second conv: conv2
        C_in2 = C_out1
        C_out2 = conv2_weight.shape[0]
        H2 = H1
        W2 = W1
        y2_pre = torch.empty((B, C_out2, H2, W2), device=device, dtype=dtype)

        total_out2 = B * C_out2 * H2 * W2
        grid2 = (triton.cdiv(total_out2, 1),)
        conv3x3_nchw_fp32[grid2](
            y1_silu, conv2_weight.to(dtype).contiguous(), y2_pre,
            B, C_in2, H2, W2, C_out2, H2, W2,
            BLOCK_IN=64
        )

        # GroupNorm 2
        y2_gn = torch.empty((B, C_out2, H2, W2), device=device, dtype=dtype)
        groupnorm_affine_kernel[(B * 32,)](
            y2_pre, y2_gn, norm2_weight.to(dtype).contiguous(), norm2_bias.to(dtype).contiguous(),
            B, C_out2, H2, W2, 32, self.eps, BLOCK_HW=256
        )

        # SiLU 2
        y2_silu = torch.empty((B, C_out2, H2, W2), device=device, dtype=dtype)
        total2 = y2_silu.numel()
        silu_kernel[(triton.cdiv(total2, silu_block),)](y2_gn, y2_silu, total2, BLOCK=silu_block)

        # Residual addition: add original x to final output (elementwise)
        x0_fp32 = x0
        out_add = torch.empty_like(x0_fp32, device=device, dtype=dtype)
        total_add = x0_fp32.numel()
        add_residual_kernel[(triton.cdiv(total_add, silu_block),)](
            y2_silu, x0_fp32, out_add, total_add, BLOCK=silu_block
        )

        return out_add


def run(*args):
    return ModelNew()(*args)
