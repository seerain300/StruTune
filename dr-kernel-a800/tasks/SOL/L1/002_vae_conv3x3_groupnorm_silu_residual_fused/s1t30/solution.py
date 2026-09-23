import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_fp32(
    x_ptr, w_ptr, y_ptr,
    B, C_IN, C_OUT, H, W, H_OUT, W_OUT,
    BLOCK_IN: tl.constexpr
):
    # Each program computes one output element y[n, c_out, h_out, w_out]
    pid = tl.program_id(0)
    total = B * C_OUT * H_OUT * W_OUT
    if pid >= total:
        return

    # Decode indices
    n = pid // (C_OUT * H_OUT * W_OUT)
    tmp = pid % (C_OUT * H_OUT * W_OUT)
    co = tmp // (H_OUT * W_OUT)
    tmp2 = tmp % (H_OUT * W_OUT)
    h_out = tmp2 // W_OUT
    w_out = tmp2 % W_OUT

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels in chunks
    for c_start in range(0, C_IN, BLOCK_IN):
        c_offsets = c_start + tl.arange(0, BLOCK_IN)
        c_mask = c_offsets < C_IN

        # Loop over 3x3 window
        for kh in range(3):
            ih = h_out + kh - 1
            mask_ih = (ih >= 0) & (ih < H)
            for kw in range(3):
                iw = w_out + kw - 1
                mask_iw = (iw >= 0) & (iw < W)
                valid = mask_ih & mask_iw

                # Compute input offsets for this (n, c chunk, ih, iw)
                # input is NCHW: flatten offset = (((n * C_IN + c) * H + ih) * W + iw)
                x_off = (((n * C_IN + c_offsets) * H + ih) * W + iw)
                x_vals = tl.load(x_ptr + x_off, mask=c_mask & valid, other=0.0)  # [BLOCK_IN]

                # weights: w[co, c, kh, kw] -> flattened offset = (((co * C_IN + c) * 9) + (kh * 3 + kw))
                w_off = (((co * C_IN + c_offsets) * 9) + (kh * 3 + kw))
                w_vals = tl.load(w_ptr + w_off, mask=c_mask, other=0.0)  # [BLOCK_IN]

                # Accumulate
                acc += tl.sum(x_vals * w_vals, axis=0)

    # Store output y[n, co, h_out, w_out]
    y_off = (((n * C_OUT + co) * H_OUT + h_out) * W_OUT + w_out)
    tl.store(y_ptr + y_off, acc)


@triton.jit
def groupnorm_affine_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    B, C, H, W, num_groups, eps, H_OUT, W_OUT,
    BLOCK_HW: tl.constexpr
):
    # One program per (n, group)
    pid = tl.program_id(0)
    groups = B * num_groups
    if pid >= groups:
        return
    n = pid // num_groups
    group = pid % num_groups

    group_size = C // num_groups
    channels_start = group * group_size
    spatial = H_OUT * W_OUT

    # Compute sum and sum of squares over group channels and all spatial positions
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for c in range(0, group_size):
        c_abs = channels_start + c
        for s_start in range(0, spatial, BLOCK_HW):
            s_offsets = s_start + tl.arange(0, BLOCK_HW)
            s_mask = s_offsets < spatial

            h_out_vec = s_offsets // W_OUT
            w_out_vec = s_offsets % W_OUT

            x_off = (((n * C + c_abs) * H_OUT + h_out_vec) * W_OUT + w_out_vec)
            x_vals = tl.load(x_ptr + x_off, mask=s_mask, other=0.0)
            sum_val += tl.sum(x_vals, axis=0)
            sum_sq += tl.sum(x_vals * x_vals, axis=0)

    numel = group_size * spatial
    mean = sum_val / numel
    var = sum_sq / numel - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine, then store
    for c in range(0, group_size):
        c_abs = channels_start + c
        weight = tl.load(weight_ptr + c_abs)
        bias = tl.load(bias_ptr + c_abs)
        for s_start in range(0, spatial, BLOCK_HW):
            s_offsets = s_start + tl.arange(0, BLOCK_HW)
            s_mask = s_offsets < spatial

            h_out_vec = s_offsets // W_OUT
            w_out_vec = s_offsets % W_OUT

            x_off = (((n * C + c_abs) * H_OUT + h_out_vec) * W_OUT + w_out_vec)
            x_vals = tl.load(x_ptr + x_off, mask=s_mask, other=0.0)
            y_vals = (x_vals - mean) * rstd
            y_vals = y_vals * weight + bias
            tl.store(y_ptr + x_off, y_vals, mask=s_mask)


@triton.jit
def silu_kernel(in_ptr, out_ptr, total_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def add_residual_kernel(a_ptr, b_ptr, out_ptr, total_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, a + b, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups=32, eps=1e-5, block_in=32, block_hw=1024):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.block_in = block_in
        self.block_hw = block_hw

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor):
        # Ensure contiguous and fp32 for compute
        device = x.device
        x_fp32 = x.contiguous().to(torch.float32)
        B, C, H, W = x_fp32.shape
        C_out1 = conv1_weight.shape[0]
        C_in2 = C_out1
        C_out2 = conv2_weight.shape[0]

        # First conv: (B, C, H, W) -> (B, C_out1, H, W)
        y1 = torch.empty((B, C_out1, H, W), device=device, dtype=torch.float32)
        grid_conv1 = (B * C_out1 * H * W,)
        conv3x3_nchw_fp32[grid_conv1](
            x_fp32, conv1_weight.contiguous().to(torch.float32),
            y1, B, C, C_out1, H, W, H, W,
            BLOCK_IN=self.block_in
        )

        # GroupNorm1 with affine
        y1_norm = torch.empty_like(y1, device=device, dtype=torch.float32)
        grid_gn1 = (B * self.num_groups,)
        groupnorm_affine_kernel[grid_gn1](
            y1, y1_norm, norm1_weight.contiguous().to(torch.float32),
            norm1_bias.contiguous().to(torch.float32),
            B, C_out1, H, W, self.num_groups, self.eps, H, W,
            BLOCK_HW=self.block_hw
        )

        # SiLU1
        y1_silu = torch.empty_like(y1_norm, device=device, dtype=torch.float32)
        total1 = y1_norm.numel()
        grid_silu1 = (triton.cdiv(total1, 1024),)
        silu_kernel[grid_silu1](y1_norm, y1_silu, total1, 1024)

        # Second conv: (B, C_out1, H, W) -> (B, C_out2, H, W)
        y2 = torch.empty((B, C_out2, H, W), device=device, dtype=torch.float32)
        grid_conv2 = (B * C_out2 * H * W,)
        conv3x3_nchw_fp32[grid_conv2](
            y1_silu, conv2_weight.contiguous().to(torch.float32),
            y2, B, C_in2, C_out2, H, W, H, W,
            BLOCK_IN=self.block_in
        )

        # GroupNorm2 with affine
        y2_norm = torch.empty_like(y2, device=device, dtype=torch.float32)
        grid_gn2 = (B * self.num_groups,)
        groupnorm_affine_kernel[grid_gn2](
            y2, y2_norm, norm2_weight.contiguous().to(torch.float32),
            norm2_bias.contiguous().to(torch.float32),
            B, C_out2, H, W, self.num_groups, self.eps, H, W,
            BLOCK_HW=self.block_hw
        )

        # SiLU2
        y2_silu = torch.empty_like(y2_norm, device=device, dtype=torch.float32)
        total2 = y2_norm.numel()
        grid_silu2 = (triton.cdiv(total2, 1024),)
        silu_kernel[grid_silu2](y2_norm, y2_silu, total2, 1024)

        # Residual addition: add original input x (cast to fp32) to y2_silu
        total_add = y2_silu.numel()
        out = torch.empty_like(y2_silu, device=device, dtype=torch.float32)
        grid_add = (triton.cdiv(total_add, 1024),)
        add_residual_kernel[grid_add](x_fp32, y2_silu, out, total_add, 1024)

        return out


def run(*args):
    return ModelNew()(*args)
