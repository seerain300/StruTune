import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_fp32(x_ptr, w_ptr, y_ptr,
                      B, C_in, H_in, W_in, C_out, H_out, W_out,
                      BLOCK_IN: tl.constexpr):
    # Each program computes one output element: y[n, c_out, h_out, w_out]
    pid = tl.program_id(axis=0)

    # Compute n, c_out, h_out, w_out from program id
    total_per_n = C_out * H_out * W_out
    n = pid // total_per_n
    rem = pid % total_per_n
    c_out = rem // (H_out * W_out)
    rem2 = rem % (H_out * W_out)
    h_out = rem2 // W_out
    w_out = rem2 % W_out

    # Accumulator
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels in chunks
    for ic0 in range(0, C_in, BLOCK_IN):
        offs_ic = ic0 + tl.arange(0, BLOCK_IN)  # vector of input channels
        mask_ic = offs_ic < C_in

        # Loop over 3x3 neighborhood with masks for padding
        for kh in range(0, 3):
            h_in = h_out * 1 + kh - 1  # padding=1
            mask_h = (h_in >= 0) & (h_in < H_in)
            for kw in range(0, 3):
                w_in = w_out * 1 + kw - 1  # padding=1
                mask_w = (w_in >= 0) & (w_in < W_in)

                # Load input vector x[n, offs_ic, h_in, w_in]
                x_off = (
                    n * (C_in * H_in * W_in)
                    + offs_ic * (H_in * W_in)
                    + h_in * W_in
                    + w_in
                )
                mask_x = mask_ic & mask_h & mask_w
                x_vals = tl.load(x_ptr + x_off, mask=mask_x, other=0.0)  # shape: [BLOCK_IN]

                # Load weight vector w[offs_ic, c_out, kh, kw]
                w_off = (
                    offs_ic * (C_out * 9)
                    + c_out * 9
                    + kh * 3 + kw
                )
                w_vals = tl.load(w_ptr + w_off, mask=mask_ic, other=0.0)  # shape: [BLOCK_IN]

                # Accumulate: sum over BLOCK_IN
                acc += tl.sum(x_vals * w_vals, axis=0)

    # Write output: y[n, c_out, h_out, w_out]
    y_off = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h_out * W_out + w_out
    tl.store(y_ptr + y_off, acc)


@triton.jit
def groupnorm_affine_kernel(x_ptr, y_ptr, weight_ptr, bias_ptr,
                            B, C, H, W, num_groups, eps,
                            BLOCK_HW: tl.constexpr):
    # Each program handles one (n, group) pair
    pid = tl.program_id(axis=0)
    n = pid // num_groups
    group = pid % num_groups

    channels_per_group = C // num_groups
    c_start = group * channels_per_group

    # Pass 1: compute sum and sum of squares over group
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)
    for ic in range(0, channels_per_group):
        c = c_start + ic
        for p in range(0, H * W, BLOCK_HW):
            offs = p + tl.arange(0, BLOCK_HW)
            mask = offs < (H * W)
            hw = H * W
            x_off = n * (C * hw) + c * hw + offs
            x_vals = tl.load(x_ptr + x_off, mask=mask, other=0.0)
            sum_val += tl.sum(x_vals, axis=0)
            sumsq_val += tl.sum(x_vals * x_vals, axis=0)

    hw = H * W
    mean = sum_val / (channels_per_group * hw)
    var = sumsq_val / (channels_per_group * hw) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and apply affine, write to y
    for ic in range(0, channels_per_group):
        c = c_start + ic
        scale = tl.load(weight_ptr + c)  # per-channel scale
        bias = tl.load(bias_ptr + c)     # per-channel bias
        for p in range(0, H * W, BLOCK_HW):
            offs = p + tl.arange(0, BLOCK_HW)
            mask = offs < (H * W)
            x_off = n * (C * hw) + c * hw + offs
            x_vals = tl.load(x_ptr + x_off, mask=mask, other=0.0)
            y_vals = (x_vals - mean) * inv_std
            y_vals = y_vals * scale + bias
            y_off = n * (C * hw) + c * hw + offs
            tl.store(y_ptr + y_off, y_vals, mask=mask)


@triton.jit
def silu_kernel(x_ptr, y_ptr, total_elems, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def add_residual_fp32(a_ptr, b_ptr, out_ptr, total_elems, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elems
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    out = a + b
    tl.store(out_ptr + offs, out, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Kernel tuning parameters
        self.conv_block_in = 32
        self.groupnorm_block_hw = 1024
        self.silu_block = 1024
        self.add_block = 1024

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        # Ensure device and dtype
        assert x.is_cuda, "Input must be on CUDA for Triton kernels"
        device = x.device
        dtype = torch.float32

        # Save original input for residual
        x0 = x.to(dtype).contiguous()

        # Shapes
        B, C_in, H_in, W_in = x0.shape
        # conv1: output channels = conv1_weight.shape[0]
        C_out1 = conv1_weight.shape[0]
        # conv2: output channels = conv2_weight.shape[0]
        C_out2 = conv2_weight.shape[0]

        # First path: Conv3x3 -> GroupNorm -> SiLU
        x1 = torch.empty((B, C_in, H_in, W_in), device=device, dtype=dtype)  # input to conv1
        # Launch conv1
        grid_conv1 = (B * C_in * H_in * W_in,)
        conv3x3_nchw_fp32[grid_conv1](
            x0, conv1_weight.to(dtype).contiguous(), x1,
            B, C_in, H_in, W_in, C_out1, H_in, W_in,
            BLOCK_IN=self.conv_block_in
        )

        # GroupNorm 1
        y1_gn = torch.empty_like(x1, device=device, dtype=dtype)
        num_groups = 32
        grid_gn1 = (B * num_groups,)
        groupnorm_affine_kernel[grid_gn1](
            x1, y1_gn, norm1_weight.to(dtype).contiguous(), norm1_bias.to(dtype).contiguous(),
            B, C_out1, H_in, W_in, num_groups, eps,
            BLOCK_HW=self.groupnorm_block_hw
        )

        # SiLU 1
        y1_silu = torch.empty_like(y1_gn, device=device, dtype=dtype)
        total1 = y1_gn.numel()
        grid_silu1 = (triton.cdiv(total1, self.silu_block),)
        silu_kernel[grid_silu1](y1_gn, y1_silu, total1, BLOCK=self.silu_block)

        # Second path: Conv3x3 -> GroupNorm -> SiLU
        x2 = torch.empty((B, C_out1, H_in, W_in), device=device, dtype=dtype)  # input to conv2
        grid_conv2 = (B * C_out1 * H_in * W_in,)
        conv3x3_nchw_fp32[grid_conv2](
            y1_silu, conv2_weight.to(dtype).contiguous(), x2,
            B, C_out1, H_in, W_in, C_out2, H_in, W_in,
            BLOCK_IN=self.conv_block_in
        )

        # GroupNorm 2
        y2_gn = torch.empty_like(x2, device=device, dtype=dtype)
        grid_gn2 = (B * num_groups,)
        groupnorm_affine_kernel[grid_gn2](
            x2, y2_gn, norm2_weight.to(dtype).contiguous(), norm2_bias.to(dtype).contiguous(),
            B, C_out2, H_in, W_in, num_groups, eps,
            BLOCK_HW=self.groupnorm_block_hw
        )

        # SiLU 2
        y2_silu = torch.empty_like(y2_gn, device=device, dtype=dtype)
        total2 = y2_gn.numel()
        grid_silu2 = (triton.cdiv(total2, self.silu_block),)
        silu_kernel[grid_silu2](y2_gn, y2_silu, total2, BLOCK=self.silu_block)

        # Residual addition: add original input x0 to final output
        total_add = x0.numel()
        out = torch.empty_like(x0, device=device, dtype=dtype)
        grid_add = (triton.cdiv(total_add, self.add_block),)
        add_residual_fp32[grid_add](x0, y2_silu, out, total_add, BLOCK=self.add_block)

        return out


def run(*args):
    return ModelNew()(*args)
