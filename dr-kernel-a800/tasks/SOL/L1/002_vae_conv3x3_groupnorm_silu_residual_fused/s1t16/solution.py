import triton
import triton.language as tl

# Conv3x3 NCHW, stride=1, padding=1, no bias
@triton.jit
def conv3x3_nchw_fp32(
    x_ptr,            # *float32, input [B, C_in, H, W]
    w_ptr,            # *float32, weight [C_out, C_in, 3, 3]
    y_ptr,            # *float32, output [B, C_out, H, W]
    B: tl.constexpr,  # int
    C_in: tl.constexpr,  # int
    H_in: tl.constexpr,  # int
    W_in: tl.constexpr,  # int
    C_out: tl.constexpr,  # int
    H_out: tl.constexpr,  # int
    W_out: tl.constexpr,  # int
    BLOCK_IN: tl.constexpr,  # int
):
    n = tl.program_id(0)  # batch
    c_out = tl.program_id(1)  # output channel
    h_out = tl.program_id(2)  # output height
    w_out = tl.program_id(3)  # output width

    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels in chunks
    for ic0 in range(0, C_in, BLOCK_IN):
        offs_in = ic0 + tl.arange(0, BLOCK_IN)
        mask_ic = offs_in < C_in

        # accumulate over 3x3 window
        for kh in range(3):
            for kw in range(3):
                h_in = h_out * 1 + kh - 1  # padding=1
                w_in = w_out * 1 + kw - 1  # padding=1
                valid = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)

                # compute input offsets for vector of input channels
                # x layout: [B, C_in, H, W] -> linear index = ((n*C_in + ic)*H + h)*W + w
                x_offs = (((n * C_in + offs_in) * H_in + h_in) * W_in + w_in)
                x_mask = mask_ic & valid

                # load input vector (masked)
                x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

                # load corresponding weight scalar: w[c_out, offs_in, kh, kw]
                w_off = (c_out * C_in + offs_in) * 9 + (kh * 3 + kw)
                w_vals = tl.load(w_ptr + w_off, mask=mask_ic, other=0.0)

                # multiply-accumulate
                # w_vals is length BLOCK_IN, x_vals is length BLOCK_IN, broadcast multiply
                acc += tl.sum(w_vals[:, None] * x_vals[None, :], axis=0)

    # store output: y layout [B, C_out, H_out, W_out] -> linear index = ((n*C_out + c_out)*H_out + h_out)*W_out + w_out
    y_offset = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out
    tl.store(y_ptr + y_offset, acc)


# GroupNorm + affine, per (n, group). Assumes x is [B, C, H*W] flattened.
@triton.jit
def groupnorm_affine_fp32(
    x_ptr,          # *float32, input flattened per group [B, C, H*W]
    scale_ptr,      # *float32, per-channel scale [C]
    bias_ptr,       # *float32, per-channel bias [C]
    y_ptr,          # *float32, output [B, C, H*W]
    B: tl.constexpr,  # int
    C: tl.constexpr,  # int
    HW: tl.constexpr,  # int
    num_groups: tl.constexpr,  # int
    group_size: tl.constexpr,  # int
    eps: tl.constexpr,  # float
    BLOCK_HW: tl.constexpr,  # int
):
    n = tl.program_id(0)  # batch
    group = tl.program_id(1)  # group id in [0, num_groups)
    c0 = group * group_size  # starting channel for this group

    # compute mean and variance over channels in group and all HW
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # loop over channels in the group
    for ic in range(group_size):
        c = c0 + ic
        for hw0 in range(0, HW, BLOCK_HW):
            offs = hw0 + tl.arange(0, BLOCK_HW)
            mask = offs < HW
            x_off = ((n * C + c) * HW) + offs
            x_vals = tl.load(x_ptr + x_off, mask=mask, other=0.0)
            # reduce this chunk
            sum_val += tl.sum(x_vals, axis=0)
            sum_sq += tl.sum(x_vals * x_vals, axis=0)

    m = group_size * HW  # elements per (n, group)
    mean = sum_val / m
    var = sum_sq / m - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # second pass: normalize and apply affine, then store
    for ic in range(group_size):
        c = c0 + ic
        for hw0 in range(0, HW, BLOCK_HW):
            offs = hw0 + tl.arange(0, BLOCK_HW)
            mask = offs < HW
            x_off = ((n * C + c) * HW) + offs
            x_vals = tl.load(x_ptr + x_off, mask=mask, other=0.0)
            normed = (x_vals - mean) * inv_std
            scale = tl.load(scale_ptr + c)
            bias = tl.load(bias_ptr + c)
            y_vals = normed * scale + bias
            y_off = ((n * C + c) * HW) + offs
            tl.store(y_ptr + y_off, y_vals, mask=mask)


# Elementwise SiLU: y = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
@triton.jit
def silu_fp32_elementwise(x_ptr, y_ptr, total: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)


# Residual addition: y = a + b, elementwise on float32 tensors of same shape
@triton.jit
def add_residual_fp32(a_ptr, b_ptr, y_ptr, total: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    y = a + b
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Triton parameters can be tuned; defaults chosen for robustness
        self.block_in = 64
        self.block_hw = 256
        self.silu_block = 1024

    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        # Ensure device and dtype compatibility
        device = x.device
        dtype = torch.float32

        # First conv: y0 = conv3x3(x)
        B, C, H, W = x.shape
        C_out1 = conv1_weight.shape[0]
        H_out1 = H
        W_out1 = W
        y0 = torch.empty((B, C_out1, H_out1, W_out1), device=device, dtype=dtype)

        # Launch conv1
        grid_conv1 = (B, C_out1, H_out1, W_out1)
        conv3x3_nchw_fp32[grid_conv1](
            x.contiguous().to(dtype),
            conv1_weight.contiguous().to(dtype),
            y0,
            B, C, H, W, C_out1, H_out1, W_out1,
            BLOCK_IN=self.block_in,
            num_warps=4,
        )

        # GroupNorm 1 (num_groups=32) on y0
        num_groups = 32
        assert C_out1 % num_groups == 0, "Output channels must be divisible by num_groups (32)"
        group_size1 = C_out1 // num_groups
        y0_flat = y0.view(B, C_out1, H_out1 * W_out1).contiguous()
        y0_norm = torch.empty((B, C_out1, H_out1 * W_out1), device=device, dtype=dtype)

        grid_gn1 = (B, num_groups)
        groupnorm_affine_fp32[grid_gn1](
            y0_flat,
            norm1_weight.contiguous().to(dtype),
            norm1_bias.contiguous().to(dtype),
            y0_norm,
            B, C_out1, H_out1 * W_out1, num_groups, group_size1, eps,
            BLOCK_HW=self.block_hw,
            num_warps=4,
        )
        y0_norm = y0_norm.view(B, C_out1, H_out1, W_out1)

        # SiLU on y0_norm
        y1_silu = torch.empty_like(y0_norm, device=device, dtype=dtype)
        total1 = B * C_out1 * H_out1 * W_out1
        grid_silu1 = (triton.cdiv(total1, self.silu_block),)
        silu_fp32_elementwise[grid_silu1](y0_norm, y1_silu, total1, BLOCK=self.silu_block, num_warps=4)

        # Second conv: y1 = conv3x3(y1_silu)
        C_in2 = conv2_weight.shape[1]  # must equal C_out1
        assert C_in2 == C_out1, "conv2_weight second dim must match output channels of first conv"
        C_out2 = conv2_weight.shape[0]
        H_out2 = H_out1
        W_out2 = W_out1
        y1 = torch.empty((B, C_out2, H_out2, W_out2), device=device, dtype=dtype)

        grid_conv2 = (B, C_out2, H_out2, W_out2)
        conv3x3_nchw_fp32[grid_conv2](
            y1_silu, conv2_weight.contiguous().to(dtype), y1,
            B, C_in2, H_out2, W_out2, C_out2, H_out2, W_out2,
            BLOCK_IN=self.block_in,
            num_warps=4,
        )

        # GroupNorm 2 (num_groups=32) on y1
        assert C_out2 % num_groups == 0, "Output channels must be divisible by num_groups (32)"
        group_size2 = C_out2 // num_groups
        y1_flat = y1.view(B, C_out2, H_out2 * W_out2).contiguous()
        y1_norm = torch.empty((B, C_out2, H_out2 * W_out2), device=device, dtype=dtype)

        grid_gn2 = (B, num_groups)
        groupnorm_affine_fp32[grid_gn2](
            y1_flat,
            norm2_weight.contiguous().to(dtype),
            norm2_bias.contiguous().to(dtype),
            y1_norm,
            B, C_out2, H_out2 * W_out2, num_groups, group_size2, eps,
            BLOCK_HW=self.block_hw,
            num_warps=4,
        )
        y1_norm = y1_norm.view(B, C_out2, H_out2, W_out2)

        # SiLU on y1_norm
        y2_silu = torch.empty_like(y1_norm, device=device, dtype=dtype)
        total2 = B * C_out2 * H_out2 * W_out2
        grid_silu2 = (triton.cdiv(total2, self.silu_block),)
        silu_fp32_elementwise[grid_silu2](y1_norm, y2_silu, total2, BLOCK=self.silu_block, num_warps=4)

        # Residual addition: add original x (cast to fp32) to y2_silu. Shapes must match (B, C, H, W).
        x_fp32 = x.contiguous().to(dtype)
        residual = torch.empty_like(y2_silu, device=device, dtype=dtype)
        total_res = B * C * H * W  # must match original x shape
        grid_res = (triton.cdiv(total_res, self.silu_block),)
        add_residual_fp32[grid_res](x_fp32, y2_silu, residual, total_res, BLOCK=self.silu_block, num_warps=4)

        return residual


def run(*args):
    return ModelNew()(*args)
