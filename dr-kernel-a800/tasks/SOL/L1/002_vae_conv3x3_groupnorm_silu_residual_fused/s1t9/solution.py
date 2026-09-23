import torch
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_fp32(
    x_ptr,            # *f32, shape (B, C_in, H, W)
    w_ptr,            # *f32, shape (C_in, C_out, 3, 3)
    y_ptr,            # *f32, shape (B, C_out, H, W)
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_IN: tl.constexpr,
):
    # Each program computes one output element y[n, c_out, h_out, w_out]
    pid = tl.program_id(0)
    total = B * C_out * H * W
    n = pid // (C_out * H * W)
    rem = pid % (C_out * H * W)
    c_out = rem // (H * W)
    rem2 = rem % (H * W)
    h_out = rem2 // W
    w_out = rem2 % W

    acc = 0.0
    # Loop over input channels in chunks
    for ic_base in range(0, C_in, BLOCK_IN):
        for ic in range(0, BLOCK_IN):
            ic_idx = ic_base + ic
            # mask for valid input channel
            mask_ic = ic_idx < C_in
            # base pointer for input channel
            x_base = x_ptr + n * C_in * H * W + ic_idx * H * W
            # accumulate over 3x3 window with padding=1
            for kh in range(0, 3):
                ih = h_out + kh - 1
                valid_ih = (ih >= 0) & (ih < H)
                for kw in range(0, 3):
                    iw = w_out + kw - 1
                    valid_iw = (iw >= 0) & (iw < W)
                    valid = mask_ic & valid_ih & valid_iw
                    x_off = ih * W + iw
                    x_val = tl.load(x_base + x_off, mask=valid, other=0.0)
                    # load corresponding weights for this (ic, c_out, kh, kw)
                    w_off = ic_idx * C_out * 9 + c_out * 9 + kh * 3 + kw
                    w_val = tl.load(w_ptr + w_off)
                    acc += x_val * w_val

    # store result
    y_off = n * C_out * H * W + c_out * H * W + h_out * W + w_out
    tl.store(y_ptr + y_off, acc)


@triton.jit
def groupnorm_affine_kernel(
    in_ptr,           # *f32, shape (B, C, H*W) flattened
    out_ptr,          # *f32, shape (B, C, H*W) flattened
    weight_ptr,       # *f32, shape (C,) scale per channel
    bias_ptr,         # *f32, shape (C,) bias per channel
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    NUM_GROUPS: tl.constexpr, EPS: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    GROUP_SIZE = C // NUM_GROUPS
    c0 = g * GROUP_SIZE

    # Compute mean and variance over this group and all spatial positions
    sum_val = 0.0
    sum_sq = 0.0
    for ic in range(0, GROUP_SIZE):
        c = c0 + ic
        hw = H * W
        for start in range(0, hw, BLOCK_HW):
            offs = start + tl.arange(0, BLOCK_HW)
            mask = offs < hw
            h = offs // W
            w = offs % W
            in_offs = (n * C + c) * hw + offs
            x = tl.load(in_ptr + in_offs, mask=mask, other=0.0)
            sum_val += tl.sum(x, axis=0)
            sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / (GROUP_SIZE * H * W)
    var = sum_sq / (GROUP_SIZE * H * W) - mean * mean
    inv_std = 1.0 / tl.sqrt(var + EPS)

    # Normalize and apply affine, then store
    for ic in range(0, GROUP_SIZE):
        c = c0 + ic
        hw = H * W
        for start in range(0, hw, BLOCK_HW):
            offs = start + tl.arange(0, BLOCK_HW)
            mask = offs < hw
            h = offs // W
            w = offs % W
            in_offs = (n * C + c) * hw + offs
            x = tl.load(in_ptr + in_offs, mask=mask, other=0.0)
            y = (x - mean) * inv_std
            scale = tl.load(weight_ptr + c, mask=True, other=1.0)
            bias = tl.load(bias_ptr + c, mask=True, other=0.0)
            y = y * scale + bias
            out_offs = (n * C + c) * hw + offs
            tl.store(out_ptr + out_offs, y, mask=mask)


@triton.jit
def silu_kernel(in_ptr, out_ptr, TOTAL_ELEMS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL_ELEMS
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def add_residual_kernel(a_ptr, b_ptr, out_ptr, TOTAL_ELEMS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL_ELEMS
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    y = a + b
    tl.store(out_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Triton tuning constants (can be adjusted)
        self.block_in = 32
        self.block_hw = 256
        self.silu_block = 1024

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        # Assume input x is (B, C, H, W). We will compute in float32.
        B, C, H, W = x.shape
        device = x.device
        dtype = torch.float32

        # Ensure contiguity and dtype
        x0 = x.to(dtype).contiguous()

        # Dummy conv to produce a residual tensor with the same shape as x
        # Weights shape: (C_in, C_out, 3, 3). Here conv1_weight is used to produce shape (B, C, H, W).
        C_in1 = conv1_weight.shape[0]
        C_out1 = conv1_weight.shape[1]
        # Output of conv1 (used as residual input)
        x0 = x0.contiguous()  # original input as residual
        # First conv: output shape (B, C_out1, H, W)
        y1 = torch.empty((B, C_out1, H, W), device=device, dtype=dtype)
        total_out1 = B * C_out1 * H * W
        grid1 = (total_out1,)
        conv3x3_nchw_fp32[grid1](
            x0, conv1_weight.to(dtype).contiguous(),
            y1,
            B, C_in1, C_out1, H, W,
            self.block_in
        )

        # GroupNorm 1 with affine
        y1_flat = y1.view(B, C_out1, H * W).contiguous()
        gn_y1 = torch.empty_like(y1_flat, device=device, dtype=dtype)
        grid_gn1 = (B, 32)  # num_groups fixed at 32
        groupnorm_affine_kernel[grid_gn1](
            y1_flat, gn_y1,
            norm1_weight.to(dtype).contiguous(), norm1_bias.to(dtype).contiguous(),
            B, C_out1, H, W, 32, eps, self.block_hw
        )

        # SiLU 1
        silu_y1 = torch.empty_like(gn_y1, device=device, dtype=dtype)
        total1 = C_out1 * H * W
        grid_silu1 = (triton.cdiv(total1, self.silu_block),)
        silu_kernel[grid_silu1](gn_y1, silu_y1, total1, BLOCK=self.silu_block)

        # Second conv: output shape (B, C, H, W) to match residual x0
        # conv2_weight shape is (C_in2, C, 3, 3). We need C_in2 == C_out1 (consistent with common convs).
        C_in2 = conv2_weight.shape[0]
        C_out2 = conv2_weight.shape[1]
        assert C_in2 == C_out1, "For the dummy conv to match shape, conv2_weight C_in must equal conv1_weight C_out."
        y2 = torch.empty((B, C_out2, H, W), device=device, dtype=dtype)
        total_out2 = B * C_out2 * H * W
        grid2 = (total_out2,)
        conv3x3_nchw_fp32[grid2](
            silu_y1.view(B, C_out2, H, W), conv2_weight.to(dtype).contiguous(),
            y2,
            B, C_in2, C_out2, H, W,
            self.block_in
        )

        # GroupNorm 2 with affine
        y2_flat = y2.view(B, C_out2, H * W).contiguous()
        gn_y2 = torch.empty_like(y2_flat, device=device, dtype=dtype)
        grid_gn2 = (B, 32)
        groupnorm_affine_kernel[grid_gn2](
            y2_flat, gn_y2,
            norm2_weight.to(dtype).contiguous(), norm2_bias.to(dtype).contiguous(),
            B, C_out2, H, W, 32, eps, self.block_hw
        )

        # SiLU 2
        silu_y2 = torch.empty_like(gn_y2, device=device, dtype=dtype)
        total2 = C_out2 * H * W
        grid_silu2 = (triton.cdiv(total2, self.silu_block),)
        silu_kernel[grid_silu2](gn_y2, silu_y2, total2, BLOCK=self.silu_block)

        # Residual addition: add x0 (original input) to silu_y2 (flattened). Shapes must match (B, C, H, W).
        # Ensure silu_y2 is reshaped back to (B, C, H, W)
        silu_y2 = silu_y2.view(B, C_out2, H, W)
        total_final = B * C_out2 * H * W
        out = torch.empty_like(silu_y2, device=device, dtype=dtype)
        grid_add = (triton.cdiv(total_final, self.silu_block),)
        add_residual_kernel[grid_add](silu_y2, x0, out, total_final, BLOCK=self.silu_block)

        return out


def run(*args):
    return ModelNew()(*args)
