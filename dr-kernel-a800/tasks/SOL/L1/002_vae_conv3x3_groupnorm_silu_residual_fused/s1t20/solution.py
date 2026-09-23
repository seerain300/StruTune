import torch
import triton
import triton.language as tl

# Conv3x3 NCHW, stride=1, padding=1, no bias
@triton.jit
def conv3x3_nchw_fp32(
    x_ptr,           # *float32, input [N, C_in, H, W]
    w_ptr,           # *float32, weight [C_out, C_in, 3, 3]
    y_ptr,           # *float32, output [N, C_out, H_out, W_out]
    N, C_in, H, W, C_out, H_out, W_out,
    BLOCK_IN: tl.constexpr,
    BLOCK_KH: tl.constexpr = 3,  # constexpr for kh
    BLOCK_KW: tl.constexpr = 3,  # constexpr for kw
):
    # program ids: one program per output element
    pid = tl.program_id(0)
    hw = H_out * W_out
    nc = C_out * hw
    n = pid // nc
    rem = pid % nc
    c_out = rem // hw
    rem2 = rem % hw
    h_out = rem2 // W_out
    w_out = rem2 % W_out

    if n >= N:
        return

    acc = tl.zeros((), dtype=tl.float32)

    # loop over input channels in chunks
    for c_in_start in range(0, C_in, BLOCK_IN):
        c_in_offsets = c_in_start + tl.arange(0, BLOCK_IN)
        mask_in = c_in_offsets < C_in

        # accumulate over 3x3 window with padding masks
        for kh in range(0, BLOCK_KH):
            for kw in range(0, BLOCK_KW):
                h_in = h_out + kh - 1
                w_in = w_out + kw - 1
                valid_h = (h_in >= 0) & (h_in < H)
                valid_w = (w_in >= 0) & (w_in < W)
                valid = valid_h & valid_w

                x_offsets = (((n * C_in + c_in_offsets) * H + h_in) * W + w_in)
                x_mask = mask_in & valid
                x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

                w_offsets = (((c_out * C_in + c_in_offsets) * (BLOCK_KH * BLOCK_KW)) + (kh * BLOCK_KW + kw))
                w_vals = tl.load(w_ptr + w_offsets, mask=mask_in, other=0.0)

                prod = w_vals * x_vals
                acc += tl.sum(prod, axis=0)

    y_offset = (((n * C_out + c_out) * H_out) + h_out) * W_out + w_out
    tl.store(y_ptr + y_offset, acc)


# GroupNorm with affine per (n, group)
@triton.jit
def groupnorm_affine_kernel_fp32(
    x_ptr,           # *float32, input [N, C, H, W] (conv output)
    gamma_ptr,       # *float32, scale [C]
    beta_ptr,        # *float32, bias [C]
    y_ptr,           # *float32, output [N, C, H, W]
    N, C, H, W,      # dimensions
    group_id,        # current group id for this program
    group_size,      # number of channels per group
    num_groups,      # total groups = C // group_size (here 32)
    eps,             # epsilon for numerical stability
    BLOCK_HW: tl.constexpr,
):
    c_start = group_id * group_size
    # Pass 1: compute sum and sum of squares over the group and all H*W
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    for c in range(0, group_size):
        c_idx = c_start + c
        for start in range(0, H * W, BLOCK_HW):
            offs = start + tl.arange(0, BLOCK_HW)
            mask = offs < (H * W)
            h = offs // W
            w = offs % W
            x_base = n * C + c_idx
            x_offsets = x_base * (H * W) + offs
            x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
            sum_val += tl.sum(x_vals, axis=0)
            sum_sq += tl.sum(x_vals * x_vals, axis=0)

    hw = H * W
    mean = sum_val / hw
    var = sum_sq / hw - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and apply affine, store
    for c in range(0, group_size):
        c_idx = c_start + c
        for start in range(0, H * W, BLOCK_HW):
            offs = start + tl.arange(0, BLOCK_HW)
            mask = offs < (H * W)
            h = offs // W
            w = offs % W
            x_base = n * C + c_idx
            x_offsets = x_base * (H * W) + offs
            x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
            norm = (x_vals - mean) * rstd
            gamma = tl.load(gamma_ptr + c_idx)
            beta = tl.load(beta_ptr + c_idx)
            y_vals = norm * gamma + beta
            y_base = n * C + c_idx
            y_offsets = y_base * (H * W) + offs
            tl.store(y_ptr + y_offsets, y_vals, mask=mask)


# SiLU elementwise
@triton.jit
def silu_kernel_fp32(x_ptr, y_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offs, y, mask=mask)


# Residual addition: elementwise add two float32 tensors (x and y must have same shape)
@triton.jit
def add_residual_kernel_fp32(a_ptr, b_ptr, out_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    out = a + b
    tl.store(out_ptr + offs, out, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_groups = 32
        self.eps = 1e-5

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        # Ensure CUDA tensors
        device = x.device
        assert device.type == 'cuda', "Triton kernels require CUDA tensors"
        # Cast to float32 and ensure contiguous
        x_in = x.contiguous().to(torch.float32)  # [N, C, H, W]
        N, C_in, H, W = x_in.shape
        # Prepare weights as float32 contiguous
        conv1_w = conv1_weight.contiguous().to(torch.float32)  # [C, C, 3, 3]
        conv2_w = conv2_weight.contiguous().to(torch.float32)  # [C, C, 3, 3]

        # First conv: output y1 has shape [N, C, H, W]
        y1 = torch.empty((N, C_in, H, W), device=device, dtype=torch.float32)

        # Launch conv1: one program per output element
        total = N * C_in * H * W
        grid_conv = (total,)
        conv3x3_nchw_fp32[grid_conv](
            x_in, conv1_w, y1,
            N, C_in, H, W, C_in, H, W,
            BLOCK_IN=16,
        )

        # GroupNorm1
        y1_out = torch.empty_like(y1, device=device, dtype=torch.float32)
        group_size = C_in // self.num_groups  # num_groups=32 -> group_size=2
        for group_id in range(self.num_groups):
            grid_gn = (1,)
            groupnorm_affine_kernel_fp32[grid_gn](
                y1, norm1_weight, norm1_bias, y1_out,
                N, C_in, H, W,
                group_id, group_size, self.num_groups, self.eps,
                BLOCK_HW=128,
            )

        # SiLU1
        y1_silu = torch.empty_like(y1_out, device=device, dtype=torch.float32)
        total_silu1 = y1_out.numel()
        grid_silu1 = (triton.cdiv(total_silu1, 1024),)
        silu_kernel_fp32[grid_silu1](y1_out, y1_silu, total_silu1, BLOCK=1024)

        # Second conv: output y2 has shape [N, C, H, W] (stride=1, padding=1 preserves dims and channels)
        y2 = torch.empty((N, C_in, H, W), device=device, dtype=torch.float32)
        conv3x3_nchw_fp32[grid_conv](
            y1_silu, conv2_w, y2,
            N, C_in, H, W, C_in, H, W,
            BLOCK_IN=16,
        )

        # GroupNorm2
        y2_out = torch.empty_like(y2, device=device, dtype=torch.float32)
        group_size2 = C_in // self.num_groups
        for group_id in range(self.num_groups):
            grid_gn2 = (1,)
            groupnorm_affine_kernel_fp32[grid_gn2](
                y2, norm2_weight, norm2_bias, y2_out,
                N, C_in, H, W,
                group_id, group_size2, self.num_groups, self.eps,
                BLOCK_HW=128,
            )

        # SiLU2
        y2_silu = torch.empty_like(y2_out, device=device, dtype=torch.float32)
        total_silu2 = y2_out.numel()
        grid_silu2 = (triton.cdiv(total_silu2, 1024),)
        silu_kernel_fp32[grid_silu2](y2_out, y2_silu, total_silu2, BLOCK=1024)

        # Final residual addition: original x has shape (N, C, H, W) as conv preserves dims and channels
        x_for_add = x_in  # same shape as y2_silu
        total_add = y2_silu.numel()
        final_out = torch.empty_like(y2_silu, device=device, dtype=torch.float32)
        grid_add = (triton.cdiv(total_add, 1024),)
        add_residual_kernel_fp32[grid_add](y2_silu, x_for_add, final_out, total_add, BLOCK=1024)

        return final_out


# Notes:
# - conv3x3_nchw_fp32 uses one program per output element and masks for padding, avoiding illegal memory access.
# - GroupNorm uses num_groups=32 and group_size=C//32=2


def run(*args):
    return ModelNew()(*args)
