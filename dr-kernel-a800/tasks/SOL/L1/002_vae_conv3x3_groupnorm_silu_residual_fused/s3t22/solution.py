import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 conv forward (no bias), stride=1, padding=1
# x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), y: (B, C_out, H, W)
# Grid: (B, C_out, H_out, tiles over W_out). Each program handles one (b, oc, h_out) and a vector of w_out.
@triton.jit
def conv3x3_triton(
    x_ptr, w_ptr, y_ptr,
    B, C_in, H, W, C_out, H_out, W_out,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK_OC: tl.constexpr,   # e.g., 32
    BLOCK_W: tl.constexpr,    # e.g., 64
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    h_out = tl.program_id(2)
    tile_w = tl.program_id(3)

    w_vec = tile_w * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_vec < W_out

    # accumulator for each output channel (vector over BLOCK_W positions)
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    # loop over input channels in blocks
    for ic_base in range(0, C_in, BLOCK_OC):
        ic_vec = ic_base + tl.arange(0, BLOCK_OC)
        ic_mask = ic_vec < C_in

        # accumulate contributions for each 3x3 kernel position
        for kh in range(-1, 2):  # 3x3: kh in [-1,0,1]
            for kw in range(-1, 2):  # kw in [-1,0,1]
                # compute input coordinates with padding=1
                h_in = h_out + kh
                w_in = w_vec + kw
                # valid mask for x loads: always true for valid h_out, w_out (h_in in [0, H-1], w_in in [0, W-1])
                x_ptrs = x_ptr \
                         + b * x_stride_b \
                         + ic_vec[:, None] * x_stride_c \
                         + h_in * x_stride_h \
                         + w_in[None, :] * x_stride_w
                x_mask = ic_mask[:, None] & w_mask[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_OC, BLOCK_W]

                # load weights for these ic_vec and oc, at current (kh, kw)
                w_ptrs = w_ptr \
                         + ic_vec * w_stride_cin \
                         + oc * w_stride_cout \
                         + kh * w_stride_kh \
                         + kw * w_stride_kw
                w_mask = ic_mask
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_OC]

                # accumulate: for each ic, multiply x_vals[ic, :] by w_vals[ic] and add to acc
                for i in range(0, BLOCK_OC):
                    acc += x_vals[i, :] * w_vals[i]

    # store results to y for this (b, oc, h_out, w_vec)
    y_ptrs = y_ptr \
             + b * y_stride_b \
             + oc * y_stride_c \
             + h_out * y_stride_h \
             + w_vec * y_stride_w
    tl.store(y_ptrs, acc, mask=w_mask)


# Triton kernel: GroupNorm forward + affine (scale, bias) + SiLU
# Assumes num_groups=32 and C % 32 == 0.
# Grid: (B, 32). Each program handles one (b, group).
@triton.jit
def group_norm_affine_silu(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W, num_groups: tl.constexpr, eps: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,  # reduction tile
):
    b = tl.program_id(0)
    g = tl.program_id(1)
    group_channels = C // num_groups
    group_elements = group_channels * H * W

    # First pass: compute sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    for start in range(0, group_elements, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < group_elements
        ch = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr \
                 + b * x_stride_b \
                 + ch * x_stride_c \
                 + h * x_stride_h \
                 + w * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        sum_val += tl.sum(x_vals, axis=0)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    mean = sum_val / group_elements
    var = sum_sq / group_elements - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize, affine, SiLU
    for start in range(0, group_elements, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < group_elements
        ch = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        x_ptrs = x_ptr \
                 + b * x_stride_b \
                 + ch * x_stride_c \
                 + h * x_stride_h \
                 + w * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        norm_vals = (x_vals - mean) * inv_std

        scale = tl.load(weight_ptr + ch, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + ch, mask=mask, other=0.0)
        z = norm_vals * scale + bias

        s = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * s

        y_ptrs = y_ptr \
                 + b * y_stride_b \
                 + ch * y_stride_c \
                 + h * y_stride_h \
                 + w * y_stride_w
        tl.store(y_ptrs, y_vals, mask=mask)


# Triton kernel: elementwise residual add y = y + x
@triton.jit
def add_residual_kernel(out_ptr, y_ptr, x_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y_vals + x_vals, mask=mask)


# Example usage in ModelNew.forward (host code)
class ModelNew(torch.nn.Module):
    def __init__(self, eps=1e-5, num_groups=32):
        super().__init__()
        self.eps = eps
        self.num_groups = num_groups

    def forward(self, x, conv1_weight, norm1_weight, norm1_bias, conv2_weight, norm2_weight, norm2_bias):
        # Ensure tensors are on CUDA
        assert x.is_cuda and conv1_weight.is_cuda and norm1_weight.is_cuda and norm1_bias.is_cuda \
               and conv2_weight.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, \
            "All tensors must be on CUDA device for Triton kernels."

        # Ensure contiguous
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        B, C, H, W = x.shape
        Cw1 = conv1_weight.shape[0]  # input channels for conv1 must equal C
        Cw2 = conv2_weight.shape[0]  # input channels for conv2 must equal Cw1 == C
        # Output sizes for conv with stride=1, padding=1
        H_out = H
        W_out = W

        # conv1
        y1 = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)
        # choose tile sizes
        BLOCK_OC1 = 32
        BLOCK_W1 = 64
        grid1 = (B, C, H_out, triton.cdiv(W_out, BLOCK_W1))
        conv3x3_triton[grid1](
            x, conv1_weight, y1,
            B, C, H, W, C, H_out, W_out,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_OC=BLOCK_OC1, BLOCK_W=BLOCK_W1,
        )

        # GroupNorm + affine + SiLU after conv1
        y1_gn = torch.empty_like(y1)
        grid_gn1 = (B, self.num_groups)
        group_norm_affine_silu[grid_gn1](
            y1, norm1_weight, norm1_bias, y1_gn,
            B, C, H_out, W_out, self.num_groups, self.eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_gn.stride(0), y1_gn.stride(1), y1_gn.stride(2), y1_gn.stride(3),
            BLOCK=1024,  # reduction tile
        )

        # conv2
        y2 = torch.empty((B, C, H_out, W_out), device=x.device, dtype=x.dtype)
        BLOCK_OC2 = 32
        BLOCK_W2 = 64
        grid2 = (B, C, H_out, triton.cdiv(W_out, BLOCK_W2))
        conv3x3_triton[grid2](
            y1_gn, conv2_weight, y2,
            B, C, H_out, W_out, C, H_out, W_out,
            y1_gn.stride(0), y1_gn.stride(1), y1_gn.stride(2), y1_gn.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_OC=BLOCK_OC2, BLOCK_W=BLOCK_W2,
        )

        # GroupNorm + affine + SiLU after conv2
        y2_gn = torch.empty_like(y2)
        grid_gn2 = (B, self.num_groups)
        group_norm_affine_silu[grid_gn2](
            y2, norm2_weight, norm2_bias, y2_gn,
            B, C, H_out, W_out, self.num_groups, self.eps,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_gn.stride(0), y2_gn.stride(1), y2_gn.stride(2), y2_gn.stride(3),
            BLOCK=1024,
        )

        # Final residual: y2_gn + x
        out = torch.empty_like(y2_gn)
        N = out.numel()
        grid_add = (triton.cdiv(N, 1024),)
        add_residual_kernel[grid_add](out, y2_gn, x, N, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
