import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 conv forward (no bias), stride=1, padding=1
# x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), y: (B, C_out, H, W)
@triton.jit
def conv3x3_triton(
    x_ptr, w_ptr, y_ptr,
    B, C_in, H, W, C_out,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr, BLOCK_IC: tl.constexpr,
):
    # program ids: batch, output channel tile, spatial tile
    b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_start = pid_oc * BLOCK_OC
    oc_vec = oc_start + tl.arange(0, BLOCK_OC)
    sp_start = pid_sp * BLOCK_SP

    # Create vectorized spatial indices for this tile
    h_vec = sp_start // W
    w_vec = sp_start % W
    # Initialize accumulator for [BLOCK_OC, BLOCK_SP]
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Loop over input channels in blocks
    for ic_start in range(0, C_in, BLOCK_IC):
        ic_vec = ic_start + tl.arange(0, BLOCK_IC)
        mask_ic = ic_vec < C_in

        # For each 3x3 neighborhood, accumulate contributions
        for kh in range(3):
            for kw in range(3):
                # compute input coordinates with padding=1
                hi = h_vec + kh - 1  # [BLOCK_SP]
                wi = w_vec + kw - 1  # [BLOCK_SP]
                in_range = (hi >= 0) & (hi < H) & (wi >= 0) & (wi < W) & mask_ic
                # x[b, ic, hi, wi]
                x_ptrs = x_ptr \
                    + b * x_stride_b \
                    + ic_vec[:, None] * x_stride_c \
                    + hi[None, :] * x_stride_h \
                    + wi[None, :] * x_stride_w
                x_mask = in_range[:, None]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_IC, BLOCK_SP]

                # weights: w[ic, oc, kh, kw]
                w_ptrs = w_ptr \
                    + ic_vec[:, None] * w_stride_cin \
                    + oc_vec[None, :] * w_stride_cout \
                    + kh * w_stride_kh \
                    + kw * w_stride_kw
                w_mask = mask_ic[:, None]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_IC, BLOCK_OC]

                # For each input channel in this block, accumulate into acc
                for i in range(BLOCK_IC):
                    # scalar mask for this ic
                    if (ic_start + i) < C_in:
                        x_row = x_vals[i, :]  # [BLOCK_SP]
                        w_row = w_vals[i, :]  # [BLOCK_OC]
                        # Outer product accumulate: acc += w_row[:, None] * x_row[None, :]
                        acc += w_row[:, None] * x_row[None, :]

    # store results y[b, oc, h, w]
    y_ptrs = y_ptr \
        + b * y_stride_b \
        + oc_vec[:, None] * y_stride_c \
        + h_vec[None, :] * y_stride_h \
        + w_vec[None, :] * y_stride_w
    mask_oc = oc_vec < C_out
    mask_store = mask_oc[:, None] & (h_vec < H) & (w_vec < W)
    tl.store(y_ptrs, acc, mask=mask_store)


# Triton kernel: GroupNorm forward + affine (scale, bias) + SiLU
# Assumes num_groups=32 and C % 32 == 0. Input x, output y are (B, C, H, W).
@triton.jit
def group_norm_affine_silu(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W, num_groups: tl.constexpr, eps: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    g = tl.program_id(1)

    group_channels = C // num_groups
    group_elements = group_channels * H * W

    # First pass: compute sum and sum of squares over the group
    sum_val = 0.0
    sum_sq = 0.0
    for start in range(0, group_elements, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < group_elements

        c_vec = g * group_channels + (idx // (H * W))  # channel index
        s_vec = idx % (H * W)                         # spatial index within the group

        h_vec = s_vec // W
        w_vec = s_vec % W

        x_ptrs = x_ptr + b * x_stride_b + c_vec * x_stride_c + h_vec * x_stride_h + w_vec * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        sum_val += tl.sum(x_vals, axis=0)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    mean = sum_val / group_elements
    var = sum_sq / group_elements - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine + SiLU
    for start in range(0, group_elements, BLOCK):
        idx = start + tl.arange(0, BLOCK)
        mask = idx < group_elements

        c_vec = g * group_channels + (idx // (H * W))
        s_vec = idx % (H * W)
        h_vec = s_vec // W
        w_vec = s_vec % W

        x_ptrs = x_ptr + b * x_stride_b + c_vec * x_stride_c + h_vec * x_stride_h + w_vec * x_stride_w
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        norm_vals = (x_vals - mean) * inv_std

        scale = tl.load(weight_ptr + c_vec, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + c_vec, mask=mask, other=0.0)
        z = norm_vals * scale + bias

        # SiLU: z * sigmoid(z)
        s = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * s

        y_ptrs = y_ptr + b * y_stride_b + c_vec * y_stride_c + h_vec * y_stride_h + w_vec * y_stride_w
        tl.store(y_ptrs, y_vals, mask=mask)


# Triton kernel: elementwise residual addition y = y + x (flattened)
@triton.jit
def add_residual_kernel(
    out_ptr, y_ptr, x_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    r = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, y + r, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups=32, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor,
                eps: float):
        # All tensors must be CUDA for Triton
        assert x.is_cuda, "Input tensor must be on CUDA for Triton kernels."
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Conv weights must be on CUDA."
        assert norm1_weight.is_cuda and norm1_bias.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "Norm params must be on CUDA."

        B, C, H, W = x.shape
        assert C % self.num_groups == 0, "Channels must be divisible by num_groups (32) for GroupNorm."

        # 1) Conv1: Triton 3x3 conv (no bias)
        y1 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)

        # Strides
        x_s0, x_s1, x_s2, x_s3 = x.stride()
        w1_s0, w1_s1, w1_s2, w1_s3 = conv1_weight.stride()
        y1_s0, y1_s1, y1_s2, y1_s3 = y1.stride()

        # Grid: (B, tiles over C, tiles over H*W)
        BLOCK_OC = 32
        BLOCK_SP = 64
        grid_conv1 = (B, triton.cdiv(C, BLOCK_OC), triton.cdiv(H * W, BLOCK_SP))
        conv3x3_triton[grid_conv1](
            x, conv1_weight, y1,
            B, C, H, W, C,
            x_s0, x_s1, x_s2, x_s3,
            w1_s0, w1_s1, w1_s2, w1_s3,
            y1_s0, y1_s1, y1_s2, y1_s3,
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP, BLOCK_IC=32,
            num_warps=4, num_stages=2,
        )

        # 2) GroupNorm + affine (norm1) + SiLU on y1
        group_norm_affine_silu[(B, self.num_groups)](
            y1, norm1_weight, norm1_bias, y1,
            B, C, H, W, self.num_groups, self.eps,
            y1_s0, y1_s1, y1_s2, y1_s3,
            y1_s0, y1_s1, y1_s2, y1_s3,
            BLOCK=1024,
            num_warps=4, num_stages=2,
        )

        # 3) Conv2: Triton 3x3 conv (no bias) applied to y1
        y2 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)

        y1_s0, y1_s1, y1_s2, y1_s3 = y1.stride()
        w2_s0, w2_s1, w2_s2, w2_s3 = conv2_weight.stride()
        y2_s0, y2_s1, y2_s2, y2_s3 = y2.stride()

        grid_conv2 = (B, triton.cdiv(C, BLOCK_OC), triton.cdiv(H * W, BLOCK_SP))
        conv3x3_triton[grid_conv2](
            y1, conv2_weight, y2,
            B, C, H, W, C,
            y1_s0, y1_s1, y1_s2, y1_s3,
            w2_s0, w2_s1, w2_s2, w2_s3,
            y2_s0, y2_s1, y2_s2, y2_s3,
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP, BLOCK_IC=32,
            num_warps=4, num_stages=2,
        )

        # 4) GroupNorm + affine (norm2) + SiLU on y2
        group_norm_affine_silu[(B, self.num_groups)](
            y2, norm2_weight, norm2_bias, y2,
            B, C, H, W, self.num_groups, self.eps,
            y2_s0, y2_s1, y2_s2, y2_s3,
            y2_s0, y2_s1, y2_s2, y2_s3,
            BLOCK=1024,
            num_warps=4, num_stages=2,
        )

        # 5) Residual add: y2 = y2 + x (launch Triton elementwise)
        N = B * C * H * W
        out = torch.empty_like(y2)
        add_residual_kernel[(triton.cdiv(N, 1024),)](
            out, y2, x,
            N,
            BLOCK=1024,
            num_warps=4, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
