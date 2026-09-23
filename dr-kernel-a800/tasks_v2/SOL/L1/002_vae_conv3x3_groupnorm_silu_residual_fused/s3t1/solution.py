import torch
import triton
import triton.language as tl


def _compute_out_shape(H, W):
    # Conv3x3 with padding=1, stride=1: output size equals input size
    return H, W


# Triton kernel: conv3x3 without bias
# x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), y: (B, C_out, H, W)
@triton.jit
def conv3x3_triton_generic(
    x_ptr,          # *const float
    w_ptr,          # *const float
    y_ptr,          # *float
    B: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    # Strides for x (NCHW)
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    # Strides for w (CIN, COUT, 3, 3)
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
    # Strides for y (NCHW)
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # Grid: (B, ceil(C_OUT/BLOCK_OC), ceil(H*W/BLOCK_SP))
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    total_sp = H * W
    num_oc_tiles = (C_OUT + BLOCK_OC - 1) // BLOCK_OC
    num_sp_tiles = (total_sp + BLOCK_SP - 1) // BLOCK_SP

    oc_start = pid_oc * BLOCK_OC
    oc_vec = oc_start + tl.arange(0, BLOCK_OC)
    mask_oc = oc_vec < C_OUT

    sp_start = pid_sp * BLOCK_SP
    sp_offsets = sp_start + tl.arange(0, BLOCK_SP)
    mask_sp = sp_offsets < total_sp

    # map sp_offsets to (h, w)
    h_vec = sp_offsets // W
    w_vec = sp_offsets % W

    # initialize accumulator
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # loop over input channels in blocks
    for ic_start in range(0, C_IN, BLOCK_OC):
        ic_vec = ic_start + tl.arange(0, BLOCK_OC)
        mask_ic = ic_vec < C_IN

        # For each 3x3 kernel position
        for kh in range(3):
            for kw in range(3):
                # Compute input h,w for this kernel position (padding=1, so -1/+0/+1)
                h_i = h_vec + kh - 1
                w_i = w_vec + kw - 1

                # mask for valid input indices (0..H-1, 0..W-1)
                mask_hw = (h_i >= 0) & (h_i < H) & (w_i >= 0) & (w_i < W)

                # load input tile: shape [BLOCK_SP]
                x_ptrs = x_ptr + pid_b * x_stride_b \
                           + ic_vec[:, None] * x_stride_c \
                           + h_i[None, :] * x_stride_h \
                           + w_i[None, :] * x_stride_w
                x_mask = mask_ic[:, None] & mask_sp[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

                # load weight vector for this (ic_vec, oc_vec, kh, kw): shape [BLOCK_OC]
                w_ptrs = w_ptr + ic_vec * w_stride_cin \
                           + oc_vec * w_stride_cout \
                           + kh * w_stride_kh \
                           + kw * w_stride_kw
                w_mask = mask_ic & (oc_vec < C_OUT)  # oc_vec is valid by mask_oc tile
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_IC]

                # outer product accumulate
                acc += w_vals[:, None] * x_vals  # [BLOCK_OC, BLOCK_SP]

    # store results to y
    y_ptrs = y_ptr + pid_b * y_stride_b \
               + oc_vec[:, None] * y_stride_c \
               + h_vec[None, :] * y_stride_h \
               + w_vec[None, :] * y_stride_w
    mask_store = mask_oc[:, None] & mask_sp[None, :]
    tl.store(y_ptrs, acc, mask=mask_store)


# Triton kernel: GroupNorm forward + affine (scale, bias) + SiLU
# We assume num_groups=32 and C % 32 == 0. Each program handles one (b, group).
@triton.jit
def group_norm_affine_silu(
    x_ptr,           # *const float input (B, C, H, W)
    weight_ptr,      # *const float per-channel scale (C,)
    bias_ptr,        # *const float per-channel bias (C,)
    y_ptr,           # *float output (B, C, H, W)
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    num_groups: tl.constexpr,
    eps: tl.constexpr,
    # Strides for x and y (NCHW)
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,  # reduction/block size
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

        # Map idx -> (c, h, w) within the group
        c_vec = g * group_channels + (idx // (H * W))
        hw = idx % (H * W)
        h_vec = hw // W
        w_vec = hw % W

        x_ptrs = x_ptr + b * x_stride_b + c_vec * x_stride_c + h_vec * x_stride_h + w_vec * x_stride_w
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

        c_vec = g * group_channels + (idx // (H * W))
        hw = idx % (H * W)
        h_vec = hw // W
        w_vec = hw % W

        x_ptrs = x_ptr + b * x_stride_b + c_vec * x_stride_c + h_vec * x_stride_h + w_vec * x_stride_w
        y_ptrs = y_ptr + b * y_stride_b + c_vec * y_stride_c + h_vec * y_stride_h + w_vec * y_stride_w

        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
        # normalize per channel using c_vec
        norm_vals = (x_vals - mean) * inv_std
        # affine
        scale = tl.load(weight_ptr + c_vec, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + c_vec, mask=mask, other=0.0)
        z = norm_vals * scale + bias
        # SiLU: z * sigmoid(z)
        s = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * s
        tl.store(y_ptrs, y_vals, mask=mask)


# Triton elementwise residual addition kernel: y = y + x
@triton.jit
def add_residual_kernel(
    out_ptr, y_ptr, x_ptr,
    N,  # total number of elements
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
        # Ensure CUDA and float32 compute
        assert x.is_cuda, "Input tensor must be on CUDA for Triton kernels."
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Conv weights must be on CUDA."
        assert norm1_weight.is_cuda and norm1_bias.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "Norm params must be on CUDA."

        B, C


def run(*args):
    return ModelNew()(*args)
