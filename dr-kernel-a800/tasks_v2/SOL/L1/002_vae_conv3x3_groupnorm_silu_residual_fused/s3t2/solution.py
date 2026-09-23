import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 conv without bias
# x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), y: (B, C_out, H, W)
@triton.jit
def conv3x3_triton_1(
    x_ptr,          # *const float
    w_ptr,          # *const float (conv1_weight)
    y_ptr,          # *float (output conv1)
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

    h_vec = sp_offsets // W
    w_vec = sp_offsets % W

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # loop over input channels in blocks (C_IN)
    for ic_start in range(0, C_IN, BLOCK_OC):
        ic_vec = ic_start + tl.arange(0, BLOCK_OC)
        mask_ic = ic_vec < C_IN

        # 3x3 neighborhood
        for kh in range(3):
            for kw in range(3):
                h_i = h_vec + kh - 1
                w_i = w_vec + kw - 1
                mask_hw = (h_i >= 0) & (h_i < H) & (w_i >= 0) & (w_i < W)

                x_ptrs = x_ptr + pid_b * x_stride_b \
                           + ic_vec[:, None] * x_stride_c \
                           + h_i[None, :] * x_stride_h \
                           + w_i[None, :] * x_stride_w
                x_mask = mask_ic[:, None] & mask_sp[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_IC, BLOCK_SP]

                w_ptrs = w_ptr + ic_vec * w_stride_cin \
                           + oc_vec * w_stride_cout \
                           + kh * w_stride_kh \
                           + kw * w_stride_kw
                w_mask = mask_ic & mask_oc
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_IC]

                acc += w_vals[:, None] * x_vals  # outer product accumulate

    y_ptrs = y_ptr + pid_b * y_stride_b \
               + oc_vec[:, None] * y_stride_c \
               + h_vec[None, :] * y_stride_h \
               + w_vec[None, :] * y_stride_w
    mask_store = mask_oc[:, None] & mask_sp[None, :]
    tl.store(y_ptrs, acc, mask=mask_store)


# Triton kernel: 3x3 conv without bias (for conv2)
@triton.jit
def conv3x3_triton_2(
    x_ptr,          # *const float (conv1 output or other input for conv2)
    w_ptr,          # *const float (conv2_weight)
    y_ptr,          # *float (output conv2)
    B: tl.constexpr,
    C_IN: tl.constexpr,
    C_OUT: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    # Strides
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
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

    h_vec = sp_offsets // W
    w_vec = sp_offsets % W

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    for ic_start in range(0, C_IN, BLOCK_OC):
        ic_vec = ic_start + tl.arange(0, BLOCK_OC)
        mask_ic = ic_vec < C_IN

        for kh in range(3):
            for kw in range(3):
                h_i = h_vec + kh - 1
                w_i = w_vec + kw - 1
                mask_hw = (h_i >= 0) & (h_i < H) & (w_i >= 0) & (w_i < W)

                x_ptrs = x_ptr + pid_b * x_stride_b \
                           + ic_vec[:, None] * x_stride_c \
                           + h_i[None, :] * x_stride_h \
                           + w_i[None, :] * x_stride_w
                x_mask = mask_ic[:, None] & mask_sp[None, :]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_IC, BLOCK_SP]

                w_ptrs = w_ptr + ic_vec * w_stride_cin \
                           + oc_vec * w_stride_cout \
                           + kh * w_stride_kh \
                           + kw * w_stride_kw
                w_mask = mask_ic & mask_oc
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_IC]

                acc += w_vals[:, None] * x_vals

    y_ptrs = y_ptr + pid_b * y_stride_b \
               + oc_vec[:, None] * y_stride_c \
               + h_vec[None, :] * y_stride_h \
               + w_vec[None, :] * y_stride_w
    mask_store = mask_oc[:, None] & mask_sp[None, :]
    tl.store(y_ptrs, acc, mask=mask_store)


# Triton kernel: GroupNorm forward + affine (scale, bias) + SiLU
# Assumes num_groups=32, and C % 32 == 0.
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
    num_groups: tl.constexpr,  # 32
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
        norm_vals = (x_vals - mean) * inv_std
        scale = tl.load(weight_ptr + c_vec, mask=mask, other=1.0)
        bias = tl.load(bias_ptr + c_vec, mask=mask, other=0.0)
        z = norm_vals * scale + bias
        s = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * s
        tl.store(y_ptrs, y_vals, mask=mask)


# Triton elementwise residual addition kernel: y = y + x (over flattened)
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

        B, C, H, W = x.shape
        # Ensure num_groups=32, C divisible
        assert C % self.num_groups == 0, "C must be divisible by num_groups (32)."

        # Make tensors contiguous and use float32
        x = x.contiguous().to(torch.float32)
        conv1_weight = conv1_weight.contiguous().to(torch.float32)
        conv2_weight = conv2_weight.contiguous().to(torch.float32)
        norm1_weight = norm1_weight.contiguous().to(torch.float32)
        norm1_bias = norm1_bias.contiguous().to(torch.float32)
        norm2_weight = norm2_weight.contiguous().to(torch.float32)
        norm2_bias = norm2_bias.contiguous().to(torch.float32)

        # Kernel tiling
        BLOCK_OC = 32
        BLOCK_SP = 128
        BLOCK_RED = 1024  # reduction block size for GroupNorm

        # Output buffers
        y1 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        y1n = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)

        # Launch conv1 Triton kernel
        x_stride_b, x_stride_c, x_stride_h, x_stride_w = x.stride()
        w1_stride_cin, w1_stride_cout, w1_stride_kh, w1_stride_kw = conv1_weight.stride()
        y1_stride_b, y1_stride_c, y1_stride_h, y1_stride_w = y1.stride()

        num_oc_tiles = (C + BLOCK_OC - 1) // BLOCK_OC
        num_sp_tiles = (H * W + BLOCK_SP - 1) // BLOCK_SP
        grid_conv1 = (B, num_oc_tiles, num_sp_tiles)

        conv3x3_triton_1[grid_conv1](
            x, conv1_weight, y1,
            B, C, C, H, W,  # conv1_weight output channels = C (same as input C)
            x_stride_b, x_stride_c, x_stride_h, x_stride_w,
            w1_stride_cin, w1_stride_cout, w1_stride_kh, w1_stride_kw,
            y1_stride_b, y1_stride_c, y1_stride_h, y1_stride_w,
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
            num_warps=4, num_stages=2,
        )

        # GroupNorm1 + SiLU
        group_norm_affine_silu[(B, self.num_groups)](
            y1, norm1_weight, norm1_bias, y1n,
            B, C, H, W, self.num_groups, self.eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1n.stride(0), y1n.stride(1), y1n.stride(2), y1n.stride(3),
            BLOCK=BLOCK_RED,
            num_warps=4, num_stages=2,
        )

        # Conv2 Triton kernel
        y2 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)

        # Use y1n as input for conv2
        y1n_stride_b, y1n_stride_c, y1n_stride_h, y1n_stride_w = y1n.stride()
        w2_stride_cin, w2_stride_cout, w2_stride_kh, w2_stride_kw = conv2_weight.stride()
        y2_stride_b, y2_stride_c, y2_stride_h, y2_stride_w = y2.stride()

        grid_conv2 = (B, num_oc_tiles, num_sp_tiles)

        conv3x3_triton_2[grid_conv2](
            y1n, conv2_weight, y2,
            B, C, C, H, W,  # conv2 output channels = C
            y1n_stride_b, y1n_stride_c, y1n_stride_h, y1n_stride_w,
            w2_stride_cin, w2_stride_cout, w2_stride_kh, w2_stride_kw,
            y2_stride_b, y2_stride_c, y2_stride_h, y2_stride_w,
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
            num_warps=4, num_stages=2,
        )

        # GroupNorm2 + SiLU
        y2n = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)

        group_norm_affine_silu[(B, self.num_groups)](
            y2, norm2_weight, norm2_bias, y2n,
            B, C, H, W, self.num_groups, self.eps,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2n.stride(0), y2n.stride(1), y2n.stride(2), y2n.stride(3),
            BLOCK=BLOCK_RED,
            num_warps=4, num_stages=2,
        )

        # Residual add (x is (B,C,H,W), y2n is conv2 output normalized+SiLU)
        # Final output = y2n + x
        out = torch.empty_like(y2n)

        N = B * C * H * W
        grid_add = (triton.cdiv(N, 1024),)

        add_residual_kernel[grid_add](
            out, y2n, x,
            N,
            BLOCK=1024,
            num_warps=4, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
