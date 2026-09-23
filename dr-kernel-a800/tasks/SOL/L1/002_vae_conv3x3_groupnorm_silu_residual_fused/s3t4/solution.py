import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 conv forward (no bias), stride=1, padding=1
# x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), y: (B, C_out, H, W)
# Grid: (B, C_out, ceil(H*W / BLOCK_SP))
# Each program handles one (b, oc) and a tile of spatial positions of size BLOCK_SP.
# We process all output channels of this tile at once. Note: in this simple version, we set tiles = 1
# and loop over all spatial positions within the program. This keeps the code simple and correct.
@triton.jit
def conv3x3_triton(
    x_ptr, w_ptr, y_ptr,
    B, C_in, H, W, C_out,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK_OC: tl.constexpr,  # number of output channels handled per program
    BLOCK_SP: tl.constexpr,  # number of spatial elements handled per program
):
    b = tl.program_id(0)
    oc_start = tl.program_id(1)
    # We set tiles to 1 in grid: third dimension is 0 (no tiling over spatial)
    # Compute spatial offsets 0..BLOCK_SP-1
    sp_off = tl.arange(0, BLOCK_SP)  # vector of spatial indices within this program

    # Prepare oc vector
    oc_vec = oc_start + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]

    # Output spatial mapping to output H, W (H_out = H, W_out = W due to padding=1, stride=1)
    # We'll iterate over all H_out*W_out positions. This kernel handles a tile of size BLOCK_SP.
    total_sp = H * W

    # Accumulator for output: [BLOCK_OC, BLOCK_SP]
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Loop over input channels in blocks
    ic_base = 0
    while ic_base < C_in:
        ic_vec = ic_base + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
        # Broadcast pointers for w and x loads
        # For each spatial position in the tile
        # We process all H*W positions. Use a loop for clarity (no tile over spatial).
        for sp_i in range(0, total_sp):
            h_out = sp_i // W
            w_out = sp_i % W

            # Build x pointers for current (b, ic_vec, h_out+kh-1, w_out+kw-1) over 3x3 neighborhood
            # For kh, kw in [0,2]
            for kh in range(0, 3):
                h_in = h_out + kh - 1  # -1 due to padding=1
                for kw in range(0, 3):
                    w_in = w_out + kw - 1

                    x_ptrs = x_ptr \
                              + b * x_stride_b \
                              + ic_vec[:, None] * x_stride_c \
                              + h_in * x_stride_h \
                              + w_in * x_stride_w

                    # Mask for valid ic and spatial (always valid since our H_out=H and W_out=W)
                    mask_ic = ic_vec < C_in  # shape [BLOCK_OC]
                    mask_sp = sp_i < total_sp  # scalar, but we need per-element mask for [BLOCK_OC, 1]
                    # We load 1D along ic_vec; mask is per ic only. For spatial, since we loop sp_i, we treat as valid.
                    x_vals = tl.load(x_ptrs, mask=mask_ic[:, None], other=0.0)  # [BLOCK_OC, 1]

                    # Load w for current ic-block and oc-block
                    w_ptrs = w_ptr \
                              + ic_base * w_stride_cin \
                              + oc_vec[:, None] * w_stride_cout \
                              + kh * w_stride_kh \
                              + kw * w_stride_kw

                    mask_w = (ic_base + tl.arange(0, BLOCK_OC)) < C_in  # equivalent to mask_ic if ic_base < C_in
                    w_vals = tl.load(w_ptrs, mask=mask_w[:, None], other=0.0)  # [BLOCK_OC, 1]

                    # Accumulate: w_vals [BLOCK_OC,1] * x_vals [BLOCK_OC,1] -> [BLOCK_OC,1], broadcast along SP
                    acc += w_vals * x_vals

        ic_base += BLOCK_OC

    # Store results to y for oc_vec and spatial positions
    # y[b, oc, h_out, w_out] = acc[oc, sp_i]
    # Build y pointers for the tile of spatial positions
    for sp_i in range(0, total_sp):
        h_out = sp_i // W
        w_out = sp_i % W

        y_ptrs = y_ptr \
                 + b * y_stride_b \
                 + oc_vec[:, None] * y_stride_c \
                 + h_out * y_stride_h \
                 + w_out * y_stride_w

        mask_store = (oc_vec[:, None] < C_out) & (sp_i < total_sp)
        tl.store(y_ptrs, acc, mask=mask_store)


# Triton kernel: GroupNorm + affine (scale, bias) + SiLU per (batch, group)
# Assumes num_groups is a constexpr (we use 32). C % num_groups == 0.
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

        # Map linear idx to (channel, h, w)
        ch = g * group_channels + (idx // (H * W))
        sp = idx % (H * W)
        h = sp // W
        w = sp % W

        # Compute pointers to x[b, ch, h, w]
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

        # SiLU: z * sigmoid(z)
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
        # Ensure CUDA tensors and dtype float32 for numerical stability
        assert x.is_cuda, "Input tensor must be on CUDA for Triton kernels."
        assert conv1_weight.is_cuda and conv2_weight.is_cuda, "Conv weights must be on CUDA."
        assert norm1_weight.is_cuda and norm1_bias.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "Norm params must be on CUDA."

        # Cast to float32 for Triton compute
        x = x.contiguous().float()
        conv1_weight = conv1_weight.contiguous().float()
        conv2_weight = conv2_weight.contiguous().float()
        norm1_weight = norm1_weight.contiguous().float()
        norm1_bias = norm1_bias.contiguous().float()
        norm2_weight = norm2_weight.contiguous().float()
        norm2_bias = norm2_bias.contiguous().float()

        B, C, H, W = x.shape
        Cw1 = conv1_weight.shape[0]  # in_channels for conv1
        Cw2 = conv2_weight.shape[0]  # in_channels for conv2
        assert Cw1 == C and Cw2 == C, "Input channels must match for both convs."

        # Allocate outputs
        y1 = torch.empty_like(x)  # conv1 output
        y2 = torch.empty_like(x)  # conv2 output

        # Launch conv1: (B, C, H, W) -> (B, C, H, W)
        # Grid: (B, C, 1)  — process all spatial in-kernel
        conv3x3_triton[(B, C, 1)](
            x, conv1_weight, y1,
            B, C, H, W, C,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_OC=64,  # tile over output channels, adjust if needed
            BLOCK_SP=1024,  # tile over spatial; we loop over all H*W in-kernel
            num_warps=4, num_stages=2,
        )

        # GroupNorm + affine + SiLU for y1, num_groups=32
        y1_norm = torch.empty_like(y1)
        group_norm_affine_silu[(B, self.num_groups)](
            y1, norm1_weight, norm1_bias, y1_norm,
            B, C, H, W,
            self.num_groups, self.eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            BLOCK=1024,  # reduction tile
            num_warps=4, num_stages=2,
        )

        # Launch conv2: (B, C, H, W) -> (B, C, H, W), input is y1_norm
        conv3x3_triton[(B, C, 1)](
            y1_norm, conv2_weight, y2,
            B, C, H, W, C,
            y1_norm.stride(0), y1_norm.stride(1), y1_norm.stride(2), y1_norm.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_OC=64,
            BLOCK_SP=1024,
            num_warps=4, num_stages=2,
        )

        # GroupNorm + affine + SiLU for y2
        y2_norm = torch.empty_like(y2)
        group_norm_affine_silu[(B, self.num_groups)](
            y2, norm2_weight, norm2_bias, y2_norm,
            B, C, H, W,
            self.num_groups, self.eps,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_norm.stride(0), y2_norm.stride(1), y2_norm.stride(2), y2_norm.stride(3),
            BLOCK=1024,
            num_warps=4, num_stages=2,
        )

        # Residual add: out = y2_norm + x
        out = torch.empty_like(x)
        N = B * C * H * W
        add_residual_kernel[(triton.cdiv(N, 1024),)](
            out, y2_norm, x,
            N,
            BLOCK=1024,
            num_warps=4, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
