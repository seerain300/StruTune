import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 conv forward (no bias), stride=1, padding=1
# x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), y: (B, C_out, H, W)
# Grid: (B, C_out, H_out, tiles over W_out). Each program handles one (b, oc, h_out) and a vector of W_out positions.
@triton.jit
def conv3x3_triton(
    x_ptr, w_ptr, y_ptr,
    B, C_in, H, W, C_out, H_out, W_out,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK_OC: tl.constexpr,  # e.g., 32
    BLOCK_W: tl.constexpr,   # e.g., 128
):
    b = tl.program_id(0)
    oc_base = tl.program_id(1)
    h_out = tl.program_id(2)
    tile_id = tl.program_id(3)

    # vector of output width indices for this tile
    w_out_vec = tile_id * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_out_vec < W_out

    # accumulator over output channels (vector of length BLOCK_OC)
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # loop over input channels in blocks
    for ic_base in range(0, C_in, BLOCK_OC):
        oc_vec = oc_base + tl.arange(0, BLOCK_OC)
        oc_mask = oc_vec < C_out

        # reinitialize acc for these oc
        acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

        # loop over 3x3 neighborhood (padding=1, stride=1)
        for kh in range(3):
            for kw in range(3):
                # compute input coords with padding
                h_in = h_out + (kh - 1)
                w_in_vec = w_out_vec + (kw - 1)

                # Load weights: shape [BLOCK_OC]
                w_ptrs = w_ptr \
                         + (ic_base + tl.arange(0, BLOCK_OC)) * w_stride_cin \
                         + oc_vec * w_stride_cout \
                         + kh * w_stride_kh \
                         + kw * w_stride_kw
                # mask for valid ic and oc
                w_mask_vec = ((ic_base + tl.arange(0, BLOCK_OC)) < C_in) & oc_mask
                w_vals = tl.load(w_ptrs, mask=w_mask_vec, other=0.0)  # [BLOCK_OC]

                # For each ic in the block, load x and accumulate
                for ic in range(BLOCK_OC):
                    ic_idx = ic_base + ic
                    ic_valid = ic_idx < C_in
                    # pointers for this ic
                    x_ptrs = x_ptr \
                             + b * x_stride_b \
                             + ic_idx * x_stride_c \
                             + h_in * x_stride_h \
                             + w_in_vec * x_stride_w
                    x_vals = tl.load(x_ptrs, mask=w_mask & ic_valid, other=0.0)  # [BLOCK_W]
                    # accumulate: acc[ic] += sum(x_vals * w_vals[ic])
                    acc += x_vals * (w_vals[ic] if ic_valid else 0.0)

        # Store results for this (b, h_out, w_out_vec) across oc_vec
        for oc_idx in range(BLOCK_OC):
            oc = oc_base + oc_idx
            oc_valid = oc < C_out
            y_ptrs = y_ptr \
                     + b * y_stride_b \
                     + oc * y_stride_c \
                     + h_out * y_stride_h \
                     + w_out_vec * y_stride_w
            # write only if oc_valid and w_mask
            tl.store(y_ptrs, acc[oc_idx], mask=w_mask & oc_valid)

    # Note: When C_out > BLOCK_OC, we should loop over oc_base in steps of BLOCK_OC in the host code.
    # Here we assume the host sets oc_base to cover all channels.


# Triton kernel: GroupNorm (num_groups=32) + affine + SiLU
# x: (B, C, H, W), weight: (C,), bias: (C,), y: (B, C, H, W)
# Grid: (B, num_groups). Each program handles one (batch, group).
@triton.jit
def group_norm_affine_silu(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W, num_groups, eps,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,  # reduction tile size
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

    # Second pass: normalize, affine, SiLU, store
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


class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        conv1_weight: torch.Tensor,
        norm1_weight: torch.Tensor,
        norm1_bias: torch.Tensor,
        conv2_weight: torch.Tensor,
        norm2_weight: torch.Tensor,
        norm2_bias: torch.Tensor,
        eps: float,
    ):
        # Ensure CUDA tensors
        assert x.is_cuda and conv1_weight.is_cuda and norm1_weight.is_cuda and norm1_bias.is_cuda \
            and conv2_weight.is_cuda and norm2_weight.is_cuda and norm2_bias.is_cuda, "Tensors must be on CUDA."
        # Shapes
        B, C, H, W = x.shape

        # conv1
        y1 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        BLOCK_OC = 32
        BLOCK_W = 128
        grid1 = (B, C, H, triton.cdiv(W, BLOCK_W))
        conv3x3_triton[grid1](
            x, conv1_weight, y1,
            B, C, H, W, C, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_OC=BLOCK_OC, BLOCK_W=BLOCK_W,
        )

        # GroupNorm1 + affine + SiLU (Triton)
        y1_gn = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid_gn1 = (B, 32)
        group_norm_affine_silu[grid_gn1](
            y1, norm1_weight, norm1_bias, y1_gn,
            B, C, H, W, 32, eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_gn.stride(0), y1_gn.stride(1), y1_gn.stride(2), y1_gn.stride(3),
            BLOCK=1024,
        )

        # conv2
        y2 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid2 = (B, C, H, triton.cdiv(W, BLOCK_W))
        conv3x3_triton[grid2](
            y1_gn, conv2_weight, y2,
            B, C, H, W, C, H, W,
            y1_gn.stride(0), y1_gn.stride(1), y1_gn.stride(2), y1_gn.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_OC=BLOCK_OC, BLOCK_W=BLOCK_W,
        )

        # GroupNorm2 + affine + SiLU (Triton)
        y2_gn = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid_gn2 = (B, 32)
        group_norm_affine_silu[grid_gn2](
            y2, norm2_weight, norm2_bias, y2_gn,
            B, C, H, W, 32, eps,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_gn.stride(0), y2_gn.stride(1), y2_gn.stride(2), y2_gn.stride(3),
            BLOCK=1024,
        )

        # Residual add (Triton)
        out = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        N = B * C * H * W
        grid_add = (triton.cdiv(N, 1024),)
        add_residual_kernel[grid_add](out, y2_gn, x, N, BLOCK=1024)

        return out


def run(*args):
    return ModelNew()(*args)
