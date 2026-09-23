import torch
import triton
import triton.language as tl


# Triton kernel: 3x3 conv forward (no bias), stride=1, padding=1
# x: (B, C_in, H, W), w: (C_in, C_out, 3, 3), y: (B, C_out, H, W)
# Grid: (B, ceil_div(C_out, BLOCK_OC), tiles over H*W)
@triton.jit
def conv3x3_triton(
    x_ptr, w_ptr, y_ptr,
    B, C_in, H, W, C_out,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK_OC: tl.constexpr,  # e.g., 32
    BLOCK_SP: tl.constexpr,  # e.g., 256
):
    b = tl.program_id(0)
    oc_chunk = tl.program_id(1)
    tile_id = tl.program_id(2)

    # compute oc range for this chunk
    oc_start = oc_chunk * BLOCK_OC
    oc_vec = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_vec < C_out

    # tile over spatial dimension
    start_sp = tile_id * BLOCK_SP
    sp = start_sp + tl.arange(0, BLOCK_SP)
    sp_mask = sp < (H * W)

    # map sp to (h, w)
    h = sp // W
    w = sp % W

    # initialize accumulator for [BLOCK_OC, BLOCK_SP]
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # loop over input channels in blocks
    for ic_base in range(0, C_in, BLOCK_OC):
        # inner loop over channels in this block
        for k in range(0, BLOCK_OC):
            ic = ic_base + k
            valid_ic = ic < C_in

            # 3x3 neighborhood with padding=1
            for kh in range(3):
                for kw in range(3):
                    ih = h + (kh - 1)
                    iw = w + (kw - 1)

                    # 2D pointer for x for this (b, ic, ih, iw), vectorized over sp and oc_vec
                    x_ptrs = x_ptr \
                             + b * x_stride_b \
                             + ic * x_stride_c \
                             + ih * x_stride_h \
                             + iw * x_stride_w \
                             + oc_vec[:, None] * 0  # oc_vec is not a dimension in x, just broadcast
                    # We need to broadcast ic, h, w across oc_vec. Triton supports broadcasting via [:, None] and [None, :].
                    # Correct x pointer should be: b*stride_b + ic*stride_c + ih*stride_h + iw*stride_w, independent of oc.
                    # However, we load x for each (ic, ih, iw) and multiply by weight for each oc.
                    # The above line is incorrect: we should form 2D pointers for x using broadcasting properly.

                    # Fix: form 2D pointer for x using broadcasting for oc and sp
                    # Pointer for x at (b, ic, ih, iw), we don't use oc in x (since we're loading per ic, per sp),
                    # but we need to load x_vals of shape [BLOCK_SP] for each (ic, kh, kw).
                    # We can compute x_ptrs as a 1D pointer vector for sp only, then load x_vals.

                    # Compute 1D pointer for x for this (ic, kh, kw), vectorized over sp
                    x_ptrs_1d = x_ptr \
                                + b * x_stride_b \
                                + ic * x_stride_c \
                                + (h + (kh - 1)) * x_stride_h \
                                + (w + (kw - 1)) * x_stride_w

                    x_mask_1d = sp_mask
                    x_vals = tl.load(x_ptrs_1d, mask=x_mask_1d, other=0.0)  # [BLOCK_SP]

                    # load weight for this (ic, oc_vec, kh, kw)
                    w_ptrs = w_ptr \
                             + ic * w_stride_cin \
                             + oc_vec * w_stride_cout \
                             + kh * w_stride_kh \
                             + kw * w_stride_kw  # oc_vec may be out-of-range if valid_ic=False; handle by mask
                    w_mask = oc_mask
                    w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_OC]

                    # outer product and accumulate into acc
                    # acc[:, j] += w_vals * x_vals[j]
                    # Implement outer product add
                    for jj in range(BLOCK_SP):
                        if (start_sp + jj) < (H * W):
                            # contribution for each oc in this chunk
                            # Note: we need to multiply w_vals (BLOCK_OC) with x_vals[jj] and add to each row
                            # We can vectorize by broadcasting: acc += w_vals[:, None] * x_vals[None, :]
                            # But x_vals is 1D; we need to broadcast x_vals[jj] across oc axis.
                            # Do it elementwise per oc:
                            # However, Triton does not support Python for-loops over dynamic ranges directly.
                            # Instead, we'll rely on broadcasting via tl.dot-like accumulation using vectorized operations.
                            # Simpler approach: construct a 1D vector for this jj and add to acc.
                            pass  # we need a better way to implement outer product; fix below

                    # We'll implement outer product explicitly:
                    # Loop over k2 in 0..BLOCK_OC-1 for w_vals and add to each row of acc[:, jj]
                    # To do that, we need to access x_vals[jj] for each jj. Instead, we use vectorized broadcast:
                    # Create x_col = x_vals[:, None] then multiply with w_vals[None, :]. But x_vals is 1D.
                    # We need a 2D broadcast: create a 2D vector by repeating x_vals across oc dimension.
                    # Triton allows elementwise multiply; we can compute contribution per oc:
                    # For each oc in this chunk, contribution = w_vals[k] * x_vals[jj]
                    # Then add to acc[row_k, jj] where row_k = oc_vec[k]?
                    # The above is unclear. Fix by using an explicit outer accumulation loop:
                    # For jj in 0..BLOCK_SP-1: acc[:, jj] += w_vals * x_vals[jj]

    # Now, perform the outer accumulation properly
    # For each sp position jj in this tile, compute contribution = w_vals * x_vals[jj] and add to acc[:, jj]
    for jj in range(BLOCK_SP):
        if (start_sp + jj) < (H * W):
            x_item = x_vals[jj]  # scalar
            # Multiply each w_vals with x_item and add to corresponding rows of acc
            # We need to map k to oc_vec[k] row index. However, acc is indexed by oc_vec, so acc[:, jj] corresponds to oc rows.
            # We can iterate over BLOCK_OC and add to acc[row_k, jj].
            # Note: k is a compile-time unrolled loop here since we loop over BLOCK_OC elements of w_vals.
            # We need to know which oc each w_vals belongs to. We loop k over BLOCK_OC and map to oc_start + k (masked by oc_mask).
            for k in range(BLOCK_OC):
                oc_k = oc_start + k
                if oc_k < C_out:
                    # Add w_vals[k] * x_item to acc[oc_k, jj]
                    # acc is a 2D tensor with shape [BLOCK_OC, BLOCK_SP]; we access acc[k, jj].
                    # However, Triton doesn't allow dynamic indexing like this; instead, we must use pointer arithmetic to store.
                    # Triton supports tl.store to a pointer computed at runtime.
                    # Compute y_ptrs for this (b, oc_k, h, w) where h,w correspond to sp[jj]
                    h_j = (start_sp + jj) // W
                    w_j = (start_sp + jj) % W
                    y_ptrs = y_ptr \
                             + b * y_stride_b \
                             + oc_k * y_stride_c \
                             + h_j * y_stride_h \
                             + w_j * y_stride_w
                    contrib = w_vals[k] * x_item
                    # Store contrib into y at [oc_k, h_j, w_j]
                    tl.store(y_ptrs, contrib, mask=oc_mask[k])

                    # Accumulate into acc[k, jj] as well (if we had acc), but we don't have an acc tensor to store into.
                    # Instead, we directly stored into y. The outer accumulation is done by writing to y per (oc, sp).
                    # So we don't need to maintain acc in memory; we write results directly to y.


# Note: The above conv kernel is still a bit cumbersome. A simpler, robust approach is:
# - For each (b, oc, tile), loop over input channels and 3x3 neighborhood, compute acc vector [BLOCK_SP], then store to y per oc.
# Let's implement that properly.

# Re-implement conv3x3_triton cleanly:
@triton.jit
def conv3x3_triton_clean(
    x_ptr, w_ptr, y_ptr,
    B, C_in, H, W, C_out,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_cin, w_stride_cout, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK_SP: tl.constexpr,  # e.g., 256
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    tile_id = tl.program_id(2)

    start_sp = tile_id * BLOCK_SP
    sp = start_sp + tl.arange(0, BLOCK_SP)
    sp_mask = sp < (H * W)

    h = sp // W
    w = sp % W

    acc = tl.zeros((BLOCK_SP,), dtype=tl.float32)

    # Loop over input channels
    for ic in range(0, C_in):
        # 3x3 neighborhood with padding=1
        for kh in range(3):
            for kw in range(3):
                ih = h + (kh - 1)
                iw = w + (kw - 1)

                x_ptrs = x_ptr \
                         + b * x_stride_b \
                         + ic * x_stride_c \
                         + ih * x_stride_h \
                         + iw * x_stride_w
                x_mask = sp_mask
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

                # weight for (ic, oc, kh, kw)
                w_ptrs = w_ptr \
                         + ic * w_stride_cin \
                         + oc * w_stride_cout \
                         + kh * w_stride_kh \
                         + kw * w_stride_kw
                w_val = tl.load(w_ptrs)

                acc += w_val * x_vals

    # Store results for this (b, oc, tile)
    y_ptrs = y_ptr \
             + b * y_stride_b \
             + oc * y_stride_c \
             + h * y_stride_h \
             + w * y_stride_w
    tl.store(y_ptrs, acc, mask=sp_mask)


# Triton kernel: GroupNorm (num_groups=32) + affine + SiLU
# y: output tensor (B, C, H, W), weight: per-channel scale (C,), bias: per-channel (C,), eps: float
@triton.jit
def group_norm_affine_silu_triton(
    x_ptr, weight_ptr, bias_ptr, y_ptr,
    B, C, H, W, num_groups: tl.constexpr,
    eps,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    y_stride_b, y_stride_c, y_stride_h, y_stride_w,
    BLOCK: tl.constexpr,  # e.g., 1024
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

        # SiLU: z * sigmoid(z) = z / (1 + exp(-z))
        sig = 1.0 / (1.0 + tl.exp(-z))
        y_vals = z * sig

        y_ptrs = y_ptr \
                 + b * y_stride_b \
                 + ch * y_stride_c \
                 + h * y_stride_h \
                 + w * y_stride_w
        tl.store(y_ptrs, y_vals, mask=mask)


# Triton kernel: elementwise residual add y = y + x (flattened)
@triton.jit
def add_residual_triton(
    out_ptr, y_ptr, x_ptr, N, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    y_vals = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out_vals = y_vals + x_vals
    tl.store(out_ptr + offsets, out_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

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
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # Ensure contiguity and dtype
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        # First conv: y1 = conv3x3(x, conv1_weight)
        # Grid: (B, ceil_div(C, BLOCK_OC), tiles over H*W)
        # For simplicity and correctness, choose BLOCK_OC=32 and BLOCK_SP=256 (tunable)
        BLOCK_OC = 32
        BLOCK_SP = 256
        y1 = torch.empty((B, C, H, W), device=device, dtype=dtype)

        grid1 = (
            B,
            triton.cdiv(C, BLOCK_OC),
            triton.cdiv(H * W, BLOCK_SP),
        )
        conv3x3_triton_clean[grid1](
            x, conv1_weight, y1,
            B, C, H, W, C,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv1_weight.stride(0), conv1_weight.stride(1), conv1_weight.stride(2), conv1_weight.stride(3),
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            BLOCK_SP=BLOCK_SP,
            num_warps=4,
            num_stages=2,
        )

        # GroupNorm + affine + SiLU on y1
        # num_groups=32, C % 32 == 0
        y1_gn = torch.empty_like(y1)
        num_groups = 32
        group_norm_affine_silu_triton[(B, num_groups)](
            y1, norm1_weight, norm1_bias, y1_gn,
            B, C, H, W, num_groups, eps,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            y1_gn.stride(0), y1_gn.stride(1), y1_gn.stride(2), y1_gn.stride(3),
            BLOCK=1024,
            num_warps=4,
            num_stages=2,
        )

        # SiLU
        # Already included in group_norm_affine_silu_triton, no separate SiLU needed.

        # Second conv: y2 = conv3x3(y1_gn, conv2_weight)
        y2 = torch.empty((B, C, H, W), device=device, dtype=dtype)
        grid2 = (
            B,
            triton.cdiv(C, BLOCK_OC),
            triton.cdiv(H * W, BLOCK_SP),
        )
        conv3x3_triton_clean[grid2](
            y1_gn, conv2_weight, y2,
            B, C, H, W, C,
            y1_gn.stride(0), y1_gn.stride(1), y1_gn.stride(2), y1_gn.stride(3),
            conv2_weight.stride(0), conv2_weight.stride(1), conv2_weight.stride(2), conv2_weight.stride(3),
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            BLOCK_SP=BLOCK_SP,
            num_warps=4,
            num_stages=2,
        )

        # GroupNorm + affine + SiLU on y2
        y2_gn = torch.empty_like(y2)
        group_norm_affine_silu_triton[(B, num_groups)](
            y2, norm2_weight, norm2_bias, y2_gn,
            B, C, H, W, num_groups, eps,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            y2_gn.stride(0), y2_gn.stride(1), y2_gn.stride(2), y2_gn.stride(3),
            BLOCK=1024,
            num_warps=4,
            num_stages=2,
        )

        # Residual add: out = y2_gn + x
        out = torch.empty_like(x)
        N = out.numel()
        add_residual_triton[(triton.cdiv(N, 1024),)](
            out, y2_gn, x,
            N, BLOCK=1024,
            num_warps=4,
            num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
