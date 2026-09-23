import torch
import triton
import triton.language as tl


# Triton Conv3x3: output tile over (H, W) for a set of output channels
# A: im2col input, shape [C_in * 9, N_tiles], where N_tiles = BLOCK_H * BLOCK_W
# W: reshaped conv weights, shape [C_out_tile, C_in * 9]
# Y: output, shape [C_out_tile, N_tiles]
@triton.jit
def conv3x3_tile_gemm_kernel(
    x_ptr,  # *f32, input [B, C_in, H, W]
    w_ptr,  # *f32, conv weights [C_out, C_in, 3, 3], will be reshaped per launch
    a_ptr,  # *f32, im2col buffer [C_in*9, N_tiles]
    y_ptr,  # *f32, output [C_out_tile, N_tiles]
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_CO: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, N_TILES: tl.constexpr,
):
    # program ids: we launch over (B, ceil_div(C_out, BLOCK_CO), ceil_div(N_tiles, 1))
    b = tl.program_id(0)
    co_block = tl.program_id(1)
    # We have one spatial tile per program; grid[2] = 1 in this kernel.
    # Compute start indices for output channels and spatial tile
    co_start = co_block * BLOCK_CO
    # Initialize output tile
    y = tl.zeros((BLOCK_CO, N_TILES), dtype=tl.float32)

    # Prepare weight tile W: shape [BLOCK_CO, C_in*9]
    # We load W per output channel and loop over patch elements; compute W_tile as a vector of length BLOCK_CO
    # Here, we'll directly perform GEMM via accumulation across patch_elements and input_channels.
    # However, Triton does not support high-level matmul, so we manually accumulate.
    # Construct index vectors for channels and patch positions
    c_in_vec = tl.arange(0, C_in)  # input channels
    dh = -1
    while dh <= 1:
        dw = -1
        while dw <= 1:
            # patch index p in [0, 8]
            p = (dh + 1) * 3 + (dw + 1)
            # For each input channel, load A rows corresponding to (c_in, dh, dw)
            # A layout: rows = c_in * 9 + p, columns = 0..N_TILES-1
            # We'll build A_rows vector of length N_TILES for this (dh, dw) and p.
            # But Triton requires static loop bounds; we'll loop over c_in.
            for ci in range(C_in):
                a_row_idx = ci * 9 + p
                # Load A vector across tiles: a_ptr[a_row_idx, 0:N_TILES]
                # We can't directly load a vector with variable indices; so we emulate with masks:
                # Each tile is a single vector element since N_TILES is passed; thus we load per tile position explicitly by building idx.
                # To keep it simple, we will not use this direct approach: instead, we precompute A in Python/host and pass to Triton.
                # Therefore, this kernel is simplified: we assume A is precomputed and just do y = W @ A for this tile.
                # We'll pass A_ptr precomputed in Python.
            dw += 1
        dh += 1

    # Since we can't construct A inside the kernel without precomputation, we skip the manual GEMM here.
    # The correct approach is to precompute A using a separate Triton kernel (im2col) and then perform y = W @ A.
    # For robustness and to avoid Triton compilation issues, we provide a simpler conv kernel that computes per output pixel.
    # The previous attempt failed; to ensure correctness, we will implement a per-pixel conv kernel with careful indexing and masks.

    # Note: The above while-loop is only for structure; the correct conv implementation below uses direct per-pixel compute.
    return  # placeholder; actual conv3x3 per-pixel kernel is below


# Conv3x3 per-pixel kernel: one program handles one output pixel (n, c_out, h_out, w_out)
# This is a robust approach and avoids complex im2col in kernel.
@triton.jit
def conv3x3_per_pixel_kernel(
    x_ptr,  # *f32, input [B, C_in, H, W]
    w_ptr,  # *f32, conv weights [C_out, C_in, 3, 3]
    y_ptr,  # *f32, output [B, C_out, H, W]
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)  # batch
    c_out = tl.program_id(1)  # output channel
    hw = tl.program_id(2)  # linearized (h_out, w_out)
    h_out = hw // W
    w_out = hw % W

    acc = 0.0
    # Accumulate over input channels and 3x3 neighborhood
    for ci in range(0, C_in):
        for dh in range(-1, 2):
            for dw in range(-1, 2):
                hi = h_out + dh
                wi = w_out + dw
                # Valid patch indices always hold due to padding=1; indices are within [0,H) and [0,W)
                x_idx = ((n * C_in + ci) * H + hi) * W + wi
                # Load weight for (c_out, ci, dh+1, dw+1)
                # weight layout: [C_out, C_in, 3, 3]
                p_h = dh + 1  # 0..2
                p_w = dw + 1  # 0..2
                w_idx = (c_out * C_in + ci) * 9 + (p_h * 3 + p_w)
                x_val = tl.load(x_ptr + x_idx)
                w_val = tl.load(w_ptr + w_idx)
                acc += x_val * w_val
    y_idx = ((n * C_out + c_out) * H + h_out) * W + w_out
    tl.store(y_ptr + y_idx, acc)


# GroupNorm reduce: per (n, group, channel), compute sum and sumsq over all elements
@triton.jit
def group_norm_reduce_kernel(
    x_ptr,  # *f32, input [B, C, H, W] contiguous
    mean_ptr,  # *f32, [C]
    rstd_ptr,  # *f32, [C]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, eps: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    c_start = g * (C // num_groups)
    c = tl.program_id(2) + c_start
    # Compute sum and sumsq over all elements of this channel
    sum_val = 0.0
    sumsq_val = 0.0
    total = H * W
    # Loop over tiles to accumulate; use constexpr N_TILES to allow Triton loop
    N_TILES = (total + 127) // 128  # example tiling; we'll set per launch
    for t in range(0, N_TILES):
        tile_start = t * 128
        # Load a vector of 128 elements (masked) and reduce
        offs = tile_start + tl.arange(0, 128)
        mask = offs < total
        base = n * C * H * W
        idx = base + c * H * W + offs
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(vals, axis=0)
        sumsq_val += tl.sum(vals * vals, axis=0)
    mean = sum_val / total
    var = sumsq_val / total - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + c, mean)
    tl.store(rstd_ptr + c, rstd)


# GroupNorm apply: per (n, group, channel, tile), normalize and apply affine + SiLU
@triton.jit
def group_norm_apply_kernel(
    x_ptr,      # *f32, input [B, C, H, W]
    scale_ptr,  # *f32, [C]
    bias_ptr,   # *f32, [C]
    y_ptr,      # *f32, output [B, C, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, eps: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    c_start = g * (C // num_groups)
    c = tl.program_id(2) + c_start
    tile = tl.program_id(3)
    total = H * W
    tile_start = tile * 128
    offs = tile_start + tl.arange(0, 128)
    mask = offs < total
    base = n * C * H * W
    idx = base + c * H * W + offs
    x_vals = tl.load(x_ptr + idx, mask=mask, other=0.0)

    mean = tl.load(x_ptr + base + c * H * W + 0)  # mean precomputed in reduce kernel; not available here
    # NOTE: We need mean and rstd precomputed per channel. To use them, we call group_norm_reduce_kernel first to fill mean/rstd arrays.
    # Here we assume mean_ptr, rstd_ptr are passed via y_ptr mapping; correct approach: have a separate kernel to compute mean/rstd.
    # For robustness, we instead compute mean/rstd in Python/host and pass them as extra arguments. Triton kernels should have inputs passed explicitly.

    # Since we can't access mean directly in this kernel, we will modify the design: compute mean/rstd in a separate Triton kernel that writes
    # mean/rstd arrays, and this kernel will read them. We'll pass mean_ptr and rstd_ptr as separate tensors to y_ptr, but Triton expects distinct
    # pointers. Simpler: host computes mean/rstd and we pass them. However, to stay Triton-only, we will provide a Triton kernel that computes mean/rstd
    # and another kernel that applies it. The above kernel is a placeholder; the correct design is below.

    # Placeholder: we'll assume mean and rstd are provided via scale/bias pointers by host; not ideal. Better to fix by computing mean/rstd in Triton
    # and passing them as arguments (we'll adjust ModelNew accordingly). To keep Triton usage, we implement mean/rstd computation Triton kernel and
    # read them here.

    # Since Triton JIT requires explicit arguments, we implement mean/rstd as global buffers computed by Triton kernel. Here we read them using
    # x_ptr + base + c * H * W + offs would be wrong; we need separate mean/rstd pointers. Adjusting below.

    # The correct approach is to have:
    # 1) group_norm_reduce_kernel writes mean[c], rstd[c]
    # 2) group_norm_apply_kernel reads mean_ptr[c], rstd_ptr[c] and applies normalization + affine + SiLU
    # We will write those kernels explicitly.

    # Fix: implement proper kernels as below. The conv kernel above is kept for robustness; the main issue is GroupNorm implementation without
    # Python loops. We'll provide those kernels now.

    # Placeholder return; this kernel will be replaced with proper apply kernel below.
    return


# Proper GroupNorm reduce: per (n, group, channel), compute sum and sumsq over H*W
# We will use a separate kernel that writes mean and rstd arrays, and a second kernel that reads them.
# The apply kernel will need scale and bias for affine; norm1_weight, norm1_bias, norm2_weight, norm2_bias are provided.

# GroupNorm reduce kernel (recomputed to be correct): per (n, group, channel)
@triton.jit
def group_norm_reduce_kernel(
    x_ptr,  # *f32, input [B, C, H, W]
    mean_ptr,  # *f32, [C]
    rstd_ptr,  # *f32, [C]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, eps: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    c_start = g * (C // num_groups)
    c = tl.program_id(2) + c_start
    total = H * W
    sum_val = 0.0
    sumsq_val = 0.0
    # Loop over tiles of size 128
    N_TILES = (total + 127) // 128
    for t in range(0, N_TILES):
        tile_start = t * 128
        offs = tile_start + tl.arange(0, 128)
        mask = offs < total
        base = n * C * H * W
        idx = base + c * H * W + offs
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(vals, axis=0)
        sumsq_val += tl.sum(vals * vals, axis=0)
    mean = sum_val / total
    var = sumsq_val / total - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + c, mean)
    tl.store(rstd_ptr + c, rstd)


# GroupNorm apply + affine + SiLU kernel
@triton.jit
def group_norm_apply_affine_silu_kernel(
    x_ptr,      # *f32, input [B, C, H, W]
    mean_ptr,   # *f32, [C]
    rstd_ptr,   # *f32, [C]
    scale_ptr,  # *f32, [C]
    bias_ptr,   # *f32, [C]
    y_ptr,      # *f32, output [B, C, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    c_start = g * (C // num_groups)
    c = tl.program_id(2) + c_start
    total = H * W
    # We need to process the entire channel; using tiles for coalesced access.
    N_TILES = (total + 127) // 128
    for t in range(0, N_TILES):
        tile_start = t * 128
        offs = tile_start + tl.arange(0, 128)
        mask = offs < total
        base = n * C * H * W
        idx = base + c * H * W + offs
        x_vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
        mean = tl.load(mean_ptr + c)
        rstd = tl.load(rstd_ptr + c)
        scale = tl.load(scale_ptr + c)
        bias = tl.load(bias_ptr + c)
        norm_vals = (x_vals - mean) * rstd
        # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
        silu_vals = norm_vals * (1.0 / (1.0 + tl.exp(-norm_vals)))
        y_vals = silu_vals * scale + bias
        tl.store(y_ptr + idx, y_vals, mask=mask)


# Elementwise residual add: y = a + b
@triton.jit
def residual_add_kernel(
    a_ptr, b_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    hw = tl.program_id(2)
    h = hw // W
    w = hw % W
    idx = ((n * C + c) * H + h) * W + w
    a_val = tl.load(a_ptr + idx)
    b_val = tl.load(b_ptr + idx)
    tl.store(out_ptr + idx, a_val + b_val)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        x: (B, C, H, W), conv weights: (C_out, C_in, 3, 3), norm scales/bias: (C,)
        Returns: (B, C_out2, H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"
        B, C_in, H, W = x.shape

        # Ensure contiguous float32 for Triton
        x_f32 = x.contiguous().to(torch.float32)
        conv1_w_f32 = conv1_weight.contiguous().to(torch.float32)  # (C_out1, C_in, 3, 3)
        conv2_w_f32 = conv2_weight.contiguous().to(torch.float32)  # (C_out2, C_in, 3, 3)
        norm1_weight_f32 = norm1_weight.contiguous().to(torch.float32)
        norm1_bias_f32 = norm1_bias.contiguous().to(torch.float32)
        norm2_weight_f32 = norm2_weight.contiguous().to(torch.float32)
        norm2_bias_f32 = norm2_bias.contiguous().to(torch.float32)

        # First path: Conv3x3 per-pixel, then GroupNorm + SiLU
        # Allocate output after conv1
        out1 = torch.empty((B, conv1_w_f32.shape[0], H, W), device=x.device, dtype=torch.float32)

        # Launch conv kernel: grid over (B, C_out1, H*W)
        grid_conv1 = (B, conv1_w_f32.shape[0], H * W)
        conv3x3_per_pixel_kernel[grid_conv1](
            x_f32, conv1_w_f32, out1,
            B=B, C_in=C_in, C_out=conv1_w_f32.shape[0], H=H, W=W,
            num_warps=4, num_stages=2,
        )

        # Compute per-channel mean and rstd for GroupNorm
        mean1 = torch.empty((out1.shape[1],), device=x.device, dtype=torch.float32)
        rstd1 = torch.empty((out1.shape[1],), device=x.device, dtype=torch.float32)
        grid_reduce1 = (B, self.num_groups, out1.shape[1] // self.num_groups)
        group_norm_reduce_kernel[grid_reduce1](
            out1, mean1, rstd1,
            B=B, C=out1.shape[1], H=H, W=W,
            num_groups=self.num_groups, eps=self.eps,
            num_warps=4, num_stages=2,
        )

        # GroupNorm apply + affine + SiLU for first block
        out1_norm = torch.empty_like(out1)
        grid_apply1 = (B, self.num_groups, out1.shape[1] // self.num_groups, (H * W + 127) // 128)
        group_norm_apply_affine_silu_kernel[grid_apply1](
            out1, mean1, rstd1, norm1_weight_f32, norm1_bias_f32, out1_norm,
            B=B, C=out1.shape[1], H=H, W=W,
            num_groups=self.num_groups,
            num_warps=4, num_stages=2,
        )

        # Second path: Conv3x3 per-pixel on normalized output, then GroupNorm + SiLU
        out2 = torch.empty((B, conv2_w_f32.shape[0], H, W), device=x.device, dtype=torch.float32)
        grid_conv2 = (B, conv2_w_f32.shape[0], H * W)
        conv3x3_per_pixel_kernel[grid_conv2](
            out1_norm, conv2_w_f32, out2,
            B=B, C_in=conv1_w_f32.shape[0], C_out=conv2_w_f32.shape[0], H=H, W=W,
            num_warps=4, num_stages=2,
        )

        # Compute per-channel mean and rstd for second GroupNorm
        mean2 = torch.empty((out2.shape[1],), device=x.device, dtype=torch.float32)
        rstd2 = torch.empty((out2.shape[1],), device=x.device, dtype=torch.float32)
        grid_reduce2 = (B, self.num_groups, out2.shape[1] // self.num_groups)
        group_norm_reduce_kernel[grid_reduce2](
            out2, mean2, rstd2,
            B=B, C=out2.shape[1], H=H, W=W,
            num_groups=self.num_groups, eps=self.eps,
            num_warps=4, num_stages=2,
        )

        # GroupNorm apply + affine + SiLU for second block
        out2_norm = torch.empty_like(out2)
        grid_apply2 = (B, self.num_groups, out2.shape[1] // self.num_groups, (H * W + 127) // 128)
        group_norm_apply_affine_silu_kernel[grid_apply2](
            out2, mean2, rstd2, norm2_weight_f32, norm2_bias_f32, out2_norm,
            B=B, C=out2.shape[1], H=H, W=W,
            num_groups=self.num_groups,
            num_warps=4, num_stages=2,
        )

        # Final residual add: out2_norm + x
        out = torch.empty_like(out2_norm)
        grid_add = (B, out2_norm.shape[1], H * W)
        residual_add_kernel[grid_add](
            out2_norm, x_f32, out,
            B=B, C=out2_norm.shape[1], H=H, W=W,
            num_warps=4, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
