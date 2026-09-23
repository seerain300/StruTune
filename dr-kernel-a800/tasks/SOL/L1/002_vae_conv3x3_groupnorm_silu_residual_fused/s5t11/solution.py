import torch
import triton
import triton.language as tl


# Conv3x3: y = x * w, stride=1, padding=1, no bias
# Grid: (B, C_out, H*W). Each program handles one output pixel (n, c_out, h_out, w_out).
@triton.jit
def conv3x3_kernel(
    x_ptr,             # *f32, input [B, C_in, H, W], contiguous NCHW
    w_ptr,             # *f32, weight [C_out, C_in, 3, 3], contiguous
    y_ptr,             # *f32, output [B, C_out, H, W], contiguous
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    hw = tl.program_id(2)
    h_out = hw // W
    w_out = hw % W

    # Accumulator
    acc = 0.0

    # Loop over input channels and 3x3 neighborhood
    for c_in in range(0, C_in):
        for dh in range(-1, 2):
            for dw in range(-1, 2):
                hi = h_out + dh
                wi = w_out + dw
                # Input index: (((n * C_in) + c_in) * H + hi) * W + wi
                x_idx = (((n * C_in) + c_in) * H + hi) * W + wi
                # Weight index: c_out * (C_in * 9) + c_in * 9 + (dh + 1) * 3 + (dw + 1)
                w_idx = c_out * (C_in * 9) + c_in * 9 + (dh + 1) * 3 + (dw + 1)
                x_val = tl.load(x_ptr + x_idx)  # float32
                w_val = tl.load(w_ptr + w_idx)  # float32
                acc += x_val * w_val

    # Store result to y[n, c_out, h_out, w_out]
    y_idx = (((n * C_out) + c_out) * H + h_out) * W + w_out
    tl.store(y_ptr + y_idx, acc)


# GroupNorm reduction: compute per-channel mean and rstd for each sample
# Grid: (B, num_groups, channels_per_group). Each program computes mean/rstd for one channel in the group.
@triton.jit
def group_norm_reduce_kernel(
    x_ptr,             # *f32, input tensor [B, C, H, W] flattened per sample
    mean_ptr,          # *f32, output mean [C]
    rstd_ptr,          # *f32, output rstd [C]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, channels_per_group: tl.constexpr,
    N_TILES: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    b = tl.program_id(0)
    g = tl.program_id(1)
    c = tl.program_id(2)

    # Determine channel indices for this group
    # group channels are [g*channels_per_group, (g+1)*channels_per_group)
    # but we only handle one c per (b,g) in this simplified implementation.
    # We compute total sum and sumsq over the whole tensor (flattened) for this channel c.
    total_sum = 0.0
    total_sumsq = 0.0

    # Loop over tiles over H*W
    for t in range(0, N_TILES):
        tile_start = t * BLOCK_HW
        hw_vec = tile_start + tl.arange(0, BLOCK_HW)
        mask = hw_vec < (H * W)
        # Flattened pointer for this sample: (((b * C) + c) * H + h) * W + w
        base = ((b * C) + c) * (H * W)
        x_vec = tl.load(x_ptr + base + hw_vec, mask=mask, other=0.0)
        total_sum += tl.sum(x_vec, axis=0)
        total_sumsq += tl.sum(x_vec * x_vec, axis=0)

    N = H * W
    mean = total_sum / N
    var = total_sumsq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)  # epsilon for numerical stability

    # Store mean and rstd for this channel
    tl.store(mean_ptr + c, mean)
    tl.store(rstd_ptr + c, rstd)


# GroupNorm apply + SiLU: for each element, normalize with per-channel mean/rstd and scale/bias, then SiLU
# Grid: (B, C, H*W). Each program handles one element.
@triton.jit
def group_norm_apply_silu_kernel(
    x_ptr,             # *f32, input [B, C, H, W] flattened per sample
    mean_ptr,          # *f32, mean [C]
    rstd_ptr,          # *f32, rstd [C]
    scale_ptr,         # *f32, weight [C]
    bias_ptr,          # *f32, bias [C]
    y_ptr,             # *f32, output [B, C, H, W] flattened per sample
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    hw = tl.program_id(2)

    # Load normalization params and affine params
    mean = tl.load(mean_ptr + c)
    rstd = tl.load(rstd_ptr + c)
    scale = tl.load(scale_ptr + c)
    bias = tl.load(bias_ptr + c)

    # Load input element
    base = ((b * C) + c) * (H * W)
    x_val = tl.load(x_ptr + base + hw)

    # Normalize and affine
    y_norm = (x_val - mean) * rstd
    y_affine = y_norm * scale + bias

    # SiLU: y * sigmoid(y)
    # sigmoid(y) = 1 / (1 + exp(-y))
    sig = 1.0 / (1.0 + tl.exp(-y_affine))
    y_out = y_affine * sig

    tl.store(y_ptr + base + hw, y_out)


# Elementwise residual add: out = out + x
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

    idx = (((n * C) + c) * H + h) * W + w
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
        x: (B, C, H, W)
        conv1_weight: (C_out1, C_in, 3, 3)
        norm1_weight, norm1_bias: (C_out1,)
        conv2_weight: (C_out2, C_out1, 3, 3)
        norm2_weight, norm2_bias: (C_out2,)
        Returns: (B, C_out2, H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"
        B, C_in, H, W = x.shape

        # Make tensors contiguous and float32
        x_f32 = x.contiguous().to(torch.float32)
        conv1_w_f32 = conv1_weight.contiguous().to(torch.float32)  # (C_out1, C_in, 3, 3)
        conv2_w_f32 = conv2_weight.contiguous().to(torch.float32)  # (C_out2, C_out1, 3, 3)
        norm1_weight_f32 = norm1_weight.contiguous().to(torch.float32)
        norm1_bias_f32 = norm1_bias.contiguous().to(torch.float32)
        norm2_weight_f32 = norm2_weight.contiguous().to(torch.float32)
        norm2_bias_f32 = norm2_bias.contiguous().to(torch.float32)

        # Output tensors
        out1 = torch.empty((B, conv1_w_f32.shape[0], H, W), device=x.device, dtype=torch.float32)
        out2 = torch.empty((B, conv2_w_f32.shape[0], H, W), device=x.device, dtype=torch.float32)

        # Launch conv1
        grid_conv1 = (B, conv1_w_f32.shape[0], H * W)
        conv3x3_kernel[grid_conv1](
            x_f32, conv1_w_f32, out1,
            B=B, C_in=C_in, C_out=conv1_w_f32.shape[0],
            H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        # GroupNorm + SiLU for first block
        C_out1 = conv1_w_f32.shape[0]
        assert C_out1 % self.num_groups == 0, "C_out1 must be divisible by num_groups"
        channels_per_group1 = C_out1 // self.num_groups

        mean1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        rstd1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)

        # Choose tiling parameters. We cover H*W with tiles of size BLOCK_HW.
        BLOCK_HW = 256  # tile size over HW
        N_TILES1 = (H * W + BLOCK_HW - 1) // BLOCK_HW  # number of tiles (compile-time constant for Triton)
        grid_reduce1 = (B, self.num_groups, channels_per_group1)
        group_norm_reduce_kernel[grid_reduce1](
            out1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group1,
            N_TILES=N_TILES1, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2
        )

        # Apply GroupNorm + SiLU
        out1_norm = torch.empty_like(out1)
        grid_apply1 = (B, C_out1, H * W)
        group_norm_apply_silu_kernel[grid_apply1](
            out1, mean1, rstd1, norm1_weight_f32, norm1_bias_f32, out1_norm,
            B=B, C=C_out1, H=H, W=W,
            num_warps=4, num_stages=2
        )

        # Conv2 on the normalized output
        grid_conv2 = (B, conv2_w_f32.shape[0], H * W)
        conv3x3_kernel[grid_conv2](
            out1_norm, conv2_w_f32, out2,
            B=B, C_in=C_out1, C_out=conv2_w_f32.shape[0],
            H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        # GroupNorm + SiLU for second block
        C_out2 = conv2_w_f32.shape[0]
        assert C_out2 % self.num_groups == 0, "C_out2 must be divisible by num_groups"
        channels_per_group2 = C_out2 // self.num_groups

        mean2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        rstd2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)

        N_TILES2 = (H * W + BLOCK_HW - 1) // BLOCK_HW
        grid_reduce2 = (B, self.num_groups, channels_per_group2)
        group_norm_reduce_kernel[grid_reduce2](
            out2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group2,
            N_TILES=N_TILES2, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2
        )

        out2_norm = torch.empty_like(out2)
        grid_apply2 = (B, C_out2, H * W)
        group_norm_apply_silu_kernel[grid_apply2](
            out2, mean2, rstd2, norm2_weight_f32, norm2_bias_f32, out2_norm,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4, num_stages=2
        )

        # Final residual add
        out = torch.empty_like(out2_norm)
        grid_add = (B, C_out2, H * W)
        residual_add_kernel[grid_add](
            out2_norm, x_f32, out,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4, num_stages=2
        )

        # Cast back to original dtype if needed
        if x.dtype != torch.float32:
            out = out.to(x.dtype)

        return out


def run(*args):
    return ModelNew()(*args)
