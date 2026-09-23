import torch
import triton
import triton.language as tl


# Conv3x3 via im2col + reduction over K = C_in * 9, tile over HW.
# Input x: (B, C_in, H, W), weight w: (C_out, C_in, 3, 3), output y: (B, C_out, H, W)
@triton.jit
def conv3x3_im2col_kernel(
    x_ptr, w_ptr, y_ptr,
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)  # batch
    c_out = tl.program_id(1)  # output channel
    tile_id = tl.program_id(2)  # tile over HW

    hw_start = tile_id * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = hw_offsets < (H * W)

    h = hw_offsets // W
    w = hw_offsets % W

    # Build A matrix [K, BLOCK_HW] where K = C_in * 9
    K = C_in * 9
    # Initialize output accumulator
    y_vec = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Accumulate over input channels and 3x3 neighborhood
    for k in range(K):
        cin = k // 9
        dh = k % 3 - 1  # -1, 0, 1
        dw = (k % 9) // 3 - 1  # -1, 0, 1, -1, 0, 1, -1, 0, 1 -> after k//3
        # Map k -> dh,dw properly: k = cin*9 + i, i in 0..8 -> (dh,dw) in (-1,0,1)
        i = k - cin * 9
        dh = i // 3 - 1
        dw = (i % 3) - 1

        hi = h + dh  # padded
        wi = w + dw  # padded
        # Valid mask due to padding=1, stride=1 -> indices are always in [0, H-1] and [0, W-1]
        # Compute input linear index: (((n*C_in) + cin)*H + hi)*W + wi
        in_idx = (((n * C_in) + cin) * H + hi) * W + wi

        # Load input vector (masked not needed since padding ensures valid)
        x_vec = tl.load(x_ptr + in_idx, mask=mask_hw, other=0.0)

        # Load weight scalar: w[c_out, cin, dh+1, dw+1]
        w_off = c_out * (C_in * 9) + k  # linearize over (C_in, 3, 3)
        w_val = tl.load(w_ptr + w_off)

        y_vec += x_vec * w_val

    # Store output tile: y[n, c_out, h, w]
    y_off = (((n * C_out) + c_out) * H + h) * W + w
    tl.store(y_ptr + y_off, y_vec, mask=mask_hw)


# Triton GroupNorm reduce: per-channel mean and rstd for each (n, group).
# We compute across all spatial elements for each (n, group, channel_in_group).
@triton.jit
def group_norm_reduce_kernel(
    x_ptr, mean_ptr, rstd_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr, channels_per_group: tl.constexpr,
    N_TILES: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)  # group id in [0, num_groups)
    channel_in_group = tl.program_id(2)  # which channel within the group

    c = channel_in_group * channels_per_group  # absolute channel index

    # Accumulate sum and sum of squares over all HW elements
    total_sum = tl.zeros((), dtype=tl.float32)
    total_sumsq = tl.zeros((), dtype=tl.float32)

    for t in range(N_TILES):
        hw_start = t * BLOCK_HW
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        mask_hw = hw_offsets < (H * W)

        h = hw_offsets // W
        w = hw_offsets % W

        idx = (((n * C) + c) * H + h) * W + w
        x_vec = tl.load(x_ptr + idx, mask=mask_hw, other=0.0)
        total_sum += tl.sum(x_vec, axis=0)
        total_sumsq += tl.sum(x_vec * x_vec, axis=0)

    m = H * W
    mean = total_sum / m
    # sum(x^2) - m * mean^2
    var = total_sumsq / m - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    tl.store(mean_ptr + c, mean)
    tl.store(rstd_ptr + c, rstd)


# Triton GroupNorm apply + affine + SiLU: per-channel per spatial tile.
@triton.jit
def group_norm_apply_silu_kernel(
    x_ptr, mean_ptr, rstd_ptr, weight_ptr, bias_ptr, y_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    N_TILES: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)  # output channel
    tile_id = tl.program_id(2)

    hw_start = tile_id * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = hw_offsets < (H * W)

    h = hw_offsets // W
    w = hw_offsets % W

    idx = (((n * C) + c) * H + h) * W + w
    x_vec = tl.load(x_ptr + idx, mask=mask_hw, other=0.0)

    mean = tl.load(mean_ptr + c)
    rstd = tl.load(rstd_ptr + c)
    gamma = tl.load(weight_ptr + c)
    beta = tl.load(bias_ptr + c)

    # GroupNorm affine
    y_vec = (x_vec - mean) * rstd
    y_vec = y_vec * gamma + beta

    # SiLU activation: y = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-y_vec))
    y_vec = y_vec * sig

    tl.store(y_ptr + idx, y_vec, mask=mask_hw)


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
        conv1_weight: (C_out1, C, 3, 3)
        norm1_weight, norm1_bias: (C_out1,)
        conv2_weight: (C_out2, C_out1, 3, 3)
        norm2_weight, norm2_bias: (C_out2,)
        Returns: (B, C_out2, H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"
        B, C, H, W = x.shape

        # Prepare weights and params as float32
        x_f32 = x.contiguous().to(torch.float32)
        conv1_w_f32 = conv1_weight.contiguous().to(torch.float32)  # (C_out1, C, 3, 3)
        conv2_w_f32 = conv2_weight.contiguous().to(torch.float32)  # (C_out2, C_out1, 3, 3)
        norm1_weight_f32 = norm1_weight.contiguous().to(torch.float32)
        norm1_bias_f32 = norm1_bias.contiguous().to(torch.float32)
        norm2_weight_f32 = norm2_weight.contiguous().to(torch.float32)
        norm2_bias_f32 = norm2_bias.contiguous().to(torch.float32)

        # Output tensors
        out1 = torch.empty((B, conv1_w_f32.shape[0], H, W), device=x.device, dtype=torch.float32)
        out2 = torch.empty((B, conv2_w_f32.shape[0], H, W), device=x.device, dtype=torch.float32)

        # Conv1
        C_out1 = conv1_w_f32.shape[0]
        assert C_out1 % self.num_groups == 0, "C_out1 must be divisible by num_groups"
        channels_per_group1 = C_out1 // self.num_groups

        BLOCK_HW = 256
        N_TILES1 = (H * W + BLOCK_HW - 1) // BLOCK_HW

        # Launch conv1: grid over (B, C_out1, N_TILES)
        grid_conv1 = (B, C_out1, N_TILES1)
        conv3x3_im2col_kernel[grid_conv1](
            x_f32, conv1_w_f32, out1,
            B=B, C_in=C, C_out=C_out1, H=H, W=W,
            BLOCK_HW=BLOCK_HW,
            num_warps=4,
            num_stages=2,
        )

        # GroupNorm + SiLU for first block
        mean1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        rstd1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)

        grid_reduce1 = (B, self.num_groups, channels_per_group1)
        group_norm_reduce_kernel[grid_reduce1](
            out1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group1,
            N_TILES=N_TILES1, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2
        )

        out1_norm = torch.empty_like(out1)
        grid_apply1 = (B, C_out1, N_TILES1)
        group_norm_apply_silu_kernel[grid_apply1](
            out1, mean1, rstd1, norm1_weight_f32, norm1_bias_f32, out1_norm,
            B=B, C=C_out1, H=H, W=W,
            N_TILES=N_TILES1, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2
        )

        # Conv2 on normalized output
        C_out2 = conv2_w_f32.shape[0]
        assert C_out2 % self.num_groups == 0, "C_out2 must be divisible by num_groups"
        channels_per_group2 = C_out2 // self.num_groups

        # Launch conv2: grid over (B, C_out2, N_TILES)
        grid_conv2 = (B, C_out2, N_TILES1)
        conv3x3_im2col_kernel[grid_conv2](
            out1_norm, conv2_w_f32, out2,
            B=B, C_in=C_out1, C_out=C_out2, H=H, W=W,
            BLOCK_HW=BLOCK_HW,
            num_warps=4,
            num_stages=2,
        )

        # GroupNorm + SiLU for second block
        mean2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        rstd2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)

        grid_reduce2 = (B, self.num_groups, channels_per_group2)
        group_norm_reduce_kernel[grid_reduce2](
            out2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group2,
            N_TILES=N_TILES1, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2
        )

        out2_norm = torch.empty_like(out2)
        grid_apply2 = (B, C_out2, N_TILES1)
        group_norm_apply_silu_kernel[grid_apply2](
            out2, mean2, rstd2, norm2_weight_f32, norm2_bias_f32, out2_norm,
            B=B, C=C_out2, H=H, W=W,
            N_TILES=N_TILES1, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2
        )

        # Final residual add: out2_norm + x
        out = torch.empty_like(out2_norm)
        grid_add = (B, C_out2, H * W)
        residual_add_kernel[grid_add](
            out2_norm, x_f32, out,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4,
            num_stages=2,
        )

        # Cast back to original dtype if needed
        if x.dtype != torch.float32:
            out = out.to(x.dtype)

        return out


def run(*args):
    return ModelNew()(*args)
