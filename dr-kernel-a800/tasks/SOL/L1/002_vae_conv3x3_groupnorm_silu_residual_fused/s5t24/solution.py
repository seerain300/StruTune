import torch
import triton
import triton.language as tl


# Conv3x3: y = x * w, stride=1, padding=1, no bias
# Grid: (B, C_out, H*W // BLOCK_HW). Each program handles one (n, c_out) and a tile of spatial positions.
@triton.jit
def conv3x3_kernel(
    x_ptr,               # *f32, input [B, C_in, H, W]
    w_ptr,               # *f32, weight [C_out, C_in, 3, 3]
    y_ptr,               # *f32, output [B, C_out, H, W]
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    tile = tl.program_id(2)

    hw = H * W
    offs = tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < hw
    h = offs // W
    w = offs % W

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Loop over input channels
    for c_in in range(C_in):
        # Accumulate over 3x3 neighborhood
        for dh in range(3):
            for dw in range(3):
                h_in = h + dh - 1  # padding=1
                w_in = w + dw - 1
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W) & mask

                # Compute input indices and load with masking
                x_idx = (((n * C_in) + c_in) * H + h_in) * W + w_in
                x_val = tl.load(x_ptr + x_idx, mask=in_bounds, other=0.0)

                # Weight is [C_out, C_in, 3, 3]; index for fixed (c_out, c_in, dh, dw)
                w_idx = (c_out * C_in + c_in) * (3 * 3) + (dh * 3 + dw)
                w_val = tl.load(w_ptr + w_idx)  # scalar

                acc += x_val * w_val

    # Store results
    y_idx = (((n * C_out) + c_out) * H + h) * W + w
    tl.store(y_ptr + y_idx, acc, mask=mask)


# GroupNorm reduction: compute per-channel sum and sumsq over spatial positions (H*W),
# grouped by num_groups, for each (n, group, channel).
@triton.jit
def group_norm_reduce_kernel(
    x_ptr,               # *f32, input [B, C, H, W]
    mean_ptr,            # *f32, output mean [B, C]
    rstd_ptr,            # *f32, output rstd [B, C]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,  # fixed 32
    N_TILES: tl.constexpr,     # tiles over H*W, e.g., 1 or 4
):
    n = tl.program_id(0)
    group = tl.program_id(1)
    c_per_group = C // num_groups
    c = tl.program_id(2)  # within group

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # Iterate over tiles of spatial positions
    for t in range(N_TILES):
        hw = H * W
        offs = t * (hw // N_TILES) + tl.arange(0, hw // N_TILES)
        mask = offs < hw
        h = offs // W
        w = offs % W

        x_idx = (((n * C) + (group * c_per_group + c)) * H + h) * W + w
        x_vec = tl.load(x_ptr + x_idx, mask=mask, other=0.0)

        sum_val += tl.sum(x_vec, axis=0)
        sum_sq += tl.sum(x_vec * x_vec, axis=0)

    N = H * W
    mean = sum_val / N
    var = sum_sq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    mean_ptr_idx = (n * C) + (group * c_per_group + c)
    rstd_ptr_idx = (n * C) + (group * c_per_group + c)
    tl.store(mean_ptr + mean_ptr_idx, mean)
    tl.store(rstd_ptr + rstd_ptr_idx, rstd)


# GroupNorm apply + affine + SiLU: for each (n, c), process tiles of H*W
@triton.jit
def apply_groupnorm_affine_silu_kernel(
    x_ptr,               # *f32, input [B, C, H, W]
    mean_ptr,            # *f32, mean [B, C]
    rstd_ptr,            # *f32, rstd [B, C]
    weight_ptr,          # *f32, scale [C]
    bias_ptr,            # *f32, bias [C]
    y_ptr,               # *f32, output [B, C, H, W]
    C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    N_TILES: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    tile = tl.program_id(2)

    hw = H * W
    offs = tile * (hw // N_TILES) + tl.arange(0, hw // N_TILES)
    mask = offs < hw
    h = offs // W
    w = offs % W

    x_idx = (((n * C) + c) * H + h) * W + w
    x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)

    mean = tl.load(mean_ptr + ((n * C) + c))
    rstd = tl.load(rstd_ptr + ((n * C) + c))
    gamma = tl.load(weight_ptr + c)
    beta = tl.load(bias_ptr + c)

    norm = (x_val - mean) * rstd
    norm = norm * gamma + beta

    # SiLU: x * sigmoid(x)
    sig = 1.0 / (1.0 + tl.exp(-norm))
    out = norm * sig

    tl.store(y_ptr + x_idx, out, mask=mask)


# Elementwise residual add: out = a + b
@triton.jit
def residual_add_kernel(
    a_ptr, b_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    hw = H * W
    offs = tl.program_id(2) * 1024 + tl.arange(0, 1024)
    mask = offs < hw
    h = offs // W
    w = offs % W
    idx = (((n * C) + c) * H + h) * W + w
    a = tl.load(a_ptr + idx, mask=mask, other=0.0)
    b = tl.load(b_ptr + idx, mask=mask, other=0.0)
    tl.store(out_ptr + idx, a + b, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self,
                x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        x: (B, C_in, H, W)
        conv1_weight: (C_out1, C_in, 3, 3)
        norm1_weight, norm1_bias: (C_out1,)
        conv2_weight: (C_out2, C_out1, 3, 3)
        norm2_weight, norm2_bias: (C_out2,)
        Returns: (B, C_out2, H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"
        B, C_in, H, W = x.shape

        # Cast to float32 and make contiguous for Triton
        x_f32 = x.contiguous().to(torch.float32)
        conv1_w_f32 = conv1_weight.contiguous().to(torch.float32)  # (C_out1, C_in, 3, 3)
        conv2_w_f32 = conv2_weight.contiguous().to(torch.float32)  # (C_out2, C_out1, 3, 3)
        norm1_weight_f32 = norm1_weight.contiguous().to(torch.float32)
        norm1_bias_f32 = norm1_bias.contiguous().to(torch.float32)
        norm2_weight_f32 = norm2_weight.contiguous().to(torch.float32)
        norm2_bias_f32 = norm2_bias.contiguous().to(torch.float32)

        # Compute first conv and GroupNorm + SiLU
        out1 = torch.empty((B, conv1_w_f32.shape[0], H, W), device=x.device, dtype=torch.float32)
        grid_conv1 = (B, conv1_w_f32.shape[0], triton.cdiv(H * W, 1024))
        conv3x3_kernel[grid_conv1](
            x_f32, conv1_w_f32, out1,
            B=B, C_in=C_in, C_out=conv1_w_f32.shape[0], H=H, W=W,
            BLOCK_HW=1024, num_warps=4, num_stages=2,
        )

        # GroupNorm stats for first conv output
        C_out1 = conv1_w_f32.shape[0]
        assert C_out1 % self.num_groups == 0, "C_out1 must be divisible by num_groups (32)"
        cpg1 = C_out1 // self.num_groups
        mean1 = torch.empty((B, C_out1), device=x.device, dtype=torch.float32)
        rstd1 = torch.empty((B, C_out1), device=x.device, dtype=torch.float32)

        # We need N_TILES consistent; choose 1 since H*W is usually small in given workloads.
        N_TILES1 = 1
        grid_reduce1 = (B, self.num_groups, cpg1)
        group_norm_reduce_kernel[grid_reduce1](
            out1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W, num_groups=self.num_groups, N_TILES=N_TILES1,
            num_warps=2, num_stages=1,
        )

        # Apply GroupNorm + affine + SiLU
        out1_norm = torch.empty_like(out1)
        grid_apply1 = (B, C_out1, N_TILES1)
        apply_groupnorm_affine_silu_kernel[grid_apply1](
            out1, mean1, rstd1, norm1_weight_f32, norm1_bias_f32, out1_norm,
            C=C_out1, H=H, W=W, N_TILES=N_TILES1,
            num_warps=2, num_stages=1,
        )

        # Second conv
        out2 = torch.empty((B, conv2_w_f32.shape[0], H, W), device=x.device, dtype=torch.float32)
        grid_conv2 = (B, conv2_w_f32.shape[0], triton.cdiv(H * W, 1024))
        conv3x3_kernel[grid_conv2](
            out1_norm, conv2_w_f32, out2,
            B=B, C_in=C_out1, C_out=conv2_w_f32.shape[0], H=H, W=W,
            BLOCK_HW=1024, num_warps=4, num_stages=2,
        )

        # GroupNorm stats for second conv output
        C_out2 = conv2_w_f32.shape[0]
        assert C_out2 % self.num_groups == 0, "C_out2 must be divisible by num_groups (32)"
        cpg2 = C_out2 // self.num_groups
        mean2 = torch.empty((B, C_out2), device=x.device, dtype=torch.float32)
        rstd2 = torch.empty((B, C_out2), device=x.device, dtype=torch.float32)

        N_TILES2 = 1
        grid_reduce2 = (B, self.num_groups, cpg2)
        group_norm_reduce_kernel[grid_reduce2](
            out2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W, num_groups=self.num_groups, N_TILES=N_TILES2,
            num_warps=2, num_stages=1,
        )

        # Apply GroupNorm + affine + SiLU
        out2_norm = torch.empty_like(out2)
        grid_apply2 = (B, C_out2, N_TILES2)
        apply_groupnorm_affine_silu_kernel[grid_apply2](
            out2, mean2, rstd2, norm2_weight_f32, norm2_bias_f32, out2_norm,
            C=C_out2, H=H, W=W, N_TILES=N_TILES2,
            num_warps=2, num_stages=1,
        )

        # Residual add: out2_norm + x_f32
        out = torch.empty_like(out2_norm)
        grid_add = (B, C_out2, triton.cdiv(H * W, 1024))
        residual_add_kernel[grid_add](
            out2_norm, x_f32, out,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
