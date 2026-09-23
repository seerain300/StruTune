import torch
import triton
import triton.language as tl


# Triton kernel: GroupNorm reduction per channel
# For each (n, group, c in the group), compute sum and sumsq over all spatial positions H*W.
# Grid: (B, num_groups, channels_per_group). We loop over tiles via constexpr N_TILES.
@triton.jit
def group_norm_reduce_kernel(
    x_ptr,            # *f32, input [B, C, H, W]
    mean_ptr,         # *f32, output [C], will store mean per channel
    sumsq_ptr,        # *f32, output [C], will store sum of squares per channel
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    num_groups: tl.constexpr,
    channels_per_group: tl.constexpr,
    N_TILES: tl.constexpr,       # number of tiles across H*W
    BLOCK_HW: tl.constexpr,      # tile size across H*W
):
    n = tl.program_id(0)
    group = tl.program_id(1)
    c = tl.program_id(2)  # c is within this group (group * channels_per_group + c)

    total = H * W
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over tiles
    for t in range(N_TILES):
        start = t * BLOCK_HW
        hw_idx = start + tl.arange(0, BLOCK_HW)
        mask = hw_idx < total

        # Map flattened hw_idx -> (h, w)
        h = hw_idx // W
        w = hw_idx % W

        base = ((n * C) + c) * total
        x_vec = tl.load(x_ptr + base + hw_idx, mask=mask, other=0.0)
        sum_val += tl.sum(x_vec, axis=0)
        sum_sq += tl.sum(x_vec * x_vec, axis=0)

    # Compute mean and write
    mean = sum_val / total
    rstd = 1.0 / tl.sqrt(sum_sq / total + 1e-5)  # eps = 1e-5 (same as original)
    tl.store(mean_ptr + c, mean)
    tl.store(sumsq_ptr + c, sum_sq)  # not used further; kept for potential debugging


# Triton kernel: GroupNorm apply + affine + SiLU
# For each (n, group, c in the group, tile), normalize and apply per-channel affine and SiLU.
@triton.jit
def group_norm_apply_silu_kernel(
    x_ptr,            # *f32, input [B, C, H, W]
    y_ptr,            # *f32, output [B, C, H, W]
    mean_ptr,         # *f32, [C], mean per channel
    rstd_ptr,         # *f32, [C], rstd per channel
    gamma_ptr,        # *f32, [C], norm weight (scale)
    beta_ptr,         # *f32, [C], norm bias
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    num_groups: tl.constexpr,
    channels_per_group: tl.constexpr,
    N_TILES: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    group = tl.program_id(1)
    c = tl.program_id(2)           # within this group
    t = tl.program_id(3)           # tile index

    total = H * W
    start = t * BLOCK_HW
    hw_idx = start + tl.arange(0, BLOCK_HW)
    mask = hw_idx < total

    h = hw_idx // W
    w = hw_idx % W

    base = ((n * C) + c) * total
    x_vec = tl.load(x_ptr + base + hw_idx, mask=mask, other=0.0)

    mean_c = tl.load(mean_ptr + c)
    rstd_c = tl.load(rstd_ptr + c)
    gamma_c = tl.load(gamma_ptr + c)
    beta_c = tl.load(beta_ptr + c)

    # Normalize + affine
    y_vec = (x_vec - mean_c) * rstd_c
    y_vec = y_vec * gamma_c + beta_c

    # SiLU: y * sigmoid(y)
    # sigmoid(y) = 1 / (1 + exp(-y))
    sig = 1.0 / (1.0 + tl.exp(-y_vec))
    y_vec = y_vec * sig

    tl.store(y_ptr + base + hw_idx, y_vec, mask=mask)


# Triton elementwise residual add: out = a + b
@triton.jit
def residual_add_kernel(
    a_ptr, b_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    hw = tl.program_id(2)
    total = H * W
    idx = (((n * C) + c) * total) + hw
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

        # Ensure divisibility for GroupNorm
        assert C_in % self.num_groups == 0, "C must be divisible by num_groups for GroupNorm"

        # Prepare tensors as float32 for Triton
        x_f32 = x.contiguous().to(torch.float32)  # (B, C_in, H, W)
        conv1_weight_f32 = conv1_weight.contiguous().to(torch.float32)  # (C_out1, C_in, 3, 3)
        conv2_weight_f32 = conv2_weight.contiguous().to(torch.float32)  # (C_out2, C_in, 3, 3)

        # First conv
        out1 = torch.nn.functional.conv2d(
            x_f32, conv1_weight_f32, bias=None, stride=1, padding=1, dilation=1
        )  # (B, C_out1, H, W)

        # GroupNorm + SiLU for first block
        C_out1 = conv1_weight_f32.shape[0]
        channels_per_group = C_out1 // self.num_groups

        # Allocate buffers for mean and rstd
        mean1 = torch.empty((C_out1,), device=x.device, dtype=torch.float32)
        rstd1 = torch.empty((C_out1,), device=x.device, dtype=torch.float32)
        sumsq1 = torch.empty((C_out1,), device=x.device, dtype=torch.float32)

        # Launch reduction kernel
        BLOCK_HW = 256  # tile size across H*W
        N_TILES = triton.cdiv(H * W, BLOCK_HW)
        grid_reduce = (B, self.num_groups, channels_per_group)
        group_norm_reduce_kernel[grid_reduce](
            out1, mean1, sumsq1,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups,
            channels_per_group=channels_per_group,
            N_TILES=N_TILES,
            BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2,
        )

        # Compute rstd from sumsq (note: sumsq includes sum of squares; we need sum of squares over H*W)
        # We store sumsq in the reduction kernel; compute rstd here.
        # rstd = 1 / sqrt(mean_sq + eps)
        # Here mean_sq = sumsq / (H*W), since sumsq is sum of squares in the kernel.
        # However, the reduction kernel computed sum and sumsq separately. We need to fix that: we'll recompute sum_sq via out1.sum() and (out1*out1).sum() in torch.
        # To keep Triton-only, we can compute sumsq per channel by summing (out1^2) across spatial dims. We'll do this in a small torch reduction but it's only for rstd. It's not Triton math.
        # But to strictly adhere to "Triton-only" requirement, we'll restructure: the reduction kernel will directly compute mean and rstd without needing sumsq separately.
        # Therefore, we modify the reduction kernel to store rstd as well.

        # Correction: re-launch reduction kernel that stores rstd correctly.
        # We'll do this by computing total elements and using sumsq computed inside the kernel. However, Triton cannot directly write rstd with the sumsq; we need to pass sumsq or compute mean and rstd inside the kernel. The previous version mistakenly returned sumsq and did not use it. Fix: directly compute rstd inside kernel using sumsq (which we will write).

        # Since we cannot fix the kernel after definition, we will implement a corrected reduction kernel that computes mean and rstd and store them. We redefine the kernel here with correct behavior.

        # Redefine reduction kernel: compute mean and rstd per channel and store them.
        # We'll relaunch with this corrected kernel.

        # Corrected reduction kernel below:

        @triton.jit
        def group_norm_reduce_correct_kernel(
            x_ptr, mean_ptr, rstd_ptr,
            B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
            num_groups: tl.constexpr, channels_per_group: tl.constexpr,
            N_TILES: tl.constexpr, BLOCK_HW: tl.constexpr
        ):
            n = tl.program_id(0)
            group = tl.program_id(1)
            c = tl.program_id(2)  # channel in this group
            total = H * W
            sum_val = 0.0
            sum_sq = 0.0

            for t in range(N_TILES):
                start = t * BLOCK_HW
                hw_idx = start + tl.arange(0, BLOCK_HW)
                mask = hw_idx < total
                h = hw_idx // W
                w = hw_idx % W
                base = ((n * C) + c) * total
                x_vec = tl.load(x_ptr + base + hw_idx, mask=mask, other=0.0)
                sum_val += tl.sum(x_vec, axis=0)
                sum_sq += tl.sum(x_vec * x_vec, axis=0)

            mean = sum_val / total
            var = sum_sq / total
            rstd = 1.0 / tl.sqrt(var + 1e-5)
            tl.store(mean_ptr + c, mean)
            tl.store(rstd_ptr + c, rstd)

        # Launch corrected reduction kernel
        mean1 = torch.empty((C_out1,), device=x.device, dtype=torch.float32)
        rstd1 = torch.empty((C_out1,), device=x.device, dtype=torch.float32)
        BLOCK_HW = 256
        N_TILES = triton.cdiv(H * W, BLOCK_HW)
        grid_reduce = (B, self.num_groups, channels_per_group)
        group_norm_reduce_correct_kernel[grid_reduce](
            out1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group,
            N_TILES=N_TILES, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2,
        )

        # Apply + SiLU for first block
        out1_norm = torch.empty_like(out1)
        gamma1 = norm1_weight_f32.contiguous().to(torch.float32)
        beta1 = norm1_bias_f32.contiguous().to(torch.float32)
        grid_apply = (B, self.num_groups, channels_per_group, N_TILES)
        group_norm_apply_silu_kernel[grid_apply](
            out1, out1_norm, mean1, rstd1, gamma1, beta1,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group,
            N_TILES=N_TILES, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2,
        )

        # Second conv on out1_norm
        out2 = torch.nn.functional.conv2d(
            out1_norm, conv2_weight_f32, bias=None, stride=1, padding=1, dilation=1
        )  # (B, C_out2, H, W)

        # GroupNorm + SiLU for second block
        C_out2 = conv2_weight_f32.shape[0]
        channels_per_group2 = C_out2 // self.num_groups
        mean2 = torch.empty((C_out2,), device=x.device, dtype=torch.float32)
        rstd2 = torch.empty((C_out2,), device=x.device, dtype=torch.float32)
        BLOCK_HW2 = 256
        N_TILES2 = triton.cdiv(H * W, BLOCK_HW2)

        group_norm_reduce_correct_kernel[(B, self.num_groups, channels_per_group2)](
            out2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group2,
            N_TILES=N_TILES2, BLOCK_HW=BLOCK_HW2,
            num_warps=4, num_stages=2,
        )

        out2_norm = torch.empty_like(out2)
        gamma2 = norm2_weight_f32.contiguous().to(torch.float32)
        beta2 = norm2_bias_f32.contiguous().to(torch.float32)
        group_norm_apply_silu_kernel[(B, self.num_groups, channels_per_group2, N_TILES2)](
            out2, out2_norm, mean2, rstd2, gamma2, beta2,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups, channels_per_group=channels_per_group2,
            N_TILES=N_TILES2, BLOCK_HW=BLOCK_HW2,
            num_warps=4, num_stages=2,
        )

        # Final residual add: out2_norm + x_f32
        out = torch.empty_like(out2_norm)
        residual_add_kernel[(B, C_out2, H * W)](
            out2_norm, x_f32, out,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
