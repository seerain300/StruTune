import torch
import triton
import triton.language as tl


# Triton kernel: compute per-channel mean and rstd over H*W (GroupNorm-like reduction)
@triton.jit
def groupnorm_reduce_per_channel(
    x_ptr,           # *f32, input [B, C, H, W]
    mean_ptr,        # *f32, [C]
    rstd_ptr,        # *f32, [C]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_HW: tl.constexpr, N_TILES: tl.constexpr
):
    n = tl.program_id(0)
    c = tl.program_id(1)

    sum_val = 0.0
    sum_sq = 0.0

    for t in range(N_TILES):
        tile_start = t * BLOCK_HW
        offs = tile_start + tl.arange(0, BLOCK_HW)
        mask = offs < (H * W)
        h = offs // W
        w = offs % W
        idx = (((n * C) + c) * H + h) * W + w
        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    m = H * W
    mean = sum_val / m
    var = sum_sq / m - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    tl.store(mean_ptr + c, mean)
    tl.store(rstd_ptr + c, rstd)


# Triton kernel: apply GroupNorm normalization + affine + SiLU (per-channel)
@triton.jit
def groupnorm_apply_per_channel(
    x_ptr,           # *f32, input [B, C, H, W]
    y_ptr,           # *f32, output [B, C, H, W]
    mean_ptr,        # *f32, [C]
    rstd_ptr,        # *f32, [C]
    gamma_ptr,       # *f32, [C] (norm weight)
    beta_ptr,        # *f32, [C] (norm bias)
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_HW: tl.constexpr, N_TILES: tl.constexpr
):
    n = tl.program_id(0)
    c = tl.program_id(1)

    mean = tl.load(mean_ptr + c)
    rstd = tl.load(rstd_ptr + c)
    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)

    for t in range(N_TILES):
        tile_start = t * BLOCK_HW
        offs = tile_start + tl.arange(0, BLOCK_HW)
        mask = offs < (H * W)
        h = offs // W
        w = offs % W
        idx = (((n * C) + c) * H + h) * W + w

        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        z = (x - mean) * rstd
        z = z * gamma + beta

        # SiLU activation: z * sigmoid(z)
        sig = 1.0 / (1.0 + tl.exp(-z))
        y = z * sig

        tl.store(y_ptr + idx, y, mask=mask)


# Triton kernel: elementwise residual add y = y + x
@triton.jit
def residual_add_kernel(
    a_ptr, b_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr
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
        conv weights: (C_out, C_in, 3, 3)
        norm scales/bias: (C,)
        Returns: (B, C_out2, H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"
        B, C_in, H, W = x.shape

        # Ensure float32 and contiguous
        x_f32 = x.contiguous().to(torch.float32)

        # First Conv: use PyTorch for correctness and speed
        C_out1 = int(conv1_weight.shape[0])
        out1 = torch.nn.functional.conv2d(x_f32, conv1_weight.to(torch.float32), bias=None, stride=1, padding=1)

        # GroupNorm + SiLU per-channel using Triton
        mean1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        rstd1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)

        BLOCK_HW = 1024
        N_TILES1 = (H * W + BLOCK_HW - 1) // BLOCK_HW

        groupnorm_reduce_per_channel[(B, C_out1)](
            out1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W,
            BLOCK_HW=BLOCK_HW, N_TILES=N_TILES1,
            num_warps=4, num_stages=2,
        )

        out1_norm = torch.empty_like(out1)

        groupnorm_apply_per_channel[(B, C_out1)](
            out1, out1_norm, mean1, rstd1, norm1_weight.to(torch.float32), norm1_bias.to(torch.float32),
            B=B, C=C_out1, H=H, W=W,
            BLOCK_HW=BLOCK_HW, N_TILES=N_TILES1,
            num_warps=4, num_stages=2,
        )

        # Second Conv: PyTorch
        C_out2 = int(conv2_weight.shape[0])
        out2 = torch.nn.functional.conv2d(out1_norm, conv2_weight.to(torch.float32), bias=None, stride=1, padding=1)

        # GroupNorm + SiLU per-channel using Triton
        mean2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        rstd2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)

        BLOCK_HW2 = 1024
        N_TILES2 = (H * W + BLOCK_HW2 - 1) // BLOCK_HW2

        groupnorm_reduce_per_channel[(B, C_out2)](
            out2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W,
            BLOCK_HW=BLOCK_HW2, N_TILES=N_TILES2,
            num_warps=4, num_stages=2,
        )

        out2_norm = torch.empty_like(out2)

        groupnorm_apply_per_channel[(B, C_out2)](
            out2, out2_norm, mean2, rstd2, norm2_weight.to(torch.float32), norm2_bias.to(torch.float32),
            B=B, C=C_out2, H=H, W=W,
            BLOCK_HW=BLOCK_HW2, N_TILES=N_TILES2,
            num_warps=4, num_stages=2,
        )

        # Final residual add: Triton
        out = torch.empty_like(out2_norm)
        residual_add_kernel[(B, C_out2, H * W)](
            out2_norm, x_f32, out,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
