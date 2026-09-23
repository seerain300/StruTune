import torch
import triton
import triton.language as tl


@triton.jit
def groupnorm_reduce_kernel(
    x_ptr,                # *f32, input [B, C, H, W]
    mean_ptr,             # *f32, output [C]
    rstd_ptr,             # *f32, output [C]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    N_TILES: tl.constexpr,  # number of tiles to cover H*W
):
    # program ids
    n = tl.program_id(0)
    g = tl.program_id(1)
    c = tl.program_id(2)

    channels_per_group = C // num_groups
    # map program c to its channel within the group
    start_c = g * channels_per_group
    c_in_group = c - start_c
    ch = start_c + c_in_group

    total_sum = 0.0
    total_sumsq = 0.0
    for t in range(N_TILES):
        tile_start = t * BLOCK_HW
        offs = tile_start + tl.arange(0, BLOCK_HW)
        mask = offs < (H * W)
        base = (n * C + ch) * (H * W)
        x_vec = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        total_sum += tl.sum(x_vec, axis=0)
        total_sumsq += tl.sum(x_vec * x_vec, axis=0)

    numel = H * W
    mean = total_sum / numel
    var = total_sumsq / numel - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(mean_ptr + ch, mean)
    tl.store(rstd_ptr + ch, rstd)


@triton.jit
def groupnorm_apply_silu_kernel(
    x_ptr,                # *f32, input [B, C, H, W] (post-conv)
    mean_ptr,             # *f32, [C]
    rstd_ptr,             # *f32, [C]
    gamma_ptr,            # *f32, [C] (norm1_weight / norm2_weight)
    beta_ptr,             # *f32, [C] (norm1_bias / norm2_bias)
    y_ptr,                # *f32, output [B, C, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    N_TILES: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    c = tl.program_id(2)
    t = tl.program_id(3)  # tile index over spatial HW

    channels_per_group = C // num_groups
    start_c = g * channels_per_group
    c_in_group = c - start_c
    ch = start_c + c_in_group

    mean = tl.load(mean_ptr + ch)
    rstd = tl.load(rstd_ptr + ch)
    gamma = tl.load(gamma_ptr + ch)
    beta = tl.load(beta_ptr + ch)

    tile_start = t * BLOCK_HW
    offs = tile_start + tl.arange(0, BLOCK_HW)
    mask = offs < (H * W)

    base = (n * C + ch) * (H * W)
    x_vec = tl.load(x_ptr + base + offs, mask=mask, other=0.0)

    # normalized affine
    norm = (x_vec - mean) * rstd
    norm = norm * gamma + beta

    # SiLU: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-norm))
    y_vec = norm * sig

    tl.store(y_ptr + base + offs, y_vec, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32):
        super().__init__()
        self.num_groups = num_groups

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

        # Ensure all tensors are float32 and contiguous for Triton/PyTorch ops
        x_f32 = x.contiguous().to(torch.float32)

        # First path: Conv3x3 -> GroupNorm -> SiLU
        out1 = torch.nn.functional.conv2d(
            x_f32, conv1_weight.contiguous().to(torch.float32), bias=None, stride=1, padding=1
        )  # shape (B, C_in, H, W) — this is conv1 output's channels, not necessarily same as input C_in
        C1 = conv1_weight.shape[1]  # out channels of conv1
        assert (C1 % self.num_groups) == 0, "C1 must be divisible by num_groups for GroupNorm"

        # GroupNorm reduction
        BLOCK_HW = 1024
        N_TILES = triton.cdiv(H * W, BLOCK_HW)
        mean1 = torch.empty(C1, device=x.device, dtype=torch.float32)
        rstd1 = torch.empty(C1, device=x.device, dtype=torch.float32)

        groupnorm_reduce_kernel[(B, self.num_groups, C1 // self.num_groups)](
            out1, mean1, rstd1,
            B=B, C=C1, H=H, W=W,
            num_groups=self.num_groups,
            BLOCK_HW=BLOCK_HW,
            N_TILES=N_TILES,
            num_warps=4,
            num_stages=2,
        )

        # GroupNorm apply + SiLU
        out1_norm = torch.empty_like(out1)
        groupnorm_apply_silu_kernel[(B, self.num_groups, C1 // self.num_groups, N_TILES)](
            out1, mean1, rstd1, norm1_weight.contiguous().to(torch.float32), norm1_bias.contiguous().to(torch.float32),
            out1_norm,
            B=B, C=C1, H=H, W=W,
            num_groups=self.num_groups,
            BLOCK_HW=BLOCK_HW,
            N_TILES=N_TILES,
            num_warps=4,
            num_stages=2,
        )

        # Second path: Conv3x3 -> GroupNorm -> SiLU
        out2 = torch.nn.functional.conv2d(
            out1_norm, conv2_weight.contiguous().to(torch.float32), bias=None, stride=1, padding=1
        )  # shape (B, C2, H, W)
        C2 = conv2_weight.shape[0]  # final output channels
        assert (C2 % self.num_groups) == 0, "C2 must be divisible by num_groups for GroupNorm"

        mean2 = torch.empty(C2, device=x.device, dtype=torch.float32)
        rstd2 = torch.empty(C2, device=x.device, dtype=torch.float32)

        groupnorm_reduce_kernel[(B, self.num_groups, C2 // self.num_groups)](
            out2, mean2, rstd2,
            B=B, C=C2, H=H, W=W,
            num_groups=self.num_groups,
            BLOCK_HW=BLOCK_HW,
            N_TILES=N_TILES,
            num_warps=4,
            num_stages=2,
        )

        out2_norm = torch.empty_like(out2)
        groupnorm_apply_silu_kernel[(B, self.num_groups, C2 // self.num_groups, N_TILES)](
            out2, mean2, rstd2, norm2_weight.contiguous().to(torch.float32), norm2_bias.contiguous().to(torch.float32),
            out2_norm,
            B=B, C=C2, H=H, W=W,
            num_groups=self.num_groups,
            BLOCK_HW=BLOCK_HW,
            N_TILES=N_TILES,
            num_warps=4,
            num_stages=2,
        )

        # Final residual add: out2_norm + x
        out = out2_norm + x_f32

        return out


def run(*args):
    return ModelNew()(*args)
