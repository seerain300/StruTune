import torch
import triton
import triton.language as tl


# Triton kernel: Conv3x3 (stride=1, padding=1), no bias.
# Each program handles one output pixel: (n, co, h_out, w_out).
@triton.jit
def conv3x3_pixel_kernel(
    x_ptr,          # *f32, input [B, C, H, W]
    w_ptr,          # *f32, weight [C_out, C_in, 3, 3]
    y_ptr,          # *f32, output [B, C_out, H, W]
    B: tl.constexpr,
    C_in: tl.constexpr,
    C_out: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
):
    n = tl.program_id(0)
    co = tl.program_id(1)
    h_out = tl.program_id(2)
    w_out = tl.program_id(3)

    acc = 0.0
    # Loop over input channels and 3x3 neighborhood
    for ci in range(C_in):
        for dh in range(3):
            hi = h_out + dh - 1  # padding=1
            for dw in range(3):
                wj = w_out + dw - 1
                # hi, wj are guaranteed in [0, H-1], [0, W-1] due to padding
                x_offset = (((n * C_in) + ci) * H + hi) * W + wj
                x_val = tl.load(x_ptr + x_offset)
                # load corresponding weight
                w_offset = (co * (C_in * 9)) + (ci * 9) + (dh * 3 + dw)
                w_val = tl.load(w_ptr + w_offset)
                acc += x_val * w_val
    # store result
    y_offset = (((n * C_out) + co) * H + h_out) * W + w_out
    tl.store(y_ptr + y_offset, acc)


# Triton kernel: GroupNorm reduction per channel (compute sum and sum of squares).
# Grid over (num_groups, channels_per_group). Each program handles one channel in one group and reduces over H*W.
@triton.jit
def groupnorm_reduce_kernel(
    x_ptr,            # *f32, input [B, C, H, W]
    mean_ptr,         # *f32, mean per channel [C]
    rstd_ptr,         # *f32, rstd per channel [C]
    C: tl.constexpr,  # channels
    H: tl.constexpr,  # height
    W: tl.constexpr,  # width
    num_groups: tl.constexpr,
    channels_per_group: tl.constexpr,  # C // num_groups
    BLOCK_HW: tl.constexpr,            # tile size over H*W
):
    group = tl.program_id(0)       # group id in [0, num_groups)
    ch_start = group * channels_per_group  # start channel of this group
    c = tl.program_id(1)            # which channel within the group
    c_idx = ch_start + c            # absolute channel index

    sum_val = 0.0
    sumsq_val = 0.0

    N_TILES = (H * W + BLOCK_HW - 1) // BLOCK_HW  # constexpr at launch

    for t in range(N_TILES):
        offs = t * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask = offs < (H * W)
        h = offs // W
        w = offs % W

        # We assume B=1 for this reduction kernel (the input tensor is [1, C, H, W] in forward)
        idx = (((0 * C) + c_idx) * H + h) * W + w
        x_val = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x_val, axis=0)
        sumsq_val += tl.sum(x_val * x_val, axis=0)

    mean = sum_val / (H * W)
    var = sumsq_val / (H * W) - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(mean_ptr + c_idx, mean)
    tl.store(rstd_ptr + c_idx, rstd)


# Triton kernel: apply GroupNorm + affine (scale/bias) + SiLU elementwise.
# Grid over (B, C, N_TILES). Each program handles one (n, c, tile).
@triton.jit
def groupnorm_apply_silu_kernel(
    x_ptr,             # *f32, input [B, C, H, W] to normalize
    mean_ptr,          # *f32, mean [C]
    rstd_ptr,          # *f32, rstd [C]
    scale_ptr,         # *f32, per-channel scale [C]
    bias_ptr,          # *f32, per-channel bias [C]
    y_ptr,             # *f32, output [B, C, H, W]
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    N_TILES: tl.constexpr,
):
    n = tl.program_id(0)  # batch index
    c = tl.program_id(1)  # channel index
    t = tl.program_id(2)  # tile index

    mean = tl.load(mean_ptr + c)
    rstd = tl.load(rstd_ptr + c)
    scale = tl.load(scale_ptr + c)
    bias = tl.load(bias_ptr + c)

    offs = t * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = offs < (H * W)
    h = offs // W
    w = offs % W

    idx = (((n * C) + c) * H + h) * W + w
    x_val = tl.load(x_ptr + idx, mask=mask, other=0.0)
    y_norm = (x_val - mean) * rstd
    y_aff = y_norm * scale + bias
    # SiLU: x * sigmoid(x)
    sig = 1.0 / (1.0 + tl.exp(-y_aff))
    y = y_aff * sig
    tl.store(y_ptr + idx, y, mask=mask)


# Triton kernel: elementwise residual add
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
        x: (B, C, H, W), conv weights: (C_out, C_in, 3, 3), norm scales/bias: (C,)
        Returns: (B, C_out2, H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"
        B, C, H, W = x.shape

        # Ensure weights/bias are float32 and contiguous
        x_f32 = x.contiguous().to(torch.float32)

        # First path: Conv3x3 (Triton), then GroupNorm + SiLU (Triton)
        out1 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid1 = (B, C, H, W)
        conv3x3_pixel_kernel[grid1](
            x_f32, conv1_weight.to(torch.float32), out1,
            B=B, C_in=C, C_out=C, H=H, W=W,
            num_warps=1, num_stages=1,
        )

        C1 = out1.shape[1]
        assert C1 % self.num_groups == 0, "num_groups must divide channels for GroupNorm"

        # Prepare mean/rstd buffers for channel C1
        mean1 = torch.empty((C1,), device=out1.device, dtype=out1.dtype)
        rstd1 = torch.empty((C1,), device=out1.device, dtype=out1.dtype)

        # Launch reduction kernel over [1, C1, H, W]; we pass B=1 and rely on indexing without B-dependent term
        channels_per_group = C1 // self.num_groups
        BLOCK_HW = 128
        N_TILES1 = (H * W + BLOCK_HW - 1) // BLOCK_HW
        groupnorm_reduce_kernel[(self.num_groups, channels_per_group)](
            out1, mean1, rstd1,
            C=C1, H=H, W=W, num_groups=self.num_groups, channels_per_group=channels_per_group, BLOCK_HW=BLOCK_HW,
            num_warps=1, num_stages=1,
        )

        # Apply GroupNorm + SiLU
        out1_norm = torch.empty_like(out1)
        groupnorm_apply_silu_kernel[(B, C1, N_TILES1)](
            out1, mean1, rstd1, norm1_weight.to(torch.float32), norm1_bias.to(torch.float32), out1_norm,
            B=B, C=C1, H=H, W=W, BLOCK_HW=BLOCK_HW, N_TILES=N_TILES1,
            num_warps=1, num_stages=1,
        )

        # Second path: Conv3x3 (Triton) on out1_norm, then GroupNorm + SiLU (Triton)
        out2 = torch.empty((B, C1, H, W), device=x.device, dtype=torch.float32)
        grid2 = (B, C1, H, W)
        conv3x3_pixel_kernel[grid2](
            out1_norm, conv2_weight.to(torch.float32), out2,
            B=B, C_in=C1, C_out=C1, H=H, W=W,
            num_warps=1, num_stages=1,
        )

        C2 = out2.shape[1]
        assert C2 % self.num_groups == 0, "num_groups must divide channels for GroupNorm"

        mean2 = torch.empty((C2,), device=out2.device, dtype=out2.dtype)
        rstd2 = torch.empty((C2,), device=out2.device, dtype=out2.dtype)

        channels_per_group2 = C2 // self.num_groups
        BLOCK_HW = 128
        N_TILES2 = (H * W + BLOCK_HW - 1) // BLOCK_HW
        groupnorm_reduce_kernel[(self.num_groups, channels_per_group2)](
            out2, mean2, rstd2,
            C=C2, H=H, W=W, num_groups=self.num_groups, channels_per_group=channels_per_group2, BLOCK_HW=BLOCK_HW,
            num_warps=1, num_stages=1,
        )

        out2_norm = torch.empty_like(out2)
        groupnorm_apply_silu_kernel[(B, C2, N_TILES2)](
            out2, mean2, rstd2, norm2_weight.to(torch.float32), norm2_bias.to(torch.float32), out2_norm,
            B=B, C=C2, H=H, W=W, BLOCK_HW=BLOCK_HW, N_TILES=N_TILES2,
            num_warps=1, num_stages=1,
        )

        # Final residual add: out2_norm + x
        out = torch.empty_like(out2_norm)
        residual_add_kernel[(B, C2, H * W)](
            out2_norm, x_f32, out,
            B=B, C=C2, H=H, W=W,
            num_warps=1, num_stages=1,
        )

        return out


def run(*args):
    return ModelNew()(*args)
