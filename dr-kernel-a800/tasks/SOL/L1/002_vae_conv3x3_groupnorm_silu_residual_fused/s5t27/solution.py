import torch
import triton
import triton.language as tl


# Conv3x3: per-pixel reduction
# y[n, co, h_out, w_out] = sum_{ci, dh in {0,1,2}, dw in {0,1,2}} x[n, ci, h_out+dh, w_out+dw] * w[co, ci, 3+dh, 3+dw]
@triton.jit
def conv3x3_pixel_kernel(
    x_ptr,            # *f32, input [B, C_in, H, W]
    w_ptr,            # *f32, weights [C_out, C_in, 3, 3]
    y_ptr,            # *f32, output [B, C_out, H, W]
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    co = tl.program_id(1)  # output channel index
    h_out = tl.program_id(2)
    w_out = tl.program_id(3)

    acc = 0.0  # scalar accumulation for this (n, co, h_out, w_out)

    # Loop over input channels
    for ci in range(0, C_in):
        # Loop over 3x3 neighborhood
        for dh in range(0, 3):
            for dw in range(0, 3):
                h = h_out + dh
                w = w_out + dw
                # Check bounds (padding=1: always valid, but keep for generality)
                if (h >= 0) & (h < H) & (w >= 0) & (w < W):
                    x_idx = (((n * C_in) + ci) * H + h) * W + w
                    x_val = tl.load(x_ptr + x_idx)
                    w_idx = (((co * C_in) + ci) * 9 + (dh * 3 + dw))
                    w_val = tl.load(w_ptr + w_idx)
                    acc += x_val * w_val

    out_idx = (((n * C_out) + co) * H + h_out) * W + w_out
    tl.store(y_ptr + out_idx, acc)


# GroupNorm reduction: per (n, group, channel) compute sum and sumsq across spatial HW
@triton.jit
def group_norm_reduce_single(
    x_ptr, mean_ptr, rstd_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
):
    n = tl.program_id(0)
    group = tl.program_id(1)
    c = tl.program_id(2)

    channels_per_group = C // num_groups
    sum_val = 0.0
    sumsq_val = 0.0

    TILE_SIZE = 1024  # chunk for reduction over H*W
    for tile in range(0, (H * W) // TILE_SIZE):
        base = tile * TILE_SIZE
        offs = base + tl.arange(0, TILE_SIZE)
        mask = offs < (H * W)

        h_vec = offs // W
        w_vec = offs % W
        idx = (((n * C) + c) * H + h_vec) * W + w_vec

        x_vec = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sum_tile = tl.sum(x_vec, axis=0)
        sumsq_tile = tl.sum(x_vec * x_vec, axis=0)

        sum_val += sum_tile
        sumsq_val += sumsq_tile

    M = H * W
    mean = sum_val / M
    var = sumsq_val / M - mean * mean
    # numerical guard
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + 1e-12)

    # Store per-(n, c) mean and rstd
    tl.store(mean_ptr + (n * C + c), mean)
    tl.store(rstd_ptr + (n * C + c), rstd)


# GroupNorm apply: y = ((x - mean[c]) * rstd[c]) * gamma[c] + beta[c], then SiLU
@triton.jit
def group_norm_apply_kernel(
    x_ptr, mean_ptr, rstd_ptr, weight_ptr, bias_ptr, y_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    n = tl.program_id(0)
    group = tl.program_id(1)
    c = tl.program_id(2)
    tile = tl.program_id(3)

    channels_per_group = C // num_groups

    base = tile * TILE_SIZE
    offs = base + tl.arange(0, TILE_SIZE)
    mask = offs < (H * W)

    h_vec = offs // W
    w_vec = offs % W
    idx = (((n * C) + c) * H + h_vec) * W + w_vec

    x_vec = tl.load(x_ptr + idx, mask=mask, other=0.0)
    mean_c = tl.load(mean_ptr + (n * C + c))
    rstd_c = tl.load(rstd_ptr + (n * C + c))
    gamma_c = tl.load(weight_ptr + c)
    beta_c = tl.load(bias_ptr + c)

    norm_vec = (x_vec - mean_c) * rstd_c
    # affine
    norm_vec = norm_vec * gamma_c + beta_c
    # SiLU: norm * sigmoid(norm) = norm / (1 + exp(-norm))
    sig = 1.0 / (1.0 + tl.exp(-norm_vec))
    y_vec = norm_vec * sig

    tl.store(y_ptr + idx, y_vec, mask=mask)


# Elementwise residual add: out = out + x
@triton.jit
def residual_add_kernel(
    a_ptr, b_ptr, out_ptr,
    total_elems: tl.constexpr, CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * CHUNK + tl.arange(0, CHUNK)
    mask = offs < total_elems
    a_val = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b_val = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a_val + b_val, mask=mask)


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
        # Ensure CUDA tensors
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"
        B, C_in, H, W = x.shape

        # Work in float32 for numerical stability
        x_f32 = x.contiguous().to(torch.float32)
        conv1_w_f32 = conv1_weight.contiguous().to(torch.float32)  # (C_out1, C_in, 3, 3)
        conv2_w_f32 = conv2_weight.contiguous().to(torch.float32)  # (C_out2, C_in, 3, 3)
        norm1_weight_f32 = norm1_weight.contiguous().to(torch.float32)
        norm1_bias_f32 = norm1_bias.contiguous().to(torch.float32)
        norm2_weight_f32 = norm2_weight.contiguous().to(torch.float32)
        norm2_bias_f32 = norm2_bias.contiguous().to(torch.float32)

        # Output tensors
        out1 = torch.empty((B, conv1_w_f32.shape[0], H, W), device=x.device, dtype=torch.float32)
        out2 = torch.empty((B, conv2_w_f32.shape[0], H, W), device=x.device, dtype=torch.float32)

        # Launch conv1: per-pixel reduction
        grid_conv1 = (B, conv1_w_f32.shape[0], H, W)
        conv3x3_pixel_kernel[grid_conv1](
            x_f32, conv1_w_f32, out1,
            B=B, C_in=C_in, C_out=conv1_w_f32.shape[0], H=H, W=W,
            num_warps=4, num_stages=2,
        )

        # GroupNorm + SiLU for first block
        C_out1 = conv1_w_f32.shape[0]
        channels_per_group1 = C_out1 // self.num_groups
        assert C_out1 % self.num_groups == 0, "num_groups must divide C_out for GroupNorm"

        mean1 = torch.empty((B * C_out1), device=x.device, dtype=torch.float32)
        rstd1 = torch.empty((B * C_out1), device=x.device, dtype=torch.float32)

        grid_reduce1 = (B, self.num_groups, C_out1 // self.num_groups)
        group_norm_reduce_single[grid_reduce1](
            out1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W, num_groups=self.num_groups,
            num_warps=4, num_stages=2,
        )

        y1 = torch.empty_like(out1)
        TILE_SIZE = 1024
        grid_apply1 = (B, self.num_groups, C_out1 // self.num_groups, (H * W + TILE_SIZE - 1) // TILE_SIZE)
        group_norm_apply_kernel[grid_apply1](
            out1, mean1, rstd1, norm1_weight_f32, norm1_bias_f32, y1,
            B=B, C=C_out1, H=H, W=W, num_groups=self.num_groups, TILE_SIZE=TILE_SIZE,
            num_warps=4, num_stages=2,
        )

        # Launch conv2: per-pixel reduction
        grid_conv2 = (B, conv2_w_f32.shape[0], H, W)
        conv3x3_pixel_kernel[grid_conv2](
            y1, conv2_w_f32, out2,
            B=B, C_in=C_out1, C_out=conv2_w_f32.shape[0], H=H, W=W,
            num_warps=4, num_stages=2,
        )

        # GroupNorm + SiLU for second block
        C_out2 = conv2_w_f32.shape[0]
        channels_per_group2 = C_out2 // self.num_groups
        assert C_out2 % self.num_groups == 0, "num_groups must divide C_out for GroupNorm"

        mean2 = torch.empty((B * C_out2), device=x.device, dtype=torch.float32)
        rstd2 = torch.empty((B * C_out2), device=x.device, dtype=torch.float32)

        grid_reduce2 = (B, self.num_groups, C_out2 // self.num_groups)
        group_norm_reduce_single[grid_reduce2](
            out2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W, num_groups=self.num_groups,
            num_warps=4, num_stages=2,
        )

        y2 = torch.empty_like(out2)
        TILE_SIZE = 1024
        grid_apply2 = (B, self.num_groups, C_out2 // self.num_groups, (H * W + TILE_SIZE - 1) // TILE_SIZE)
        group_norm_apply_kernel[grid_apply2](
            out2, mean2, rstd2, norm2_weight_f32, norm2_bias_f32, y2,
            B=B, C=C_out2, H=H, W=W, num_groups=self.num_groups, TILE_SIZE=TILE_SIZE,
            num_warps=4, num_stages=2,
        )

        # Residual add: y2 + x
        out = torch.empty_like(y2)
        total_elems = B * C_out2 * H * W
        CHUNK = 4096
        grid_add = ((total_elems + CHUNK - 1) // CHUNK,)
        residual_add_kernel[grid_add](
            y2, x_f32, out,
            total_elems=total_elems, CHUNK=CHUNK,
            num_warps=4, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
