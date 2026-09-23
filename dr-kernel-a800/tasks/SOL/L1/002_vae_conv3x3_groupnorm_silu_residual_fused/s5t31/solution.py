import torch
import triton
import triton.language as tl


# Conv3x3 via im2col + GEMM-like Triton kernel:
# For each (n, c_out in [c_start, c_start+BLOCK_CO), h_out tile, w_out tile),
# compute output tile for each input channel and 3x3 patch, accumulate.
@triton.jit
def conv3x3_tile_gemm_kernel(
    x_ptr,          # *f32, [B, C_in, H, W] contiguous
    w_ptr,          # *f32, [C_out, C_in, 3, 3] contiguous
    y_ptr,          # *f32, [B, C_out, H, W] contiguous
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    C_out: tl.constexpr,
    BLOCK_CO: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    c_start = tl.program_id(1) * BLOCK_CO
    h_tile = tl.program_id(2)
    w_tile = tl.program_id(3)

    h_offs = h_tile * BLOCK_HW // W  # number of h tiles, but Triton uses one-dim tile; derive from h_tile, w_tile
    w_offs = w_tile * BLOCK_HW % W
    hw_offs = h_offs * W + w_offs + tl.arange(0, BLOCK_HW)
    total = H * W
    N_TILES = (total + BLOCK_HW - 1) // BLOCK_HW

    # Accumulator for output tile
    acc = tl.zeros([BLOCK_CO, BLOCK_HW], dtype=tl.float32)

    # Loop over input channels and 3x3 patch
    for cin in range(0, C_in):
        for dh in range(-1, 2):
            for dw in range(-1, 2):
                # Compute input spatial indices for this 3x3 position
                h_in = hw_offs // W + dh
                w_in = hw_offs % W + dw
                valid = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                base_x = n * C_in * H * W + cin * H * W
                idx = base_x + h_in * W + w_in
                x_vals = tl.load(x_ptr + idx, mask=valid, other=0.0)

                # Load corresponding weights for each output channel in tile
                co_range = c_start + tl.arange(0, BLOCK_CO)
                mask_co = co_range < C_out
                base_w = 0
                # Unroll over patch offset: (dh, dw) maps to weight's spatial index (3+dh, 3+dw)
                # weight layout: [C_out, C_in, 3, 3] contiguous => per (co, cin, dh2, dw2)
                for dh2 in range(0, 3):
                    for dw2 in range(0, 3):
                        if (dh2 == dh) and (dw2 == dw):
                            w_idx = base_w + co_range * (C_in * 9) + cin * 9 + (dh2 * 3 + dw2)
                            w_vals = tl.load(w_ptr + w_idx, mask=mask_co, other=0.0)  # shape [BLOCK_CO]
                            acc += w_vals[:, None] * x_vals[None, :]
    # Write back accumulated outputs
    for i in range(0, BLOCK_CO):
        co = c_start + i
        co_mask = co < C_out
        y_base = n * C_out * H * W + co * H * W
        y_idx = y_base + hw_offs
        tl.store(y_ptr + y_idx, acc[i, :], mask=co_mask)


# GroupNorm reduction: per (n, group, channel), compute sum and sumsq across H*W
@triton.jit
def group_norm_reduce_kernel(
    x_ptr,      # *f32, [B, C, H, W] contiguous
    mean_ptr,   # *f32, [C]
    rstd_ptr,   # *f32, [C]
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


# GroupNorm apply + affine + SiLU: per (n, group, channel, tile) over H*W
@triton.jit
def group_norm_apply_affine_silu_kernel(
    x_ptr,          # *f32, [B, C, H, W]
    mean_ptr,       # *f32, [C]
    rstd_ptr,       # *f32, [C]
    scale_ptr,      # *f32, [C]
    bias_ptr,       # *f32, [C]
    y_ptr,          # *f32, [B, C, H, W]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    c_start = g * (C // num_groups)
    c = tl.program_id(2) + c_start
    total = H * W
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
        y_vals = (x_vals - mean) * rstd
        # SiLU: y * sigmoid(y)
        s = 1.0 / (1.0 + tl.exp(-y_vals))
        y_vals = y_vals * s
        y_vals = y_vals * scale + bias
        tl.store(y_ptr + idx, y_vals, mask=mask)


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

    def forward(self,
                x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        x: (B, C_in, H, W), conv weights: (C_out, C_in, 3, 3), norm scales/bias: (C,)
        Returns: (B, C_out2, H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"
        B, C_in, H, W = x.shape

        # Ensure contiguous and float32
        x_f32 = x.contiguous().to(torch.float32)

        # First conv3x3
        C_out1 = conv1_weight.shape[0]
        out1 = torch.empty((B, C_out1, H, W), device=x.device, dtype=torch.float32)
        conv1_w_f32 = conv1_weight.contiguous().to(torch.float32)  # (C_out1, C_in, 3, 3)

        BLOCK_CO = 16
        BLOCK_HW = 128  # tile across spatial
        grid_conv1 = (B, triton.cdiv(C_out1, BLOCK_CO), triton.cdiv(H * W, BLOCK_HW))
        conv3x3_tile_gemm_kernel[grid_conv1](
            x_f32, conv1_w_f32, out1,
            B=B, C_in=C_in, H=H, W=W,
            C_out=C_out1,
            BLOCK_CO=BLOCK_CO, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2,
        )

        # GroupNorm + SiLU for first block
        out1_norm = torch.empty_like(out1)
        # Compute mean/rstd
        mean1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        rstd1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        group_norm_reduce_kernel[(B, self.num_groups, C_out1 // self.num_groups)](
            out1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups, eps=self.eps,
            num_warps=4, num_stages=2,
        )
        # Apply affine + SiLU
        group_norm_apply_affine_silu_kernel[(B, self.num_groups, C_out1 // self.num_groups, (H * W + 127) // 128)](
            out1, mean1, rstd1, norm1_weight_f32, norm1_bias_f32, out1_norm,
            B=B, C=C_out1, H=H, W=W,
            num_groups=self.num_groups,
            num_warps=4, num_stages=2,
        )

        # Second conv3x3
        C_out2 = conv2_weight.shape[0]
        out2 = torch.empty((B, C_out2, H, W), device=x.device, dtype=torch.float32)
        conv2_w_f32 = conv2_weight.contiguous().to(torch.float32)  # (C_out2, C_out1, 3, 3)

        grid_conv2 = (B, triton.cdiv(C_out2, BLOCK_CO), triton.cdiv(H * W, BLOCK_HW))
        conv3x3_tile_gemm_kernel[grid_conv2](
            out1_norm, conv2_w_f32, out2,
            B=B, C_in=C_out1, H=H, W=W,
            C_out=C_out2,
            BLOCK_CO=BLOCK_CO, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2,
        )

        # GroupNorm + SiLU for second block
        out2_norm = torch.empty_like(out2)
        mean2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        rstd2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        group_norm_reduce_kernel[(B, self.num_groups, C_out2 // self.num_groups)](
            out2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups, eps=self.eps,
            num_warps=4, num_stages=2,
        )
        group_norm_apply_affine_silu_kernel[(B, self.num_groups, C_out2 // self.num_groups, (H * W + 127) // 128)](
            out2, mean2, rstd2, norm2_weight_f32, norm2_bias_f32, out2_norm,
            B=B, C=C_out2, H=H, W=W,
            num_groups=self.num_groups,
            num_warps=4, num_stages=2,
        )

        # Final residual add
        out = torch.empty_like(out2_norm)
        residual_add_kernel[(B, C_out2, H * W)](
            out2_norm, x_f32, out,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
