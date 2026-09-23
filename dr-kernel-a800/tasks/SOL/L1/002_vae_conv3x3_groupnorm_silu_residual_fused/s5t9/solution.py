import torch
import triton
import triton.language as tl


# Conv3x3: y[n, c_out, h_out, w_out] = sum_{c_in=0..C_in-1} sum_{dh,dw in 3x3} x[n, c_in, h_out+dh, w_out+dw] * w[c_out, c_in, 3+dh, 3+dw]
# Stride=1, padding=1, no bias. Each program computes one output pixel. Robust for dynamic shapes.
@triton.jit
def conv3x3_kernel(
    x_ptr, w_ptr, y_ptr,
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    h_out = tl.program_id(2)
    w_out = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and 3x3 neighborhood
    for c_in in range(C_in):
        for dh in range(3):
            h_in = h_out + dh - 1  # padding=1: indices always valid
            for dw in range(3):
                w_in = w_out + dw - 1
                # Load x value
                x_idx = ((n * C_in) + c_in) * (H * W) + h_in * W + w_in
                x_val = tl.load(x_ptr + x_idx)  # scalar load
                # Load weight value
                w_idx = ((c_out * C_in) + c_in) * 9 + (dh * 3 + dw)  # weight layout: (C_out, C_in, 3, 3)
                w_val = tl.load(w_ptr + w_idx)
                acc += x_val * w_val

    # Store result
    y_idx = ((n * C_out) + c_out) * (H * W) + h_out * W + w_out
    tl.store(y_ptr + y_idx, acc)


# GroupNorm reduction: compute per-channel mean and rstd over spatial HW for each (n, group, c)
@triton.jit
def group_norm_reduce_kernel(
    x_ptr, mean_ptr, rstd_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)  # group index
    c = tl.program_id(2)  # channel index within the group
    group_size = C // num_groups  # channels per group
    ch = g * group_size + c

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)
    HW = H * W
    # Loop over tiles in the spatial plane
    for tile in range(N_TILES):  # N_TILES is provided as tl.constexpr meta
        start = tile * BLOCK_HW
        offsets = start + tl.arange(0, BLOCK_HW)
        mask = offsets < HW
        ptr = x_ptr + (((n * C) + ch) * HW) + offsets
        vals = tl.load(ptr, mask=mask, other=0.0)
        # Reduce sum and sumsq for this tile
        # Note: tl.sum expects a vector; we reduce over BLOCK_HW
        sum_val += tl.sum(vals, axis=0)
        sum_sq += tl.sum(vals * vals, axis=0)

    mean = sum_val / HW
    var = sum_sq / HW - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(mean_ptr + ch, mean)
    tl.store(rstd_ptr + ch, rstd)


# GroupNorm apply: apply normalization + affine (scale, bias) + SiLU for each tile
@triton.jit
def group_norm_apply_kernel(
    x_ptr, mean_ptr, rstd_ptr, scale_ptr, bias_ptr, y_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    c = tl.program_id(2)
    tile = tl.program_id(3)
    group_size = C // num_groups
    ch = g * group_size + c

    mean = tl.load(mean_ptr + ch)
    rstd = tl.load(rstd_ptr + ch)
    scale = tl.load(scale_ptr + ch)
    bias = tl.load(bias_ptr + ch)

    start = tile * BLOCK_HW
    offsets = start + tl.arange(0, BLOCK_HW)
    mask = offsets < (H * W)
    HW = H * W

    x_vals = tl.load(x_ptr + (((n * C) + ch) * HW) + offsets, mask=mask, other=0.0)
    # Normalize: (x - mean) * rstd
    norm = (x_vals - mean) * rstd
    # Affine
    norm = norm * scale + bias
    # SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    sig = 1.0 / (1.0 + tl.exp(-norm))
    out = norm * sig

    tl.store(y_ptr + (((n * C) + ch) * HW) + offsets, out, mask=mask)


# Elementwise residual add in Triton: out = y + x
@triton.jit
def residual_add_kernel(
    y_ptr, x_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    hw = tl.program_id(2)
    h = hw // W
    w = hw % W
    idx = (((n * C) + c) * H + h) * W + w
    y_val = tl.load(y_ptr + idx)
    x_val = tl.load(x_ptr + idx)
    tl.store(out_ptr + idx, y_val + x_val)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        x: (B, C_in, H, W), conv weights: (C_out, C_in, 3, 3), norm scales/bias: (C_out,)
        Returns: (B, C_out2, H, W)
        """
        assert x.is_cuda, "Inputs must be on CUDA for Triton kernels"
        assert x.dim() == 4, "x must be (B, C, H, W)"

        # Ensure dtype and contiguity
        B, C_in, H, W = x.shape
        x_f32 = x.contiguous().to(torch.float32)

        # 1) First conv
        C_out1 = conv1_weight.shape[0]
        y1 = torch.empty((B, C_out1, H, W), device=x.device, dtype=torch.float32)
        grid1 = (B, C_out1, H, W)
        conv3x3_kernel[grid1](
            x_f32, conv1_weight.contiguous().to(torch.float32), y1,
            B=B, C_in=C_in, C_out=C_out1, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        # 2) GroupNorm + SiLU for first block
        N_TILES1 = _compute_ntiles(H * W, BLOCK_HW)
        mean1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        rstd1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        grid_reduce1 = (B, self.num_groups, C_out1 // self.num_groups)
        group_norm_reduce_kernel[grid_reduce1](
            y1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W, num_groups=self.num_groups,
            N_TILES=N_TILES1, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2,
        )
        y1_norm = torch.empty_like(y1)
        grid_apply1 = (B, self.num_groups, C_out1 // self.num_groups, N_TILES1)
        group_norm_apply_kernel[grid_apply1](
            y1, mean1, rstd1, norm1_weight.to(torch.float32), norm1_bias.to(torch.float32), y1_norm,
            B=B, C=C_out1, H=H, W=W, num_groups=self.num_groups,
            N_TILES=N_TILES1, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2,
        )

        # 3) Second conv: conv2 on y1_norm
        C_out2 = conv2_weight.shape[0]
        y2 = torch.empty((B, C_out2, H, W), device=x.device, dtype=torch.float32)
        grid2 = (B, C_out2, H, W)
        conv3x3_kernel[grid2](
            y1_norm, conv2_weight.contiguous().to(torch.float32), y2,
            B=B, C_in=C_out1, C_out=C_out2, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        # 4) GroupNorm + SiLU for second block
        N_TILES2 = _compute_ntiles(H * W, BLOCK_HW)
        mean2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        rstd2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        grid_reduce2 = (B, self.num_groups, C_out2 // self.num_groups)
        group_norm_reduce_kernel[grid_reduce2](
            y2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W, num_groups=self.num_groups,
            N_TILES=N_TILES2, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2,
        )
        y2_norm = torch.empty_like(y2)
        grid_apply2 = (B, self.num_groups, C_out2 // self.num_groups, N_TILES2)
        group_norm_apply_kernel[grid_apply2](
            y2, mean2, rstd2, norm2_weight.to(torch.float32), norm2_bias.to(torch.float32), y2_norm,
            B=B, C=C_out2, H=H, W=W, num_groups=self.num_groups,
            N_TILES=N_TILES2, BLOCK_HW=BLOCK_HW,
            num_warps=4, num_stages=2,
        )

        # 5) Residual add: out = y2_norm + x
        out = torch.empty_like(y2_norm)
        grid_add = (B, C_out2, H * W)
        residual_add_kernel[grid_add](
            y2_norm, x_f32, out,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4, num_stages=2,
        )

        return out


def _compute_ntiles(hw: int, block_hw: int) -> int:
    # Number of tiles to cover HW with BLOCK_HW
    return (hw + block_hw - 1) // block_hw


# Example usage (to be replaced by evaluation harness):
# model = ModelNew(num_groups=32, eps=1e-5).cuda()
# x = torch.randn(16, 64, 64, 64, device='cuda', dtype=torch.float32)
# conv1_w = torch.randn(128, 64, 3, 3, device='cuda', dtype=torch.float32)
# norm1_w = torch.randn(128, device='cuda', dtype=torch.float32)
# norm1_b = torch.randn(128, device='cuda', dtype=torch.float32)
# conv2_w = torch.randn(256, 128, 3, 3, device='cuda', dtype=torch.float32)
# norm2_w = torch.randn(256, device='cuda', dtype=torch.float32)
# norm2_b = torch.randn(256, device='cuda', dtype=torch.float32)
# out = model(x, conv1_w, norm1_w, norm1_b, conv2_w, norm2_w, norm2_b)


def run(*args):
    return ModelNew()(*args)
