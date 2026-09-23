import torch
import triton
import triton.language as tl


def _compute_ntiles(hw: int, block_hw: int) -> int:
    # Number of tiles to cover HW elements
    return (hw + block_hw - 1) // block_hw


# GroupNorm reduction kernel: compute per-channel mean and rstd for a group
@triton.jit
def group_norm_reduce_kernel(
    x_ptr,            # *f32, input [B, C, H, W]
    mean_ptr,         # *f32, output [C]
    rstd_ptr,         # *f32, output [C]
    B: tl.constexpr,  # batch
    C: tl.constexpr,  # channels
    H: tl.constexpr,  # height
    W: tl.constexpr,  # width
    num_groups: tl.constexpr,
    N_TILES: tl.constexpr,  # number of tiles over H*W
    BLOCK_HW: tl.constexpr, # tile size over H*W
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    c = tl.program_id(2)  # one channel per program within the group

    group_size = C // num_groups
    c_start = g * group_size
    c = c_start + c

    sum_val = 0.0
    sumsq_val = 0.0

    for t in range(N_TILES):
        start = t * BLOCK_HW
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < (H * W)
        h = offs // W
        w = offs % W

        base = ((n * C) + c) * (H * W)
        idx = base + offs

        x = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    hw_total = H * W
    mean = sum_val / hw_total
    var = sumsq_val / hw_total - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)  # eps

    tl.store(mean_ptr + c, mean)
    tl.store(rstd_ptr + c, rstd)


# GroupNorm apply kernel: normalize + affine + SiLU per channel per tile
@triton.jit
def group_norm_apply_kernel(
    x_ptr,             # *f32, input [B, C, H, W]
    mean_ptr,          # *f32, [C]
    rstd_ptr,          # *f32, [C]
    scale_ptr,         # *f32, per-channel affine scale [C]
    bias_ptr,          # *f32, per-channel affine bias [C]
    y_ptr,             # *f32, output [B, C, H, W]
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    num_groups: tl.constexpr,
    N_TILES: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    c = tl.program_id(2)  # one channel per program within the group

    group_size = C // num_groups
    c_start = g * group_size
    c = c_start + c

    mean = tl.load(mean_ptr + c)
    rstd = tl.load(rstd_ptr + c)
    scale = tl.load(scale_ptr + c)
    bias = tl.load(bias_ptr + c)

    for t in range(N_TILES):
        start = t * BLOCK_HW
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < (H * W)
        h = offs // W
        w = offs % W

        base = ((n * C) + c) * (H * W)
        idx = base + offs

        x = tl.load(x_ptr + idx, mask=mask, other=0.0)

        # Normalize
        norm = (x - mean) * rstd

        # Affine
        y = norm * scale + bias

        # SiLU: y * sigmoid(y), sigmoid(y) = 1 / (1 + exp(-y))
        sig = 1.0 / (1.0 + tl.exp(-y))
        y = y * sig

        tl.store(y_ptr + idx, y, mask=mask)


# Conv3x3 via im2col + reduction: input x_ptr, weights w_ptr, output y_ptr
# Grid is (B, C_out, N_TILES), we vectorize over output spatial positions and reduce over input channels and 3x3 neighborhood.
@triton.jit
def conv3x3_im2col_gemm_kernel(
    x_ptr,            # *f32, input [B, C_in, H, W]
    w_ptr,            # *f32, weights [C_out, C_in, 3, 3]
    y_ptr,            # *f32, output [B, C_out, H, W]
    B: tl.constexpr,
    C_in: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C_out: tl.constexpr,
    N_TILES: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    t = tl.program_id(2)

    start = t * BLOCK_HW
    offs = start + tl.arange(0, BLOCK_HW)
    mask = offs < (H * W)
    h = offs // W
    w = offs % W

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # For each input channel and 3x3 neighborhood, accumulate into acc
    for c_in in range(C_in):
        for dh in (-1, 0, 1):
            for dw in (-1, 0, 1):
                h_in = h + dh
                w_in = w + dw  # padding=1 guarantees bounds
                base_in = ((n * C_in) + c_in) * (H * W)
                idx_in = base_in + h_in * W + w_in
                x_val = tl.load(x_ptr + idx_in, mask=mask, other=0.0)
                # weight for (c_out, c_in, dh+1, dw+1)
                dh_i = dh + 1
                dw_i = dw + 1
                w_idx = c_out * (C_in * 9) + c_in * 9 + dh_i * 3 + dw_i
                w_val = tl.load(w_ptr + w_idx)  # scalar
                acc += x_val * w_val

    base_out = ((n * C_out) + c_out) * (H * W)
    idx_out = base_out + offs
    tl.store(y_ptr + idx_out, acc, mask=mask)


# Residual add kernel: out = out + x, elementwise
@triton.jit
def residual_add_kernel(
    out_ptr, x_ptr, B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr
):
    n = tl.program_id(0)
    c = tl.program_id(1)
    hw = tl.program_id(2)

    h = hw // W
    w = hw % W
    idx = (((n * C) + c) * H + h) * W + w

    a = tl.load(out_ptr + idx)
    b = tl.load(x_ptr + idx)
    tl.store(out_ptr + idx, a + b)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5, block_hw: int = 1024, num_warps: int = 4):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.block_hw = block_hw
        self.num_warps = num_warps

    def forward(self, x: torch.Tensor,
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
        assert x.is_cuda, "Inputs must be CUDA tensors for Triton kernels"
        B, C_in, H, W = x.shape

        # Ensure all inputs are float32 and contiguous
        x_f32 = x.contiguous().to(torch.float32)
        conv1_w = conv1_weight.contiguous().to(torch.float32)  # (C_out1, C_in, 3, 3)
        conv2_w = conv2_weight.contiguous().to(torch.float32)  # (C_out2, C_out1, 3, 3)
        norm1_scale = norm1_weight.contiguous().to(torch.float32)
        norm1_bias = norm1_bias.contiguous().to(torch.float32)
        norm2_scale = norm2_weight.contiguous().to(torch.float32)
        norm2_bias = norm2_bias.contiguous().to(torch.float32)

        C_out1 = conv1_w.shape[0]
        C_out2 = conv2_w.shape[0]

        # 1) Conv1: y1 = Conv3x3(x)
        y1 = torch.empty((B, C_out1, H, W), device=x.device, dtype=torch.float32)
        N_TILES1 = _compute_ntiles(H * W, self.block_hw)
        grid1 = (B, C_out1, N_TILES1)
        conv3x3_im2col_gemm_kernel[grid1](
            x_f32, conv1_w, y1,
            B=B, C_in=C_in, H=H, W=W, C_out=C_out1, N_TILES=N_TILES1, BLOCK_HW=self.block_hw,
            num_warps=self.num_warps, num_stages=2,
        )

        # 2) GroupNorm + SiLU on y1
        mean1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        rstd1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        grid_reduce1 = (B, self.num_groups, C_out1 // self.num_groups)
        group_norm_reduce_kernel[grid_reduce1](
            y1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W, num_groups=self.num_groups, N_TILES=1, BLOCK_HW=self.block_hw,
            num_warps=self.num_warps, num_stages=2,
        )
        y1_norm = torch.empty_like(y1)
        grid_apply1 = (B, self.num_groups, C_out1 // self.num_groups, N_TILES1)
        group_norm_apply_kernel[grid_apply1](
            y1, mean1, rstd1, norm1_scale, norm1_bias, y1_norm,
            B=B, C=C_out1, H=H, W=W, num_groups=self.num_groups, N_TILES=N_TILES1, BLOCK_HW=self.block_hw,
            num_warps=self.num_warps, num_stages=2,
        )

        # 3) Conv2: y2 = Conv3x3(y1_norm)
        y2 = torch.empty((B, C_out2, H, W), device=x.device, dtype=torch.float32)
        N_TILES2 = _compute_ntiles(H * W, self.block_hw)
        grid2 = (B, C_out2, N_TILES2)
        conv3x3_im2col_gemm_kernel[grid2](
            y1_norm, conv2_w, y2,
            B=B, C_in=C_out1, H=H, W=W, C_out=C_out2, N_TILES=N_TILES2, BLOCK_HW=self.block_hw,
            num_warps=self.num_warps, num_stages=2,
        )

        # 4) GroupNorm + SiLU on y2
        mean2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        rstd2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        grid_reduce2 = (B, self.num_groups, C_out2 // self.num_groups)
        group_norm_reduce_kernel[grid_reduce2](
            y2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W, num_groups=self.num_groups, N_TILES=1, BLOCK_HW=self.block_hw,
            num_warps=self.num_warps, num_stages=2,
        )
        y2_norm = torch.empty_like(y2)
        grid_apply2 = (B, self.num_groups, C_out2 // self.num_groups, N_TILES2)
        group_norm_apply_kernel[grid_apply2](
            y2, mean2, rstd2, norm2_scale, norm2_bias, y2_norm,
            B=B, C=C_out2, H=H, W=W, num_groups=self.num_groups, N_TILES=N_TILES2, BLOCK_HW=self.block_hw,
            num_warps=self.num_warps, num_stages=2,
        )

        # 5) Residual add
        out = torch.empty_like(y2_norm)
        grid_add = (B, C_out2, H * W)
        residual_add_kernel[grid_add](
            y2_norm, x_f32, B=B, C=C_out2, H=H, W=W,
            num_warps=self.num_warps, num_stages=2,
        )

        return out


def run(*args):
    return ModelNew()(*args)
