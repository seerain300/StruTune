import torch
import triton
import triton.language as tl


# Conv3x3: y = x * w, stride=1, padding=1, no bias
@triton.jit
def conv3x3_kernel(
    x_ptr,           # *f32, input [B, C_in, H, W], contiguous
    w_ptr,           # *f32, weight [C_out, C_in, 3, 3], contiguous
    y_ptr,           # *f32, output [B, C_out, H, W], contiguous
    B: tl.constexpr, C_in: tl.constexpr, C_out: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
):
    n = tl.program_id(0)
    c_out = tl.program_id(1)
    hw = tl.program_id(2)
    h_out = hw // W
    w_out = hw % W

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels
    for cin in range(0, C_in):
        # 3x3 neighborhood with padding=1
        for dh in range(-1, 2):
            for dw in range(-1, 2):
                h_in = h_out + dh
                w_in = w_out + dw
                # input index: ((n * C_in + cin) * H + h_in) * W + w_in
                x_idx = ((n * C_in + cin) * H + h_in) * W + w_in
                x_val = tl.load(x_ptr + x_idx)
                # weight index: (c_out * C_in + cin) * (3*3) + (dh+1)*3 + (dw+1)
                w_idx = (c_out * C_in + cin) * 9 + (dh + 1) * 3 + (dw + 1)
                w_val = tl.load(w_ptr + w_idx)
                acc += x_val * w_val

    # output index: ((n * C_out + c_out) * H + h_out) * W + w_out
    y_idx = ((n * C_out + c_out) * H + h_out) * W + w_out
    tl.store(y_ptr + y_idx, acc)


# GroupNorm reduction: compute per-(n, group, channel) sum and sumsq over HW
@triton.jit
def group_norm_reduce_kernel(
    x_ptr,           # *f32, input [B, C, H, W]
    mean_ptr,        # *f32, per-channel mean [C]
    rstd_ptr,        # *f32, per-channel rstd [C]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
):
    n = tl.program_id(0)
    group = tl.program_id(1)
    c = tl.program_id(2)
    cpg = C // num_groups
    # Accumulate sum and sumsq over all HW elements of x[n, c, :, :]
    total = tl.zeros((), dtype=tl.float32)
    total2 = tl.zeros((), dtype=tl.float32)
    BLOCK = 1024
    N_TILES = tl.cdiv(H * W, BLOCK)
    for t in range(0, N_TILES):
        tile_start = t * BLOCK
        offs = tile_start + tl.arange(0, BLOCK)
        mask = offs < (H * W)
        h = offs // W
        w = offs % W
        idx = ((n * C + c) * H + h) * W + w
        x_vec = tl.load(x_ptr + idx, mask=mask, other=0.0)
        total += tl.sum(x_vec, axis=0)
        total2 += tl.sum(x_vec * x_vec, axis=0)
    hw = H * W
    mean = total / hw
    var = total2 / hw - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(mean_ptr + c, mean)
    tl.store(rstd_ptr + c, rstd)


# GroupNorm apply + affine + SiLU per tile
@triton.jit
def group_norm_apply_kernel_affine_silu(
    x_ptr, y_ptr, gamma_ptr, beta_ptr, mean_ptr, rstd_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    num_groups: tl.constexpr,
):
    n = tl.program_id(0)
    group = tl.program_id(1)
    c = tl.program_id(2)
    tile = tl.program_id(3)
    cpg = C // num_groups
    BLOCK = 1024
    tile_start = tile * BLOCK
    offs = tile_start + tl.arange(0, BLOCK)
    mask = offs < (H * W)
    h = offs // W
    w = offs % W

    gamma = tl.load(gamma_ptr + c)
    beta = tl.load(beta_ptr + c)
    mean = tl.load(mean_ptr + c)
    rstd = tl.load(rstd_ptr + c)

    idx = ((n * C + c) * H + h) * W + w
    x_vec = tl.load(x_ptr + idx, mask=mask, other=0.0)
    norm = (x_vec - mean) * rstd
    norm = norm * gamma + beta

    # SiLU: norm * sigmoid(norm)
    sig = 1.0 / (1.0 + tl.exp(-norm))
    y_vec = norm * sig

    tl.store(y_ptr + idx, y_vec, mask=mask)


# Residual add: out = a + b, elementwise
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
    idx = ((n * C + c) * H + h) * W + w
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

        # Ensure float32 and contiguous
        x_f32 = x.contiguous().to(torch.float32)

        # Prepare weights
        conv1_w = conv1_weight.contiguous().to(torch.float32)  # (C_out1, C_in, 3, 3)
        conv2_w = conv2_weight.contiguous().to(torch.float32)  # (C_out2, C_out1, 3, 3)
        norm1_gamma = norm1_weight.contiguous().to(torch.float32)  # (C_out1,)
        norm1_beta = norm1_bias.contiguous().to(torch.float32)     # (C_out1,)
        norm2_gamma = norm2_weight.contiguous().to(torch.float32)  # (C_out2,)
        norm2_beta = norm2_bias.contiguous().to(torch.float32)     # (C_out2,)

        B, C_in, H, W = x_f32.shape
        C_out1 = conv1_w.shape[0]
        C_out2 = conv2_w.shape[0]

        # First conv: out1 = conv3x3(x)
        out1 = torch.empty((B, C_out1, H, W), device=x.device, dtype=torch.float32)
        grid1 = (B, C_out1, H * W)
        conv3x3_kernel[grid1](
            x_f32, conv1_w, out1,
            B=B, C_in=C_in, C_out=C_out1, H=H, W=W,
            num_warps=4, num_stages=2
        )

        # GroupNorm + SiLU for first block
        mean1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        rstd1 = torch.empty(C_out1, device=x.device, dtype=torch.float32)
        grid_r1 = (B, self.num_groups, C_out1 // self.num_groups)
        group_norm_reduce_kernel[grid_r1](
            out1, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W, num_groups=self.num_groups,
            num_warps=4, num_stages=2
        )
        out1_norm = torch.empty_like(out1)
        grid_a1 = (B, self.num_groups, C_out1 // self.num_groups, triton.cdiv(H * W, 1024))
        group_norm_apply_kernel_affine_silu[grid_a1](
            out1, out1_norm, norm1_gamma, norm1_beta, mean1, rstd1,
            B=B, C=C_out1, H=H, W=W, num_groups=self.num_groups,
            num_warps=4, num_stages=2
        )

        # Second conv: out2 = conv3x3(out1_norm)
        out2 = torch.empty((B, C_out2, H, W), device=x.device, dtype=torch.float32)
        grid2 = (B, C_out2, H * W)
        conv3x3_kernel[grid2](
            out1_norm, conv2_w, out2,
            B=B, C_in=C_out1, C_out=C_out2, H=H, W=W,
            num_warps=4, num_stages=2
        )

        # GroupNorm + SiLU for second block
        mean2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        rstd2 = torch.empty(C_out2, device=x.device, dtype=torch.float32)
        grid_r2 = (B, self.num_groups, C_out2 // self.num_groups)
        group_norm_reduce_kernel[grid_r2](
            out2, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W, num_groups=self.num_groups,
            num_warps=4, num_stages=2
        )
        out2_norm = torch.empty_like(out2)
        grid_a2 = (B, self.num_groups, C_out2 // self.num_groups, triton.cdiv(H * W, 1024))
        group_norm_apply_kernel_affine_silu[grid_a2](
            out2, out2_norm, norm2_gamma, norm2_beta, mean2, rstd2,
            B=B, C=C_out2, H=H, W=W, num_groups=self.num_groups,
            num_warps=4, num_stages=2
        )

        # Final residual add: out = out2_norm + x
        out = torch.empty_like(out2_norm)
        residual_add_kernel[(B, C_out2, H * W)](
            out2_norm, x_f32, out,
            B=B, C=C_out2, H=H, W=W,
            num_warps=4, num_stages=2
        )

        return out


def run(*args):
    return ModelNew()(*args)
