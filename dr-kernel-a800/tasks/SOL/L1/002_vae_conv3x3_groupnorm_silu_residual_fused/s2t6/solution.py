import torch
import triton
import triton.language as tl


def _check_valid_shapes(C: int, num_groups: int):
    if C % num_groups != 0:
        raise ValueError(f"num_groups={num_groups} must divide C={C}")


# Triton kernel: Conv3x3 NCHW, stride=1, padding=1, no bias
# out[n, co, h, w] = sum_{ci} sum_{dh in [0,1,2], dw in [0,1,2]} x[n, ci, h+dh, w+dw] * w[co, ci, dh+1, dw+1]
@triton.jit
def conv3x3_nchw_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C_IN, H: tl.constexpr, W: tl.constexpr, C_OUT: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,   # 3x3
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,  # 1
    BLOCK_W: tl.constexpr,                 # tile size along W
):
    # Grid: (N, C_OUT, H, ceil_div(W, BLOCK_W))
    n = tl.program_id(0)
    co = tl.program_id(1)
    h = tl.program_id(2)
    w_blk = tl.program_id(3)

    w_start = w_blk * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Loop over input channels and 3x3 neighborhood
    for ci in range(0, C_IN):
        for dh in range(0, KH):
            h_in = h + dh - PAD_H  # -1, 0, 1
            for dw in range(0, KW):
                w_in = w_offsets + dw - PAD_W
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W) & mask_w

                x_off = ((n * C_IN + ci) * H + h_in) * W + w_in
                x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)

                # weight layout: [C_OUT, C_IN, 3, 3]
                w_off = co * (C_IN * 3 * 3) + ci * (3 * 3) + (dh + 1) * 3 + (dw + 1)
                w_val = tl.load(w_ptr + w_off)

                acc += x_val * w_val

    out_off = ((n * C_OUT + co) * H + h) * W + w_offsets
    tl.store(out_ptr + out_off, acc, mask=mask_w)


# Triton kernel: compute sum and sum of squares for each (n, group) across channels in group and all H*W
@triton.jit
def groupnorm_sums_kernel(
    x_ptr, sums_ptr, sumsq_ptr,
    N, C, H: tl.constexpr, W: tl.constexpr, num_groups: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
):
    # One program per (n, group)
    pid = tl.program_id(0)  # 0 .. (N * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    start_ci = g * C_PER_GROUP
    s = tl.float32(0.0)
    s2 = tl.float32(0.0)

    for ci in range(start_ci, start_ci + C_PER_GROUP):
        for hi in range(0, H):
            for wi in range(0, W):
                idx = ((n * C + ci) * H + hi) * W + wi
                x_val = tl.load(x_ptr + idx)
                s += x_val
                s2 += x_val * x_val

    tl.store(sums_ptr + out_idx, s)
    tl.store(sumsq_ptr + out_idx, s2)


# Triton kernel: compute inverse std per (n, group)
@triton.jit
def groupnorm_invstd_kernel(
    sums_ptr, sumsq_ptr, invstd_ptr,
    N, C, H: tl.constexpr, W: tl.constexpr, num_groups: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (N * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    s = tl.load(sums_ptr + out_idx)
    s2 = tl.load(sumsq_ptr + out_idx)
    group_size = C_PER_GROUP * H * W
    mean = s / group_size
    var = s2 / group_size - mean * mean
    invstd = 1.0 / tl.sqrt(var + 1e-5)  # epsilon
    tl.store(invstd_ptr + out_idx, invstd)


# Triton kernel: normalize + affine + SiLU per (n, group)
@triton.jit
def groupnorm_silu_apply_kernel(
    x_ptr, out_ptr, norm_w_ptr, norm_b_ptr, invstd_ptr,
    N, C, H: tl.constexpr, W: tl.constexpr, num_groups: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (N * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g
    invstd = tl.load(invstd_ptr + out_idx)

    start_ci = g * C_PER_GROUP
    for ci in range(start_ci, start_ci + C_PER_GROUP):
        w = tl.load(norm_w_ptr + ci)
        b = tl.load(norm_b_ptr + ci)
        for hi in range(0, H):
            for wi in range(0, W):
                idx = ((n * C + ci) * H + hi) * W + wi
                x_val = tl.load(x_ptr + idx)
                # normalize: (x - b) * invstd * w
                norm = (x_val - b) * invstd * w
                # SiLU: norm * sigmoid(norm)
                sig = 1.0 / (1.0 + tl.exp(-norm))
                out_val = norm * sig
                tl.store(out_ptr + idx, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int, eps: float):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor,
                norm1_weight: torch.Tensor,
                norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor,
                norm2_weight: torch.Tensor,
                norm2_bias: torch.Tensor):
        """
        Triton-only implementation of:
          y1 = Conv3x3(x, conv1_weight, stride=1, padding=1)
          y1 = GroupNorm(num_groups, affine=True, eps=eps)(y1) then SiLU
          y2 = Conv3x3(y1, conv2_weight, stride=1, padding=1)
          y2 = GroupNorm(num_groups, affine=True, eps=eps)(y2) then SiLU
          out = y2 + x
        All heavy ops are implemented in Triton kernels.
        """
        B, C, H, W = x.shape
        _check_valid_shapes(C, self.num_groups)

        # Ensure contiguous and float32 for Triton
        x = x.contiguous().to(torch.float32)

        # Allocate output for first conv
        y1 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)

        # Launch Triton conv1
        BLOCK_W = 64  # tile along W
        grid_conv = (B, C, H, triton.cdiv(W, BLOCK_W))
        conv3x3_nchw_kernel[grid_conv](
            x, conv1_weight.to(torch.float32).contiguous(), y1,
            B, C, H=H, W=W, C_OUT=C,
            KH=3, KW=3, PAD_H=1, PAD_W=1, BLOCK_W=BLOCK_W,
            num_warps=4, num_stages=2,
        )

        # Triton GroupNorm + SiLU for y1
        C_PER_GROUP = C // self.num_groups
        N = B
        sums = torch.empty((N * self.num_groups,), device=x.device, dtype=torch.float32)
        sumsq = torch.empty((N * self.num_groups,), device=x.device, dtype=torch.float32)
        grid_reduce = (N * self.num_groups,)
        groupnorm_sums_kernel[grid_reduce](
            y1, sums, sumsq,
            N, C, H=H, W=W, num_groups=self.num_groups,
            C_PER_GROUP=C_PER_GROUP,
            num_warps=4, num_stages=2,
        )

        invstd = torch.empty((N * self.num_groups,), device=x.device, dtype=torch.float32)
        groupnorm_invstd_kernel[grid_reduce](
            sums, sumsq, invstd,
            N, C, H=H, W=W, num_groups=self.num_groups,
            C_PER_GROUP=C_PER_GROUP,
            num_warps=4, num_stages=2,
        )

        norm1_w = norm1_weight.to(torch.float32).contiguous()
        norm1_b = norm1_bias.to(torch.float32).contiguous()

        out1 = torch.empty_like(y1)  # normalized + affine + SiLU result
        groupnorm_silu_apply_kernel[grid_reduce](
            y1, out1, norm1_w, norm1_b, invstd,
            N, C, H=H, W=W, num_groups=self.num_groups,
            C_PER_GROUP=C_PER_GROUP,
            num_warps=4, num_stages=2,
        )

        # Add residual: out1 += x
        out1 = out1 + x

        # Second conv: conv2 on out1
        y2 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid_conv2 = (B, C, H, triton.cdiv(W, BLOCK_W))
        conv3x3_nchw_kernel[grid_conv2](
            out1, conv2_weight.to(torch.float32).contiguous(), y2,
            B, C, H=H, W=W, C_OUT=C,
            KH=3, KW=3, PAD_H=1, PAD_W=1, BLOCK_W=BLOCK_W,
            num_warps=4, num_stages=2,
        )

        # Triton GroupNorm + SiLU for y2
        sums2 = torch.empty((N * self.num_groups,), device=x.device, dtype=torch.float32)
        sumsq2 = torch.empty((N * self.num_groups,), device=x.device, dtype=torch.float32)
        groupnorm_sums_kernel[grid_reduce](
            y2, sums2, sumsq2,
            N, C, H=H, W=W, num_groups=self.num_groups,
            C_PER_GROUP=C_PER_GROUP,
            num_warps=4, num_stages=2,
        )

        invstd2 = torch.empty((N * self.num_groups,), device=x.device, dtype=torch.float32)
        groupnorm_invstd_kernel[grid_reduce](
            sums2, sumsq2, invstd2,
            N, C, H=H, W=W, num_groups=self.num_groups,
            C_PER_GROUP=C_PER_GROUP,
            num_warps=4, num_stages=2,
        )

        norm2_w = norm2_weight.to(torch.float32).contiguous()
        norm2_b = norm2_bias.to(torch.float32).contiguous()

        out2 = torch.empty_like(y2)
        groupnorm_silu_apply_kernel[grid_reduce](
            y2, out2, norm2_w, norm2_b, invstd2,
            N, C, H=H, W=W, num_groups=self.num_groups,
            C_PER_GROUP=C_PER_GROUP,
            num_warps=4, num_stages=2,
        )

        # Residual add
        out = out2 + out1
        return out


def run(*args):
    return ModelNew()(*args)
