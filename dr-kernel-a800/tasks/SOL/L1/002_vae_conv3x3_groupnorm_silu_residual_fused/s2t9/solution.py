import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3x3_nchw_kernel(
    x_ptr, w_ptr, out_ptr,
    B: tl.constexpr, C_in: tl.constexpr, H: tl.constexpr, W: tl.constexpr, C_out: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Grid: (B, C_out, H, ceil_div(W, BLOCK_W))
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w_blk = tl.program_id(3)

    h = pid_h
    w_start = pid_w_blk * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Loop over input channels
    for ci in range(0, C_in):
        # Accumulate over 3x3 neighborhood with padding=1
        for dh in range(-PAD_H, PAD_H + KH):
            for dw in range(-PAD_W, PAD_W + KW):
                h_in = h + dh
                w_in = w_offsets + dw
                in_bounds = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W) & mask_w
                # NCHW contiguous layout:
                x_index = ((pid_n * C_in + ci) * H + h_in) * W + w_in
                x_val = tl.load(x_ptr + x_index, mask=in_bounds, other=0.0)
                # Weight layout: (C_out, C_in, 3, 3), contiguous
                w_index = pid_co * (C_in * 3 * 3) + ci * (3 * 3) + (dh + 1) * 3 + (dw + 1)
                w_val = tl.load(w_ptr + w_index)
                acc += x_val * w_val

    out_index = ((pid_n * C_out + pid_co) * H + h) * W + w_offsets
    tl.store(out_ptr + out_index, acc, mask=mask_w)


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
    start_ci = g * C_PER_GROUP
    total_elems = C_PER_GROUP * H * W

    s = tl.float32(0.0)
    s2 = tl.float32(0.0)

    for ci in range(start_ci, start_ci + C_PER_GROUP):
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)  # float32
                s += x_val
                s2 += x_val * x_val

    tl.store(sums_ptr + pid, s)
    tl.store(sumsq_ptr + pid, s2)


@triton.jit
def groupnorm_invstd_kernel(
    sums_ptr, sumsq_ptr, invstd_ptr,
    N, C, H: tl.constexpr, W: tl.constexpr, num_groups: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (N * num_groups - 1)
    s = tl.load(sums_ptr + pid)
    s2 = tl.load(sumsq_ptr + pid)
    total_elems = C_PER_GROUP * H * W
    mean = s / total_elems
    var = s2 / total_elems - mean * mean
    invstd = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(invstd_ptr + pid, invstd)


@triton.jit
def groupnorm_silu_apply_kernel(
    x_ptr, norm_w_ptr, norm_b_ptr, out_ptr, invstd_ptr,
    N, C, H: tl.constexpr, W: tl.constexpr, num_groups: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (N * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    invstd = tl.load(invstd_ptr + pid)
    start_ci = g * C_PER_GROUP

    for ci in range(start_ci, start_ci + C_PER_GROUP):
        # apply affine
        w = tl.load(norm_w_ptr + ci)
        b = tl.load(norm_b_ptr + ci)
        # iterate H*W
        for h in range(0, H):
            for w_idx in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w_idx
                x_val = tl.load(x_ptr + idx)
                normed = (x_val - b) * invstd
                y = normed * w  # affine
                # SiLU: y * sigmoid(y)
                sig = 1.0 / (1.0 + tl.exp(-y))
                out_val = y * sig
                tl.store(out_ptr + idx, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self, num_groups: int = 32, eps: float = 1e-5):
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
        # Ensure inputs are float32 and contiguous
        x = x.contiguous().to(torch.float32)
        conv1_weight = conv1_weight.contiguous().to(torch.float32)
        conv2_weight = conv2_weight.contiguous().to(torch.float32)
        norm1_w = norm1_weight.contiguous().to(torch.float32)
        norm1_b = norm1_bias.contiguous().to(torch.float32)
        norm2_w = norm2_weight.contiguous().to(torch.float32)
        norm2_b = norm2_bias.contiguous().to(torch.float32)

        N, C, H, W = x.shape
        C1 = conv1_weight.shape[1]  # input channels for conv1
        C2 = conv2_weight.shape[1]  # input channels for conv2 (should equal C_out of conv1)
        _assert_divisible(C, self.num_groups)
        C_per_group = C // self.num_groups

        # Conv1: Triton kernel
        y1 = torch.empty((N, C1, H, W), device=x.device, dtype=torch.float32)
        grid1 = (N, C1, H, triton.cdiv(W, 64))
        conv3x3_nchw_kernel[grid1](
            x, conv1_weight, y1,
            B=N, C_in=C, H=H, W=W, C_out=C1,
            KH=3, KW=3, PAD_H=1, PAD_W=1,
            BLOCK_W=64,
            num_warps=4, num_stages=2,
        )

        # GroupNorm + SiLU for y1
        sums1 = torch.empty(N * self.num_groups, device=x.device, dtype=torch.float32)
        sumsq1 = torch.empty(N * self.num_groups, device=x.device, dtype=torch.float32)
        grid_reduce1 = (N * self.num_groups,)
        groupnorm_sums_kernel[grid_reduce1](
            y1, sums1, sumsq1,
            N, C, H=H, W=W, num_groups=self.num_groups,
            C_PER_GROUP=C_per_group,
        )
        invstd1 = torch.empty(N * self.num_groups, device=x.device, dtype=torch.float32)
        groupnorm_invstd_kernel[grid_reduce1](
            sums1, sumsq1, invstd1,
            N, C, H=H, W=W, num_groups=self.num_groups,
            C_PER_GROUP=C_per_group,
        )
        y1_gn_silu = torch.empty_like(y1)
        groupnorm_silu_apply_kernel[grid_reduce1](
            y1, norm1_w, norm1_b, y1_gn_silu, invstd1,
            N, C, H=H, W=W, num_groups=self.num_groups,
            C_PER_GROUP=C_per_group,
            num_warps=4, num_stages=2,
        )

        # Conv2: Triton kernel using y1_gn_silu as input
        y2 = torch.empty((N, C2, H, W), device=x.device, dtype=torch.float32)
        grid2 = (N, C2, H, triton.cdiv(W, 64))
        conv3x3_nchw_kernel[grid2](
            y1_gn_silu, conv2_weight, y2,
            B=N, C_in=C1, H=H, W=W, C_out=C2,
            KH=3, KW=3, PAD_H=1, PAD_W=1,
            BLOCK_W=64,
            num_warps=4, num_stages=2,
        )

        # GroupNorm + SiLU for y2
        sums2 = torch.empty(N * self.num_groups, device=x.device, dtype=torch.float32)
        sumsq2 = torch.empty(N * self.num_groups, device=x.device, dtype=torch.float32)
        groupnorm_sums_kernel[grid_reduce1](
            y2, sums2, sumsq2,
            N, C2, H=H, W=W, num_groups=self.num_groups,
            C_PER_GROUP=C_per_group,  # C2 == C, same num_groups
        )
        invstd2 = torch.empty(N * self.num_groups, device=x.device, dtype=torch.float32)
        groupnorm_invstd_kernel[grid_reduce1](
            sums2, sumsq2, invstd2,
            N, C2, H=H, W=W, num_groups=self.num_groups,
            C_PER_GROUP=C_per_group,
        )
        y2_gn_silu = torch.empty_like(y2)
        groupnorm_silu_apply_kernel[grid_reduce1](
            y2, norm2_w, norm2_b, y2_gn_silu, invstd2,
            N, C2, H=H, W=W, num_groups=self.num_groups,
            C_PER_GROUP=C_per_group,
            num_warps=4, num_stages=2,
        )

        # Residual: x + y2_gn_silu
        out = y2_gn_silu + x

        return out


def run(*args):
    return ModelNew()(*args)
