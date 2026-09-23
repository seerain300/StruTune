import torch
import triton
import triton.language as tl


def _assert_divisible(dividend: int, divisor: int):
    if dividend % divisor != 0:
        raise ValueError(f"{dividend} is not divisible by {divisor}")


# Triton kernel: Conv3x3 NCHW, stride=1, padding=1, no bias
# Computes out[n, co, h, w] = sum_{ci=0..C-1} sum_{dh in [-1,0,1]} sum_{dw in [-1,0,1]} x[n, ci, h+dh, w+dw] * w[co, ci, 1+dh, 1+dw]
@triton.jit
def conv3x3_nchw_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C, H, W, C_OUT,
    BLOCK_W: tl.constexpr,  # tile size along W
):
    # Grid: (B, C_OUT, H, ceil_div(W, BLOCK_W))
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w_blk = tl.program_id(3)

    h = pid_h
    w_start = pid_w_blk * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W

    # Accumulator for output row
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Loop over input channels; C is constexpr (specialized per input)
    for ci in range(0, C):
        # Accumulate over 3x3 neighborhood with padding=1
        for dh in range(-1, 2):  # constexpr loop
            h_in = h + dh
            h_in_in_bounds = (h_in >= 0) & (h_in < H)
            for dw in range(-1, 2):  # constexpr loop
                w_in = w_offsets + dw
                w_in_in_bounds = (w_in >= 0) & (w_in < W)
                mask = mask_w & h_in_in_bounds & w_in_in_bounds

                # Compute input pointers: ((n*C + ci)*H + h_in)*W + w_in
                x_off = ((pid_n * C + ci) * H + h_in) * W + w_in
                x_val = tl.load(x_ptr + x_off, mask=mask, other=0.0)

                # Compute weight pointer: ((co*C + ci)*3 + (1+dh))*3 + (1+dw)
                # weight layout is (C_OUT, C, 3, 3)
                w_off = (pid_co * C + ci) * 9 + (1 + dh) * 3 + (1 + dw)
                w_val = tl.load(w_ptr + w_off)  # scalar weight

                acc += x_val * w_val

    # Store results: out layout is (B, C_OUT, H, W)
    out_off = (pid_n * C_OUT + pid_co) * (H * W) + h * W + w_offsets
    tl.store(out_ptr + out_off, acc, mask=mask_w)


# Triton kernel: per (n, group) compute sum and sumsq across channels in the group and spatial H*W
@triton.jit
def groupnorm_sums_kernel(
    x_ptr, sums_ptr, sumsq_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    start_ci = g * C_PER_GROUP
    s = tl.float32(0.0)
    s2 = tl.float32(0.0)

    for ci in range(start_ci, start_ci + C_PER_GROUP):
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)
                s += x_val
                s2 += x_val * x_val

    tl.store(sums_ptr + out_idx, s)
    tl.store(sumsq_ptr + out_idx, s2)


# Triton kernel: per (n, group) compute inverse std from sums and sumsq
@triton.jit
def groupnorm_invstd_kernel(
    sums_ptr, sumsq_ptr, invstd_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    s = tl.load(sums_ptr + out_idx)
    s2 = tl.load(sumsq_ptr + out_idx)
    group_size = C_PER_GROUP * H * W
    mean = s / group_size
    var = s2 / group_size - mean * mean
    invstd = 1.0 / tl.sqrt(var + 1e-5)  # epsilon for stability
    tl.store(invstd_ptr + out_idx, invstd)


# Triton kernel: normalize + affine + SiLU per (n, group), using precomputed invstd
@triton.jit
def groupnorm_silu_apply_kernel(
    x_ptr, norm_w_ptr, norm_b_ptr, out_ptr, invstd_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g
    invstd = tl.load(invstd_ptr + out_idx)

    start_ci = g * C_PER_GROUP
    for ci in range(start_ci, start_ci + C_PER_GROUP):
        w = tl.load(norm_w_ptr + ci)
        b = tl.load(norm_b_ptr + ci)
        for h in range(0, H):
            for w_idx in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w_idx
                x_val = tl.load(x_ptr + idx)
                # normalize and affine
                y = (x_val - mean) * invstd
                y = y * w + b
                # SiLU: y * sigmoid(y)
                sig = 1.0 / (1.0 + tl.exp(-y))
                out_val = y * sig
                out_off = (n * C + ci) * (H * W) + h * W + w_idx
                tl.store(out_ptr + out_off, out_val)


# Triton kernel: elementwise add (used for residual)
@triton.jit
def add_residual_kernel(
    in_ptr, res_ptr, out_ptr,
    B, C, H, W,
):
    pid = tl.program_id(0)
    total = B * C * H * W
    idx = pid
    # Simple linear indexing over contiguous tensors
    while idx < total:
        val = tl.load(in_ptr + idx)
        res = tl.load(res_ptr + idx)
        tl.store(out_ptr + idx, val + res)
        idx += 1


class ModelNew(torch.nn.Module):
    def __init__(self, conv1_weight, conv2_weight, norm1_weight, norm1_bias, norm2_weight, norm2_bias, eps=1e-5):
        super().__init__()
        # Store weights
        self.conv1_weight = conv1_weight
        self.conv2_weight = conv2_weight
        self.norm1_weight = norm1_weight
        self.norm1_bias = norm1_bias
        self.norm2_weight = norm2_weight
        self.norm2_bias = norm2_bias
        self.eps = eps
        # Validate num_groups divides C
        C = conv1_weight.shape[1]  # input channels
        num_groups = 32
        _assert_divisible(C, num_groups)

    def _launch_triton_conv(self, x, weight, out):
        B, C, H, W = x.shape
        C_OUT = weight.shape[0]
        # Triton conv kernel launch: grid over (B, C_OUT, H, ceil_div(W, BLOCK_W))
        BLOCK_W = 64  # tile along W; can be tuned
        grid = (B, C_OUT, H, (W + BLOCK_W - 1) // BLOCK_W)
        conv3x3_nchw_kernel[grid](
            x, weight, out,
            B, C, H, W, C_OUT,
            BLOCK_W=BLOCK_W,
        )

    def _launch_groupnorm_silu(self, inp, norm_weight, norm_bias, out):
        B, C, H, W = inp.shape
        num_groups = 32
        C_PER_GROUP = C // num_groups
        _assert_divisible(C, num_groups)

        # Allocate scratch for sums and invstd
        sums = torch.empty(B * num_groups, device=inp.device, dtype=torch.float32)
        sumsq = torch.empty(B * num_groups, device=inp.device, dtype=torch.float32)
        invstd = torch.empty(B * num_groups, device=inp.device, dtype=torch.float32)

        # 1) compute sums and sumsq
        grid_reduce = (B * num_groups,)
        groupnorm_sums_kernel[grid_reduce](
            inp, sums, sumsq,
            B, C, H, W, num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )
        # 2) compute invstd
        groupnorm_invstd_kernel[grid_reduce](
            sums, sumsq, invstd,
            B, C, H, W, num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )
        # 3) apply normalization + affine + SiLU
        grid_apply = (B * num_groups,)
        groupnorm_silu_apply_kernel[grid_apply](
            inp, norm_weight, norm_bias, out, invstd,
            B, C, H, W, num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )

    def forward(self, x):
        # Ensure contiguous and float32 for Triton
        if not x.is_contiguous():
            x = x.contiguous()
        if x.dtype != torch.float32:
            x = x.float()

        # Stage 1: Conv1
        out1 = torch.empty_like(x)
        self._launch_triton_conv(x, self.conv1_weight, out1)

        # Stage 1: GroupNorm + SiLU
        out1_gn = torch.empty_like(out1)
        self._launch_groupnorm_silu(out1, self.norm1_weight, self.norm1_bias, out1_gn)

        # Stage 2: Conv2
        out2 = torch.empty_like(x)
        self._launch_triton_conv(out1_gn, self.conv2_weight, out2)

        # Stage 2: GroupNorm + SiLU
        out2_gn = torch.empty_like(out2)
        self._launch_groupnorm_silu(out2, self.norm2_weight, self.norm2_bias, out2_gn)

        # Residual add using Triton elementwise add
        B, C, H, W = x.shape
        out = torch.empty_like(x)
        total = B * C * H * W
        grid_add = (triton.cdiv(total, 1024),)
        add_residual_kernel[grid_add](
            out2_gn, x, out,
            B, C, H, W,
        )

        return out

    def _assert_divisible(self, dividend: int, divisor: int):
        if dividend % divisor != 0:
            raise ValueError(f"{dividend} is not divisible by {divisor}")


def run(*args):
    return ModelNew()(*args)
