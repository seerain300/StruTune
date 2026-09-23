import torch
import triton
import triton.language as tl


# Triton kernel: Conv2d 3x3, stride=1, padding=1, NCHW layout, no bias
# Computes out[n, co, h, w] = sum_{ci=0..C-1} sum_{dh=-1..1} sum_{dw=-1..1} x[n, ci, h+dh, w+dw] * w[co, ci, 1+dh, 1+dw]
# Assumes x_ptr, w_ptr, out_ptr point to contiguous float32 tensors.
@triton.jit
def conv3x3_nchw_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C, H, W, C_OUT,
    BLOCK_W: tl.constexpr,  # tile size along W (e.g., 64 or 128)
    C_CONST: tl.constexpr,  # input channels (compile-time for loop)
    H_CONST: tl.constexpr,  # height (compile-time for loop)
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

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Loop over input channels and 3x3 neighborhood; constexpr loops are OK here.
    for ci in range(0, C_CONST):
        for dh in range(-1, 1 + 1):
            for dw in range(-1, 1 + 1):
                in_h = h + dh
                in_w = w_offsets + dw
                # Validity masks for padding
                mask_h = (in_h >= 0) & (in_h < H_CONST)
                mask = mask_w & mask_h
                x_off = ((pid_n * C_CONST + ci) * H_CONST + in_h) * W + in_w
                x_val = tl.load(x_ptr + x_off, mask=mask, other=0.0)

                # Load weight for this (co, ci, 1+dh, 1+dw)
                w_off = (pid_co * C_CONST + ci) * 9 + (1 + dh) * 3 + (1 + dw)
                w_val = tl.load(w_ptr + w_off)

                acc += x_val * w_val

    # Store results
    out_off = ((pid_n * C_OUT + pid_co) * H * W + h * W + w_offsets)
    tl.store(out_ptr + out_off, acc, mask=mask_w)


# Triton kernel: compute sums and sum of squares per (n, group) for GroupNorm
@triton.jit
def groupnorm_sums_kernel(
    x_ptr, sums_ptr, sumsq_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,  # channels per group
    H_CONST: tl.constexpr,      # height (compile-time for loop)
    W_CONST: tl.constexpr,      # width (compile-time for loop)
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    s = tl.float32(0.0)
    s2 = tl.float32(0.0)

    start_ci = g * C_PER_GROUP
    for ci in range(start_ci, start_ci + C_PER_GROUP):
        for h in range(0, H_CONST):
            for w in range(0, W_CONST):
                off = ((n * C + ci) * H_CONST + h) * W_CONST + w
                x_val = tl.load(x_ptr + off)
                s += x_val
                s2 += x_val * x_val

    tl.store(sums_ptr + out_idx, s)
    tl.store(sumsq_ptr + out_idx, s2)


# Triton kernel: compute inverse std per (n, group) from sums and sumsq
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
    invstd = 1.0 / tl.sqrt(var + 1e-5)
    tl.store(invstd_ptr + out_idx, invstd)


# Triton kernel: normalize + affine + SiLU per (n, group), using precomputed invstd
@triton.jit
def groupnorm_silu_apply_kernel(
    x_ptr, out_ptr, norm_w_ptr, norm_b_ptr, invstd_ptr,
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
                off = ((n * C + ci) * H + h) * W + w_idx
                x_val = tl.load(x_ptr + off)
                # normalize and affine
                y = (x_val - mean) * invstd * w + b
                # SiLU: y * sigmoid(y) = y / (1 + exp(-y))
                sig = 1.0 / (1.0 + tl.exp(-y))
                silu = y * sig
                tl.store(out_ptr + off, silu)


# Simple Triton elementwise add kernel: out = x + y
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    a = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, a + b, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, conv1_weight: torch.Tensor, conv2_weight: torch.Tensor,
                 norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                 norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                 eps: float, batch_size: int, height: int, width: int, C: int, num_groups: int):
        super().__init__()
        # Save parameters
        self.conv1_weight = conv1_weight
        self.conv2_weight = conv2_weight
        self.norm1_weight = norm1_weight
        self.norm1_bias = norm1_bias
        self.norm2_weight = norm2_weight
        self.norm2_bias = norm2_bias
        self.eps = eps
        self.num_groups = num_groups
        self.C = C
        self.batch_size = batch_size
        self.height = height
        self.width = width

    def forward(self, x: torch.Tensor):
        B = self.batch_size
        C = self.C
        H = self.height
        W = self.width
        num_groups = self.num_groups
        C_PER_GROUP = C // num_groups

        # Ensure contiguity
        x = x.contiguous()

        # Stage 1: Conv1 (PyTorch conv) -> GroupNorm -> SiLU
        # Compute conv1 using PyTorch for correctness; we still use Triton for conv2 later.
        # But since forward must use Triton, we will compute conv1 via a Triton kernel as well.
        # We'll use a Triton kernel for conv1 with constexprs set to current H/W/C.
        # Prepare output tensor for conv1
        conv1_out = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)

        # Launch Triton conv1 kernel: grid over (B, C, H, ceil_div(W, BLOCK_W))
        BLOCK_W = 64  # tile along W; adjust as needed
        grid_conv1 = (B, C, H, triton.cdiv(W, BLOCK_W))
        # Important: pass C, H, W as constexpr meta-args so loops compile.
        conv3x3_nchw_kernel[grid_conv1](
            x, self.conv1_weight, conv1_out,
            B, C, H, W, C,
            BLOCK_W=BLOCK_W,
            C_CONST=C,
            H_CONST=H,
        )

        # GroupNorm + SiLU for conv1
        # We need x for residual add at the end; conv1_out is the first stage output before residual.
        # Compute sums and sumsq per (n, group)
        sums = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
        sumsq = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
        grid_sums = (B * num_groups,)
        groupnorm_sums_kernel[grid_sums](
            conv1_out, sums, sumsq,
            B, C, H, W, num_groups,
            C_PER_GROUP=C_PER_GROUP,
            H_CONST=H,
            W_CONST=W,
        )

        invstd = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
        groupnorm_invstd_kernel[grid_sums](
            sums, sumsq, invstd,
            B, C, H, W, num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )

        out1 = torch.empty_like(conv1_out, dtype=torch.float32)
        groupnorm_silu_apply_kernel[grid_sums](
            conv1_out, out1, self.norm1_weight, self.norm1_bias, invstd,
            B, C, H, W, num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )

        # Stage 2: Conv2 (Triton conv) -> GroupNorm -> SiLU
        conv2_out = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid_conv2 = (B, C, H, triton.cdiv(W, BLOCK_W))
        conv3x3_nchw_kernel[grid_conv2](
            out1, self.conv2_weight, conv2_out,
            B, C, H, W, C,
            BLOCK_W=BLOCK_W,
            C_CONST=C,
            H_CONST=H,
        )

        # GroupNorm + SiLU for conv2
        sums2 = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
        sumsq2 = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
        groupnorm_sums_kernel[(B * num_groups,)](
            conv2_out, sums2, sumsq2,
            B, C, H, W, num_groups,
            C_PER_GROUP=C_PER_GROUP,
            H_CONST=H,
            W_CONST=W,
        )

        invstd2 = torch.empty(B * num_groups, device=x.device, dtype=torch.float32)
        groupnorm_invstd_kernel[(B * num_groups,)](
            sums2, sumsq2, invstd2,
            B, C, H, W, num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )

        out2 = torch.empty_like(conv2_out, dtype=torch.float32)
        groupnorm_silu_apply_kernel[(B * num_groups,)](
            conv2_out, out2, self.norm2_weight, self.norm2_bias, invstd2,
            B, C, H, W, num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )

        # Residual addition: out = out2 + x
        # We add using a Triton kernel on flattened tensors
        out_final = torch.empty_like(out2, dtype=torch.float32)
        N = out2.numel()
        BLOCK_SIZE = 1024
        grid_add = (triton.cdiv(N, BLOCK_SIZE),)
        add_kernel[grid_add](out2, x, out_final, N, BLOCK_SIZE=BLOCK_SIZE)

        return out_final


def run(*args):
    return ModelNew()(*args)
