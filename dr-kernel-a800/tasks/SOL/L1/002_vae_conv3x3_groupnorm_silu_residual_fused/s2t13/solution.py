import torch
import triton
import triton.language as tl


def _assert_divisible(dividend: int, divisor: int):
    if dividend % divisor != 0:
        raise ValueError(f"{dividend} must be divisible by {divisor}")


# Triton kernel: Conv3x3 NCHW, stride=1, padding=1, no bias
# Computes out[n, co, h, w] = sum_{ci=0..C-1} sum_{dh in [-1,0,1]} sum_{dw in [-1,0,1]} x[n, ci, h+dh, w+dw] * w[co, ci, 1+dh, 1+dw]
@triton.jit
def conv3x3_nchw_per_pixel_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C, H, W, C_OUT,
):
    # Grid: (B, C_OUT, H, W)
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    n = pid_n
    co = pid_co
    h = pid_h
    w = pid_w

    acc = tl.float32(0.0)

    # Iterate over input channels
    for ci in range(0, C):
        # Iterate over 3x3 neighborhood with padding=1
        for dh in range(-1, 2):
            h_in = h + dh
            # Valid only if 0 <= h_in < H
            h_valid = (h_in >= 0) & (h_in < H)
            for dw in range(-1, 2):
                w_in = w + dw
                w_valid = (w_in >= 0) & (w_in < W)
                in_bounds = h_valid & w_valid

                # Compute input and weight pointers
                # x[n, ci, h_in, w_in], w[co, ci, 1+dh, 1+dw]
                # Indexing: ((n*C + ci)*H + h_in)*W + w_in
                x_idx = ((n * C + ci) * H + h_in) * W + w_in
                w_idx = ((co * C + ci) * 3 + (1 + dh)) * 3 + (1 + dw)  # flatten (co, ci, dh, dw)

                x_val = tl.load(x_ptr + x_idx, mask=in_bounds, other=0.0)
                w_val = tl.load(w_ptr + w_idx)

                acc += x_val * w_val

    out_idx = ((n * C_OUT + co) * H + h) * W + w
    tl.store(out_ptr + out_idx, acc)


# Triton kernel: compute sum and sum of squares per (n, channel) over H*W (GroupNorm reduction)
@triton.jit
def groupnorm_reduce_kernel(
    x_ptr, sums_ptr, sumsq_ptr,
    B, C, H, W,
):
    # One program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    s = tl.float32(0.0)
    s2 = tl.float32(0.0)

    for h in range(0, H):
        for w in range(0, W):
            idx = ((n * C + c) * H + h) * W + w
            x_val = tl.load(x_ptr + idx)
            s += x_val
            s2 += x_val * x_val

    out_idx = n * C + c
    tl.store(sums_ptr + out_idx, s)
    tl.store(sumsq_ptr + out_idx, s2)


# Triton kernel: compute inverse std per (n, channel) from sums and sumsq
@triton.jit
def groupnorm_invstd_kernel(
    sums_ptr, sumsq_ptr, invstd_ptr,
    B, C, H, W,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    s = tl.load(sums_ptr + (n * C + c))
    s2 = tl.load(sumsq_ptr + (n * C + c))
    group_size = H * W
    mean = s / group_size
    var = s2 / group_size - mean * mean
    invstd = 1.0 / tl.sqrt(var + 1e-5)  # eps for stability
    tl.store(invstd_ptr + (n * C + c), invstd)


# Triton kernel: apply GroupNorm + affine + SiLU per (n, channel)
@triton.jit
def groupnorm_silu_apply_kernel(
    x_ptr, norm_w_ptr, norm_b_ptr, out_ptr, invstd_ptr,
    B, C, H, W,
):
    # grid: (B, C)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    invstd = tl.load(invstd_ptr + (n * C + c))
    w = tl.load(norm_w_ptr + c)
    b = tl.load(norm_b_ptr + c)

    for h in range(0, H):
        for w_idx in range(0, W):
            idx = ((n * C + c) * H + h) * W + w_idx
            x_val = tl.load(x_ptr + idx)
            # normalize + affine
            z = x_val * invstd * w + b
            # SiLU: z * sigmoid(z), sigmoid(z) = 1 / (1 + exp(-z))
            sig = 1.0 / (1.0 + tl.exp(-z))
            y = z * sig
            tl.store(out_ptr + idx, y)


# Triton kernel: elementwise residual add out += x
@triton.jit
def add_residual_kernel(
    out_ptr, x_ptr,
    B, C, H, W,
):
    # grid: (B, C, H, W)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    n = pid_n
    c = pid_c
    h = pid_h
    w = pid_w

    out_idx = ((n * C + c) * H + h) * W + w
    out_val = tl.load(out_ptr + out_idx)
    x_val = tl.load(x_ptr + out_idx)  # same layout as out
    out_val += x_val
    tl.store(out_ptr + out_idx, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        num_groups = 32
        _assert_divisible(64, num_groups)  # enforce C=64, num_groups=32
        self.num_groups = num_groups
        self.C = 64
        self.C_per_group = self.C // self.num_groups  # 2

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor):
        """
        x: (B, C, H, W), float32
        convX_weight: (C, C, 3, 3), float32
        normX_weight, normX_bias: (C,), float32
        """
        assert x.ndim == 4, "x must be (B, C, H, W)"
        B, C, H, W = x.shape
        assert C == self.C, f"Expected C={self.C}, got {C}"
        # Ensure contiguous NCHW
        x = x.contiguous()

        # Allocate outputs
        out1 = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)
        out2 = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)

        # Launch conv1
        conv3x3_nchw_per_pixel_kernel[(B, C, H, W)](
            x, conv1_weight, out1,
            B, C, H, W, C,
            num_warps=1, num_stages=1,
        )

        # GroupNorm + SiLU stage 1
        # Reduction
        sums1 = torch.empty((B, C), dtype=torch.float32, device=x.device)
        sumsq1 = torch.empty((B, C), dtype=torch.float32, device=x.device)
        groupnorm_reduce_kernel[(B, C)](
            out1, sums1, sumsq1,
            B, C, H, W,
            num_warps=1, num_stages=1,
        )
        # invstd
        invstd1 = torch.empty((B, C), dtype=torch.float32, device=x.device)
        groupnorm_invstd_kernel[(B, C)](
            sums1, sumsq1, invstd1,
            B, C, H, W,
            num_warps=1, num_stages=1,
        )
        # Apply affine + SiLU
        out_silu1 = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)
        groupnorm_silu_apply_kernel[(B, C)](
            out1, norm1_weight, norm1_bias, out_silu1, invstd1,
            B, C, H, W,
            num_warps=1, num_stages=1,
        )

        # Residual add: out_silu1 += x
        add_residual_kernel[(B, C, H, W)](
            out_silu1, x,
            B, C, H, W,
            num_warps=1, num_stages=1,
        )

        # conv2
        conv3x3_nchw_per_pixel_kernel[(B, C, H, W)](
            out_silu1, conv2_weight, out2,
            B, C, H, W, C,
            num_warps=1, num_stages=1,
        )

        # GroupNorm + SiLU stage 2
        # Reduction
        sums2 = torch.empty((B, C), dtype=torch.float32, device=x.device)
        sumsq2 = torch.empty((B, C), dtype=torch.float32, device=x.device)
        groupnorm_reduce_kernel[(B, C)](
            out2, sums2, sumsq2,
            B, C, H, W,
            num_warps=1, num_stages=1,
        )
        # invstd
        invstd2 = torch.empty((B, C), dtype=torch.float32, device=x.device)
        groupnorm_invstd_kernel[(B, C)](
            sums2, sumsq2, invstd2,
            B, C, H, W,
            num_warps=1, num_stages=1,
        )
        # Apply affine + SiLU
        out_silu2 = torch.empty((B, C, H, W), dtype=torch.float32, device=x.device)
        groupnorm_silu_apply_kernel[(B, C)](
            out2, norm2_weight, norm2_bias, out_silu2, invstd2,
            B, C, H, W,
            num_warps=1, num_stages=1,
        )

        # Residual add: out_silu2 += x
        add_residual_kernel[(B, C, H, W)](
            out_silu2, x,
            B, C, H, W,
            num_warps=1, num_stages=1,
        )

        return out_silu2


def run(*args):
    return ModelNew()(*args)
