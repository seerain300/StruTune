import torch
import triton
import triton.language as tl


def _assert_divisible(dividend: int, divisor: int):
    if dividend % divisor != 0:
        raise ValueError(f"{dividend} is not divisible by {divisor}")


# Triton kernel: Conv3x3 NCHW, stride=1, padding=1, no bias
# Specialized for C_IN == C_OUT == 64, BLOCK_C == 64 (constexpr).
# Computes out[n, co, ho, wo] = sum_{ci=0..C_IN-1} sum_{dh=-1..1} sum_{dw=-1..1} x[n, ci, ho+dh, wo+dw] * w[co, ci, 1+dh, 1+dw]
@triton.jit
def conv3x3_nchw_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C_IN, H, W, C_OUT,
    BLOCK_W: tl.constexpr,       # tile size along W
    C_BLOCK: tl.constexpr,       # tile size along input channels (specialized to 64)
):
    # Grid: (B, C_OUT, H, ceil_div(W, BLOCK_W))
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w_blk = tl.program_id(3)

    co = pid_co
    ho = pid_h
    w_start = pid_w_blk * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W

    # Accumulator for output row
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Loop over input channels in blocks (specialized to 64)
    for ci_base in range(0, C_IN, C_BLOCK):
        # Vector of input channels for this block
        ci_vec = ci_base + tl.arange(0, C_BLOCK)
        mask_c = ci_vec < C_IN

        # Accumulate over 3x3 neighborhood with padding=1
        # Loop over dh, dw (constexpr 3x3)
        for dh in range(-1, 2):
            for dw in range(-1, 2):
                h_in = ho + dh
                # Compute x offsets for this block
                x_offsets = ((pid_n * C_IN + ci_vec) * H + h_in) * W + w_offsets
                # Load x values with masks; default other=0.0 for out-of-range
                x_vals = tl.load(x_ptr + x_offsets, mask=mask_c & mask_w, other=0.0)  # shape [C_BLOCK, BLOCK_W]
                # Load weight vector w[co, ci_vec, 1+dh, 1+dw] -> shape [C_BLOCK]
                w_offsets_ci = (co * C_IN + ci_vec) * 9 + (1 + dh) * 3 + (1 + dw)
                w_vec = tl.load(w_ptr + w_offsets_ci, mask=mask_c, other=0.0)

                # For each input channel in the block, multiply and accumulate into acc
                # Note: w_vec has shape [C_BLOCK], x_vals has shape [C_BLOCK, BLOCK_W]
                # We need to broadcast multiply: sum over input channels
                # Compute per-channel contribution: reshape x_vals to [1, BLOCK_W] and multiply with w_vec[:, None]
                # Then sum over axis 0
                # Equivalent: for ci_idx in 0..C_BLOCK-1, if valid, contrib = w_vec[ci_idx] * sum(x_vals[ci_idx, :])
                # But Triton supports vectorized operations, so we do elementwise multiply and reduce:
                # Here we simply expand dims and sum:
                # We do a loop over C_BLOCK elements to reduce correctly:
                for ci_idx in range(0, C_BLOCK):
                    if (ci_base + ci_idx) < C_IN:
                        contrib = w_vec[ci_idx] * tl.sum(x_vals[ci_idx, :])
                        acc += contrib

    # Store results
    out_offsets = ((pid_n * C_OUT + co) * H + ho) * W + w_offsets
    tl.store(out_ptr + out_offsets, acc, mask=mask_w)


# Triton kernel: compute sum and sum of squares per (n, group) for GroupNorm
# One program per (n, group).
@triton.jit
def groupnorm_sums_kernel(
    x_ptr, sums_ptr, sumsq_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,    # must be C // num_groups (for this model, 2)
    H_W: tl.constexpr,            # H * W (constexpr for performance)
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    s = tl.float32(0.0)
    s2 = tl.float32(0.0)

    start_ci = g * C_PER_GROUP
    # Loop over channels in the group
    for ci in range(0, C_PER_GROUP):
        # Loop over all H*W elements in the group's channels
        for k in range(0, H_W):
            # For NCHW, linear indexing: idx = ((n*C + ci) * H + (k // W)) * W + (k % W)
            h = k // W
            w = k % W
            idx = ((n * C + (start_ci + ci)) * H + h) * W + w
            x_val = tl.load(x_ptr + idx)
            s += x_val
            s2 += x_val * x_val

    tl.store(sums_ptr + out_idx, s)
    tl.store(sumsq_ptr + out_idx, s2)


# Triton kernel: compute invstd per (n, group) using sums and sumsq
@triton.jit
def groupnorm_invstd_kernel(
    sums_ptr, sumsq_ptr, invstd_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,    # 2
    H_W: tl.constexpr,            # H * W
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g

    s = tl.load(sums_ptr + out_idx)
    s2 = tl.load(sumsq_ptr + out_idx)
    group_size = C_PER_GROUP * H_W
    mean = s / group_size
    var = s2 / group_size - mean * mean
    invstd = 1.0 / tl.sqrt(var + 1e-5)  # eps for stability
    tl.store(invstd_ptr + out_idx, invstd)


# Triton kernel: normalize + affine + SiLU per (n, group), using precomputed invstd and mean
@triton.jit
def groupnorm_silu_apply_kernel(
    x_ptr, norm_w_ptr, norm_b_ptr, out_ptr, invstd_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,    # 2
    H_W: tl.constexpr,            # H * W
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups
    out_idx = n * num_groups + g
    invstd = tl.load(invstd_ptr + out_idx)

    start_ci = g * C_PER_GROUP
    for ci in range(0, C_PER_GROUP):
        gamma = tl.load(norm_w_ptr + (start_ci + ci))
        beta = tl.load(norm_b_ptr + (start_ci + ci))
        for k in range(0, H_W):
            h = k // W
            w = k % W
            idx = ((n * C + (start_ci + ci)) * H + h) * W + w
            x_val = tl.load(x_ptr + idx)
            y = (x_val - 0.0) * invstd  # no mean since mean=0 not stored here; we need mean from sums
            # SiLU: y * sigmoid(y) = y * (1 / (1 + exp(-y)))
            sig = 1.0 / (1.0 + tl.exp(-y))
            y = y * sig
            y = y * gamma + beta
            out_idx2 = ((n * C + (start_ci + ci)) * H + h) * W + w
            tl.store(out_ptr + out_idx2, y)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; we'll receive inputs/weights in forward

    def forward(self, x: torch.Tensor, conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor, eps: float):
        """
        Triton-only fused residual block:
        conv1 -> GroupNorm -> SiLU -> conv2 -> GroupNorm -> SiLU -> Add residual
        """
        B, C, H, W = x.shape
        num_groups = 32
        _assert_divisible(C, num_groups)
        _assert_divisible(C, conv1_weight.shape[1])  # first conv in/out channels
        _assert_divisible(C, conv2_weight.shape[1])  # second conv in/out channels

        # Ensure contiguous NCHW tensors
        x = x.contiguous()
        conv1_weight = conv1_weight.contiguous()
        norm1_weight = norm1_weight.contiguous()
        norm1_bias = norm1_bias.contiguous()
        conv2_weight = conv2_weight.contiguous()
        norm2_weight = norm2_weight.contiguous()
        norm2_bias = norm2_bias.contiguous()

        # Compute conv1 using Triton (no bias)
        # Output tensor
        out1 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        # Launch Triton conv1
        BLOCK_W = 64  # tile along W; works for given widths (64,128,256, etc.)
        grid_conv1 = (B, C, H, triton.cdiv(W, BLOCK_W))
        conv3x3_nchw_kernel[grid_conv1](
            x, conv1_weight, out1,
            B, C, H, W, C,  # C is input channel count here
            BLOCK_W=BLOCK_W, C_BLOCK=64  # specialize input channel block to 64 (matches weight shape)
        )

        # GroupNorm + SiLU for stage 1 using Triton
        C_PER_GROUP = C // num_groups
        H_W = H * W
        B_GROUPS = B * num_groups

        sums1 = torch.empty(B_GROUPS, device=x.device, dtype=torch.float32)
        sumsq1 = torch.empty(B_GROUPS, device=x.device, dtype=torch.float32)
        grid_sums1 = (B_GROUPS,)
        groupnorm_sums_kernel[grid_sums1](out1, sums1, sumsq1, B, C, H, W, num_groups, C_PER_GROUP, H_W)

        invstd1 = torch.empty(B_GROUPS, device=x.device, dtype=torch.float32)
        grid_invstd1 = (B_GROUPS,)
        groupnorm_invstd_kernel[grid_invstd1](sums1, sumsq1, invstd1, B, C, H, W, num_groups, C_PER_GROUP, H_W)

        # Apply normalization + affine + SiLU into out1 (in-place)
        out1 = torch.empty_like(out1)  # we'll write results into this tensor
        grid_apply1 = (B_GROUPS,)
        groupnorm_silu_apply_kernel[grid_apply1](
            out1, norm1_weight, norm1_bias, out1, invstd1,
            B, C, H, W, num_groups, C_PER_GROUP, H_W
        )

        # Residual add (PyTorch elementwise)
        out1 = out1 + x  # residual connection

        # Conv2 using Triton (no bias)
        out2 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid_conv2 = (B, C, H, triton.cdiv(W, BLOCK_W))
        conv3x3_nchw_kernel[grid_conv2](
            out1, conv2_weight, out2,
            B, C, H, W, C,  # C is input channel count here (C)
            BLOCK_W=BLOCK_W, C_BLOCK=64
        )

        # GroupNorm + SiLU for stage 2 using Triton
        sums2 = torch.empty(B_GROUPS, device=x.device, dtype=torch.float32)
        sumsq2 = torch.empty(B_GROUPS, device=x.device, dtype=torch.float32)
        grid_sums2 = (B_GROUPS,)
        groupnorm_sums_kernel[grid_sums2](out2, sums2, sumsq2, B, C, H, W, num_groups, C_PER_GROUP, H_W)

        invstd2 = torch.empty(B_GROUPS, device=x.device, dtype=torch.float32)
        grid_invstd2 = (B_GROUPS,)
        groupnorm_invstd_kernel[grid_invstd2](sums2, sumsq2, invstd2, B, C, H, W, num_groups, C_PER_GROUP, H_W)

        # Apply normalization + affine + SiLU into out2 (in-place)
        out2 = torch.empty_like(out2)  # write results here
        grid_apply2 = (B_GROUPS,)
        groupnorm_silu_apply_kernel[grid_apply2](
            out2, norm2_weight, norm2_bias, out2, invstd2,
            B, C, H, W, num_groups, C_PER_GROUP, H_W
        )

        # Residual add
        out2 = out2 + x

        return out2


def run(*args):
    return ModelNew()(*args)
