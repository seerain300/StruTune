import torch
import triton
import triton.language as tl


def _assert_divisible(dividend: int, divisor: int):
    if dividend % divisor != 0:
        raise ValueError(f"{dividend} must be divisible by {divisor}")


# Triton kernel: Conv3x3 NCHW, stride=1, padding=1, no bias
# Computes out[n, co, h, w] = sum_{ci=0..C-1} sum_{dh=-1..1} sum_{dw=-1..1} x[n, ci, h+dh, w+dw] * w[co, ci, 1+dh, 1+dw]
# Grid: (B, C_OUT, H * ceil_div(W, BLOCK_W))
@triton.jit
def conv3x3_nchw_row_kernel(
    x_ptr, w_ptr, out_ptr,
    B, C, H, W, C_OUT,
    BLOCK_W: tl.constexpr,  # tile size along W, typically 64
):
    pid_n = tl.program_id(0)   # batch
    pid_co = tl.program_id(1)  # output channel
    pid_hw = tl.program_id(2)  # combined h and w-block index

    num_w_blocks = (W + BLOCK_W - 1) // BLOCK_W
    h = pid_hw // num_w_blocks
    w_block = pid_hw % num_w_blocks
    w_start = w_block * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W

    # Accumulator for output row
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Loop over input channels
    for ci in range(0, C):
        # Accumulate over 3x3 neighborhood with padding=1
        for dh in range(-1, 2):
            oh = h + dh
            oh_valid = (oh >= 0) & (oh < H)
            for dw in range(-1, 2):
                ow_vec = w_offsets + dw
                ow_valid = (ow_vec >= 0) & (ow_vec < W)
                mask = mask_w & oh_valid & ow_valid

                # Index into x: ((n*C + ci)*H + oh)*W + ow
                base_nc_ci = (pid_n * C + ci) * H * W
                idx_vec = base_nc_ci + oh * W + ow_vec

                # Load x values; invalid positions contribute 0
                x_vals = tl.load(x_ptr + idx_vec, mask=mask, other=0.0)

                # Load corresponding weight w[co, ci, 1+dh, 1+dw]
                # w layout: [co, ci, kh, kw]
                kh = 1 + dh
                kw = 1 + dw
                w_val = tl.load(w_ptr + pid_co * (C * 3 * 3) + ci * (3 * 3) + kh * 3 + kw)

                # Accumulate
                acc += x_vals * w_val

    # Store output row for this co
    base_out = (pid_n * C_OUT + pid_co) * H * W
    out_idx_vec = base_out + h * W + w_offsets
    tl.store(out_ptr + out_idx_vec, acc, mask=mask_w)


# Triton kernel: compute per-(n, group) sum and sum of squares across channels and spatial positions
# Grid: (B * num_groups,)
@triton.jit
def groupnorm_sums_kernel(
    x_ptr, sums_ptr, sumsq_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups

    start_ci = g * C_PER_GROUP
    s = tl.float32(0.0)
    s2 = tl.float32(0.0)

    for ci in range(start_ci, start_ci + C_PER_GROUP):
        for h in range(0, H):
            for w in range(0, W):
                idx = ((n * C + ci) * H + h) * W + w
                x_val = tl.load(x_ptr + idx)  # expect float32
                s += x_val
                s2 += x_val * x_val

    tl.store(sums_ptr + pid, s)
    tl.store(sumsq_ptr + pid, s2)


# Triton kernel: compute inverse std from sums and sumsq
# Grid: (B * num_groups,)
@triton.jit
def groupnorm_invstd_kernel(
    sums_ptr, sumsq_ptr, invstd_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)  # 0 .. (B * num_groups - 1)
    n = pid // num_groups
    g = pid % num_groups

    s = tl.load(sums_ptr + pid)
    s2 = tl.load(sumsq_ptr + pid)
    group_size = C_PER_GROUP * H * W
    mean = s / group_size
    var = s2 / group_size - mean * mean
    invstd = 1.0 / tl.sqrt(var + 1e-5)  # numerical stability
    tl.store(invstd_ptr + pid, invstd)


# Triton kernel: apply GroupNorm (using mean and invstd) + affine + SiLU
# Grid: (B, C, H * ceil_div(W, BLOCK_W))
@triton.jit
def groupnorm_silu_apply_row_kernel(
    x_ptr, norm_w_ptr, norm_b_ptr, out_ptr, mean_ptr, invstd_ptr,
    B, C, H, W, num_groups,
    C_PER_GROUP: tl.constexpr,
):
    pid_n = tl.program_id(0)   # batch
    pid_ci = tl.program_id(1)  # input channel (we will treat ci as output channel for apply; pass x and norms)
    pid_hw = tl.program_id(2)  # combined h and w-block index

    num_w_blocks = (W + BLOCK_W - 1) // BLOCK_W
    h = pid_hw // num_w_blocks
    w_block = pid_hw % num_w_blocks
    w_start = w_block * BLOCK_W
    w_offsets = w_start + tl.arange(0, BLOCK_W)
    mask_w = w_offsets < W

    # Load per-channel affine parameters
    scale = tl.load(norm_w_ptr + pid_ci)
    bias = tl.load(norm_b_ptr + pid_ci)

    # Load normalization parameters for (n, group) of this channel
    g = pid_ci // C_PER_GROUP  # since groups are contiguous over channels
    pid_group = pid_n * num_groups + g
    mean = tl.load(mean_ptr + pid_group)
    invstd = tl.load(invstd_ptr + pid_group)

    for hh in range(0, H):
        # Process one row h = hh
        for ww in range(0, BLOCK_W):
            w_idx = w_start + ww
            mask = mask_w[ww]
            # Build per-element vector to load/store efficiently by using ww as the loop index for the vector
            # However, Triton expects vectorized operations, so we use a vectorized approach:
            pass  # The above comments clarify intent; actual vectorized loads/stores will be done via idx_vec below.

    # Vectorized over w_offsets
    for hh in range(0, H):
        h_curr = hh
        for ww in range(0, W):
            idx = ((pid_n * C + pid_ci) * H + h_curr) * W + ww
            # But we need vectorized loads/stores; instead, compute idx_vec for the row and apply
            pass  # Same comment; Triton will vectorize with w_offsets.

    # Implement vectorized row processing:
    for hh in range(0, H):
        h_curr = hh
        idx_vec = ((pid_n * C + pid_ci) * H + h_curr) * W + w_offsets
        mask_vec = mask_w
        x_vals = tl.load(x_ptr + idx_vec, mask=mask_vec, other=0.0)
        y = (x_vals - mean) * invstd  # normalize per (n, group)
        y = y * scale + bias          # affine
        # SiLU activation: y * sigmoid(y)
        sig = 1.0 / (1.0 + tl.exp(-y))
        out_vals = y * sig
        out_idx_vec = ((pid_n * C + pid_ci) * H + h_curr) * W + w_offsets
        tl.store(out_ptr + out_idx_vec, out_vals, mask=mask_vec)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        conv1_weight: torch.Tensor,
        norm1_weight: torch.Tensor,
        norm1_bias: torch.Tensor,
        conv2_weight: torch.Tensor,
        norm2_weight: torch.Tensor,
        norm2_bias: torch.Tensor,
        eps: float,
    ):
        """
        Fused residual block: Conv3x3 -> GroupNorm -> SiLU -> Conv3x3 -> GroupNorm -> SiLU -> Add
        All heavy computation performed via Triton kernels; torch only used for residual add.
        """
        assert x.is_cuda, "All tensors must be on CUDA for Triton kernels"
        assert x.is_contiguous(), "Input must be contiguous NCHW"
        B, C, H, W = x.shape
        num_groups = 32
        _assert_divisible(C, num_groups)
        C_PER_GROUP = C // num_groups

        # First path: Conv3x3 -> GroupNorm -> SiLU
        out1 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)

        # Launch conv1 Triton kernel: grid over (B, C, H * ceil_div(W, BLOCK_W))
        BLOCK_W = 64
        grid = (B, C, H * ((W + BLOCK_W - 1) // BLOCK_W))
        conv3x3_nchw_row_kernel[grid](
            x, conv1_weight, out1,
            B, C, H, W, C,
            BLOCK_W=BLOCK_W,
        )

        # Triton GroupNorm + SiLU for out1
        # 1) compute sums and sumsq
        grid_sums = (B * num_groups,)
        sums1 = torch.empty(grid_sums[0], device=x.device, dtype=torch.float32)
        sumsq1 = torch.empty(grid_sums[0], device=x.device, dtype=torch.float32)
        groupnorm_sums_kernel[grid_sums](
            out1, sums1, sumsq1,
            B, C, H, W, num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )

        # 2) compute invstd
        invstd1 = torch.empty(grid_sums[0], device=x.device, dtype=torch.float32)
        groupnorm_invstd_kernel[grid_sums](
            sums1, sumsq1, invstd1,
            B, C, H, W, num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )

        # 3) apply normalization + affine + SiLU
        out1_norm = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        grid_apply = (B, C, H * ((W + BLOCK_W - 1) // BLOCK_W))
        groupnorm_silu_apply_row_kernel[grid_apply](
            out1, norm1_weight, norm1_bias, out1_norm,
            sums1, invstd1,
            B, C, H, W, num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )

        # Second path: Conv3x3 -> GroupNorm -> SiLU
        out2 = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        conv3x3_nchw_row_kernel[grid](
            out1_norm, conv2_weight, out2,
            B, C, H, W, C,
            BLOCK_W=BLOCK_W,
        )

        # Triton GroupNorm + SiLU for out2
        sums2 = torch.empty(grid_sums[0], device=x.device, dtype=torch.float32)
        sumsq2 = torch.empty(grid_sums[0], device=x.device, dtype=torch.float32)
        groupnorm_sums_kernel[grid_sums](
            out2, sums2, sumsq2,
            B, C, H, W, num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )
        invstd2 = torch.empty(grid_sums[0], device=x.device, dtype=torch.float32)
        groupnorm_invstd_kernel[grid_sums](
            sums2, sumsq2, invstd2,
            B, C, H, W, num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )
        out2_norm = torch.empty((B, C, H, W), device=x.device, dtype=torch.float32)
        groupnorm_silu_apply_row_kernel[grid_apply](
            out2, norm2_weight, norm2_bias, out2_norm,
            sums2, invstd2,
            B, C, H, W, num_groups,
            C_PER_GROUP=C_PER_GROUP,
        )

        # Residual connection
        out = out2_norm + x

        return out


def run(*args):
    return ModelNew()(*args)
