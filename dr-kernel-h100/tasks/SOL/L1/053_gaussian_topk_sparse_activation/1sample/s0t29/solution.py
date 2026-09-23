import math
import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_sum_sumsq_row_kernel(
    x_ptr,
    sum_ptr,
    sumsq_ptr,
    B, S, K,
    stride_b, stride_s, stride_k,
    BLOCK_K: tl.constexpr,
):
    # 2D grid over (B, S): each program handles one (b, s) row
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Base pointer for this row
    base = b * stride_b + s * stride_s

    # Accumulate sum and sumsq across K
    total_sum = 0.0
    total_sumsq = 0.0

    # Loop over K in tiles
    for kk in range(0, K, BLOCK_K):
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_ptr + base + offs * stride_k, mask=mask, other=0.0)
        # Accumulate in fp32
        x = x.to(tl.float32)
        total_sum += tl.sum(x, axis=0)
        x2 = x * x
        total_sumsq += tl.sum(x2, axis=0)

    # Write per-row sum and sumsq
    tl.store(sum_ptr + b * S + s, total_sum)
    tl.store(sumsq_ptr + b * S + s, total_sumsq)


@triton.jit
def _compute_threshold_row_kernel(
    sum_ptr,
    sumsq_ptr,
    threshold_ptr,
    B, S, K,  # K unused, kept for signature symmetry
    z_score,  # scalar float
    BLOCK_K: tl.constexpr,  # not used here, kept for potential future use
):
    # 2D grid over (B, S): each program handles one (b, s) row
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Load per-row sum and sumsq
    sum_val = tl.load(sum_ptr + b * S + s)
    sumsq_val = tl.load(sumsq_ptr + b * S + s)

    # Compute mean and std in fp32
    K_f32 = tl.full((), K, tl.float32)
    mean = sum_val / K_f32
    var = sumsq_val / K_f32 - mean * mean
    # std = sqrt(max(var, 0))
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # threshold = mean + std * z_score
    threshold = mean + std * z_score

    # Store per-row threshold
    tl.store(threshold_ptr + b * S + s, threshold)


@triton.jit
def _apply_threshold_relu_row_kernel(
    x_ptr,
    threshold_ptr,
    out_ptr,
    B, S, K,
    stride_b, stride_s, stride_k,
    z_score,  # scalar float (kept for signature symmetry)
    BLOCK_K: tl.constexpr,
):
    # 2D grid over (B, S): each program handles one (b, s) row
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Load per-row threshold
    t = tl.load(threshold_ptr + b * S + s)

    # Base pointers for this row
    base_in = b * stride_b + s * stride_s

    # Compute base_out for this row in 1D buffer: out is laid out as [B, S, K] flattened
    row_out_base = (b * S + s) * K

    # Iterate across K in tiles and apply y = max(0, x - t)
    for kk in range(0, K, BLOCK_K):
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(x_ptr + base_in + offs * stride_k, mask=mask, other=0.0)
        x = x.to(tl.float32)
        y = x - t
        # ReLU: max(0, y)
        zero = 0.0
        y = tl.maximum(y, zero)
        # Store to 1D output buffer
        tl.store(out_ptr + row_out_base + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9, block_k: int = 256):
        super().__init__()
        # Precompute z_score for target_sparsity = 0.9 (default), using the same definition as original
        # For 0.9, z_score ≈ 1.2815515655446004
        self.z_score = float(target_sparsity)
        self.block_k = block_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous
        x = x.contiguous()
        # We operate in float32 for numerical stability in reductions and std
        B, S, K = x.shape
        device = x.device

        # Strides in elements
        stride_b, stride_s, stride_k = x.stride()

        # Allocate per-row sum and sumsq (fp32)
        sum_row = torch.empty(B * S, dtype=torch.float32, device=device)
        sumsq_row = torch.empty(B * S, dtype=torch.float32, device=device)

        # 2D grid over (B, S) so each program handles one row
        grid = (B, S)

        # Launch reduction kernel: compute per-row sum and sumsq
        _reduce_sum_sumsq_row_kernel[grid](
            x,
            sum_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            self.block_k,
            num_warps=4,
            num_stages=2,
        )

        # Allocate per-row threshold buffer (fp32)
        threshold_row = torch.empty(B * S, dtype=torch.float32, device=device)

        # Launch compute threshold kernel: compute per-row threshold = mean + std * z_score
        _compute_threshold_row_kernel[grid](
            sum_row,
            sumsq_row,
            threshold_row,
            B, S, K,
            self.z_score,
            self.block_k,
            num_warps=4,
            num_stages=1,
        )

        # Allocate 1D output buffer (fp32) and launch apply kernel
        out_fp32 = torch.empty(B * S * K, dtype=torch.float32, device=device)

        _apply_threshold_relu_row_kernel[grid](
            x,
            threshold_row,
            out_fp32,
            B, S, K,
            stride_b, stride_s, stride_k,
            self.z_score,
            self.block_k,
            num_warps=4,
            num_stages=2,
        )

        # Reshape output to [B, S, K]
        out_fp32 = out_fp32.view(B, S, K)
        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
