import math
import torch
import triton
import triton.language as tl


# Triton kernel: reduce sum and sum of squares per (b, s) row across K features.
# One program per row (b, s).
@triton.jit
def _reduce_sum_sumsq_kernel_2d(
    x_ptr,           # *const float, base pointer
    mean_out_ptr,    # *float, per-row mean
    sumsq_out_ptr,   # *float, per-row sumsq
    B, S, K,         # int32 sizes
    stride_b, stride_s, stride_k,  # int64 strides
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)  # batch index
    s = tl.program_id(1)  # seq index

    # Base pointer for this row
    base = b * stride_b + s * stride_s

    # Accumulate sum and sumsq in fp32
    sum_val = 0.0
    sumsq_val = 0.0

    # Loop across K in tiles
    for kk in range(0, K, BLOCK_K):
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        ptrs = x_ptr + base + offs * stride_k
        vals = tl.load(ptrs, mask=mask, other=0.0)  # vals are float32
        sum_val += tl.sum(vals, axis=0)
        sumsq_val += tl.sum(vals * vals, axis=0)

    mean = sum_val / K
    sumsq = sumsq_val / K
    tl.store(mean_out_ptr + b * S + s, mean)
    tl.store(sumsq_out_ptr + b * S + s, sumsq)


# Triton kernel: elementwise apply threshold and ReLU per (b, s) row across K features.
# One program per row (b, s).
@triton.jit
def _apply_threshold_relu_kernel_2d(
    x_ptr,           # *const float, base pointer
    out_ptr,         # *float, 1D output buffer (size = B*S*K)
    mean_in_ptr,     # *const float, per-row mean
    sumsq_in_ptr,    # *const float, per-row sumsq
    B, S, K,         # int32 sizes
    stride_b, stride_s, stride_k,  # int64 strides
    z_score,          # float32 scalar: inverse normal cdf of target_sparsity
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)  # batch index
    s = tl.program_id(1)  # seq index

    # Load mean and sumsq for this row
    mean = tl.load(mean_in_ptr + b * S + s)
    sumsq = tl.load(sumsq_in_ptr + b * S + s)
    std = tl.sqrt(sumsq - mean * mean)  # population std
    std = tl.maximum(std, 0.0)
    threshold = mean + std * z_score

    base = b * stride_b + s * stride_s

    # Compute output index base for this row in 1D buffer
    row_start = (b * S + s) * K

    # Iterate across K in tiles; write to out_ptr at linear positions
    for kk in range(0, K, BLOCK_K):
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        x_ptrs = x_ptr + base + offs * stride_k
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)  # fp32
        # subtract threshold (same for all features of this row), apply ReLU
        y = x_vals - threshold
        y = tl.maximum(y, 0.0)
        out_ptrs = out_ptr + row_start + offs
        tl.store(out_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Precompute z_score for target_sparsity = 0.9 (standard normal 90th percentile)
        # This matches torch.erfinv behavior used implicitly in _ndtri approximation.
        self.z_score = float(1.2815515655446004)  # math.erf(1 - 2*target_sparsity) => z = norm.ppf(0.9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If no sparsity requested, return original as-is (original code returns same x).
        # Here we keep behavior: only sparsify if target_sparsity > 0. The original uses 0.9 by default.
        target_sparsity = 0.9
        if target_sparsity == 0.0:
            return x

        # Convert to float32 for numerical stability and make contiguous
        x_f32 = x.to(torch.float32).contiguous()
        B, S, K = x_f32.shape
        stride_b, stride_s, stride_k = x_f32.stride()

        # Per-row buffers (fp32) for mean and sumsq
        mean_row = torch.empty(B * S, dtype=torch.float32, device=x_f32.device)
        sumsq_row = torch.empty(B * S, dtype=torch.float32, device=x_f32.device)

        # Launch reduction kernel: one program per (b, s) row
        grid = (B, S)
        _reduce_sum_sumsq_kernel_2d[grid](
            x_f32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            BLOCK_K=256,
            num_warps=4,
            num_stages=2,
        )

        # Allocate 1D output buffer (fp32) and launch elementwise kernel
        out_fp32 = torch.empty(B * S * K, dtype=torch.float32, device=x_f32.device)

        _apply_threshold_relu_kernel_2d[grid](
            x_f32,
            out_fp32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            float(self.z_score),  # scalar float
            BLOCK_K=256,
            num_warps=4,
            num_stages=2,
        )

        # Reshape output to [B, S, K]
        out_fp32 = out_fp32.view(B, S, K)
        # Return in bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
