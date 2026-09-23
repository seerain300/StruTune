import math
import torch
import triton
import triton.language as tl


# Triton kernels: 3D indexing over [B, S, K] with one program per row (b, s)
@triton.jit
def _reduce_sum_sumsq_kernel_3d(
    x_ptr,            # *const float (input, fp32)
    mean_out_ptr,     # *float (per-row mean, fp32)
    sumsq_out_ptr,    # *float (per-row sumsq, fp32)
    B: tl.constexpr,  # int
    S: tl.constexpr,  # int
    K: tl.constexpr,  # int
    stride_xb,        # int
    stride_xs,        # int
    stride_xk,        # int
    BLOCK_K: tl.constexpr,
):
    # program id maps to (b, s)
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Base offsets for the row (b, s)
    base_x = b * stride_xb + s * stride_xs

    # Accumulators in fp32
    sum_val = 0.0
    sumsq_val = 0.0

    # Iterate across K in tiles
    for kk in range(0, K, BLOCK_K):
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        # Load a tile of the row
        x_vals = tl.load(x_ptr + base_x + offs * stride_xk, mask=mask, other=0.0)
        # Accumulate sum and sum of squares
        sum_val += tl.sum(x_vals, axis=0)
        sumsq_val += tl.sum(x_vals * x_vals, axis=0)

    # Compute mean and sumsq (mean = sum/K, sumsq = sumsq/K)
    mean_val = sum_val / K
    sumsq_val = sumsq_val / K

    # Store results for this row
    # Output arrays have length B*S, linear index idx = b*S + s
    idx = b * S + s
    tl.store(mean_out_ptr + idx, mean_val)
    tl.store(sumsq_out_ptr + idx, sumsq_val)


@triton.jit
def _apply_threshold_relu_kernel_3d(
    x_ptr,            # *const float (input, fp32)
    y_ptr,            # *float (output, fp32)
    mean_ptr,         # *const float (per-row mean, fp32)
    sumsq_ptr,        # *const float (per-row sumsq, fp32)
    B: tl.constexpr,  # int
    S: tl.constexpr,  # int
    K: tl.constexpr,  # int
    stride_xb,        # int
    stride_xs,        # int
    stride_xk,        # int
    stride_yb,        # int
    stride_ys,        # int
    stride_yk,        # int
    z_score,          # float (scalar)
    BLOCK_K: tl.constexpr,
):
    # program id maps to (b, s)
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Base offsets for the row (b, s)
    base_x = b * stride_xb + s * stride_xs
    base_y = b * stride_yb + s * stride_ys

    # Load per-row mean and sumsq
    idx = b * S + s
    mean_val = tl.load(mean_ptr + idx)
    sumsq_val = tl.load(sumsq_ptr + idx)

    # Compute std = sqrt(max(var, 0)), where var = sumsq - mean^2
    var = sumsq_val - mean_val * mean_val
    var = tl.maximum(var, 0.0)
    std_val = tl.sqrt(var)

    # Threshold factor: m = mean + std * z_score
    m = mean_val + std_val * z_score

    # Apply: y = max(x - m, 0) for the entire row
    for kk in range(0, K, BLOCK_K):
        offs = kk + tl.arange(0, BLOCK_K)
        mask = offs < K
        x_vals = tl.load(x_ptr + base_x + offs * stride_xk, mask=mask, other=0.0)
        y_vals = x_vals - m
        # ReLU
        y_vals = tl.maximum(y_vals, 0.0)
        tl.store(y_ptr + base_y + offs * stride_yk, y_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9, block_k: int = 256):
        super().__init__()
        # Precompute inverse normal cdf for target_sparsity
        # Using a common approximation for N(0,1) quantile; target_sparsity in (0,1)
        # For sparsity=0.9, z_score ≈ 1.2815515655446004
        self.z_score = float(math.erf(target_sparsity * math.sqrt(2.0)))
        self.z_score = (self.z_score + 1.0) * 0.5  # Correct mapping: erf to N(0,1)
        # Alternatively, use precomputed constant:
        # self.z_score = 1.2815515655446004
        self.block_k = block_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure the input is 3D [B, S, K]
        assert x.dim() == 3, f"Expected 3D input [B, S, K], got shape {tuple(x.shape)}"
        # Make input contiguous and in float32 for computation
        x_f32 = x.to(torch.float32).contiguous()
        B, S, K = x_f32.shape

        # Allocate per-row buffers (fp32) on device
        mean_row = torch.empty(B * S, dtype=torch.float32, device=x_f32.device)
        sumsq_row = torch.empty(B * S, dtype=torch.float32, device=x_f32.device)

        # Launch reduction kernel: one program per (b, s) row
        grid = (B, S)
        _reduce_sum_sumsq_kernel_3d[grid](
            x_f32,
            mean_row,
            sumsq_row,
            B, S, K,
            x_f32.stride(0), x_f32.stride(1), x_f32.stride(2),
            self.block_k,
            num_warps=4,
            num_stages=2,
        )

        # Allocate output tensor (fp32) with same shape as input
        y = torch.empty_like(x_f32)

        # Launch elementwise kernel: one program per (b, s) row
        _apply_threshold_relu_kernel_3d[grid](
            x_f32,
            y,
            mean_row,
            sumsq_row,
            B, S, K,
            x_f32.stride(0), x_f32.stride(1), x_f32.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            float(self.z_score),  # scalar float
            self.block_k,
            num_warps=4,
            num_stages=2,
        )

        # Return in bfloat16 to match original behavior
        return y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
