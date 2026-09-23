import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute sum per feature across all rows (B*S)
@triton.jit
def sum_per_feature_kernel(x_ptr, sum_ptr,
                            B, S, L,
                            BLOCK_ROWS: tl.constexpr):
    f = tl.program_id(0)
    rows = B * S
    total_sum = 0.0
    for row_start in range(0, rows, BLOCK_ROWS):
        row_offsets = row_start + tl.arange(0, BLOCK_ROWS)
        mask_rows = row_offsets < rows
        idx = row_offsets * L + f
        vals = tl.load(x_ptr + idx, mask=mask_rows, other=0.0)
        total_sum += tl.sum(vals, axis=0)
    tl.store(sum_ptr + f, total_sum)


# Triton kernel: compute sum of squares per feature across all rows (B*S)
@triton.jit
def sumsq_per_feature_kernel(x_ptr, sumsq_ptr,
                              B, S, L,
                              BLOCK_ROWS: tl.constexpr):
    f = tl.program_id(0)
    rows = B * S
    total_sumsq = 0.0
    for row_start in range(0, rows, BLOCK_ROWS):
        row_offsets = row_start + tl.arange(0, BLOCK_ROWS)
        mask_rows = row_offsets < rows
        idx = row_offsets * L + f
        vals = tl.load(x_ptr + idx, mask=mask_rows, other=0.0)
        total_sumsq += tl.sum(vals * vals, axis=0)
    tl.store(sumsq_ptr + f, total_sumsq)


# Triton kernel: compute mean, std, and per-feature threshold = mean + std * ndtri(target_sparsity)
@triton.jit
def compute_mean_std_per_feature_kernel(sum_ptr, sumsq_ptr, threshold_ptr,
                                        B, S, L, target_sparsity):
    # One program per feature
    f = tl.program_id(0)
    sum_f = tl.load(sum_ptr + f)
    sumsq_f = tl.load(sumsq_ptr + f)

    rows = B * S
    mean = sum_f / rows
    var = sumsq_f / rows - mean * mean
    var = tl.maximum(var, 0.0)  # guard against tiny negative
    std = tl.sqrt(var)

    # Compute q = ndtri(target_sparsity) using Abramowitz-Stegun 7.1.26 approximation
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    p_low = 0.02425

    p = target_sparsity  # scalar argument
    t = 1.0 - p
    # Lower region
    q_low = (((c1 * t + c2) * t + c3) / (((d1 * t + d2) * t + d3)))
    # Upper region
    q_up = -(((((c1 * t + c2) * t + c3) * t + c4) * t + c5) * t + c6) / (((((d1 * t + d2) * t + d3) * t + d4) * t + 1.0))
    # Central region
    u = tl.sqrt(-2.0 * tl.log(p)) - 1.0
    q_mid = (((((a1 * u + a2) * u + a3) * u + a4) * u + a5) * u + a6) * u / (((((b1 * u + b2) * u + b3) * u + b4) * u + b5) * u + 1.0)

    mask_low = t > p_low
    mask_high = t < 1.0 - p_low
    mask_mid = ~(mask_low | mask_high)

    q = tl.where(mask_low, q_low, tl.where(mask_high, q_up, q_mid))

    thr = mean + std * q
    tl.store(threshold_ptr + f, thr)


# Triton kernel: elementwise sparse ReLU with per-feature threshold broadcast
@triton.jit
def sparse_relu_per_feature_kernel(x_ptr, threshold_ptr,
                                    out_ptr,
                                    B, S, L):
    pid_row = tl.program_id(0)  # rows dimension
    pid_f = tl.program_id(1)    # features dimension
    rows = B * S
    mask = (pid_row < rows) & (pid_f < L)
    idx = pid_row * L + pid_f
    x_val = tl.load(x_ptr + idx, mask=mask, other=0.0)
    thr = tl.load(threshold_ptr + pid_f)
    y = x_val - thr
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + idx, y, mask=mask)


# Triton kernel: cast FP32 to BF16 elementwise
@triton.jit
def cast_bf16_kernel(in_ptr, out_ptr, N, BLOCK: tl.constexpr):
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        vals = tl.load(in_ptr + offs, mask=mask, other=0.0)
        tl.store(out_ptr + offs, vals, mask=mask)  # implicit cast on store


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.0):
        super().__init__()
        self.target_sparsity = float(target_sparsity)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Ensure inputs: [B, S, L], contiguous
        assert inputs.dim() == 3, "inputs must be 3D: [B, S, L]"
        inputs = inputs.contiguous()
        B, S, L = inputs.shape

        # Allocate per-feature buffers
        sum_f = torch.empty(L, dtype=torch.float32, device=inputs.device)
        sumsq_f = torch.empty(L, dtype=torch.float32, device=inputs.device)
        threshold_f = torch.empty(L, dtype=torch.float32, device=inputs.device)

        # 1) Reduce sum and sumsq per feature across all rows (no torch ops in forward)
        BLOCK_ROWS = 1024
        grid_sum = (L,)
        sum_per_feature_kernel[grid_sum](
            inputs, sum_f,
            B, S, L,
            BLOCK_ROWS=BLOCK_ROWS,
            num_warps=4
        )

        sumsq_per_feature_kernel[grid_sum](
            inputs, sumsq_f,
            B, S, L,
            BLOCK_ROWS=BLOCK_ROWS,
            num_warps=4
        )

        # 2) Compute mean, std, and per-feature threshold using Triton (ndtri inside kernel)
        compute_mean_std_per_feature_kernel[(L,)](
            sum_f, sumsq_f, threshold_f,
            B, S, L, self.target_sparsity,
            num_warps=1
        )

        # 3) Elementwise sparse ReLU: broadcast threshold per feature
        out_fp32 = torch.empty(B * S * L, dtype=torch.float32, device=inputs.device)
        grid_relu = (B * S, L)
        sparse_relu_per_feature_kernel[grid_relu](
            inputs, threshold_f,
            out_fp32,
            B, S, L,
            num_warps=4
        )

        # 4) Cast to bfloat16 via Triton kernel (forward MUST invoke this)
        out_bf16 = torch.empty(B * S * L, dtype=torch.bfloat16, device=inputs.device)
        BLOCK_CAST = 4096
        grid_cast = (triton.cdiv(B * S * L, BLOCK_CAST),)
        cast_bf16_kernel[grid_cast](
            out_fp32, out_bf16, B * S * L,
            BLOCK=BLOCK_CAST, num_warps=4
        )

        return out_bf16.view(B, S, L)


def run(*args):
    return ModelNew()(*args)
