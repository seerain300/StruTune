import torch
import triton
import triton.language as tl


@triton.jit
def reduce_sum_sumsq_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    B, S, D,         # int32 dimensions
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0..(B*S - 1)
    b = pid // S
    s = pid % S

    base = b * S + s
    base_idx = (base * D) + tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    sum_val = 0.0
    sumsq_val = 0.0

    offs = 0
    while offs < D:
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        ptrs = X_ptr + base_idx + idx
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)
        offs += BLOCK_SIZE

    tl.store(SUM_ptr + pid, sum_val)
    tl.store(SUMSQ_ptr + pid, sumsq_val)


@triton.jit
def compute_mean_std_kernel(
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    D,               # int32
):
    pid = tl.program_id(axis=0)
    sum_val = tl.load(SUM_ptr + pid)
    sumsq_val = tl.load(SUMSQ_ptr + pid)
    d = tl.full((), D, tl.int32)
    mean = sum_val / d
    var = sumsq_val / d - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + pid, mean)
    tl.store(STD_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(
    THRESH_SCALE_ptr,  # *float32, length 1 (scalar sparsity)
    OUT_ptr,           # *float32, length 1 (output z-score)
):
    # Load sparsity p (scalar)
    p = tl.load(THRESH_SCALE_ptr)
    # Constants for A&S 5.2.23 approximation (Abramowitz & Stegun)
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
    p_high = 1.0 - p_low

    # Lower region
    q = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
    den_low = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    z_low = poly_low / den_low

    # Central region
    q2 = p - 0.5
    r2 = q2 * q2
    poly_mid = (((((a1 * r2 + a2) * r2 + a3) * r2 + a4) * r2 + a5) * r2 + a6)
    den_mid = (((((b1 * r2 + b2) * r2 + b3) * r2 + b4) * r2 + b5) * r2 + 1.0)
    z_mid = poly_mid * q2 / den_mid

    # Upper region
    q3 = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_up = (((((c1 * q3 + c2) * q3 + c3) * q3 + c4) * q3 + c5) * q3 + c6)
    den_up = ((((d1 * q3 + d2) * q3 + d3) * q3 + d4) * q3 + 1.0)
    z_up = -poly_up / den_up

    # Select appropriate z based on p
    z = tl.where(p < p_low, z_low, tl.where(p > p_high, z_up, z_mid))

    # Store scalar z to OUT_ptr[0]
    tl.store(OUT_ptr, z)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    Z_ptr,           # *float32, length 1
    OUT_ptr,         # *float32, output [B, S, D]
    B, S, D,         # int32
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    tile = tl.program_id(axis=2)

    pid = b * S + s
    mean = tl.load(MEAN_ptr + pid)
    std = tl.load(STD_ptr + pid)
    z_score = tl.load(Z_ptr)  # scalar

    threshold = mean + std * z_score

    base = b * S + s
    base_idx = (base * D) + tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    x = tl.load(X_ptr + base_idx + offs, mask=mask, other=0.0).to(tl.float32)
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU

    out_ptrs = OUT_ptr + base_idx + offs
    tl.store(out_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of the original run function.
        All computations (reduction, mean/std, inv-ndtri, activation) are done by Triton kernels.
        Forward only performs tensor allocations and kernel launches; no torch elementwise/reduction ops.
        """
        assert inputs.dim() == 3, "inputs must be [batch_size, seq_len, intermediate_size]"
        B, S, D = inputs.shape
        inputs = inputs.contiguous()

        # Buffers (float32 for numerical stability)
        total_rows = B * S
        sum_buf = torch.empty(total_rows, dtype=torch.float32, device=inputs.device)
        sumsq_buf = torch.empty(total_rows, dtype=torch.float32, device=inputs.device)
        mean_buf = torch.empty(total_rows, dtype=torch.float32, device=inputs.device)
        std_buf = torch.empty(total_rows, dtype=torch.float32, device=inputs.device)

        # 1) Reduce: compute per-(b,s) sum and sumsq
        grid_reduce = (total_rows,)
        BLOCK_SIZE_RED = 256
        reduce_sum_sumsq_kernel[grid_reduce](inputs, sum_buf, sumsq_buf, B, S, D, BLOCK_SIZE=BLOCK_SIZE_RED, num_warps=4)

        # 2) Compute mean and std from sums
        grid_meanstd = (total_rows,)
        compute_mean_std_kernel[grid_meanstd](sum_buf, sumsq_buf, mean_buf, std_buf, D, num_warps=2)

        # 3) Compute z_score = ndtri(target_sparsity) in Triton (scalar)
        sparsity_buf = torch.tensor([float(target_sparsity)], dtype=torch.float32, device=inputs.device)
        z_score_buf = torch.empty(1, dtype=torch.float32, device=inputs.device)
        ndtri_approx_kernel[(1,)](sparsity_buf, z_score_buf, num_warps=2)

        # 4) Apply activation: y = max(0, x - (mean + std * z_score))
        out_fp32 = torch.empty((B, S, D), dtype=torch.float32, device=inputs.device)
        BLOCK_SIZE_ACT = 256
        grid_act = (B, S, triton.cdiv(D, BLOCK_SIZE_ACT))
        apply_activation_kernel[grid_act](inputs, mean_buf, std_buf, z_score_buf, out_fp32, B, S, D, BLOCK_SIZE=BLOCK_SIZE_ACT, num_warps=4)

        # 5) Cast to bfloat16 to match original behavior (original returns bfloat16)
        out_bf16 = out_fp32.to(torch.bfloat16)

        return out_bf16


def run(*args):
    return ModelNew()(*args)
