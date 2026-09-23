import torch
import triton
import triton.language as tl


# Triton kernel: compute per-row mean and population std along the last dimension (N).
# X_ptr: [rows, N], contiguous.
# MEAN_ptr: [rows], fp32 output.
# STD_ptr: [rows], fp32 output.
@triton.jit
def row_stats_kernel(
    X_ptr,
    MEAN_ptr,
    STD_ptr,
    rows: tl.int32,
    N: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Compute base offset for this row
    base = row_id * N
    # Accumulate sum and sum of squares in fp32
    sum_val = 0.0
    sum_sq = 0.0
    # Iterate over columns in chunks
    for col in range(0, N, BLOCK):
        offs = col + tl.arange(0, BLOCK)
        mask = offs < N
        ptrs = X_ptr + base + offs
        x = tl.load(ptrs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    n_float = tl.full((), N, tl.float32)
    mean = sum_val / n_float
    # population std: sqrt(E[x^2] - (E[x])^2)
    var = sum_sq / n_float - mean * mean
    # Ensure non-negative due to numerical issues
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    # Store
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Triton kernel: compute inverse normal CDF (quantile) for a scalar probability p.
# Uses Abramowitz & Stegun 5.2.23 approximation.
# Inputs:
#   P_ptr: 1-element tensor (device), dtype fp32, value in (0,1).
# Outputs:
#   Y_ptr: 1-element tensor (device), dtype fp32, inverse CDF.
@triton.jit
def ndtri_kernel(P_ptr, Y_ptr):
    # Load p
    p = tl.load(P_ptr)
    # Constants for approximation
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

    # Regions
    mask_low = p < p_low
    mask_high = p > p_high
    mask_mid = ~mask_low & ~mask_high

    # Lower region
    # q = sqrt(-2 * log(p))
    q_low = tl.sqrt(-2.0 * tl.log(p))
    y_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
            ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    y_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
            (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    y_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
             ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Select region
    # Triton doesn't have built-in where for scalar, we use masks
    y = tl.zeros((), dtype=tl.float32)
    y = tl.where(mask_low, y_low, y)
    y = tl.where(mask_mid, y_mid, y)
    y = tl.where(mask_high, y_high, y)

    tl.store(Y_ptr, y)


# Triton kernel: compute per-row threshold = mean + std * multiplier.
# MEAN_ptr, STD_ptr: vectors of length rows (fp32).
# MULTIPLIER_ptr: 1-element tensor (fp32).
# THRESH_ptr: output vector of length rows (fp32).
@triton.jit
def threshold_vec_kernel(
    MEAN_ptr,
    STD_ptr,
    MULTIPLIER_ptr,
    THRESH_ptr,
    rows: tl.int32,
):
    row_id = tl.program_id(0)
    # Load scalars
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    multiplier = tl.load(MULTIPLIER_ptr)  # scalar
    thr = mean + std * multiplier
    tl.store(THRESH_ptr + row_id, thr)


# Triton kernel: apply ReLU(x - threshold[row]) elementwise for each row.
# X_ptr: input, fp32, shape [rows, N], contiguous.
# THRESH_ptr: per-row threshold, length rows, fp32.
# OUT_ptr: output, fp32, shape [rows, N], contiguous.
@triton.jit
def relu_threshold_kernel(
    X_ptr,
    THRESH_ptr,
    OUT_ptr,
    rows: tl.int32,
    N: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Load threshold for this row
    thr = tl.load(THRESH_ptr + row_id)
    base = row_id * N
    for col in range(0, N, BLOCK):
        offs = col + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X_ptr + base + offs, mask=mask, other=0.0)
        # ReLU(x - thr)
        y = x - thr
        y = tl.where(y > 0.0, y, 0.0)
        tl.store(OUT_ptr + base + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure CUDA and contiguous
        if not inputs.is_cuda:
            inputs = inputs.cuda()
        inputs = inputs.contiguous()

        # We compute in fp32 for numerical stability; original code converts to fp32 for stats.
        x_f32 = inputs if inputs.dtype == torch.float32 else inputs.to(torch.float32)

        # Flatten leading dims: rows = batch * seq, N = last dim
        B, S, N = x_f32.shape
        rows = B * S

        # 1) Compute row-wise mean and std
        mean = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        std = torch.empty(rows, device=x_f32.device, dtype=torch.float32)

        grid_stats = (rows,)
        row_stats_kernel[grid_stats](
            x_f32, mean, std,
            rows, N,
            BLOCK=256,
            num_warps=4,
        )

        # 2) Compute multiplier = ndtri(target_sparsity) using Triton scalar kernel
        p = torch.tensor(target_sparsity, device=x_f32.device, dtype=torch.float32)  # 1-element tensor
        multiplier = torch.empty(1, device=x_f32.device, dtype=torch.float32)
        ndtri_kernel[(1,)](p, multiplier)  # single program, scalar compute

        # 3) Compute threshold per row
        threshold = torch.empty(rows, device=x_f32.device, dtype=torch.float32)
        threshold_vec_kernel[(rows,)](mean, std, multiplier, threshold, rows)

        # 4) Apply ReLU(x - threshold[row]) elementwise
        out = torch.empty_like(x_f32, device=x_f32.device, dtype=torch.float32)
        grid_relu = (rows,)
        relu_threshold_kernel[grid_relu](
            x_f32,
            threshold,
            out,
            rows,
            N,
            BLOCK=256,
            num_warps=4,
        )

        # Match original behavior: return bf16
        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
