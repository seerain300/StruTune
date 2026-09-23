import torch
import triton
import triton.language as tl


# Kernel 1: per-feature reduction across all rows: accumulate sum and sum of squares
@triton.jit
def sum_sumsq_per_feature(
    x_ptr,            # *const float32, input flattened as [rows, L] via row * L + f
    sum_ptr,          # *float32, length 1 (scalar per feature)
    sumsq_ptr,        # *float32, length 1 (scalar per feature)
    L: tl.int32,      # number of features
    rows: tl.int32,   # total rows = B * S
    BLOCK_ROWS: tl.constexpr
):
    f = tl.program_id(0)  # feature index
    acc = tl.zeros((), dtype=tl.float32)
    acc2 = tl.zeros((), dtype=tl.float32)
    row = 0
    while row < rows:
        for k in range(BLOCK_ROWS):
            r = row + k
            if r < rows:
                val = tl.load(x_ptr + r * L + f)
                acc += val
                acc2 += val * val
        row += BLOCK_ROWS
    tl.store(sum_ptr, acc)
    tl.store(sumsq_ptr, acc2)


# Kernel 2: compute mean and std per feature from sum and sum of squares
@triton.jit
def compute_mean_std_per_feature(
    sum_ptr,            # *const float32, length 1
    sumsq_ptr,          # *const float32, length 1
    mean_ptr,           # *float32, length 1
    std_ptr,            # *float32, length 1
    L: tl.int32,        # number of features
    rows: tl.int32      # total rows = B * S
):
    s = tl.load(sum_ptr)
    ss = tl.load(sumsq_ptr)
    mean = s / (rows * 1.0)
    var = ss / (rows * 1.0) - mean * mean
    std = tl.sqrt(var)  # Triton has sqrt
    tl.store(mean_ptr, mean)
    tl.store(std_ptr, std)


# Kernel 3: compute inverse standard normal CDF (Abramowitz & Stegun 7.1.26) for a 0-d tensor input p
@triton.jit
def ndtri_kernel(
    p_ptr,        # *const float32, 1-element tensor containing p (target_sparsity)
    out_ptr       # *float32, 1-element tensor to store ndtri(p)
):
    p = tl.load(p_ptr)
    # Constants for the approximation
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
    mask_low = p < p_low
    q = tl.sqrt(-2.0 * tl.log(p))
    low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
          ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Central region
    mask_mid = (p >= p_low) & (p <= p_high)
    q = p - 0.5
    r = q * q
    mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
          (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Upper region
    mask_high = p > p_high
    q = tl.sqrt(-2.0 * tl.log(1.0 - p))
    up = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
         ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    # Select based on mask (tl.where on scalar masks)
    res = tl.where(mask_low, low, mid)
    res = tl.where(mask_high, up, res)

    tl.store(out_ptr, res)


# Kernel 4: compute per-feature threshold: mean + std * std_multiplier
@triton.jit
def compute_threshold_per_feature(
    mean_ptr,          # *const float32, length 1
    std_ptr,           # *const float32, length 1
    std_multiplier_ptr,# *const float32, 1-element tensor (ndtri(target_sparsity))
    threshold_ptr      # *float32, length 1
):
    m = tl.load(mean_ptr)
    sd = tl.load(std_ptr)
    sm = tl.load(std_multiplier_ptr)
    thr = m + sd * sm
    tl.store(threshold_ptr, thr)


# Kernel 5: apply sparse ReLU per element: y = max(x - threshold, 0) for all rows
@triton.jit
def sparse_relu_per_element(
    x_ptr,             # *const float32, flattened input [rows, L]
    threshold_ptr,     # *const float32, scalar per feature (broadcast)
    y_ptr,             # *float32, flattened output [rows, L]
    L: tl.int32,       # number of features
    rows: tl.int32     # total rows = B * S
):
    row = tl.program_id(0)
    f = tl.program_id(1)
    # One program per (row, feature)
    if row < rows and f < L:
        x_val = tl.load(x_ptr + row * L + f)
        thr = tl.load(threshold_ptr)  # scalar threshold
        y_val = x_val - thr
        y_val = tl.where(y_val > 0.0, y_val, 0.0)
        tl.store(y_ptr + row * L + f, y_val)


# Kernel 6: cast FP32 output to BF16 (forward MUST invoke this kernel)
@triton.jit
def cast_fp32_to_bf16(
    inp_ptr,           # *const float32, flattened input [rows*L]
    out_ptr,           # *bfloat16, flattened output [rows*L]
    N: tl.int32        # total number of elements (rows * L)
):
    idx = tl.program_id(0)
    if idx < N:
        val = tl.load(inp_ptr + idx)
        # Triton will cast to bfloat16 when storing to out_ptr (BF16 tensor)
        tl.store(out_ptr + idx, val.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float, block_rows: int = 2048):
        super().__init__()
        self.target_sparsity = float(target_sparsity)
        self.block_rows = int(block_rows)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Ensure FP32 contiguous input for Triton kernels
        x = inputs.contiguous().to(torch.float32)
        device = x.device
        dtype_in = x.dtype  # fp32
        B, S, L = x.shape
        rows = B * S
        total = rows * L

        # Flatten input logically (we will pass pointers and use row*L + f addressing)
        # Allocate per-feature accumulators (scalars in FP32)
        sum_f = torch.zeros(1, dtype=torch.float32, device=device)
        sumsq_f = torch.zeros(1, dtype=torch.float32, device=device)

        # 1) Per-feature reduction: sum and sumsq over all rows
        # Launch one program per feature
        grid_red = (L,)
        sum_sumsq_per_feature[grid_red](
            x.view(-1), sum_f, sumsq_f, L, rows, BLOCK_ROWS=self.block_rows
        )

        # 2) Compute mean and std per feature
        mean_f = torch.empty(1, dtype=torch.float32, device=device)
        std_f = torch.empty(1, dtype=torch.float32, device=device)
        compute_mean_std_per_feature[(1,)](sum_f, sumsq_f, mean_f, std_f, L, rows)

        # 3) Compute std_multiplier = ndtri(target_sparsity) using Triton (no torch ops)
        std_multiplier = torch.empty(1, dtype=torch.float32, device=device)
        # Pass scalar p as a 1-element tensor (device-side, no torch ops in forward)
        p_scalar = torch.tensor(self.target_sparsity, dtype=torch.float32, device=device)
        ndtri_kernel[(1,)](p_scalar, std_multiplier)

        # 4) Compute threshold per feature: mean + std * std_multiplier
        threshold = torch.empty(1, dtype=torch.float32, device=device)
        compute_threshold_per_feature[(1,)](mean_f, std_f, std_multiplier, threshold)

        # 5) Apply sparse ReLU per element: y = max(x - threshold, 0)
        # We can do this either per element with broadcasting or by subtracting the same threshold.
        # To avoid another torch op, we compute x - thr and apply ReLU inside Triton:
        # Allocate FP32 output buffer
        y_fp32 = torch.empty(total, dtype=torch.float32, device=device)
        grid_act = (rows, L)
        sparse_relu_per_element[grid_act](
            x.view(-1), threshold, y_fp32, L, rows
        )

        # 6) Cast FP32 output to BF16 via Triton (forward MUST invoke this kernel)
        y_bf16 = torch.empty(total, dtype=torch.bfloat16, device=device)
        grid_cast = (triton.cdiv(total, 1024),)
        cast_fp32_to_bf16[grid_cast](y_fp32, y_bf16, total, BLOCK=1024)

        # Reshape back to [B, S, L]
        y = y_bf16.view(B, S, L)
        return y


def run(*args):
    return ModelNew()(*args)
