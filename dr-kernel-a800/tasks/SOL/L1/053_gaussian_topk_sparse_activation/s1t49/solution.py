import torch
import triton
import triton.language as tl


@triton.jit
def reduce_sum_sumsq_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    B, S, D          # int32 dimensions
):
    # One program per (b, s) row
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    base = (b * S + s) * D

    acc_sum = tl.zeros((), dtype=tl.float32)
    acc_sumsq = tl.zeros((), dtype=tl.float32)

    off = 0
    while off < D:
        idx = off + tl.arange(0, 1024)
        mask = idx < D
        x = tl.load(X_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        acc_sum += tl.sum(x, axis=0)
        acc_sumsq += tl.sum(x * x, axis=0)
        off += 1024

    tl.store(SUM_ptr + pid, acc_sum)
    tl.store(SUMSQ_ptr + pid, acc_sumsq)


@triton.jit
def compute_mean_std_kernel(
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    B, S, D          # int32 dimensions
):
    pid = tl.program_id(axis=0)
    # MEAN_ptr and STD_ptr are 1D of length B*S
    sum_val = tl.load(SUM_ptr + pid)
    sumsq_val = tl.load(SUMSQ_ptr + pid)

    mean = sum_val / D
    var_num = sumsq_val / D - mean * mean
    var_num = tl.maximum(var_num, 0.0)
    std = tl.sqrt(var_num)

    tl.store(MEAN_ptr + pid, mean)
    tl.store(STD_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(
    OUT_ptr,         # *float32, length 1
    P,               # float32 scalar target_sparsity
):
    # Abramowitz & Stegun 5.2.23 approximation (piecewise)
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

    p = P  # float32 scalar

    # Lower region
    q_low = tl.sqrt(-2.0 * tl.log(p))
    result_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
                 ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)

    # Central region
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    result_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
                 (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)

    # Upper region
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    result_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
                   ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)

    # Select piecewise result
    # Triton supports where semantics; compute final z
    z = tl.where(p < p_low, result_low, tl.where(p > p_high, result_high, result_mid))

    # Store z-score into OUT_ptr[0]
    tl.store(OUT_ptr, z)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    Z_ptr,           # *float32, length 1 (scalar z-score)
    OUT_ptr,         # *bfloat16, output [B, S, D]
    B, S, D          # int32 dimensions
):
    # 2D grid: axis 0 over (b, s), axis 1 over tiles of D
    pid0 = tl.program_id(axis=0)
    b = pid0 // S
    s = pid0 % S

    pid1 = tl.program_id(axis=1)
    off = pid1 * 1024
    idx = off + tl.arange(0, 1024)
    mask = idx < D

    mean = tl.load(MEAN_ptr + (b * S + s))
    std = tl.load(STD_ptr + (b * S + s))
    z_score = tl.load(Z_ptr)  # scalar float32
    threshold = mean + std * z_score  # float32

    base = (b * S + s) * D
    x = tl.load(X_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU

    tl.store(OUT_ptr + base + idx, y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, input_tensor: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Triton-only: ensure contiguous input and allocate buffers
        assert input_tensor.is_cuda, "Input must be on CUDA device for Triton kernels."
        input_tensor = input_tensor.contiguous()
        B, S, D = input_tensor.shape

        # Allocate buffers for sums, means, stds
        sum_buf = torch.empty(B * S, dtype=torch.float32, device=input_tensor.device)
        sumsq_buf = torch.empty(B * S, dtype=torch.float32, device=input_tensor.device)
        mean_buf = torch.empty(B * S, dtype=torch.float32, device=input_tensor.device)
        std_buf = torch.empty(B * S, dtype=torch.float32, device=input_tensor.device)

        # Output in bfloat16 to match original behavior
        out = torch.empty_like(input_tensor, dtype=torch.bfloat16)

        # Launch reduction kernel
        grid_reduce = (B * S,)
        reduce_sum_sumsq_kernel[grid_reduce](
            input_tensor, sum_buf, sumsq_buf, B, S, D,
            num_warps=8, num_stages=4
        )

        # Launch mean/std kernel
        grid_stats = (B * S,)
        compute_mean_std_kernel[grid_stats](
            sum_buf, sumsq_buf, mean_buf, std_buf, B, S, D,
            num_warps=1, num_stages=1
        )

        # Compute inverse-normal CDF for target sparsity on device
        z_buf = torch.empty(1, dtype=torch.float32, device=input_tensor.device)
        # Pass scalar as float32 (ModelNew receives target_sparsity as float)
        grid_ndtri = (1,)
        ndtri_approx_kernel[grid_ndtri](z_buf, float(target_sparsity), num_warps=1, num_stages=1)

        # Launch apply activation kernel
        grid_apply = (B * S, triton.cdiv(D, 1024))
        apply_activation_kernel[grid_apply](
            input_tensor, mean_buf, std_buf, z_buf, out, B, S, D,
            num_warps=8, num_stages=4
        )

        return out


def run(*args):
    return ModelNew()(*args)
