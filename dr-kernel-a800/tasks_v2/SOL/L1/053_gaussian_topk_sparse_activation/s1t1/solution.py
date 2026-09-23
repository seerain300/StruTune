import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    B, S, D,         # int32 dimensions
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # 0..(B*S - 1)
    b = pid // S
    s = pid % S

    # Compute base row offset as int*vector to allow pointer arithmetic with idx
    base = b * S + s
    base_idx = (base * D) + tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    sum_val = 0.0
    sumsq_val = 0.0

    offs = 0
    while offs < D:
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        ptrs = X_ptr + base_idx + idx
        x = tl.load(ptrs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        # Reduce vector to scalar for this chunk
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)
        offs += BLOCK_SIZE

    # Write results; SUM_ptr/ SUMSQ_ptr are float32 buffers of length B*S
    tl.store(SUM_ptr + pid, sum_val)
    tl.store(SUMSQ_ptr + pid, sumsq_val)


@triton.jit
def compute_mean_std_kernel(
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    D,               # int32 number of features
):
    pid = tl.program_id(axis=0)
    sum_val = tl.load(SUM_ptr + pid)
    sumsq_val = tl.load(SUMSQ_ptr + pid)
    d = tl.full((), D, tl.int32)
    mean = sum_val / d
    var = sumsq_val / d - mean * mean
    var = tl.maximum(var, 0.0)  # clamp tiny negative due to rounding
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + pid, mean)
    tl.store(STD_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(
    P_ptr,           # *float32, length 1 (single scalar sparsity)
    OUT_ptr,         # *float32, length 1 (output z-score)
):
    # Load sparsity p
    p = tl.load(P_ptr)
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

    # Lower region: q = sqrt(-2*log(p)) and rational function
    q_low = tl.sqrt(-2.0 * tl.log(p))
    poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
    denom_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
    nd_low = poly_low / denom_low

    # Central region: q = p - 0.5, rational function with q^2
    q_mid = p - 0.5
    r_mid = q_mid * q_mid
    poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
    denom_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
    nd_mid = poly_mid * q_mid / denom_mid

    # Upper region: q = sqrt(-2*log(1-p))
    q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
    denom_high = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
    nd_high = -poly_high / denom_high

    # Select region
    low_mask = p < p_low
    high_mask = p > p_high
    ndtri_approx = tl.where(low_mask, nd_low, tl.where(high_mask, nd_high, nd_mid))

    # Store result
    tl.store(OUT_ptr, ndtri_approx)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    OUT_ptr,         # *output tensor [B, S, D], contiguous (bfloat16)
    B, S, D,         # int32
    THRESH_SCALE,    # float32 scalar: precomputed z-score from ndtri
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(axis=0)   # batch index
    s = tl.program_id(axis=1)   # seq index
    tile = tl.program_id(axis=2)  # tile index across D

    base = b * S + s
    base_idx = (base * D) + tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    # Load per-row mean and std
    mean = tl.load(MEAN_ptr + (b * S + s))
    std = tl.load(STD_ptr + (b * S + s))

    # Compute threshold
    threshold = mean + std * THRESH_SCALE

    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    x_ptrs = X_ptr + base_idx + offs
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU

    out_ptrs = OUT_ptr + base_idx + offs
    tl.store(out_ptrs, y, mask=mask)  # store float32; OUT tensor is bfloat16 (Triton handles casting on store)


def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of the original run function.
        - All reductions, std/mean computation, and activation are done by Triton kernels.
        - No torch elementwise ops or reductions in forward.
        """
        assert inputs.dim() == 3, "inputs must be [batch_size, seq_len, intermediate_size]"
        B, S, D = inputs.shape
        inputs = inputs.contiguous()

        total_rows = B * S
        sum_buf = torch.empty(total_rows, dtype=torch.float32, device=inputs.device)
        sumsq_buf = torch.empty(total_rows, dtype=torch.float32, device=inputs.device)
        mean_buf = torch.empty(total_rows, dtype=torch.float32, device=inputs.device)
        std_buf = torch.empty(total_rows, dtype=torch.float32, device=inputs.device)

        block_size = min(1024, _next_power_of_two(D))
        grid = (total_rows,)

        # Reduction kernel: compute per-row sum and sumsq
        reduce_mean_std_kernel[grid](inputs, sum_buf, sumsq_buf, B, S, D, BLOCK_SIZE=block_size)

        # Compute mean and std from sums
        compute_mean_std_kernel[grid](sum_buf, sumsq_buf, mean_buf, std_buf, D)

        # Compute z-score via Triton kernel (inverse normal CDF using A&S 5.2.23)
        sparsity_buf = torch.tensor([float(target_sparsity)], dtype=torch.float32, device=inputs.device)
        z_score_buf = torch.empty(1, dtype=torch.float32, device=inputs.device)
        ndtri_approx_kernel[(1,)](sparsity_buf, z_score_buf)
        z_score = float(z_score_buf.item())  # host-side scalar for activation

        # Allocate output in bfloat16 (same as typical input dtype)
        out = torch.empty(B, S, D, dtype=torch.bfloat16, device=inputs.device)

        # Launch apply activation kernel
        apply_activation_kernel[(B, S, triton.cdiv(D, block_size))](
            inputs, mean_buf, std_buf, out, B, S, D, float(z_score), BLOCK_SIZE=block_size
        )

        return out


def run(*args):
    return ModelNew()(*args)
