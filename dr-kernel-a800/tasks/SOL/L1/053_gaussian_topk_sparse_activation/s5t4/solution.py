import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# -----------------------------
# Triton kernels
# -----------------------------
if TRITON_AVAILABLE:
    @triton.jit
    def reduce_sum_sumsq_rows_kernel(
        x_ptr,            # *const float32
        sums_ptr,         # *float32, length B*S
        sums2_ptr,        # *float32, length B*S
        B,                # int32
        S,                # int32
        F,                # int32
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per (b, s) row
        pid = tl.program_id(axis=0)
        b = pid // S
        s = pid % S
        row_start = (b * S + s) * F

        local_sum = 0.0
        local_sumsq = 0.0

        for offs in range(0, F, BLOCK_SIZE):
            idx = offs + tl.arange(0, BLOCK_SIZE)
            mask = idx < F
            vals = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
            local_sum += tl.sum(vals, axis=0)
            local_sumsq += tl.sum(vals * vals, axis=0)

        tl.store(sums_ptr + pid, local_sum)
        tl.store(sums2_ptr + pid, local_sumsq)

    @triton.jit
    def compute_mean_std_kernel(
        sums_ptr,               # *float32, length B*S
        sums2_ptr,              # *float32, length B*S
        mean_ptr,               # *float32, length B*S
        std_ptr,                # *float32, length B*S
        F: tl.constexpr,        # feature size
    ):
        pid = tl.program_id(axis=0)
        total = tl.load(sums_ptr + pid)
        sumsq = tl.load(sums2_ptr + pid)
        mean = total / F
        var = sumsq / F - mean * mean
        var = tl.maximum(var, 0.0)  # avoid tiny negative due to fp rounding
        std = tl.sqrt(var)
        tl.store(mean_ptr + pid, mean)
        tl.store(std_ptr + pid, std)

    @triton.jit
    def ndtri_approx_kernel(
        sp_ptr,                 # *float32, length 1 (scalar target_sparsity)
        out_ptr,                # *float32, length 1 (scalar output)
        a1, a2, a3, a4, a5, a6,  # constants for central region
        b1, b2, b3, b4, b5,      # constants for central region
        c1, c2, c3, c4, c5, c6,  # constants for lower and upper regions
        d1, d2, d3, d4,          # constants for lower and upper regions
        p_low, p_high,           # thresholds
        BLOCK_SIZE: tl.constexpr,
    ):
        # Single-program scalar compute for p
        p = tl.load(sp_ptr)
        # Lower region
        cond_low = p < p_low
        q_low = tl.sqrt(-2.0 * tl.log(p))
        lower = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
                ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
        lower = tl.where(cond_low, lower, 0.0)

        # Central region
        cond_mid = (p >= p_low) & (p <= p_high)
        q_mid = p - 0.5
        r_mid = q_mid * q_mid
        poly_a = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
        poly_b = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
        phi_mid = poly_a * q_mid / poly_b
        phi_mid = tl.where(cond_mid, phi_mid, 0.0)

        # Upper region
        cond_high = p > p_high
        q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
        psi_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
                   ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)
        psi_high = tl.where(cond_high, psi_high, 0.0)

        # Combine
        result = lower + phi_mid + psi_high
        tl.store(out_ptr, result)

    @triton.jit
    def sparsify_relu_kernel(
        x_ptr,            # *const float32 input
        out_ptr,          # *float32 output
        mean_ptr,         # *const float32, shape [B*S]
        std_ptr,          # *const float32, shape [B*S]
        threshold_scale,  # scalar float32 multiplier for threshold
        B,                # int32
        S,                # int32
        F,                # int32
        total_elems,      # int32
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < total_elems

        SF = S * F
        b = offs // SF
        rem = offs % SF
        s = rem // F
        f = rem % F

        x_ptrs = x_ptr + b * SF + s * F + f
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        ms = mean_ptr + b * S + s
        ss = std_ptr + b * S + s
        mean = tl.load(ms, mask=mask, other=0.0)
        std = tl.load(ss, mask=mask, other=0.0)

        threshold = mean + std * threshold_scale
        y = x_vals - threshold
        y = tl.maximum(y, 0.0)

        out_ptrs = out_ptr + b * SF + s * F + f
        tl.store(out_ptrs, y, mask=mask)


# -----------------------------
# ModelNew: Triton-only forward
# -----------------------------
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Triton-only execution
        assert TRITON_AVAILABLE, "Triton is not available"
        assert inputs.is_cuda, "Input must be on CUDA for Triton execution"

        x = inputs.contiguous().to(torch.float32)
        B, S, F = x.shape
        device = x.device

        # 1) Reduce per (b, s): sum and sum of squares
        total_rows = B * S
        sums = torch.empty(total_rows, dtype=torch.float32, device=device)
        sums2 = torch.empty(total_rows, dtype=torch.float32, device=device)

        BLOCK_SIZE_REDUCE = 1024
        grid_reduce = (total_rows,)
        reduce_sum_sumsq_rows_kernel[grid_reduce](
            x, sums, sums2, B, S, F, BLOCK_SIZE=BLOCK_SIZE_REDUCE
        )

        # 2) Compute mean and std per (b, s) in Triton
        mean = torch.empty(total_rows, dtype=torch.float32, device=device)
        std = torch.empty(total_rows, dtype=torch.float32, device=device)

        grid_mean = (total_rows,)
        compute_mean_std_kernel[grid_mean](
            sums, sums2, mean, std, F=F
        )

        # 3) Compute inverse normal CDF for target_sparsity in Triton (scalar)
        sparsity_tensor = torch.tensor([float(target_sparsity)], dtype=torch.float32, device=device)
        ndtri_output = torch.empty(1, dtype=torch.float32, device=device)

        # Abramowitz & Stegun constants (7.1.26)
        a1 = -3.969683028665376e+01; a2 = 2.209460984245205e+02; a3 = -2.759285104469687e+02
        a4 = 1.383577518672690e+02; a5 = -3.066479806614716e+01; a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01; b2 = 1.615858368580409e+02; b3 = -1.556989798598866e+02
        b4 = 6.680131188771972e+01; b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03; c2 = -3.223964580411365e-01; c3 = -2.400758277161838e+00
        c4 = -2.549732539343734e+00; c5 = 4.374664141464968e+00; c6 = 2.938163982698783e+00
        d1 = 7.784695709041462e-03; d2 = 3.224671290700398e-01; d3 = 2.445134137142996e+00
        d4 = 3.754408661907416e+00
        p_low = 0.02425; p_high = 1.0 - p_low

        grid_ndtri = (1,)
        ndtri_approx_kernel[grid_ndtri](
            sparsity_tensor, ndtri_output,
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low, p_high,
            BLOCK_SIZE=1
        )

        threshold_scale = ndtri_output[0]  # scalar

        # 4) Apply sparsification: y = max(0, x - (mean + std * threshold_scale))
        out_fp32 = torch.empty((B, S, F), dtype=torch.float32, device=device)

        total_elems = B * S * F
        BLOCK_SIZE_POINT = 4096
        grid_sparsify = (triton.cdiv(total_elems, BLOCK_SIZE_POINT),)
        sparsify_relu_kernel[grid_sparsify](
            x, out_fp32, mean, std, threshold_scale,
            B, S, F, total_elems, BLOCK_SIZE=BLOCK_SIZE_POINT
        )

        # Return as bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
