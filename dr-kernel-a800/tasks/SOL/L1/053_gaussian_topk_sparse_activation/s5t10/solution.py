import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels
if TRITON_AVAILABLE:
    @triton.jit
    def reduce_sum_sumsq_rows_kernel(
        x_ptr,            # *const float32
        sums_ptr,         # *float32, length B*S
        sums2_ptr,        # *float32, length B*S
        B: tl.constexpr,  # batch size (shape hint)
        S: tl.constexpr,  # seq_len (shape hint)
        F,                # feature_size (runtime int)
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
            ptrs = x_ptr + row_start + idx
            vals = tl.load(ptrs, mask=mask, other=0.0)
            local_sum += tl.sum(vals, axis=0)
            local_sumsq += tl.sum(vals * vals, axis=0)

        tl.store(sums_ptr + pid, local_sum)
        tl.store(sums2_ptr + pid, local_sumsq)

    @triton.jit
    def compute_mean_std_kernel(
        sums_ptr,          # *float32, length B*S
        sums2_ptr,         # *float32, length B*S
        mean_ptr,          # *float32, length B*S
        std_ptr,           # *float32, length B*S
        F,                 # feature_size (runtime int)
    ):
        pid = tl.program_id(axis=0)
        total = tl.load(sums_ptr + pid)
        sumsq = tl.load(sums2_ptr + pid)
        f = F
        mean = total / f
        var = sumsq / f - mean * mean
        var = tl.maximum(var, 0.0)
        std = tl.sqrt(var)
        tl.store(mean_ptr + pid, mean)
        tl.store(std_ptr + pid, std)

    @triton.jit
    def ndtri_approx_kernel(
        out_ptr,           # *float32, length 1
        p,                 # scalar float32 in (0, 1)
        a1, a2, a3, a4, a5, a6,   # constants for central/lower regions
        b1, b2, b3, b4, b5,       # constants for central region
        c1, c2, c3, c4, c5, c6,   # constants for lower/upper regions
        d1, d2, d3, d4,           # constants for lower/upper regions
        p_low, p_high,            # region bounds
    ):
        # Compute inverse normal CDF using A&S 7.1.26
        # lower region: p < p_low
        mask_low = p < p_low
        q_low = tl.sqrt(-2.0 * tl.log(p))
        y_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6) / \
                ((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0)
        # central region: p_low <= p <= p_high
        mask_mid = (p >= p_low) & (p <= p_high)
        q_mid = p - 0.5
        r_mid = q_mid * q_mid
        poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
        denom_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
        y_mid = poly_mid * q_mid / denom_mid
        # upper region: p > p_high
        mask_high = p > p_high
        q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
        y_high = -(((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6) / \
                 ((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0)
        # select
        y = y_low * mask_low + y_mid * mask_mid + y_high * mask_high
        tl.store(out_ptr, y)

    @triton.jit
    def sparsify_kernel(
        x_ptr,             # *const float32 input
        out_ptr,           # *float32 output
        mean_ptr,          # *const float32, shape [B*S]
        std_ptr,           # *const float32, shape [B*S]
        threshold_scale,   # scalar float32
        B: tl.constexpr,
        S: tl.constexpr,
        F: tl.constexpr,
        total_elems,       # int32
        BLOCK_SIZE: tl.constexpr,
    ):
        # 1D grid over total elements
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
        y = tl.maximum(y, 0.0)  # ReLU

        out_ptrs = out_ptr + b * SF + s * F + f
        tl.store(out_ptrs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Triton availability and device
        assert TRITON_AVAILABLE, "Triton is not available"
        assert inputs.is_cuda, "Input must be on CUDA for Triton execution"

        # Convert to float32 for statistics; ensure contiguous
        x = inputs.to(torch.float32).contiguous()
        B, S, F = x.shape
        device = x.device

        # 1) Compute per-(batch, seq) sum and sum of squares using Triton
        sums = torch.empty(B * S, dtype=torch.float32, device=device)
        sums2 = torch.empty(B * S, dtype=torch.float32, device=device)

        grid_stats = (B * S,)
        BLOCK_SIZE_STATS = 1024
        reduce_sum_sumsq_rows_kernel[grid_stats](
            x, sums, sums2, B, S, F, BLOCK_SIZE=BLOCK_SIZE_STATS
        )

        # 2) Compute mean and std per (batch, seq) using Triton
        mean = torch.empty(B * S, dtype=torch.float32, device=device)
        std = torch.empty(B * S, dtype=torch.float32, device=device)

        grid_mean_std = (B * S,)
        compute_mean_std_kernel[grid_mean_std](
            sums, sums2, mean, std, F
        )

        # 3) Compute inverse normal CDF (ndtri) for target_sparsity in Triton
        # A&S 7.1.26 constants
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


def run(*args):
    return ModelNew()(*args)
