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
        x_ptr,            # *const float32, input flattened
        sums_ptr,         # *float32, length B*S
        sums2_ptr,        # *float32, length B*S
        B: tl.constexpr,  # batch size
        S: tl.constexpr,  # seq_len
        F,                # feature_size (intermediate dimension)
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per (b, s) row
        pid = tl.program_id(axis=0)
        b = pid // S
        s = pid % S
        row_start = (b * S + s) * F

        local_sum = 0.0
        local_sumsq = 0.0

        # Loop over feature dimension in chunks
        for offs in range(0, F, BLOCK_SIZE):
            idx = offs + tl.arange(0, BLOCK_SIZE)
            mask = idx < F
            vals = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
            local_sum += tl.sum(vals, axis=0)
            local_sumsq += tl.sum(vals * vals, axis=0)

        # Store per-row sums
        tl.store(sums_ptr + pid, local_sum)
        tl.store(sums2_ptr + pid, local_sumsq)

    @triton.jit
    def ndtri_approx_kernel(
        out_ptr,          # *float32, length 1
        p,                # scalar float32, target_sparsity
        # constants for Abramowitz & Stegun 7.1.26 approximation
        a1, a2, a3, a4, a5, a6,
        b1, b2, b3, b4, b5,
        c1, c2, c3, c4, c5, c6,
        d1, d2, d3, d4,
        p_low, p_high,
    ):
        # Lower region: p <= p_low
        q = tl.sqrt(-2.0 * tl.log(p))
        t = c1 * q + c2
        t = t * q + c3
        t = t * q + c4
        t = t * q + c5
        t = t * q + c6
        d = d1 * q + d2
        d = d * q + d3
        d = d * q + d4
        approx_low = t / (d + 1.0)

        # Upper region: p >= p_high (log(1-p))
        q2 = tl.sqrt(-2.0 * tl.log(1.0 - p))
        t2 = c1 * q2 + c2
        t2 = t2 * q2 + c3
        t2 = t2 * q2 + c4
        t2 = t2 * q2 + c5
        t2 = t2 * q2 + c6
        d2 = d1 * q2 + d2
        d2 = d2 * q2 + d3
        d2 = d2 * q2 + d4
        approx_high = -t2 / (d2 + 1.0)

        # Central region: p_low < p < p_high
        q3 = p - 0.5
        r = q3 * q3
        t3 = a1 * r + a2
        t3 = t3 * r + a3
        t3 = t3 * r + a4
        t3 = t3 * r + a5
        t3 = t3 * r + a6
        d3 = b1 * r + b2
        d3 = d3 * r + b3
        d3 = d3 * r + b4
        d3 = d3 * r + b5
        central = t3 * q3 / (d3 + 1.0)

        cond_low = p <= p_low
        cond_high = p >= p_high
        final = tl.where(cond_low, approx_low, central)
        final = tl.where(cond_high, approx_high, final)

        tl.store(out_ptr, final)

    @triton.jit
    def sparsify_relu_kernel(
        x_ptr,            # *const float32, input
        out_ptr,          # *float32, output
        mean_ptr,         # *const float32, shape [B*S]
        std_ptr,          # *const float32, shape [B*S]
        threshold_scale,  # scalar float32, multiplier for ndtri
        B: tl.constexpr,
        S: tl.constexpr,
        F: tl.constexpr,
        total_elems,      # int32
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
        assert TRITON_AVAILABLE, "Triton is not available"
        assert inputs.is_cuda, "Input must be on CUDA for Triton execution"

        # Ensure float32 and contiguous
        x = inputs.contiguous().to(torch.float32)
        B, S, F = x.shape
        device = x.device

        num_rows = B * S
        total_elems = B * S * F

        # 1) Compute per-(batch, seq) sum and sum of squares using Triton
        sums = torch.empty(num_rows, dtype=torch.float32, device=device)
        sums2 = torch.empty(num_rows, dtype=torch.float32, device=device)

        BLOCK_SIZE_RED = 1024
        reduce_sum_sumsq_rows_kernel[(num_rows,)](
            x, sums, sums2, B, S, F, BLOCK_SIZE=BLOCK_SIZE_RED
        )

        # 2) Compute mean and std per row on device (host-side ops, minimal)
        mean = sums / float(F)
        var = sums2 / float(F) - mean * mean
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var)

        # Keep as 1D for kernel simplicity
        mean_1d = mean
        std_1d = std

        # 3) Compute inverse normal CDF (ndtri) for scalar target_sparsity in Triton
        # Prepare constants for Abramowitz & Stegun 7.1.26
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

        # Allocate a 1-element tensor for the scalar ndtri result
        threshold_scale = torch.empty(1, dtype=torch.float32, device=device)

        ndtri_approx_kernel[(1,)](
            threshold_scale,
            float(target_sparsity),
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
            p_low, p_high,
        )

        # 4) Apply sparsification: output = max(0, x - (mean + std * ndtri))
        out_fp32 = torch.empty((B, S, F), dtype=torch.float32, device=device)

        BLOCK_SIZE_POINT = 1024
        grid = (triton.cdiv(total_elems, BLOCK_SIZE_POINT),)
        sparsify_relu_kernel[grid](
            x, out_fp32, mean_1d, std_1d, threshold_scale[0], B, S, F, total_elems, BLOCK_SIZE=BLOCK_SIZE_POINT
        )

        # Return as bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
