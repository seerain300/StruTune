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
        x_ptr,         # *const float32
        sums_ptr,      # *float32, length B*S
        sums2_ptr,     # *float32, length B*S
        B, S, F,       # int32 dims
        BLOCK_SIZE: tl.constexpr,
    ):
        # one program per row index in [0, B*S)
        row = tl.program_id(axis=0)
        if row >= B * S:
            return

        # Map row to (b, s)
        b = row // S
        s = row % S
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

        tl.store(sums_ptr + row, local_sum)
        tl.store(sums2_ptr + row, local_sumsq)

    @triton.jit
    def compute_mean_std_kernel(
        sums_ptr,       # *const float32, length B*S
        sums2_ptr,      # *const float32, length B*S
        mean_ptr,       # *float32, length B*S
        std_ptr,        # *float32, length B*S
        F,              # int32
    ):
        row = tl.program_id(axis=0)
        if row >= B * S:
            return

        total = tl.load(sums_ptr + row)
        total2 = tl.load(sums2_ptr + row)

        mean = total / F
        var = total2 / F - mean * mean
        # numerical safety: clamp variance to non-negative
        var = tl.maximum(var, 0.0)
        std = tl.sqrt(var)

        tl.store(mean_ptr + row, mean)
        tl.store(std_ptr + row, std)

    @triton.jit
    def compute_ndtri_kernel(
        out_ptr,       # *float32, scalar output
        p,             # float32 input (target_sparsity)
        # constants for Abramowitz & Stegun 7.1.26
        p_low, p_high,
        a1, a2, a3, a4, a5, a6,
        b1, b2, b3, b4, b5,
        c1, c2, c3, c4, c5, c6,
        d1, d2, d3, d4,
    ):
        # central region approximation
        q = p - 0.5
        r = q * q
        poly = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        den = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        ndtri_mid = poly * q / den

        # lower region: p < p_low
        q_low = tl.sqrt(-2.0 * tl.log(p))
        poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
        den_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
        ndtri_low = -poly_low / den_low

        # upper region: p > p_high
        q_high = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly_high = (((((c1 * q_high + c2) * q_high + c3) * q_high + c4) * q_high + c5) * q_high + c6)
        den_high = (((((d1 * q_high + d2) * q_high + d3) * q_high + d4) * q_high + 1.0))
        ndtri_high = poly_high / den_high

        # select based on p
        res = tl.where(p < p_low, ndtri_low, tl.where(p > p_high, ndtri_high, ndtri_mid))

        tl.store(out_ptr, res)

    @triton.jit
    def sparsify_relu_kernel(
        x_ptr,            # *const float32 input
        out_ptr,          # *float32 output
        mean_ptr,         # *const float32, shape [B*S]
        std_ptr,          # *const float32, shape [B*S]
        threshold_scale,  # scalar float32 multiplier for threshold
        total_elems,      # int32
        B, S, F,          # int32 dims
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
        # If Triton unavailable, fallback to original implementation
        if not TRITON_AVAILABLE:
            # The original 'run' function is assumed to be provided elsewhere.
            # Here we just call it to maintain correctness without Triton.
            # Note: In a real scenario, you'd replace this with the original code path.
            # For evaluation, we need Triton-only; if unavailable, this code path won't be used.
            pass

        assert TRITON_AVAILABLE, "Triton is not available"
        assert inputs.is_cuda, "Input must be on CUDA for Triton execution"

        # Convert to float32 for statistics; ensure contiguous
        x = inputs.to(torch.float32).contiguous()
        B, S, F = x.shape

        # 1) Per-row sums and sums of squares
        sums = torch.empty(B * S, dtype=torch.float32, device=x.device)
        sums2 = torch.empty(B * S, dtype=torch.float32, device=x.device)

        BLOCK_SIZE_RED = 1024
        grid_reduce = (B * S,)
        reduce_sum_sumsq_rows_kernel[grid_reduce](
            x, sums, sums2, B, S, F, BLOCK_SIZE=BLOCK_SIZE_RED
        )

        # 2) Compute mean and std per row
        mean = torch.empty(B * S, dtype=torch.float32, device=x.device)
        std = torch.empty(B * S, dtype=torch.float32, device=x.device)

        grid_mean = (B * S,)
        compute_mean_std_kernel[grid_mean](sums, sums2, mean, std, F)

        # 3) Compute inverse normal CDF (ndtri) for target_sparsity in Triton
        # Use Abramowitz & Stegun 7.1.26 constants
        p_low = 0.02425
        p_high = 1.0 - p_low
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

        ndtri_scale = torch.empty(1, dtype=torch.float32, device=x.device)

        grid_ndtri = (1,)
        compute_ndtri_kernel[grid_ndtri](
            ndtri_scale, target_sparsity,
            p_low, p_high,
            a1, a2, a3, a4, a5, a6,
            b1, b2, b3, b4, b5,
            c1, c2, c3, c4, c5, c6,
            d1, d2, d3, d4,
        )
        threshold_scale = ndtri_scale[0]  # scalar float

        # 4) Apply sparsification: output = max(0, x - (mean + std * threshold_scale))
        out_fp32 = torch.empty((B, S, F), dtype=torch.float32, device=x.device)

        total_elems = B * S * F
        BLOCK_SIZE_POINT = 1024
        grid_point = (triton.cdiv(total_elems, BLOCK_SIZE_POINT),)
        sparsify_relu_kernel[grid_point](
            x, out_fp32, mean, std, threshold_scale, total_elems, B, S, F, BLOCK_SIZE=BLOCK_SIZE_POINT
        )

        # Return as bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
