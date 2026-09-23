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
        x_ptr,              # *const float32
        sums_ptr,           # *float32, length (B*S)
        sums2_ptr,          # *float32, length (B*S)
        F,                  # int32: feature size (intermediate dimension length)
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per row (b, s)
        row_id = tl.program_id(axis=0)
        row_start = row_id * F

        local_sum = 0.0
        local_sumsq = 0.0

        for offs in range(0, F, BLOCK_SIZE):
            idx = offs + tl.arange(0, BLOCK_SIZE)
            mask = idx < F
            vals = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
            local_sum += tl.sum(vals, axis=0)
            local_sumsq += tl.sum(vals * vals, axis=0)

        tl.store(sums_ptr + row_id, local_sum)
        tl.store(sums2_ptr + row_id, local_sumsq)

    @triton.jit
    def compute_mean_std_rows_kernel(
        sums_ptr,           # *const float32, length (B*S)
        sums2_ptr,          # *const float32, length (B*S)
        mean_ptr,           # *float32, length (B*S)
        std_ptr,            # *float32, length (B*S)
        F,                  # int32
    ):
        row_id = tl.program_id(axis=0)
        s = tl.load(sums_ptr + row_id)
        ss = tl.load(sums2_ptr + row_id)
        mean = s / F
        var = ss / F - mean * mean
        var = tl.maximum(var, 0.0)
        std = tl.sqrt(var)
        tl.store(mean_ptr + row_id, mean)
        tl.store(std_ptr + row_id, std)

    @triton.jit
    def inv_normal_cdf_scalar_kernel(
        out_ptr,            # *float32, length 1
        p,                  # float32 scalar probability (0,1)
        BLOCK_SIZE: tl.constexpr,
    ):
        # Abramowitz & Stegun 7.1.26 approximation for inverse normal CDF
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

        if p < 0.5:
            t = tl.sqrt(-2.0 * tl.log(p))
            poly = (((((c1 * t + c2) * t + c3) * t + c4) * t + c5) * t + c6)
            poly2 = (((((d1 * t + d2) * t + d3) * t + d4) * t + 1.0))
            x = poly / poly2
        else:
            t = tl.sqrt(-2.0 * tl.log(1.0 - p))
            poly = -(((((c1 * t + c2) * t + c3) * t + c4) * t + c5) * t + c6)
            poly2 = (((((d1 * t + d2) * t + d3) * t + d4) * t + 1.0))
            x = poly / poly2

        tl.store(out_ptr + 0, x)

    @triton.jit
    def sparsify_relu_kernel(
        x_ptr,            # *const float32 input
        out_ptr,          # *float32 output
        mean_ptr,         # *const float32, shape [B*S]
        std_ptr,          # *const float32, shape [B*S]
        threshold_scale,  # scalar float32 multiplier for threshold
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

        # Ensure float32 for statistics; ensure contiguous
        x = inputs.contiguous().to(torch.float32)
        B, S, F = x.shape
        device = x.device

        num_rows = B * S

        # 1) Compute per-(batch, seq) sum and sum of squares using Triton
        sums = torch.empty(num_rows, dtype=torch.float32, device=device)
        sums2 = torch.empty(num_rows, dtype=torch.float32, device=device)

        BLOCK_SIZE_RED = 256


def run(*args):
    return ModelNew()(*args)
