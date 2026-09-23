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
        x_ptr,                 # *const float32, flattened input
        sums_ptr,              # *float32, length B*S
        sums2_ptr,             # *float32, length B*S
        B: tl.constexpr,       # batch size
        S: tl.constexpr,       # seq_len
        F: tl.constexpr,       # feature_size
    ):
        # One program per row: pid in [0, B*S)
        pid = tl.program_id(axis=0)
        row_start = pid * F

        local_sum = 0.0
        local_sumsq = 0.0

        # Accumulate sum and sum of squares across F elements
        for i in range(0, F):
            val = tl.load(x_ptr + row_start + i)
            local_sum += val
            local_sumsq += val * val

        tl.store(sums_ptr + pid, local_sum)
        tl.store(sums2_ptr + pid, local_sumsq)

    @triton.jit
    def compute_mean_std_kernel(
        sums_ptr,              # *float32, length B*S
        sums2_ptr,             # *float32, length B*S
        mean_ptr,              # *float32, length B*S
        std_ptr,               # *float32, length B*S
        F: tl.constexpr,       # feature size
    ):
        pid = tl.program_id(axis=0)
        s = tl.load(sums_ptr + pid)
        s2 = tl.load(sums2_ptr + pid)
        mean = s / float(F)
        var = s2 / float(F) - mean * mean
        var = tl.maximum(var, 0.0)  # clamp to avoid tiny negatives
        std = tl.sqrt(var)
        tl.store(mean_ptr + pid, mean)
        tl.store(std_ptr + pid, std)

    @triton.jit
    def ndtri_scalar_kernel(
        p,                     # scalar float32, target sparsity in (0, 1)
        out_ptr,               # *float32, single element
        P_LOW: tl.constexpr = 0.02425,
    ):
        # Abramowitz and Stegun 7.1.26 approximation for ndtri(p)
        # Lower region
        mask_low = p <= P_LOW
        q_low = tl.sqrt(-2.0 * tl.log(p))

        a1 = tl.full((), -3.969683028665376e+01, tl.float32)
        a2 = tl.full((),  2.209460984245205e+02, tl.float32)
        a3 = tl.full((), -2.759285104469687e+02, tl.float32)
        a4 = tl.full((),  1.383577518672690e+02, tl.float32)
        a5 = tl.full((), -3.066479806614716e+01, tl.float32)
        a6 = tl.full((),  2.506628277459239e+00, tl.float32)

        b1 = tl.full((), -5.447609879822406e+01, tl.float32)
        b2 = tl.full((),  1.615858368580409e+02, tl.float32)
        b3 = tl.full((), -1.556989798598866e+02, tl.float32)
        b4 = tl.full((),  6.680131188771972e+01, tl.float32)
        b5 = tl.full((), -1.328068155288572e+01, tl.float32)

        c1 = tl.full((), -7.784894002430293e-03, tl.float32)
        c2 = tl.full((), -3.223964580411365e-01, tl.float32)
        c3 = tl.full((), -2.400758277161838e+00, tl.float32)
        c4 = tl.full((), -2.549732539343734e+00, tl.float32)
        c5 = tl.full((),  4.374664141464968e+00, tl.float32)
        c6 = tl.full((),  2.938163982698783e+00, tl.float32)

        d1 = tl.full((),  7.784695709041462e-03, tl.float32)
        d2 = tl.full((),  3.224671290700398e-01, tl.float32)
        d3 = tl.full((),  2.445134137142996e+00, tl.float32)
        d4 = tl.full((),  3.754408661907416e+00, tl.float32)

        # Lower region polynomial
        poly_low = (((((c1 * q_low + c2) * q_low + c3) * q_low + c4) * q_low + c5) * q_low + c6)
        denom_low = (((((d1 * q_low + d2) * q_low + d3) * q_low + d4) * q_low + 1.0))
        res_low = poly_low / denom_low

        # Central region
        mask_mid = p > P_LOW & (p < (1.0 - P_LOW))
        q_mid = p - 0.5
        r_mid = q_mid * q_mid
        poly_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6)
        denom_mid = (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)
        res_mid = poly_mid / denom_mid

        # Upper region
        mask_up = p >= (1.0 - P_LOW)
        q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
        poly_up = (((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6)
        denom_up = (((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0))
        res_up = -poly_up / denom_up  # negative sign per formula

        res = tl.where(mask_low, res_low, 0.0) + tl.where(mask_mid, res_mid, 0.0) + tl.where(mask_up, res_up, 0.0)
        tl.store(out_ptr, res)

    @triton.jit
    def sparsify_relu_kernel(
        x_ptr,                 # *const float32 input
        out_ptr,               # *float32 output
        mean_ptr,              # *const float32, length B*S
        std_ptr,               # *const float32, length B*S
        threshold_scale,       # scalar float32 multiplier for threshold
        B: tl.constexpr,
        S: tl.constexpr,
        F: tl.constexpr,
        total_elems,           # int32
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

        row_id = b * S + s  # each element maps to its (b, s) via linear index

        x_ptrs = x_ptr + b * SF + s * F + f
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        mean = tl.load(mean_ptr + row_id, mask=mask, other=0.0)
        std = tl.load(std_ptr + row_id, mask=mask, other=0.0)

        threshold = mean + std * threshold_scale
        y = x_vals - threshold
        y = tl.maximum(y, 0.0)  # ReLU

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
        # Triton-only forward, no torch reductions or elementwise ops
        assert TRITON_AVAILABLE, "Triton is not available"
        assert inputs.is_cuda, "Input must be on CUDA for Triton execution"

        # Ensure contiguous and convert to float32 for statistics
        x = inputs.to(torch.float32).contiguous()
        B, S, F = x.shape
        device = x.device

        # 1) Compute per-(batch, seq) sum and sum of squares using Triton
        sums = torch.empty(B * S, dtype=torch.float32, device=device)
        sums2 = torch.empty(B * S, dtype=torch.float32, device=device)

        grid = (B * S,)
        reduce_sum_sumsq_rows_kernel[grid](x, sums, sums2, B, S, F)

        # 2) Compute mean and std per row
        mean = torch.empty(B * S, dtype=torch.float32, device=device)
        std = torch.empty(B * S, dtype=torch.float32, device=device)

        compute_mean_std_kernel[grid](sums, sums2, mean, std, F)

        # 3) Compute ndtri(target_sparsity) in Triton and store into a 1-element tensor
        ndtri_out = torch.empty(1, dtype=torch.float32, device=device)
        P_LOW = 0.02425
        ndtri_scalar_kernel[(1,)](float(target_sparsity), ndtri_out, P_LOW)

        # 4) Apply sparsification: output = max(0, x - (mean + std * ndtri))
        out_fp32 = torch.empty((B, S, F), dtype=torch.float32, device=device)

        total_elems = B * S * F
        BLOCK_SIZE = 1024
        grid1d = (triton.cdiv(total_elems, BLOCK_SIZE),)
        sparsify_relu_kernel[grid1d](
            x, out_fp32, mean, std, float(ndtri_out), B, S, F, total_elems, BLOCK_SIZE=BLOCK_SIZE
        )

        # Return as bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
