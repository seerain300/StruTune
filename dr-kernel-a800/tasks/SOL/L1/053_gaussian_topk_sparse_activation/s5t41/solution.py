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
        x_ptr,            # *const float32 input flattened
        sums_ptr,         # *float32, length B*S
        sums2_ptr,        # *float32, length B*S
        B: tl.constexpr,  # batch size
        S: tl.constexpr,  # seq_len
        F,                # feature_size (intermediate_size)
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
    def ndtri_kernel(
        out_ptr,          # *float32, single-element output for ndtri
        p,                # float32 scalar probability
        BLOCK_SIZE: tl.constexpr,
    ):
        # Abramowitz & Stegun 7.1.26 approximation for normal PPF (inverse CDF)
        # One program instance; compute into out_ptr[0]
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

        # Scalar computation
        if p < p_low:
            q = tl.sqrt(-2.0 * tl.log(p))
            r = ((c1 * q + c2) * q + c3) * q + c4
            r = (r * q + c5) * q + c6
            s = ((d1 * q + d2) * q + d3) * q + d4
            r = r / (s * q + 1.0)
            x = -q * r
        elif p <= p_high:
            q = p - 0.5
            r = q * q
            r = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
            s = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
            x = r / s
        else:
            q = tl.sqrt(-2.0 * tl.log(1.0 - p))
            r = ((c1 * q + c2) * q + c3) * q + c4
            r = (r * q + c5) * q + c6
            s = ((d1 * q + d2) * q + d3) * q + d4
            r = -r / (s * q + 1.0)
            x = q + r

        tl.store(out_ptr, x)

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

        grid_reduce = (num_rows,)
        BLOCK_SIZE_RED = 256
        reduce_sum_sumsq_rows_kernel[grid_reduce](
            x.view(-1), sums, sums2, B, S, F, BLOCK_SIZE=BLOCK_SIZE_RED
        )

        # 2) Compute mean and std per row (on device, small 1D ops)
        mean = sums / float(F)
        std = torch.sqrt(sums2 / float(F) - mean * mean)
        std = torch.clamp(std, min=0.0)

        # 3) Compute inverse normal CDF for target_sparsity via Triton (scalar)
        threshold_scale = torch.empty(1, dtype=torch.float32, device=device)
        grid_ndtri = (1,)
        ndtri_kernel[grid_ndtri](threshold_scale, float(target_sparsity), BLOCK_SIZE=1)

        # 4) Apply sparsification: output = max(0, x - threshold) using Triton
        out_fp32 = torch.empty((B, S, F), dtype=torch.float32, device=device)

        total_elems = B * S * F
        BLOCK_SIZE_POINT = 1024
        grid_point = (triton.cdiv(total_elems, BLOCK_SIZE_POINT),)
        sparsify_relu_kernel[grid_point](
            x, out_fp32, mean, std, threshold_scale[0], B, S, F, total_elems, BLOCK_SIZE=BLOCK_SIZE_POINT
        )

        # 5) Return as bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
