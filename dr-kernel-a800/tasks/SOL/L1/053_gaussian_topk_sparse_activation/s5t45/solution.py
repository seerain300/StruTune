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
        x_ptr,                  # *const float32
        sums_ptr,               # *float32, length B*S
        sums2_ptr,              # *float32, length B*S
        B: tl.constexpr,        # int
        S: tl.constexpr,        # int
        F,                      # feature_size (last dim)
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
        sums_ptr,               # *const float32, length B*S
        sums2_ptr,              # *const float32, length B*S
        mean_ptr,               # *float32, length B*S
        std_ptr,                # *float32, length B*S
        F: tl.constexpr,        # feature_size
    ):
        pid = tl.program_id(axis=0)
        s = tl.load(sums_ptr + pid)       # sum
        s2 = tl.load(sums2_ptr + pid)     # sum of squares
        mean = s / F
        var = s2 / F - mean * mean
        var = tl.maximum(var, 0.0)        # clamp to avoid tiny negatives
        std = tl.sqrt(var)
        tl.store(mean_ptr + pid, mean)
        tl.store(std_ptr + pid, std)

    @triton.jit
    def ndtri_approx_kernel(
        out_ptr,                # *float32, length 1
        p,                      # scalar float32 in (0, 1)
        BLOCK_SIZE: tl.constexpr,
    ):
        # Abramowitz and Stegun 7.1.26 approximation for inverse normal CDF
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

        if p < p_low:
            q = tl.sqrt(-2.0 * tl.log(p))
            nd = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        elif p <= (1.0 - p_low):
            q = p - 0.5
            r = q * q
            nd = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                 (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        else:
            q = tl.sqrt(-2.0 * tl.log(1.0 - p))
            nd = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

        tl.store(out_ptr, nd)

    @triton.jit
    def sparsify_relu_kernel(
        x_ptr,                  # *const float32 input
        out_ptr,                # *float32 output
        mean_ptr,               # *const float32, shape [B*S]
        std_ptr,                # *const float32, shape [B*S]
        threshold_scale,        # scalar float32 multiplier for threshold
        B: tl.constexpr,
        S: tl.constexpr,
        F: tl.constexpr,
        total_elems,            # int32
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


# -----------------------------
# ModelNew: Triton-only forward
# -----------------------------
class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Triton-only execution: ensure CUDA and Triton availability
        assert TRITON_AVAILABLE, "Triton is not available"
        assert inputs.is_cuda, "Input must be on CUDA for Triton execution"

        # Ensure float32 and contiguous
        x = inputs.to(torch.float32).contiguous()
        B, S, F = x.shape
        device = x.device

        # 1) Per-(batch, seq) sum and sum of squares via Triton
        total_rows = B * S
        sums = torch.empty(total_rows, dtype=torch.float32, device=device)
        sums2 = torch.empty(total_rows, dtype=torch.float32, device=device)

        # Launch one program per row
        grid_reduce = (total_rows,)
        reduce_sum_sumsq_rows_kernel[grid_reduce](
            x, sums, sums2, B, S, F, BLOCK_SIZE=1024, num_warps=4
        )

        # 2) Compute mean and std per row in Triton
        mean = torch.empty(total_rows, dtype=torch.float32, device=device)
        std = torch.empty(total_rows, dtype=torch.float32, device=device)

        grid_stats = (total_rows,)
        compute_mean_std_kernel[grid_stats](sums, sums2, mean, std, F, num_warps=1)

        # 3) Compute ndtri(target_sparsity) in Triton (scalar)
        ndtri_val = torch.empty(1, dtype=torch.float32, device=device)
        p = float(target_sparsity)
        ndtri_approx_kernel[(1,)](ndtri_val, p, BLOCK_SIZE=1, num_warps=1)
        threshold_scale = ndtri_val[0]  # scalar float32 on device

        # 4) Apply sparsification: output = max(0, x - (mean + std * threshold_scale))
        out_fp32 = torch.empty_like(x)  # float32 output buffer

        total_elems = B * S * F
        grid_point = (triton.cdiv(total_elems, 1024),)
        sparsify_relu_kernel[grid_point](
            x, out_fp32, mean, std, threshold_scale,
            B, S, F, total_elems, BLOCK_SIZE=1024, num_warps=4
        )

        # Return as bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
