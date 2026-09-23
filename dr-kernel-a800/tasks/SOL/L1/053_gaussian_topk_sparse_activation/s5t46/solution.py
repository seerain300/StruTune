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
        B: tl.constexpr,        # int (compile-time for grid convenience)
        S: tl.constexpr,        # int
        F,                      # feature_size (last dim), runtime int
    ):
        # One program per (b, s) row
        pid = tl.program_id(axis=0)
        # total rows = B*S (axis=0 grid size enforces this)
        row_start = pid * F

        local_sum = 0.0
        local_sumsq = 0.0

        # Loop over feature dimension in chunks
        for offs in range(0, F, 1024):
            idx = offs + tl.arange(0, 1024)
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
        F,                      # feature_size (runtime int)
    ):
        pid = tl.program_id(axis=0)
        total = tl.load(sums_ptr + pid)
        sumsq = tl.load(sums2_ptr + pid)
        mean = total / F
        var = sumsq / F - mean * mean
        # Clamp variance to non-negative for numerical robustness
        var = tl.maximum(var, 0.0)
        std = tl.sqrt(var)
        tl.store(mean_ptr + pid, mean)
        tl.store(std_ptr + pid, std)


    @triton.jit
    def ndtri_kernel(
        out_ptr,                # *float32, single-element output
        p,                      # float32 scalar (target_sparsity)
    ):
        # Abramowitz and Stegun 7.1.26 approximation for standard normal PPF
        # Constants
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

        # Lower region
        q = tl.sqrt(-2.0 * tl.log(p))
        t = 1.0 / (1.0 - p)
        poly_low = (((((c1 * t + c2) * t + c3) * t + c4) * t + c5) * t + c6)
        den_low = (((((d1 * t + d2) * t + d3) * t + d4) * t + 1.0))
        nd_low = poly_low / den_low

        # Upper region
        q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
        t_up = 1.0 / p
        poly_up = (((((c1 * t_up + c2) * t_up + c3) * t_up + c4) * t_up + c5) * t_up + c6)
        den_up = (((((d1 * t_up + d2) * t_up + d3) * t_up + d4) * t_up + 1.0))
        nd_up = -poly_up / den_up

        # Central region
        q_mid = p - 0.5
        r = q_mid * q_mid
        poly_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
        den_mid = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        nd_mid = poly_mid / den_mid

        # Select region based on p
        nd = tl.where(p < p_low, nd_low, tl.where(p > p_high, nd_up, nd_mid))

        tl.store(out_ptr, nd)


    @triton.jit
    def sparsify_relu_kernel(
        x_ptr,            # *const float32 input
        out_ptr,          # *float32 output
        mean_ptr,         # *const float32, shape [B*S]
        std_ptr,          # *const float32, shape [B*S]
        threshold_scale,  # scalar float32 multiplier for threshold (ndtri(target_sparsity))
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
        y = tl.maximum(y, 0.0)  # ReLU

        out_ptrs = out_ptr + b * SF + s * F + f
        tl.store(out_ptrs, y, mask=mask)


# -----------------------------
# ModelNew: Triton-only forward
# -----------------------------
class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect single input tensor and scalar target_sparsity
        assert len(args) == 2, "ModelNew.forward expects (inputs, target_sparsity)"
        inputs, target_sparsity = args

        # Triton-only execution: ensure CUDA and Triton availability
        assert TRITON_AVAILABLE, "Triton is not available"
        assert inputs.is_cuda, "Input must be on CUDA for Triton execution"

        # Ensure float32 and contiguous
        x = inputs.to(torch.float32).contiguous()
        # Shape handling: [B, S, F]
        assert x.ndim == 3, "inputs must be 3D: [batch_size, seq_len, intermediate_size]"
        B, S, F = x.shape
        device = x.device

        # 1) Per-(batch, seq) sum and sum of squares via Triton
        total_rows = B * S
        sums = torch.empty(total_rows, dtype=torch.float32, device=device)
        sums2 = torch.empty(total_rows, dtype=torch.float32, device=device)

        grid_reduce = (total_rows,)
        reduce_sum_sumsq_rows_kernel[grid_reduce](
            x, sums, sums2, B=B, S=S, F=F, num_warps=4
        )

        # 2) Compute mean and std per row in Triton (unbiased=False: divide by F)
        mean = torch.empty(total_rows, dtype=torch.float32, device=device)
        std = torch.empty(total_rows, dtype=torch.float32, device=device)

        grid_stats = (total_rows,)
        compute_mean_std_kernel[grid_stats](
            sums, sums2, mean, std, F
        )

        # 3) Compute inverse normal CDF for target_sparsity in Triton (scalar)
        ndtri_scale = torch.empty(1, dtype=torch.float32, device=device)
        ndtri_kernel[(1,)](ndtri_scale, float(target_sparsity))

        # 4) Apply sparsification: output = max(0, x - (mean + std * ndtri_scale)) in Triton
        out_fp32 = torch.empty((B, S, F), dtype=torch.float32, device=device)

        total_elems = B * S * F
        BLOCK_SIZE_POINT = 1024
        grid_sp = (triton.cdiv(total_elems, BLOCK_SIZE_POINT),)
        sparsify_relu_kernel


def run(*args):
    return ModelNew()(*args)
