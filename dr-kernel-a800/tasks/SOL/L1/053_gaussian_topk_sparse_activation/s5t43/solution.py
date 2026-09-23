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
        B, S, F,          # int32 dims
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per row (b, s)
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

        # Write per-row sums
        tl.store(sums_ptr + pid, local_sum)
        tl.store(sums2_ptr + pid, local_sumsq)

    @triton.jit
    def compute_mean_std_1d_kernel(
        sums_ptr,       # *const float32, length N
        sums2_ptr,      # *const float32, length N
        mean_ptr,       # *float32, length N
        std_ptr,        # *float32, length N
        N,              # int32
        F,              # int32 (feature size)
    ):
        pid = tl.program_id(axis=0)
        total = tl.load(sums_ptr + pid)
        total2 = tl.load(sums2_ptr + pid)
        mean = total / F
        var = total2 / F - mean * mean
        var = tl.maximum(var, 0.0)
        std = tl.sqrt(var)
        tl.store(mean_ptr + pid, mean)
        tl.store(std_ptr + pid, std)

    @triton.jit
    def sparsify_relu_kernel(
        x_ptr,           # *const float32 input
        out_ptr,         # *float32 output
        mean_ptr,        # *const float32, shape [B*S] (flattened)
        std_ptr,         # *const float32, shape [B*S] (flattened)
        ndtri_scale_ptr, # *const float32, scalar [1]
        B: tl.constexpr,
        S: tl.constexpr,
        F: tl.constexpr,
        total_elems,     # int32
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

        ndtri_scale = tl.load(ndtri_scale_ptr)  # scalar
        threshold = mean + std * ndtri_scale
        y = x_vals - threshold
        # ReLU
        y = tl.maximum(y, 0.0)

        out_ptrs = out_ptr + b * SF + s * F + f
        tl.store(out_ptrs, y, mask=mask)

    @triton.jit
    def ndtri_approx_kernel(
        out_ptr,         # *float32, shape [1]
        p,               # float32 scalar target_sparsity
        # constants for Abramowitz & Stegun 7.1.26
    ):
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
        z_low = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

        # Central region
        q_mid = p - 0.5
        r_mid = q_mid * q_mid
        z_mid = (((((a1 * r_mid + a2) * r_mid + a3) * r_mid + a4) * r_mid + a5) * r_mid + a6) * q_mid / \
                (((((b1 * r_mid + b2) * r_mid + b3) * r_mid + b4) * r_mid + b5) * r_mid + 1.0)

        # Upper region
        q_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
        z_up = -(((((c1 * q_up + c2) * q_up + c3) * q_up + c4) * q_up + c5) * q_up + c6) / \
               ((((d1 * q_up + d2) * q_up + d3) * q_up + d4) * q_up + 1.0)

        # Select region based on p
        z = tl.where(p < p_low, z_low, tl.where(p > p_high, z_up, z_mid))
        tl.store(out_ptr, z)


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

        # Ensure float32 for statistics; ensure contiguous
        x = inputs.contiguous().to(torch.float32)
        B, S, F = x.shape
        device = x.device

        # 1) Compute per-(batch, seq) sum and sum of squares using Triton
        num_rows = B * S
        sums = torch.empty(num_rows, dtype=torch.float32, device=device)
        sums2 = torch.empty(num_rows, dtype=torch.float32, device=device)

        BLOCK_SIZE_RED = 256
        reduce_sum_sumsq_rows_kernel[(num_rows,)](
            x, sums, sums2, B, S, F, BLOCK_SIZE=BLOCK_SIZE_RED
        )

        # 2) Compute mean and std per row using Triton elementwise kernel on 1D arrays
        mean = torch.empty(num_rows, dtype=torch.float32, device=device)
        std = torch.empty(num_rows, dtype=torch.float32, device=device)

        # We implement mean and std calculation in Triton. This kernel is simple and vectorized.
        compute_mean_std_1d_kernel[(num_rows,)](
            sums, sums2, mean, std, num_rows, F
        )

        # Reshape to [B, S] for broadcasting in the final kernel
        mean = mean.view(B, S)
        std = std.view(B, S)

        # 3) Compute ndtri(target_sparsity) via Triton scalar approximation into a 1-element tensor
        ndtri_scale = torch.empty(1, dtype=torch.float32, device=device)
        p = torch.tensor(float(target_sparsity), dtype=torch.float32, device=device)
        ndtri_approx_kernel[(1,)](ndtri_scale, p)

        # 4) Apply sparsification: output = max(0, x - (mean + std * ndtri_scale))
        out_fp32 = torch.empty((B, S, F), dtype=torch.float32, device=device)

        total_elems = B * S * F
        BLOCK_SIZE_POINT = 1024
        sparsify_relu_kernel[(triton.cdiv(total_elems, BLOCK_SIZE_POINT),)](
            x, out_fp32, mean, std, ndtri_scale, B, S, F, total_elems, BLOCK_SIZE=BLOCK_SIZE_POINT
        )

        # Return as bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
