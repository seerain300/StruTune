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
        B: tl.constexpr,  # batch size
        S: tl.constexpr,  # seq_len
        F,                # feature_size (runtime int)
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per (b, s) row
        pid = tl.program_id(axis=0)
        b = pid // S
        s = pid % S
        base = (b * S + s) * F

        local_sum = 0.0
        local_sumsq = 0.0

        for offs in range(0, F, BLOCK_SIZE):
            idx = offs + tl.arange(0, BLOCK_SIZE)
            mask = idx < F
            ptrs = x_ptr + base + idx
            vals = tl.load(ptrs, mask=mask, other=0.0)
            local_sum += tl.sum(vals, axis=0)
            local_sumsq += tl.sum(vals * vals, axis=0)

        tl.store(sums_ptr + pid, local_sum)
        tl.store(sums2_ptr + pid, local_sumsq)

    @triton.jit
    def compute_stats_rows_kernel(
        sums_ptr,          # *const float32, length B*S
        sums2_ptr,         # *const float32, length B*S
        mean_ptr,          # *float32, length B*S
        invvar_ptr,        # *float32, length B*S (1/std^2)
        F,                 # feature_size (runtime)
    ):
        pid = tl.program_id(axis=0)
        total = tl.load(sums_ptr + pid)
        avg = total / F
        avg2 = tl.load(sums2_ptr + pid) / F
        var = avg2 - avg * avg
        # clamp var to non-negative to avoid tiny negative due to rounding
        var = tl.maximum(var, 0.0)
        inv_var = 1.0 / tl.sqrt(var)
        tl.store(mean_ptr + pid, avg)
        tl.store(invvar_ptr + pid, inv_var)

    @triton.jit
    def ndtri_approx_kernel(
        out_ptr,           # *float32, length 1
        p,                 # float32 scalar (0 < p < 1)
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

        q = p - 0.5
        r = q * q
        x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

        tl.store(out_ptr, x)

    @triton.jit
    def sparsify_kernel(
        x_ptr,            # *const float32 input
        out_ptr,          # *bfloat16 output
        mean_ptr,         # *const float32, length B*S
        invvar_ptr,       # *const float32, length B*S
        std_scale,        # scalar float32 multiplier for threshold
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

        x_ptrs = x_ptr + offs
        x_vals = tl.load(x_ptrs, mask=mask, other=0.0)

        mean_row = mean_ptr + b * S + s
        invvar_row = invvar_ptr + b * S + s
        mean = tl.load(mean_row, mask=mask, other=0.0)
        invvar = tl.load(invvar_row, mask=mask, other=0.0)
        std = 1.0 / tl.sqrt(invvar)

        threshold = mean + std * std_scale
        y = x_vals - threshold
        y = tl.maximum(y, 0.0)  # ReLU

        # Store as bfloat16
        y_bf16 = y.to(tl.bfloat16)
        out_ptrs = out_ptr + offs
        tl.store(out_ptrs, y_bf16, mask=mask)


# -----------------------------
# ModelNew: Triton-only forward
# -----------------------------
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Triton-only path
        assert TRITON_AVAILABLE, "Triton is not available"
        assert inputs.is_cuda, "Input must be on CUDA for Triton execution"

        # Ensure float32 contiguous for Triton kernels
        x = inputs.to(torch.float32).contiguous()
        B, S, F = x.shape
        device = x.device

        # 1) Compute per-(batch, seq) sum and sum of squares using Triton
        sums = torch.empty(B * S, dtype=torch.float32, device=device)
        sums2 = torch.empty(B * S, dtype=torch.float32, device=device)

        BLOCK_SIZE_RED = 1024
        grid = (B * S,)
        reduce_sum_sumsq_rows_kernel[grid](x, sums, sums2, B, S, F, BLOCK_SIZE=BLOCK_SIZE_RED)

        # 2) Compute mean and inverse variance per (b, s) in Triton
        mean = torch.empty(B * S, dtype=torch.float32, device=device)
        invvar = torch.empty(B * S, dtype=torch.float32, device=device)

        compute_stats_rows_kernel[grid](sums, sums2, mean, invvar, F)

        # 3) Compute inverse normal CDF for target_sparsity in Triton (scalar)
        std_scale = torch.empty(1, dtype=torch.float32, device=device)
        ndtri_approx_kernel[(1,)](std_scale, target_sparsity)

        # 4) Apply sparsification: y = max(0, x - threshold), threshold = mean + std * std_scale
        total_elems = B * S * F
        out_bf16 = torch.empty_like(x, dtype=torch.bfloat16, device=device)

        BLOCK_SIZE_POINT = 1024
        grid_point = (triton.cdiv(total_elems, BLOCK_SIZE_POINT),)
        sparsify_kernel[grid_point](
            x, out_bf16, mean, invvar, std_scale.item(), B, S, F, total_elems, BLOCK_SIZE=BLOCK_SIZE_POINT
        )

        return out_bf16


def run(*args):
    return ModelNew()(*args)
