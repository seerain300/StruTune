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
    def reduce_row_sum_sumsq_kernel(
        x_ptr,            # *const float32, flattened input
        sums_ptr,         # *float32, length B*S
        sums2_ptr,        # *float32, length B*S
        B: tl.constexpr,  # batch size (metadata)
        S: tl.constexpr,  # seq_len (metadata)
        F,                # feature_size (intermediate dimension)
        BLOCK_SIZE: tl.constexpr,
    ):
        # One program per (b, s) row
        pid = tl.program_id(axis=0)
        row = pid  # since grid is exactly B*S
        local_sum = 0.0
        local_sumsq = 0.0

        # Loop over feature dimension in chunks
        for offs in range(0, F, BLOCK_SIZE):
            idx = offs + tl.arange(0, BLOCK_SIZE)
            mask = idx < F
            x_ptrs = x_ptr + row * F + idx
            vals = tl.load(x_ptrs, mask=mask, other=0.0)
            local_sum += tl.sum(vals, axis=0)
            local_sumsq += tl.sum(vals * vals, axis=0)

        tl.store(sums_ptr + row, local_sum)
        tl.store(sums2_ptr + row, local_sumsq)

    @triton.jit
    def inv_erf_kernel(
        out_ptr,          # *float32, scalar output
        p_ptr,            # *const float32, scalar input p
        A1, A2, A3, A4, A5,
        B1, B2, B3, B4, B5,
        C1, C2, C3, C4, C5, C6,
        D1, D2, D3, D4,
    ):
        # Compute z = sqrt(2) * erfinv(p) using A&S approximation
        # Constants:
        # p_ptr is scalar input p in (0, 1)
        p = tl.load(p_ptr)
        # Initial approximation
        z = (((((A1*p + A2)*p + A3)*p + A4)*p + A5)*p)
        t = 1.0 - p
        y = (((((B1*t + B2)*t + B3)*t + B4)*t + B5)*t)
        z = z / (1.0 + y * p)
        # Refinement using Newton iteration (1 step)
        # erf(z) ~ 1 - (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5) * exp(-z^2)
        # where t = 1 / (1 + p * z^2), coefficients:
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        z2 = z * z
        t = 1.0 / (1.0 + p * z2)
        # poly = a1*t + a2*t^2 + ... + a5*t^5
        poly = a1 * t
        poly = poly + a2 * (t * t)
        poly = poly + a3 * (t * t * t)
        poly = poly + a4 * (t * t * t * t)
        poly = poly + a5 * (t * t * t * t * t)
        erf_approx = 1.0 - poly * tl.exp(-z2)
        delta = (erf_approx - p) / (2.0 * erf_approx * (1.0 - erf_approx))  # derivative term 2*erf(z)*exp(z^2)
        # Since erf_approx ~ 1 - p, (1 - erf_approx) ~ p, and the derivative term is ~ 2*erf(z)*exp(z^2) ≈ 2*(erf_approx)*exp(z^2),
        # we approximate derivative with 2*erf_approx, because erf_approx is close to 1 - p.
        # But we can get erf(z) via erf_approx, and compute derivative as 2*erf_approx*(1 - erf_approx) would be 2*(1 - p)*p,
        # which is small. Simpler: use z_next = z - (erf_approx - p)/sqrt(pi)/exp(z^2) is not applicable.
        # Instead, we use a safer update: if (erf_approx - p) is small, keep z; else update with a small step:
        # We recompute derivative df/dz ≈ 2*erf(z) with erf(z) ≈ erf_approx
        df_dz = 2.0 * erf_approx
        # delta = (erf_approx - p) / df_dz
        delta = (erf_approx - p) / df_dz
        z = z + delta
        # Store result
        tl.store(out_ptr, z)

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
        # ReLU: max(0, y)
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
        # Triton availability and CUDA check
        assert TRITON_AVAILABLE, "Triton is not available"
        assert inputs.is_cuda, "Input must be on CUDA for Triton execution"

        # Convert to float32 and ensure contiguous
        x = inputs.to(torch.float32).contiguous()
        B, S, F = x.shape
        total_elems = B * S * F

        device = x.device

        # 1) Compute per-(batch, seq) sum and sum of squares using Triton
        sums = torch.empty(B * S, dtype=torch.float32, device=device)
        sums2 = torch.empty(B * S, dtype=torch.float32, device=device)

        BLOCK_SIZE = 1024
        grid_reduce = (B * S,)
        reduce_row_sum_sumsq_kernel[grid_reduce](
            x, sums, sums2, B, S, F, BLOCK_SIZE=BLOCK_SIZE
        )

        # 2) Compute mean and std per (batch, seq) on host (minimal computation)
        mean = sums / float(F)
        var = sums2 / float(F) - mean * mean
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var)

        # 3) Compute inverse normal CDF (ndtri) for target_sparsity in Triton
        # inv_erf(p) = sqrt(2) * z where z is quantile
        threshold_scale_buf = torch.empty(1, dtype=torch.float32, device=device)
        # Prepare scalar p on device
        p_tensor = torch.tensor(target_sparsity, dtype=torch.float32, device=device)
        inv_erf_kernel[(1,)](
            threshold_scale_buf, p_tensor,
            0.254829592, -1.68449946324, 1.42141374139, -1.45315202755, 1.06140542904,
            0.02000029928, 0.04677942361, -0.00000271682, 0.00000319227, -0.00000037620,
            -7.78489400243e-03, -3.22396458041e-01, -2.40075827716e+00, -2.54973253934e+00,
            4.37466414146e+00, 2.93816398269e+00,
            7.78469570904e-03, 3.22467129070e-01, 2.44513413714e+00, 3.75440866191e+00
        )
        threshold_scale = threshold_scale_buf[0] * math.sqrt(2.0)  # inv_erf(p) = z, we need sqrt(2)*z

        # 4) Apply sparsification: output = max(0, x - threshold) in Triton
        out_fp32 = torch.empty((B, S, F), dtype=torch.float32, device=device)

        BLOCK_SIZE_POINT = 1024
        grid_point = (triton.cdiv(total_elems, BLOCK_SIZE_POINT),)
        sparsify_relu_kernel[grid_point](
            x, out_fp32, mean, std, threshold_scale,
            B, S, F, total_elems, BLOCK_SIZE=BLOCK_SIZE_POINT
        )

        # Return as bfloat16 to match original behavior
        return out_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
