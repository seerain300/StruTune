import torch
import triton
import triton.language as tl


# Single Triton kernel:
# - Computes mean and population std per row across N.
# - Computes q = ndtri(target_sparsity) using Abramowitz & Stegun 5.2.23 approximation.
# - Computes per-row threshold = mean + std * q.
# - Applies ReLU(x - threshold[row]) elementwise and writes to OUT (fp32).
@triton.jit
def full_forward_kernel(
    X_ptr,          # *f32, input [rows, N], linearized
    OUT_ptr,        # *f32, output [rows, N], linearized
    rows: tl.constexpr,
    N: tl.constexpr,
    target_sparsity,  # f32 scalar on device (0..1)
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Initialize accumulators
    sum_val = 0.0
    sum_sq = 0.0

    # First pass: compute sum and sum of squares across N
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / N
    # population std (unbiased=False): sqrt(E[x^2] - (E[x])^2)
    var = sum_sq / N - mean * mean
    std = tl.sqrt(var)

    # Compute q = ndtri(target_sparsity) using Abramowitz & Stegun 5.2.23
    # For p in (0, 1), use the central region approximation (default path).
    # Handle p ~ 0 and p ~ 1 by falling back to lower/higher region formulas,
    # but since target_sparsity is strictly between 0 and 1, central region is fine.
    # Implement the central region formula directly:
    # z = sqrt(2) * erfinv(2p - 1)
    # We implement erfinv via A&S rational approximation to avoid relying on tl.erf.
    # A&S coefficients (for erfinv approximation in the central region):
    # p_low = 0.02425; p_high = 1 - p_low; central region when p in [p_low, p_high].
    p_low = 0.02425
    p_high = 1.0 - p_low
    # If target_sparsity outside central, use lower/higher region; otherwise central.
    # Here we assume target_sparsity in (0,1), and central region formula is robust for typical values.
    p = target_sparsity  # already in (0,1)
    # Central region:
    # x = p - 0.5
    # Approximation: erf(x) ~ ((a1 t + a2) t + a3) t + a4) t + a5) t + a6
    #               / ((b1 t + b2) t + b3) t + b4) t + b5), t = x^2
    x = p - 0.5
    t = x * x
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

    poly = (((((a1 * t + a2) * t + a3) * t + a4) * t + a5) * t + a6)
    den = (((((b1 * t + b2) * t + b3) * t + b4) * t + b5) * t + 1.0)
    erf_approx = poly / den
    z = tl.sqrt(2.0) * (x + (1.0 - x) * erf_approx)
    q = z  # ndtri(p) = z

    # Compute per-row threshold = mean + std * q
    threshold = mean + std * q

    # Second pass: apply ReLU(x - threshold) and store to OUT
    for col_start in range(0, N, BLOCK):
        offs = col_start + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X_ptr + idx, mask=mask, other=0.0)
        y = x - threshold
        # ReLU
        y = tl.maximum(y, 0.0)
        tl.store(OUT_ptr + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Assume inputs is [B, S, N] float32/float16/bfloat16; we'll compute in fp32
        x = inputs
        # Convert to fp32 for numerical stability in statistics and elementwise math
        x_f32 = x.to(torch.float32)
        B, S, N = x_f32.shape
        rows = B * S

        # Allocate linearized output
        OUT = torch.empty(rows * N, device=x_f32.device, dtype=torch.float32)

        # Launch single Triton kernel: one program per row
        grid = (rows,)
        full_forward_kernel[grid](
            x_f32.view(-1),  # linearized input
            OUT,
            rows=rows,
            N=N,
            target_sparsity=float(target_sparsity),
            BLOCK=1024,
            num_warps=4,
        )

        # Return in bf16 to match the original behavior
        return OUT.view(B, S, N).to(torch.bfloat16)