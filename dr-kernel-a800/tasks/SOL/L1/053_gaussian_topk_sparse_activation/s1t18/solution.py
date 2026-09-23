import torch
import triton
import triton.language as tl


@triton.jit
def reduce_sum_sumsq_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    B, S, D,         # int32
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (b, s) row
    pid = tl.program_id(axis=0)  # pid in [0, B*S)
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Iterate over feature dimension D in chunks of BLOCK_SIZE
    for start in range(0, D, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)  # int32 vector
        mask = offs < D
        base_idx = pid * D  # int32
        x = tl.load(X_ptr + base_idx.to(tl.int64) + offs.to(tl.int64), mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    tl.store(SUM_ptr + pid, sum_val)
    tl.store(SUMSQ_ptr + pid, sumsq_val)


@triton.jit
def compute_mean_std_kernel(
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    D,               # int32 number of features
):
    pid = tl.program_id(axis=0)
    sum_val = tl.load(SUM_ptr + pid)
    sumsq_val = tl.load(SUMSQ_ptr + pid)
    d = tl.full((), D, tl.int32)
    mean = sum_val / d
    var = sumsq_val / d - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + pid, mean)
    tl.store(STD_ptr + pid, std)


@triton.jit
def ndtri_newton_kernel(
    P_ptr,           # *float32, length 1 (scalar p in [0,1])
    OUT_ptr,         # *float32, length 1 (output z-score)
    MAX_ITERS: tl.constexpr,  # number of Newton iterations
):
    # Load sparsity p
    p = tl.load(P_ptr)
    # We compute z = erfinv(1 - 2p). Let a = 1 - 2p in [0, 1].
    a = 1.0 - 2.0 * p

    # Initial guess: for a near 0.5, z ~ (1 - a) * 0.7978845608028654 (constant = sqrt(pi)/2)
    # For general a, use a better initial guess from Abramowitz & Stegun 7.1.26:
    # z0 = sign(a - 0.5) * ((1 - a) / (1 + a))**0.147
    # Implement for a <= 0.5 (since p >= 0.5 here). For p <= 0.5, mirror: z0 = -z0.
    # Compute z0 for a <= 0.5
    is_half = a <= 0.5
    c = 0.7978845608028654  # sqrt(pi)/2
    z0 = tl.where(is_half, c * (1.0 - a), 0.0)
    # For a > 0.5, use symmetry: z0 = -erfinv(2*(1-p) - 1) -> same formula but with (1-p)
    # However, our p is target_sparsity in [0,1], and typical usage is p in [0.5, 1). We handle a in (0.5, 1].
    # Use z0 = -z0 for a > 0.5 path; but since a = 1 - 2p, when p < 0.5 => a > 0.5, so we should set z0 negative.
    # To be precise: for a > 0.5, z0 = -abs(z0) is not correct. Instead, we set z0 = -((1 - (1 - a)) / (1 + (1 - a)))**0.147.
    # But to simplify, we compute z0 for a <= 0.5 and for a > 0.5 separately:
    # For a > 0.5: b = 2*p - 1 in (0, 0.5], z0 = -((1 - b) / (1 + b))**0.147
    b = 2.0 * p - 1.0
    z0_right = -(((1.0 - b) / (1.0 + b)) ** 0.147)
    z0 = tl.where(p <= 0.5, c * (1.0 - a), z0_right)

    # Newton iterations: solve erff(z) = a for z
    # f(z) = erff(z) - a; f'(z) = 2/sqrt(pi) * exp(-z^2) ≈ 1.1283791670955126 * exp(-z*z)
    # Update: z <- z + (a - erff(z)) / f'(z)
    two_over_sqrt_pi = 1.1283791670955126  # 2 / sqrt(pi)
    for _ in range(MAX_ITERS):
        z = z0
        e = tl.exp(-z * z)
        erfz = 1.0 - 2.0 / (1.0 + 1.0 / (1.0 + 0.3275911 * z * (1.0 + 0.2308536 / (1.0 + 0.0218138 * z * z))))  # approximation of erf(z)
        # Better approximation: use the well-known Abramowitz & Stegun formula for erf(z) via t = 1 / (1 + p*|z|)
        # But given we have z0 and want precision, we can use a simple approximation here; PyTorch's erff is accurate enough for this use.
        # To keep it exact: use torch.erf in the host? No. Implement a standard approximation:
        # erf(z) ≈ 1 - 2/(sqrt(pi)*(1 + p|z| + 3p^2|z|^3 + 5p^3|z|^5 + 7p^4|z|^7)) * exp(-z^2)
        # For simplicity and accuracy, reuse the standard approximation:
        # erf(z) ≈ 1 - 2/(1.0 + 1.0/(1.0 + 0.3275911*|z|)) * exp(-z^2)
        # We need sign(z). Use s = 1 if z >= 0 else -1
        s = tl.where(z >= 0.0, 1.0, -1.0)
        az = tl.abs(z)
        # Approximation: erf(z) ≈ s * (1 - exp(-z^2) * (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5)), t = 1/(1+p|z|)
        # Here, use simple formula: erf(z) ≈ 1 - 2/(1 + 1/(1 + 0.3275911*az)) * exp(-z^2)
        # Define t
        t = 1.0 / (1.0 + 0.3275911 * az)
        # Compute polynomial
        # Note: Triton doesn't have built-in erf; we'll use a well-known approximation:
        # erf(z) ≈ 1 - (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5) * exp(-z^2)
        # Coefficients: a1=0.254829592, a2=-0.284496736, a3=1.421413741, a4=-1.453152027, a5=1.061405429
        poly = 0.254829592 * t + (-0.284496736) * (t * t) + 1.421413741 * (t * t * t) + (-1.453152027) * (t * t * t * t) + 1.061405429 * (t * t * t * t * t)
        erfz = 1.0 - poly * tl.exp(-z * z)
        # Update z
        deriv = two_over_sqrt_pi * tl.exp(-z * z)
        z = z + (a - erfz) / deriv
        z0 = z

    tl.store(OUT_ptr, z0)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous (float32)
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    Z_ptr,           # *float32, length 1 (scalar z-score)
    OUT_ptr,         # *bfloat16, output [B, S, D] (bfloat16)
    B, S, D,         # int32
    BLOCK_SIZE: tl.constexpr,
):
    # 3D grid: (B, S, tiles of D)
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    tile = tl.program_id(axis=2)

    base = b * S + s
    base_idx = base * D  # int32

    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    mean = tl.load(MEAN_ptr + base)
    std = tl.load(STD_ptr + base)
    z_score = tl.load(Z_ptr)  # scalar float32
    threshold = mean + std * z_score

    x = tl.load(X_ptr + base_idx.to(tl.int64) + offs.to(tl.int64), mask=mask, other=0.0).to(tl.float32)
    y = x - threshold  # broadcast scalar threshold
    y = tl.maximum(y, 0.0)  # ReLU
    # Store as bfloat16
    tl.store(OUT_ptr + base_idx.to(tl.int64) + offs.to(tl.int64), y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure inputs are contiguous and on the correct device
        inputs = inputs.contiguous()
        B, S, D = inputs.shape
        device = inputs.device

        # 1) Reduce sum and sum of squares across feature dimension using Triton
        SUM = torch.empty(B * S, dtype=torch.float32, device=device)
        SUMSQ = torch.empty(B * S, dtype=torch.float32, device=device)

        reduce_sum_sumsq_kernel[(B * S,)](
            inputs, SUM, SUMSQ, B, S, D,
            BLOCK_SIZE=1024,
            num_warps=8,
        )

        # 2) Compute mean and std per (b, s) row using Triton
        MEAN = torch.empty(B * S, dtype=torch.float32, device=device)
        STD = torch.empty(B * S, dtype=torch.float32, device=device)

        compute_mean_std_kernel[(B * S,)](
            SUM, SUMSQ, MEAN, STD, D,
            num_warps=4,
        )

        # 3) Compute z-score for target sparsity via Triton scalar kernel
        # Create a 1-element float32 device tensor for p and output z
        p_buf = torch.empty(1, dtype=torch.float32, device=device)
        p_buf.fill_(float(target_sparsity))
        Z = torch.empty(1, dtype=torch.float32, device=device)

        ndtri_newton_kernel[(1,)](
            p_buf, Z,
            MAX_ITERS=8,
            num_warps=1,
        )

        # 4) Apply activation: y = max(0, x - (mean + std * z)), store as bfloat16
        out_bf16 = torch.empty((B, S, D), dtype=torch.bfloat16, device=device)
        apply_activation_kernel[(B, S, triton.cdiv(D, 1024))](
            inputs, MEAN, STD, Z, out_bf16, B, S, D,
            BLOCK_SIZE=1024,
            num_warps=8,
        )

        return out_bf16


def run(*args):
    return ModelNew()(*args)
