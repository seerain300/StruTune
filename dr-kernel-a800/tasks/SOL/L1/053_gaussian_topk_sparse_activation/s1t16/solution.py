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

    # Iterate over feature dimension in chunks
    for start in range(0, D, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        base = pid * D  # row start index
        # 64-bit indexing for robustness
        x = tl.load(X_ptr + (base + offs.to(tl.int64)), mask=mask, other=0.0).to(tl.float32)
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
    D,               # int32
):
    pid = tl.program_id(axis=0)
    sum_val = tl.load(SUM_ptr + pid)
    sumsq_val = tl.load(SUMSQ_ptr + pid)
    d = tl.full((), D, tl.int32)
    mean = sum_val / d
    var = sumsq_val / d - mean * mean
    var = tl.maximum(var, 0.0)  # clamp variance to non-negative
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + pid, mean)
    tl.store(STD_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(
    P_ptr,           # *float32, length 1 (scalar p in [0,1])
    OUT_ptr,         # *float32, length 1 (output z-score)
):
    # Load sparsity p
    p = tl.load(P_ptr)
    # Abramowitz & Stegun 5.2.23 approximation constants
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
    poly = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6)
    den = (((((d1*q + d2)*q + d3)*q + d4)*q + 1.0))
    low = poly / den

    # Central region
    q2 = p - 0.5
    r = q2 * q2
    poly_c = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)
    den_c = (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
    mid = poly_c * q2 / den_c

    # Upper region
    q3 = tl.sqrt(-2.0 * tl.log(1.0 - p))
    poly_u = (((((c1*q3 + c2)*q3 + c3)*q3 + c4)*q3 + c5)*q3 + c6)
    den_u = (((((d1*q3 + d2)*q3 + d3)*q3 + d4)*q3 + 1.0))
    up = -poly_u / den_u

    result = tl.where(p < p_low, low, tl.where(p > p_high, up, mid))
    tl.store(OUT_ptr, result)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous (float32)
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    Z_ptr,           # *float32, length 1 (scalar z-score)
    OUT_ptr,         # *float32, output [B, S, D]
    B, S, D,         # int32
    BLOCK_SIZE: tl.constexpr,
):
    # 3D grid over (B, S, tiles of D)
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

    x = tl.load(X_ptr + base_idx + offs.to(tl.int64), mask=mask, other=0.0).to(tl.float32)
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(OUT_ptr + base_idx + offs.to(tl.int64), y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float):
        # Ensure input is on GPU; this Triton code requires CUDA
        assert inputs.is_cuda, "ModelNew requires CUDA tensors"
        # Convert to float32 for statistics
        X = inputs.contiguous().to(torch.float32)
        B, S, D = X.shape

        # Allocate buffers
        SUM = torch.empty(B * S, dtype=torch.float32, device=X.device)
        SUMSQ = torch.empty(B * S, dtype=torch.float32, device=X.device)
        MEAN = torch.empty(B * S, dtype=torch.float32, device=X.device)
        STD = torch.empty(B * S, dtype=torch.float32, device=X.device)
        Z = torch.empty(1, dtype=torch.float32, device=X.device)

        # Launch reduction kernel: one program per (b, s) row
        grid_reduce = (B * S,)
        reduce_sum_sumsq_kernel[grid_reduce](
            X, SUM, SUMSQ, B, S, D,
            BLOCK_SIZE=1024,
            num_warps=8,
        )

        # Compute mean and std per row
        grid_mean_std = (B * S,)
        compute_mean_std_kernel[grid_mean_std](
            SUM, SUMSQ, MEAN, STD, D,
            num_warps=1,
        )

        # Compute ndtri(target_sparsity) on device with Triton
        p = torch.tensor(target_sparsity, dtype=torch.float32, device=X.device)
        ndtri_approx_kernel[(1,)](p, Z, num_warps=1)

        # Apply activation: y = max(0, x - (mean + std * z_score))
        OUT = torch.empty(B * S * D, dtype=torch.float32, device=X.device)
        grid_apply = (B, S, triton.cdiv(D, 1024))
        apply_activation_kernel[grid_apply](
            X, MEAN, STD, Z, OUT, B, S, D,
            BLOCK_SIZE=1024,
            num_warps=4,
        )

        # Return in bfloat16 to match original behavior
        return OUT.view(B, S, D).to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
