import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    B, S, D,         # int32
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # one program per (b, s) row
    b = pid // S
    s = pid % S
    base = (b * S + s) * D

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Iterate over feature dimension in chunks of BLOCK_SIZE
    for start in range(0, D, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        x = tl.load(X_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
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
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + pid, mean)
    tl.store(STD_ptr + pid, std)


@triton.jit
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    Z_ptr,           # *float32, length 1 (scalar z-score)
    OUT_ptr,         # *float32, output [B, S, D]
    B, S, D,         # int32
    BLOCK_SIZE: tl.constexpr,
):
    # 3D grid: (B, S, ceil_div(D, BLOCK_SIZE))
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    tile = tl.program_id(axis=2)

    base = b * S + s
    base_idx = base * D

    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    mean = tl.load(MEAN_ptr + base)
    std = tl.load(STD_ptr + base)
    z_score = tl.load(Z_ptr)  # scalar
    threshold = mean + std * z_score

    x = tl.load(X_ptr + base_idx + offs, mask=mask, other=0.0).to(tl.float32)
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(OUT_ptr + base_idx + offs, y, mask=mask)


@triton.jit
def ndtri_approx_kernel(
    P_ptr,           # *float32, length 1 (scalar p in [0,1])
    OUT_ptr,         # *float32, length 1 (output z-score)
):
    # Load p
    p = tl.load(P_ptr)
    # Abramowitz & Stegun 5.2.23 approximation for inverse normal CDF
    # Implement for p <= 0.5 branch. For p > 0.5, host will pass 1-p (symmetry).
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

    t = tl.sqrt(-2.0 * tl.log(p))
    z = (((((a1 * t + a2) * t + a3) * t + a4) * t + a5) * t + a6) / \
        (((((b1 * t + b2) * t + b3) * t + b4) * t + b5) * t + 1.0)
    tl.store(OUT_ptr, z)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Triton-optimized version of the original run() function:
        - Computes mean and std along feature dimension in float32.
        - Uses inverse normal CDF to get z_score (computed via Triton).
        - Applies y = max(0, x - (mean + std * z_score)) in float32.
        - Returns output in bfloat16 (matching original behavior).
        """
        assert inputs.is_cuda, "Input must be on CUDA device for Triton kernels."
        inputs = inputs.contiguous()
        B, S, D = inputs.shape
        device = inputs.device

        # 1) Compute per-row sum and sum of squares (float32)
        sum_buf = torch.empty(B * S, dtype=torch.float32, device=device)
        sumsq_buf = torch.empty(B * S, dtype=torch.float32, device=device)
        grid_reduce = (B * S,)
        reduce_mean_std_kernel[grid_reduce](
            inputs, sum_buf, sumsq_buf, B, S, D, BLOCK_SIZE=256, num_warps=4
        )

        # 2) Compute per-row mean and std (float32)
        mean_buf = torch.empty(B * S, dtype=torch.float32, device=device)
        std_buf = torch.empty(B * S, dtype=torch.float32, device=device)
        grid_mean_std = (B * S,)
        compute_mean_std_kernel[grid_mean_std](
            sum_buf, sumsq_buf, mean_buf, std_buf, D, num_warps=1
        )

        # 3) Compute inverse-normal CDF for target sparsity (scalar) via Triton
        # For p > 0.5, use symmetry: ndtri(p) = -ndtri(1-p).
        p = float(target_sparsity)
        if p > 0.5:
            p = 1.0 - p
        p_dev = torch.tensor(p, dtype=torch.float32, device=device)
        z_score_buf = torch.empty(1, dtype=torch.float32, device=device)
        ndtri_approx_kernel[(1,)](p_dev, z_score_buf, num_warps=1)

        # 4) Apply activation: y = max(0, x - (mean + std * z_score))
        out_f32 = torch.empty_like(inputs, dtype=torch.float32, device=device)
        grid_apply = (B, S, triton.cdiv(D, 256))
        apply_activation_kernel[grid_apply](
            inputs, mean_buf, std_buf, z_score_buf, out_f32, B, S, D, BLOCK_SIZE=256, num_warps=4
        )

        # 5) Cast to bfloat16 to match original output dtype
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
