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

    # Loop over the feature dimension in chunks
    for start in range(0, D, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        base = pid * D
        x = tl.load(X_ptr + base + offs.to(tl.int64), mask=mask, other=0.0)
        x = x.to(tl.float32)
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
    # population variance (unbiased=False): var = E[x^2] - (E[x])^2
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
    y = x - threshold  # broadcast scalar threshold
    y = tl.maximum(y, 0.0)  # ReLU
    tl.store(OUT_ptr + base_idx + offs.to(tl.int64), y, mask=mask)


@triton.jit
def ndtri_approx_kernel(
    P_ptr,           # *float32, length 1 (scalar p in [0,1])
    OUT_ptr,         # *float32, length 1 (output z-score)
):
    # Load sparsity p
    p = tl.load(P_ptr)
    # Constants for Abramowitz & Stegun 5.2.23 approximation
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
    z_low = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)

    # Upper region
    q2 = tl.sqrt(-2.0 * tl.log(1.0 - p))
    z_high = -(((((c1*q2 + c2)*q2 + c3)*q2 + c4)*q2 + c5)*q2 + c6) / ((((d1*q2 + d2)*q2 + d3)*q2 + d4)*q2 + 1.0)

    # Central region
    q_mid = p - 0.5
    r = q_mid * q_mid
    poly = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6)
    denom = (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0)
    z_mid = poly * q_mid / denom

    z = tl.where(p < p_low, z_low, tl.where(p > p_high, z_high, z_mid))
    tl.store(OUT_ptr, z)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float, block_size_reduce: int = 1024, block_size_apply: int = 1024):
        super().__init__()
        self.target_sparsity = float(target_sparsity)
        # Tunable meta-parameters
        self.block_size_reduce = block_size_reduce  # for reduction kernel
        self.block_size_apply = block_size_apply    # for apply kernel

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Ensure device and contiguity; keep dtype conversions inside kernels
        assert inputs.is_cuda, "ModelNew expects CUDA tensors"
        inputs = inputs.contiguous()
        B, S, D = inputs.shape

        # Allocate intermediate buffers (float32 for numerical stability)
        sum_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        sumsq_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        mean_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        std_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)

        # Device scalar for target sparsity (avoid any host torch ops)
        p_tensor = torch.tensor(self.target_sparsity, dtype=torch.float32, device=inputs.device)
        z_buf = torch.empty(1, dtype=torch.float32, device=inputs.device)

        # 1) Reduce to sum and sum of squares (float32 accumulation)
        grid_reduce = (B * S,)
        reduce_sum_sumsq_kernel[grid_reduce](
            inputs, sum_buf, sumsq_buf,
            B, S, D,
            BLOCK_SIZE=self.block_size_reduce,
            num_warps=4,
            num_stages=2,
        )

        # 2) Compute mean and std (population variance, unbiased=False)
        grid_mean_std = (B * S,)
        compute_mean_std_kernel[grid_mean_std](
            sum_buf, sumsq_buf, mean_buf, std_buf, D,
            num_warps=2,
            num_stages=2,
        )

        # 3) Compute inverse normal CDF (scalar z) via Triton
        ndtri_approx_kernel[(1,)](
            p_tensor, z_buf,
            num_warps=1,
            num_stages=1,
        )

        # 4) Apply activation: y = max(0, x - (mean + std * z))
        out_f32 = torch.empty((B, S, D), dtype=torch.float32, device=inputs.device)
        grid_apply = (B, S, triton.cdiv(D, self.block_size_apply))
        apply_activation_kernel[grid_apply](
            inputs, mean_buf, std_buf, z_buf, out_f32,
            B, S, D,
            BLOCK_SIZE=self.block_size_apply,
            num_warps=8,
            num_stages=2,
        )

        # Return in bfloat16 to match original function’s output dtype
        return out_f32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
