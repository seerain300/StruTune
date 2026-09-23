import torch
import triton
import triton.language as tl


@triton.jit
def reduce_mean_std_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    B, S, D,         # int32 dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (b, s) row
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Loop over D in chunks of BLOCK_SIZE
    for start in range(0, D, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < D
        base = b * S + s
        x = tl.load(X_ptr + base * D + offs, mask=mask, other=0.0).to(tl.float32)
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
def apply_activation_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    MEAN_ptr,        # *float32, length B*S
    STD_ptr,         # *float32, length B*S
    Z_ptr,           # *float32, length 1 (scalar z-score)
    OUT_ptr,         # *float32, output [B, S, D] (host will cast to bf16)
    B, S, D,         # int32 dimensions
    BLOCK_SIZE: tl.constexpr,
):
    # 3D grid: (B, S, ceil_div(D, BLOCK_SIZE))
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    tile = tl.program_id(axis=2)

    base = b * S + s
    base_idx = base * D  # scalar int32 base offset

    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    # Load mean and std for this (b, s) row
    mean = tl.load(MEAN_ptr + base)
    std = tl.load(STD_ptr + base)
    z_score = tl.load(Z_ptr)  # scalar z-score
    threshold = mean + std * z_score

    # Load input, apply activation
    x = tl.load(X_ptr + base_idx + offs, mask=mask, other=0.0).to(tl.float32)
    y = x - threshold
    y = tl.maximum(y, 0.0)  # ReLU

    tl.store(OUT_ptr + base_idx + offs, y, mask=mask)


@triton.jit
def ndtri_approx_kernel(
    P_ptr,           # *float32, length 1 (scalar sparsity p in [0,1])
    OUT_ptr,         # *float32, length 1 (output z-score)
):
    # Load sparsity p
    p = tl.load(P_ptr)

    # Abramowitz & Stegun 5.2.23 approximation (piecewise)
    p_low = 0.02425
    p_high = 1.0 - p_low

    c1 = -7.784695709041462e-03
    c2 = 3.224671290700398e-01
    c3 = 2.445134137142996e+00
    c4 = 3.754408661907416e+00

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

    # Lower region
    t_low = tl.sqrt(-2.0 * tl.log(p))
    y_low = (((((c1 * t_low + c2) * t_low + c3) * t_low + c4) * t_low + c5) * t_low + c6) / \
            ((((d1 * t_low + d2) * t_low + d3) * t_low + d4) * t_low + 1.0)

    # Central region
    q = p - 0.5
    r = q * q
    y_mid = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    # Upper region
    t_up = tl.sqrt(-2.0 * tl.log(1.0 - p))
    y_up = -(((((c1 * t_up + c2) * t_up + c3) * t_up + c4) * t_up + c5) * t_up + c6) / \
           ((((d1 * t_up + d2) * t_up + d3) * t_up + d4) * t_up + 1.0)

    res = tl.where(p < p_low, y_low, tl.where(p > p_high, y_up, y_mid))
    tl.store(OUT_ptr, res)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float):
        super().__init__()
        self.target_sparsity = float(target_sparsity)

    def forward(self, inputs: torch.Tensor):
        # Ensure CUDA tensor
        assert inputs.is_cuda, "ModelNew requires a CUDA tensor input"
        # Expect shape [B, S, D]
        assert inputs.ndim == 3, "Input must be a 3D tensor [batch_size, seq_len, intermediate_size]"
        B, S, D = inputs.shape

        # Convert to float32 for computation
        X = inputs.to(torch.float32)

        # Allocate buffers for sums, mean, std
        sums = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        sumsq = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        mean = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        std = torch.empty(B * S, dtype=torch.float32, device=inputs.device)

        # Launch reduction kernel: one program per (b, s) row
        BLOCK_SIZE = 1024
        grid_reduce = (B * S,)
        reduce_mean_std_kernel[grid_reduce](
            X, sums, sumsq, B, S, D,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )

        # Compute mean and std
        compute_mean_std_kernel[(B * S,)](
            sums, sumsq, mean, std, D,
            num_warps=4,
        )

        # Compute z-score via Triton kernel (single scalar)
        p_buf = torch.tensor([self.target_sparsity], dtype=torch.float32, device=inputs.device)
        z_score = torch.empty(1, dtype=torch.float32, device=inputs.device)
        ndtri_approx_kernel[(1,)](
            p_buf, z_score,
            num_warps=4,
        )

        # Apply activation and store as float32; cast to bfloat16 after
        OUT_fp32 = torch.empty(B, S, D, dtype=torch.float32, device=inputs.device)
        grid_apply = (B, S, triton.cdiv(D, BLOCK_SIZE))
        apply_activation_kernel[grid_apply](
            X, mean, std, z_score, OUT_fp32, B, S, D,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )

        # Cast to bfloat16 to match original output dtype
        return OUT_fp32.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
