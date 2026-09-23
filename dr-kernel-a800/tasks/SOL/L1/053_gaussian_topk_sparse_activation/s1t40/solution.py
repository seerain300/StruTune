import torch
import triton
import triton.language as tl


@triton.jit
def reduce_sum_sumsq_kernel(
    X_ptr,           # *input tensor [B, S, D], contiguous
    SUM_ptr,         # *float32, length B*S
    SUMSQ_ptr,       # *float32, length B*S
    B, S, D          # int32 dimensions
):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    base = b * S * D + s * D

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    offset = 0
    while offset < D:
        offs = offset + tl.arange(0, 1024)
        mask = offs < D
        x = tl.load(X_ptr + base + offs, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        sum_val += tl.sum(x_f32, axis=0)
        sumsq_val += tl.sum(x_f32 * x_f32, axis=0)
        offset += 1024

    tl.store(SUM_ptr + pid, sum_val)
    tl.store(SUMSQ_ptr + pid, sumsq_val)


@triton.jit
def compute_mean_std_kernel(
    SUM_ptr,        # *float32, length B*S
    SUMSQ_ptr,      # *float32, length B*S
    MEAN_ptr,       # *float32, length B*S
    STD_ptr,        # *float32, length B*S
    B, S, D         # int32 dimensions
):
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    n = D  # last-dim length
    sum_val = tl.load(SUM_ptr + pid)
    sumsq_val = tl.load(SUMSQ_ptr + pid)

    mean = sum_val / n
    var = sumsq_val / n - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    tl.store(MEAN_ptr + pid, mean)
    tl.store(STD_ptr + pid, std)


@triton.jit
def ndtri_approx_kernel(
    P_ptr,          # *float32, length 1 (target sparsity as scalar)
    Z_ptr,          # *float32, length 1 output z-score
    p_low: tl.constexpr,   # 0.02425
    p_high: tl.constexpr   # 0.97575 (1.0 - p_low)
):
    p = tl.load(P_ptr)  # scalar float32

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

    mask_low = p < p_low
    mask_high = p > p_high
    mask_mid = ~mask_low & ~mask_high

    if mask_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        result = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif mask_high:
        q = tl.sqrt(-2.0 * tl.log(1.0 - p))
        result = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                 ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    else:
        q = p - 0.5
        r = q * q
        result = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
                 (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)

    tl.store(Z_ptr, result)


@triton.jit
def apply_activation_kernel(
    X_ptr,          # *input tensor [B, S, D], contiguous
    MEAN_ptr,       # *float32, length B*S
    STD_ptr,        # *float32, length B*S
    Z_ptr,          # *float32, length 1
    OUT_ptr,        # *float32, length B*S*D (intermediate for storing f32)
    B, S, D         # int32 dimensions
):
    pid_bs = tl.program_id(axis=0)
    tile = tl.program_id(axis=1)

    b = pid_bs // S
    s = pid_bs % S

    base = b * S * D + s * D

    d_start = tile * 1024
    offs = d_start + tl.arange(0, 1024)
    mask = offs < D

    x = tl.load(X_ptr + base + offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    mean = tl.load(MEAN_ptr + pid_bs)
    std = tl.load(STD_ptr + pid_bs)
    z_score = tl.load(Z_ptr)  # scalar z-score

    threshold = mean + std * z_score
    y = x_f32 - threshold
    y = tl.maximum(y, 0.0)  # ReLU

    tl.store(OUT_ptr + base + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure CUDA and contiguous input
        if not inputs.is_cuda:
            inputs = inputs.cuda()
        inputs = inputs.contiguous()

        B, S, D = inputs.shape

        # Device buffers for stats (float32)
        sum_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        sumsq_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        mean_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)
        std_buf = torch.empty(B * S, dtype=torch.float32, device=inputs.device)

        # z_score buffer (length 1)
        z_score_buf = torch.empty(1, dtype=torch.float32, device=inputs.device)

        # Fill sparsity target into a 1-element tensor without torch.tensor (using zeros + fill_)
        p_tensor = torch.empty(1, dtype=torch.float32, device=inputs.device)
        p_tensor.zero_()
        p_tensor.fill_(float(target_sparsity))

        # Launch reduction kernel
        grid_reduce = (B * S,)
        reduce_sum_sumsq_kernel[grid_reduce](
            inputs, sum_buf, sumsq_buf, B, S, D,
            num_warps=8, num_stages=4
        )

        # Compute mean and std
        compute_mean_std_kernel[grid_reduce](
            sum_buf, sumsq_buf, mean_buf, std_buf, B, S, D,
            num_warps=1, num_stages=1
        )

        # Compute z_score via Triton (inverse normal CDF for sparsity target)
        ndtri_approx_kernel[(1,)](
            p_tensor, z_score_buf, 0.02425, 0.97575,
            num_warps=1, num_stages=1
        )

        # Output buffer in float32 for activation; host will cast to bfloat16
        out_f32 = torch.empty(B * S * D, dtype=torch.float32, device=inputs.device)

        # Apply activation per (b, s) row in tiles over D
        grid_apply = (B * S, triton.cdiv(D, 1024))
        apply_activation_kernel[grid_apply](
            inputs, mean_buf, std_buf, z_score_buf, out_f32, B, S, D,
            num_warps=8, num_stages=4
        )

        # Cast to bfloat16 to match original behavior
        out_bf16 = out_f32.to(torch.bfloat16).view(B, S, D)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
