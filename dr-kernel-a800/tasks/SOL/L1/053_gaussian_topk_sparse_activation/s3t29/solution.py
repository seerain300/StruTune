import torch
import triton
import triton.language as tl


@triton.jit
def compute_q_kernel(q_ptr, target: tl.float32):
    # Compute q = ndtri(target) using Abramowitz & Stegun 7.1.26 approximation.
    # q is inverse standard normal CDF: P(Phi < q) = target, for target in (0, 1).
    # Implement on the device without torch ops.
    # Constants:
    p = target  # in (0, 1)

    # 7.1.26 constants (A&S)
    p_low = 0.02425
    p_high = 1.0 - p_low
    p_low_safe = 1e-7  # small safety to avoid log(0)
    p = tl.maximum(p, p_low_safe)

    # Lower region
    p_region = p < p_low
    q_low = tl.sqrt(-2.0 * tl.log(p))  # sqrt(-2 ln p)
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

    t_low = 1.0 / (1.0 + 0.319381530 * q_low)
    poly_low = a1 * t_low + a2
    poly_low = poly_low * t_low + a3
    poly_low = poly_low * t_low + a4
    poly_low = poly_low * t_low + a5
    poly_low = poly_low * t_low + a6

    t_low2 = b1 * t_low + b2
    t_low2 = t_low2 * t_low + b3
    t_low2 = t_low2 * t_low + b4
    t_low2 = t_low2 * t_low + b5
    approx_low = poly_low / t_low2

    # Upper region
    p_region2 = p > p_high
    # For upper region, compute z = sqrt(-2 ln(1 - p))
    one = 1.0
    p_upper = one - p
    # Use log of max(p_upper, eps)
    p_upper = tl.maximum(p_upper, 1e-7)
    q_up = tl.sqrt(-2.0 * tl.log(p_upper))
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

    t_up = 1.0 / (1.0 + 0.2316419 * q_up)
    poly_up = c1 * t_up + c2
    poly_up = poly_up * t_up + c3
    poly_up = poly_up * t_up + c4
    poly_up = poly_up * t_up + c5
    poly_up = poly_up * t_up + c6

    t_up2 = d1 * t_up + d2
    t_up2 = t_up2 * t_up + d3
    t_up2 = t_up2 * t_up + d4
    t_up2 = t_up2 * t_up + 1.0
    approx_up = -poly_up / t_up2

    # Combine regions
    # If p <= low: use approx_low; else if p >= high: use approx_up; else default to approx_low
    # Using masks: q = where(p < p_low, approx_low, where(p > p_high, approx_up, approx_low))
    q_approx = tl.where(p_region, approx_low, tl.where(p_region2, approx_up, approx_low))

    # Store q
    tl.store(q_ptr, q_approx)


@triton.jit
def sum_per_feature_kernel(x_ptr, sum_ptr,
                            B, S, L,
                            CHAN_N: tl.constexpr):
    # One program per feature
    f = tl.program_id(0)
    rows = B * S
    acc = 0.0
    # Iterate rows in chunks of CHAN_N
    for chunk in tl.static_range(0, rows, CHAN_N):
        offs = chunk + tl.arange(0, CHAN_N)
        mask = offs < rows
        b = offs // S
        s = offs % S
        idx = b * L + s * L + f
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    tl.store(sum_ptr + f, acc)


@triton.jit
def sumsq_per_feature_kernel(x_ptr, sumsq_ptr,
                             B, S, L,
                             CHAN_N: tl.constexpr):
    # One program per feature
    f = tl.program_id(0)
    rows = B * S
    acc = 0.0
    for chunk in tl.static_range(0, rows, CHAN_N):
        offs = chunk + tl.arange(0, CHAN_N)
        mask = offs < rows
        b = offs // S
        s = offs % S
        idx = b * L + s * L + f
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
        acc += tl.sum(vals * vals, axis=0)
    tl.store(sumsq_ptr + f, acc)


@triton.jit
def mean_std_threshold_per_feature_kernel(sum_ptr, sumsq_ptr, threshold_ptr,
                                          q_ptr, B, S, L):
    f = tl.program_id(0)
    sum_f = tl.load(sum_ptr + f)
    sumsq_f = tl.load(sumsq_ptr + f)
    q_val = tl.load(q_ptr)  # scalar
    mean = sum_f / (B * S)
    var = sumsq_f / (B * S) - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    thresh = mean + std * q_val
    tl.store(threshold_ptr + f, thresh)


@triton.jit
def sparse_relu_per_feature_kernel(x_ptr, threshold_ptr, out_ptr,
                                   B, S, L):
    # 2D grid: pid_row over B*S, pid_f over L
    pid_row = tl.program_id(0)
    pid_f = tl.program_id(1)
    mask_row = pid_row < B * S
    mask_f = pid_f < L
    if not (mask_row and mask_f):
        return
    b = pid_row // S
    s = pid_row % S
    idx = b * L + s * L + pid_f
    x_val = tl.load(x_ptr + idx)
    thresh = tl.load(threshold_ptr + pid_f)
    y = x_val - thresh
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + idx, y)


@triton.jit
def cast_bf16_kernel(in_ptr, out_ptr, N, BLOCK: tl.constexpr):
    # Cast FP32 in_ptr to BF16 out_ptr elementwise
    for start in tl.static_range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        vals = tl.load(in_ptr + offs, mask=mask, other=0.0)
        # Triton doesn't have explicit bf16 cast helper; represent bf16 as fp32 and rely on store type.
        # We cast manually by reducing to fp16 then fp32 equivalent (Triton will handle appropriate type).
        # Simpler: Triton will store fp32 to bf16 pointer as bf16. Just store vals.
        tl.store(out_ptr + offs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # Ensure contiguous and compute in FP32 for accuracy
        x = inputs.contiguous()
        if x.dtype != torch.float32:
            x = x.float()
        B, S, L = x.shape
        rows = B * S

        # 1) Compute q = ndtri(target_sparsity) in Triton (no torch ops in forward)
        q_buf = torch.empty(1, dtype=torch.float32, device=x.device)
        compute_q_kernel[(1,)](q_buf, target_sparsity)

        # 2) Allocate per-feature accumulators (FP32)
        sum_f = torch.empty(L, dtype=torch.float32, device=x.device)
        sumsq_f = torch.empty(L, dtype=torch.float32, device=x.device)
        threshold_f = torch.empty(L, dtype=torch.float32, device=x.device)

        # 3) Compute per-feature sums and sumsq
        # Flatten (B,S) into a single "rows" index for kernel convenience
        # We'll pass B and S, and the kernel will map rows -> (b,s)
        CHAN_N = 128  # chunk size per loop; can tune
        sum_per_feature_kernel[(L,)](x, sum_f, B, S, L, CHAN_N, num_warps=4)
        sumsq_per_feature_kernel[(L,)](x, sumsq_f, B, S, L, CHAN_N, num_warps=4)

        # 4) Compute mean, std, and threshold per feature (Triton)
        mean_std_threshold_per_feature_kernel[(L,)](sum_f, sumsq_f, threshold_f, q_buf, B, S, L, num_warps=1)

        # 5) Apply sparse ReLU elementwise with per-feature threshold
        # Create a FP32 output buffer and launch 2D kernel
        out_fp32 = torch.empty(B * S * L, dtype=torch.float32, device=x.device)
        grid = (rows, L)
        sparse_relu_per_feature_kernel[grid](x.view(-1), threshold_f, out_fp32, B, S, L, num_warps=4)

        # 6) Cast to bfloat16 via Triton (forward MUST invoke this kernel)
        out_bf16 = torch.empty(B * S * L, dtype=torch.bfloat16, device=x.device)
        BLOCK_CAST = 4096
        grid_cast = (triton.cdiv(B * S * L, BLOCK_CAST),)
        cast_bf16_kernel[grid_cast](out_fp32, out_bf16, B * S * L, BLOCK=BLOCK_CAST, num_warps=4)

        # 7) Reshape back to [B, S, L]
        out = out_bf16.view(B, S, L)
        return out


def run(*args):
    return ModelNew()(*args)
