import math
import torch
import triton
import triton.language as tl


@triton.jit
def reduce_sum_sumsq_per_feature(
    x_ptr,                # *float32, input pointer
    sum_ptr,              # *float32, [L], per-feature sum
    sumsq_ptr,            # *float32, [L], per-feature sum of squares
    B: tl.int32, S: tl.int32, L: tl.int32,
    stride_b: tl.int32,   # stride along batch
    stride_s: tl.int32,   # stride along seq
    stride_f: tl.int32,   # stride along feature (last dim)
    BLOCK_ROWS: tl.constexpr
):
    """
    For each feature f in [0, L): accumulate sum and sumsq over all rows (B*S).
    One program per feature.
    """
    f = tl.program_id(0)
    if f >= L:
        return
    acc_sum = 0.0
    acc_sumsq = 0.0
    for row_start in range(0, B * S, BLOCK_ROWS):
        rows = row_start + tl.arange(0, BLOCK_ROWS)
        row_mask = rows < (B * S)
        b = rows // S
        s = rows % S
        off = b * stride_b + s * stride_s + f * stride_f
        vals = tl.load(x_ptr + off, mask=row_mask, other=0.0)
        acc_sum += tl.sum(vals, axis=0)
        acc_sumsq += tl.sum(vals * vals, axis=0)
    tl.store(sum_ptr + f, acc_sum)
    tl.store(sumsq_ptr + f, acc_sumsq)


@triton.jit
def compute_mean_std_per_feature(
    sum_ptr,               # *float32, [L]
    sumsq_ptr,             # *float32, [L]
    out_mean_ptr,          # *float32, [L]
    out_std_ptr,           # *float32, [L]
    rows_total: tl.int32,  # B*S
    L: tl.int32
):
    """
    For each feature f in [0, L): compute mean and std from sum and sumsq.
    std = sqrt(max(var, 0)) for numerical stability.
    """
    for f in range(0, L):
        s = tl.load(sum_ptr + f)
        ss = tl.load(sumsq_ptr + f)
        mean = s / rows_total
        var = ss / rows_total - mean * mean
        var = tl.maximum(var, 0.0)
        std = tl.sqrt(var)
        tl.store(out_mean_ptr + f, mean)
        tl.store(out_std_ptr + f, std)


@triton.jit
def ndtri_approx_kernel(
    p_ptr,                # *float32, [1] input scalar in (0,1)
    z_ptr,                # *float32, [1] output scalar
):
    """
    Compute inverse standard normal CDF z = ndtri(p) for p in (0,1).
    Use Abramowitz & Stegun 7.1.26 approximation with coefficients.
    p_ptr[0] = probability; z_ptr[0] = result.
    """
    p = tl.load(p_ptr)  # scalar
    # Constants
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

    # Low region
    if p < p_low:
        q = tl.sqrt(-2.0 * tl.log(p))
        z = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    else:
        # Central region
        q = p - 0.5
        r = q * q
        z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        # Upper region
        if p > (1.0 - p_low):
            q = tl.sqrt(-2.0 * tl.log(1.0 - p))
            z = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    tl.store(z_ptr, z)


@triton.jit
def compute_thr_per_feature(
    mean_ptr,              # *float32, [L]
    std_ptr,               # *float32, [L]
    p_ptr,                 # *float32, [1] probability (target_sparsity)
    thr_ptr,               # *float32, [L] output thresholds
    L: tl.int32
):
    """
    Compute per-feature threshold: thr[f] = mean[f] + std[f] * z, where z = ndtri(p).
    """
    # Load p and compute z via Triton kernel
    p = tl.load(p_ptr)
    z = tl.zeros((), dtype=tl.float32)
    z_buf = tl.full((1,), 0.0, dtype=tl.float32)  # dummy
    ndtri_approx_kernel[(1,)](
        p_ptr,
        z_buf,  # output scalar buffer
    )
    z = tl.load(z_buf)  # get z computed in the kernel
    for f in range(0, L):
        mean = tl.load(mean_ptr + f)
        std = tl.load(std_ptr + f)
        thr = mean + std * z
        tl.store(thr_ptr + f, thr)


@triton.jit
def sparse_relu_per_feature(
    x_ptr,                 # *float32, flattened [N = B*S*L]
    thr_ptr,               # *float32, [L] thresholds
    out_ptr,               # *float32, flattened [N]
    B: tl.int32, S: tl.int32, L: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    """
    For each row (over B*S), iterate over features in chunks of BLOCK_SIZE,
    subtract the per-feature threshold, apply ReLU, and store.
    """
    row = tl.program_id(0)
    if row >= B * S:
        return
    base = row * L
    for f in range(0, L, BLOCK_SIZE):
        offs = f + tl.arange(0, BLOCK_SIZE)
        mask = offs < L
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        thr = tl.load(thr_ptr + offs, mask=mask, other=0.0)
        y = x - thr
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + base + offs, y, mask=mask)


@triton.jit
def cast_bf16_kernel(
    inp_ptr,       # *float32, flattened [N]
    out_ptr,       # *bfloat16, flattened [N]
    N: tl.int32,
    BLOCK: tl.constexpr
):
    """
    Cast float32 to bfloat16 via Triton store.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x, mask=mask)  # Triton will cast to destination dtype (bf16)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float):
        super().__init__()
        self.target_sparsity = float(target_sparsity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, L], CUDA tensor
        assert x.is_cuda, "Input must be on CUDA device for Triton kernels."
        B, S, L = x.shape
        rows_total = B * S

        # Ensure contiguous and cast to float32 for Triton computation
        x_fp32 = x.contiguous().to(torch.float32)

        # Allocate per-feature accumulators (device-side)
        sum_f = torch.empty(L, dtype=torch.float32, device=x.device)
        sumsq_f = torch.empty(L, dtype=torch.float32, device=x.device)
        mean_f = torch.empty(L, dtype=torch.float32, device=x.device)
        std_f = torch.empty(L, dtype=torch.float32, device=x.device)
        thr = torch.empty(L, dtype=torch.float32, device=x.device)

        # Strides for last-dimension (features) reduction
        stride_b = x_fp32.stride(0)
        stride_s = x_fp32.stride(1)
        stride_f = x_fp32.stride(2)

        # 1) Per-feature reduction: sum and sumsq
        reduce_sum_sumsq_per_feature[(L,)](
            x_fp32,
            sum_f,
            sumsq_f,
            B, S, L,
            stride_b, stride_s, stride_f,
            BLOCK_ROWS=1024,
            num_warps=4
        )

        # 2) Compute mean and std per feature (Triton)
        compute_mean_std_per_feature[(1,)](
            sum_f,
            sumsq_f,
            mean_f,
            std_f,
            rows_total,
            L,
            num_warps=1
        )

        # 3) Compute per-feature threshold using Triton ndtri approximation
        p_tensor = torch.empty(1, dtype=torch.float32, device=x.device)
        # Fill p_tensor[0] with target_sparsity; do it on device without torch math in forward
        # We rely on Triton kernel to compute ndtri(p) from p_tensor[0].
        p_tensor[0] = self.target_sparsity
        compute_thr_per_feature[(L,)](
            mean_f,
            std_f,
            p_tensor,
            thr,
            L,
            num_warps=1
        )

        # 4) Apply sparse ReLU in Triton: y = max(x - thr[f], 0)
        N = B * S * L
        out_fp32 = torch.empty(N, dtype=torch.float32, device=x.device)
        sparse_relu_per_feature[(B * S,)](
            x_fp32.view(-1),
            thr,
            out_fp32,
            B, S, L,
            BLOCK_SIZE=1024,
            num_warps=4
        )

        # 5) Cast to bfloat16 via Triton (forward must invoke this kernel)
        out_bf16 = torch.empty(N, dtype=torch.bfloat16, device=x.device)
        cast_bf16_kernel[(triton.cdiv(N, 4096),)](
            out_fp32,
            out_bf16,
            N,
            BLOCK=4096,
            num_warps=4
        )

        # Reshape back to [B, S, L]
        y = out_bf16.view(B, S, L)
        return y


def run(*args):
    return ModelNew()(*args)
