import math

import torch
import triton
import triton.language as tl


def _ndtri_scalar(p: float) -> float:
    """Inverse standard-normal CDF (quantile), matching the reference _ndtri.

    Port of the Abramowitz & Stegun 26.2.23 rational approximation used by the
    task reference, evaluated in host fp64 then narrowed to fp32 at the launch
    boundary. Since ``target_sparsity`` is a scalar the resulting z-multiplier
    is shared by every row, so this is computed once on the host rather than on
    the GPU.
    """
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

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
               (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)


@triton.jit
def _gaussian_topk_kernel(
    x_ptr, out_ptr,
    stride_m,
    z,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * stride_m

    # -- Pass 1: per-row statistics (fp32 accumulators, one-pass mean/var) --
    acc_sum = 0.0
    acc_sq = 0.0
    for start in range(0, H, BLOCK_H):
        offs = start + tl.arange(0, BLOCK_H)
        mask = offs < H
        vals = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0).to(tl.float32)
        acc_sum += tl.sum(vals, axis=0)
        acc_sq += tl.sum(vals * vals, axis=0)

    mean = acc_sum / H
    var = acc_sq / H - mean * mean
    var = tl.maximum(var, 0.0)          # guard tiny-negative round-off before sqrt
    std = tl.sqrt(var)
    cutoff = mean + std * z

    # -- Pass 2: apply adaptive ReLU threshold, store bf16 (round-to-nearest) --
    for start in range(0, H, BLOCK_H):
        offs = start + tl.arange(0, BLOCK_H)
        mask = offs < H
        vals = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0).to(tl.float32)
        res = tl.maximum(vals - cutoff, 0.0)
        tl.store(out_ptr + row_start + offs, res.to(tl.bfloat16), mask=mask)


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """Gaussian-based top-k sparse activation (fused Triton kernel).

    out[b,s,h] = relu(x - (mean_row + std_row * ndtri(target_sparsity)))
    with statistics over the contiguous last dim, computed in fp32.
    """
    # Early return: no sparsity requested -> identity (matches reference).
    if target_sparsity == 0.0:
        return inputs

    H = inputs.shape[-1]
    # View as [M, H]; reshape is a no-op view for contiguous input.
    x = inputs.reshape(-1, H)
    if not x.is_contiguous():
        x = x.contiguous()
    M = x.shape[0]

    out = torch.empty_like(x)

    z = _ndtri_scalar(float(target_sparsity))

    BLOCK_H = 1024
    grid = (M,)
    _gaussian_topk_kernel[grid](
        x, out,
        x.stride(0),
        z,
        H=H,
        BLOCK_H=BLOCK_H,
        num_warps=8,
        num_stages=2,
    )

    return out.reshape(inputs.shape)
