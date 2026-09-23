# KDA A800 best solution: L1/053_gaussian_topk_sparse_activation
# candidate: c002  |  feedback: 24.93x  |  final (authoritative): 26.36x
# campaign formal-kda-20260916 (A800, g0056)  |  evaluations: 2
# source: tasks/formal-kda-20260916--sol_execbench--L1-053_gaussian_topk_sparse_activation/control/candidates/c002/solution.py (sha256-frozen snapshot)

import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Host-side scalar: inverse standard-normal CDF (quantile).
# Replicates the reference `_ndtri` (Abramowitz & Stegun 26.2.23) EXACTLY:
# same constants, same 3-region branch. Evaluated once per call in float
# precision and passed to the kernel as a single scalar (launch plumbing).
# This is NOT a computational fallback for the tensor op — it computes one
# data-independent scalar (std multiplier).
# ---------------------------------------------------------------------------
def _ndtri_scalar(p: float) -> float:
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


def _next_pow2(n: int) -> int:
    return 1 << (max(1, n) - 1).bit_length()


# ---------------------------------------------------------------------------
# Design A: one program per row. Load the whole row (BLOCK_H = next_pow2(H))
# once into registers, compute mean + population variance (divisor H) via an
# in-register two-pass, apply relu(x - (mean + std*Z)), store bf16.
# Traffic: 1 read + 1 write (memory-optimal for a fused reduction+map).
# ---------------------------------------------------------------------------
@triton.jit
def _gaussian_topk_kernel(
    in_ptr,          # *bf16  [M, H]
    out_ptr,         # *bf16  [M, H]
    H,               # int32  real row length
    Z,               # fp32   std multiplier = _ndtri(target_sparsity)
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    base = row * H

    # bf16 -> f32 load; padded lanes read 0.0 and contribute 0 to the sum.
    x = tl.load(in_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # Pass 1: mean over real lanes (padding contributes 0; divide by real H).
    mean = tl.sum(x, axis=0) / H

    # Pass 2: population variance; force padded lanes to 0 before squaring so
    # they never pollute the reduction.
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / H
    std = tl.sqrt(var)

    thr = mean + std * Z
    out = tl.maximum(x - thr, 0.0)

    tl.store(out_ptr + base + offs, out.to(tl.bfloat16), mask=mask)


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """Gaussian-based top-k sparse activation (fused Triton).

    output = relu(inputs_f32 - (row_mean + row_std * ndtri(target_sparsity)))
    with row stats over the last dim (population std, divisor H), bf16 output.
    """
    ts = float(target_sparsity)

    # Early exit: identity, bit-for-bit matches the reference (no launch).
    if ts == 0.0:
        return inputs

    orig_shape = inputs.shape
    H = orig_shape[-1]

    x2d = inputs.contiguous().view(-1, H)
    M = x2d.shape[0]
    out2d = torch.empty_like(x2d)

    if M == 0:
        return out2d.view(orig_shape)

    z = _ndtri_scalar(ts)

    BLOCK_H = _next_pow2(H)
    # c002: uniform num_warps=16. Raising the H<=8192 path from 8->16 halves the
    # per-thread element count (e.g. H=8192: 32->16 elems/thread), giving more
    # parallel loads in flight for the streaming reduction; H=12288 already used 16.
    num_warps = 16
    num_stages = 2

    grid = (M,)
    _gaussian_topk_kernel[grid](
        x2d,
        out2d,
        H,
        z,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return out2d.view(orig_shape)
