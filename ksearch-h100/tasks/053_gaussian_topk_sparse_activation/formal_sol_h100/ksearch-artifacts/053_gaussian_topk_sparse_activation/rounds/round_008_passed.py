# solution=GPT-5.6-Sol_053_gaussian_topk_sparse_activation_triton_optimized_r8 score=47.698618090933444 passed=True
import math
import struct
from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _gaussian_sparse_activation_kernel(
    inputs_ptr,
    output_ptr,
    n_cols: tl.constexpr,
    std_multiplier,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_base = row_idx * n_cols

    if n_cols == 12288:
        offsets0 = tl.arange(0, 8192)
        offsets1 = tl.arange(0, 4096) + 8192

        values0 = tl.load(inputs_ptr + row_base + offsets0).to(tl.float32)
        sum0 = tl.sum(values0, axis=0)
        square_sum0 = tl.sum(values0 * values0, axis=0)

        values1 = tl.load(inputs_ptr + row_base + offsets1).to(tl.float32)
        sum1 = tl.sum(values1, axis=0)
        square_sum1 = tl.sum(values1 * values1, axis=0)

        inv_n = 1.0 / n_cols
        mean = (sum0 + sum1) * inv_n
        mean_square = (square_sum0 + square_sum1) * inv_n
        variance = tl.maximum(mean_square - mean * mean, 0.0)
        cutoff = mean + tl.sqrt(variance) * std_multiplier

        values0 = tl.load(inputs_ptr + row_base + offsets0).to(tl.float32)
        activated0 = tl.maximum(values0 - cutoff, 0.0)
        tl.store(output_ptr + row_base + offsets0, activated0)

        values1 = tl.load(inputs_ptr + row_base + offsets1).to(tl.float32)
        activated1 = tl.maximum(values1 - cutoff, 0.0)
        tl.store(output_ptr + row_base + offsets1, activated1)

    elif n_cols == 16384:
        offsets0 = tl.arange(0, 8192)
        offsets1 = tl.arange(0, 8192) + 8192

        values0 = tl.load(inputs_ptr + row_base + offsets0).to(tl.float32)
        sum0 = tl.sum(values0, axis=0)
        square_sum0 = tl.sum(values0 * values0, axis=0)

        values1 = tl.load(inputs_ptr + row_base + offsets1).to(tl.float32)
        sum1 = tl.sum(values1, axis=0)
        square_sum1 = tl.sum(values1 * values1, axis=0)

        inv_n = 1.0 / n_cols
        mean = (sum0 + sum1) * inv_n
        mean_square = (square_sum0 + square_sum1) * inv_n
        variance = tl.maximum(mean_square - mean * mean, 0.0)
        cutoff = mean + tl.sqrt(variance) * std_multiplier

        values0 = tl.load(inputs_ptr + row_base + offsets0).to(tl.float32)
        activated0 = tl.maximum(values0 - cutoff, 0.0)
        tl.store(output_ptr + row_base + offsets0, activated0)

        values1 = tl.load(inputs_ptr + row_base + offsets1).to(tl.float32)
        activated1 = tl.maximum(values1 - cutoff, 0.0)
        tl.store(output_ptr + row_base + offsets1, activated1)

    else:
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        row_offsets = row_base + offsets

        values = tl.load(
            inputs_ptr + row_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        inv_n = 1.0 / n_cols
        mean = tl.sum(values, axis=0) * inv_n
        mean_square = tl.sum(values * values, axis=0) * inv_n
        variance = tl.maximum(mean_square - mean * mean, 0.0)
        cutoff = mean + tl.sqrt(variance) * std_multiplier

        values = tl.load(
            inputs_ptr + row_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        activated = tl.maximum(values - cutoff, 0.0)
        tl.store(output_ptr + row_offsets, activated, mask=mask)


def _to_float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


@lru_cache(maxsize=32)
def _ndtri_scalar(p: float) -> float:
    p = _to_float32(p)

    if not math.isfinite(p) or p <= 0.0 or p >= 1.0:
        return float("nan")

    a1 = -3.969683028665376e01
    a2 = 2.209460984245205e02
    a3 = -2.759285104469687e02
    a4 = 1.383577518672690e02
    a5 = -3.066479806614716e01
    a6 = 2.506628277459239e00

    b1 = -5.447609879822406e01
    b2 = 1.615858368580409e02
    b3 = -1.556989798598866e02
    b4 = 6.680131188771972e01
    b5 = -1.328068155288572e01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580700398e-01
    c3 = -2.400758277161838e00
    c4 = -2.549732539343734e00
    c5 = 4.374664141464968e00
    c6 = 2.938163982698783e00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e00
    d4 = 3.754408661907416e00

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        )

    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6)
            * q
            / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        )

    q = math.sqrt(-2.0 * math.log1p(-p))
    return -(
        (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    )


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    p = float(target_sparsity)
    if p == 0.0:
        return inputs

    n_cols = inputs.shape[-1]
    n_rows = inputs.numel() // n_cols
    output = torch.empty_like(inputs)

    block_size = triton.next_power_of_2(n_cols)
    std_multiplier = _ndtri_scalar(p)

    _gaussian_sparse_activation_kernel[(n_rows,)](
        inputs,
        output,
        n_cols=n_cols,
        std_multiplier=std_multiplier,
        BLOCK_SIZE=block_size,
        num_warps=8,
        num_stages=1,
    )
    return output