# solution=GPT-5.6-Sol_053_gaussian_topk_sparse_activation_triton_optimized_r1 score=-1.0 passed=False
import math
import struct

import torch
import triton
import triton.language as tl


@triton.jit
def _row_statistics_kernel(
    inputs_ptr,
    statistics_ptr,
    feature_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < feature_size

    values = tl.load(
        inputs_ptr + row * feature_size + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    inv_feature_size = 1.0 / feature_size
    mean = tl.sum(values, axis=0) * inv_feature_size

    centered = tl.where(mask, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) * inv_feature_size
    std = tl.sqrt(tl.maximum(variance, 0.0))

    tl.store(statistics_ptr + row * 2, mean)
    tl.store(statistics_ptr + row * 2 + 1, std)


@triton.jit
def _sparse_activation_kernel(
    inputs_ptr,
    statistics_ptr,
    output_ptr,
    std_multiplier,
    feature_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    feature_block = tl.program_id(0)
    row = tl.program_id(1)

    offsets = feature_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < feature_size

    mean = tl.load(statistics_ptr + row * 2)
    std = tl.load(statistics_ptr + row * 2 + 1)
    threshold = mean + std * std_multiplier

    values = tl.load(
        inputs_ptr + row * feature_size + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    activated = tl.maximum(values - threshold, 0.0)
    tl.store(
        output_ptr + row * feature_size + offsets,
        activated,
        mask=mask,
    )


def _to_float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _ndtri_scalar(value: float) -> float:
    p = _to_float32(value)

    if p <= 0.0 or p >= 1.0:
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
    c2 = -3.223964580411365e-01
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
        numerator = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        denominator = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        return numerator / denominator

    if p <= p_high:
        q = p - 0.5
        r = q * q
        numerator = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
        denominator = (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        return numerator / denominator

    q = math.sqrt(-2.0 * math.log(1.0 - p))
    numerator = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
    denominator = ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    return -(numerator / denominator)


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    target_sparsity_value = float(target_sparsity)

    if target_sparsity_value == 0.0:
        return inputs

    batch_size, seq_len, feature_size = inputs.shape
    row_count = batch_size * seq_len

    statistics = torch.empty(
        (row_count, 2),
        device=inputs.device,
        dtype=torch.float32,
    )
    output = torch.empty_like(inputs)

    statistics_block_size = triton.next_power_of_2(feature_size)
    statistics_num_warps = 8 if statistics_block_size >= 4096 else 4

    _row_statistics_kernel[(row_count,)](
        inputs,
        statistics,
        feature_size=feature_size,
        BLOCK_SIZE=statistics_block_size,
        num_warps=statistics_num_warps,
    )

    std_multiplier = _ndtri_scalar(target_sparsity_value)
    activation_block_size = 1024
    activation_grid = (
        triton.cdiv(feature_size, activation_block_size),
        row_count,
    )

    _sparse_activation_kernel[activation_grid](
        inputs,
        statistics,
        output,
        std_multiplier,
        feature_size=feature_size,
        BLOCK_SIZE=activation_block_size,
        num_warps=8,
    )

    return output