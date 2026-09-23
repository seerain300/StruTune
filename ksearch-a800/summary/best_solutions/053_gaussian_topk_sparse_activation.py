# task: 053_gaussian_topk_sparse_activation
# bench: SOL-L1 | batch: formal_20260914
# final eval (official evaluator, full workloads): valid=True pass=12/12 geomean=31.807x
# feedback best (5-workload sample during search): 33.329x
# torch fallback audit: 干净 (-)
# tokens: 1,506,558

import math
import struct

import torch
import triton
import triton.language as tl


def _float32(value):
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def _ndtri_scalar(p):
    p = _float32(p)

    if math.isnan(p):
        return 0.0
    if p == 0.0:
        return -math.inf
    if p == 1.0:
        return math.inf

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
        result = (
            (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        result = (
            (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
            / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        result = -(
            (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
            / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
        )

    return _float32(result)


@triton.jit
def _gaussian_sparse_kernel(
    inputs,
    output,
    std_multiplier: tl.constexpr,
    feature_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    REUSE_VALUES: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < feature_size
    row_start = row * feature_size
    input_ptrs = inputs + row_start + offsets

    if REUSE_VALUES:
        values = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
    else:
        values = tl.load(
            input_ptrs,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)

    row_sum = tl.sum(values, axis=0)
    row_square_sum = tl.sum(values * values, axis=0)

    inv_size = 1.0 / feature_size
    mean = row_sum * inv_size
    variance = row_square_sum * inv_size - mean * mean
    variance = tl.maximum(variance, 0.0)
    threshold = mean + tl.sqrt(variance) * std_multiplier

    if REUSE_VALUES:
        values_out = values
    else:
        values_out = tl.load(
            input_ptrs,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

    tl.store(
        output + row_start + offsets,
        tl.maximum(values_out - threshold, 0.0),
        mask=mask,
    )


@triton.jit
def _gaussian_sparse_4096_kernel(
    inputs,
    output,
    std_multiplier: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    row_start = row * 4096

    ptr0 = inputs + row_start + offsets
    ptr1 = ptr0 + BLOCK_SIZE

    values0 = tl.load(ptr0, eviction_policy="evict_last").to(tl.float32)
    values1 = tl.load(ptr1, eviction_policy="evict_last")

    values1_f32 = values1.to(tl.float32)
    row_sum = tl.sum(values0 + values1_f32, axis=0)
    row_square_sum = tl.sum(
        values0 * values0 + values1_f32 * values1_f32,
        axis=0,
    )

    mean = row_sum * (1.0 / 4096.0)
    variance = row_square_sum * (1.0 / 4096.0) - mean * mean
    variance = tl.maximum(variance, 0.0)
    threshold = mean + tl.sqrt(variance) * std_multiplier

    tl.store(
        output + row_start + offsets,
        tl.maximum(values0 - threshold, 0.0),
    )
    tl.store(
        output + row_start + BLOCK_SIZE + offsets,
        tl.maximum(values1.to(tl.float32) - threshold, 0.0),
    )


@triton.jit
def _gaussian_sparse_8192_kernel(
    inputs,
    output,
    std_multiplier: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    row_start = row * 8192

    ptr0 = inputs + row_start + offsets
    ptr1 = ptr0 + BLOCK_SIZE

    values0 = tl.load(ptr0).to(tl.float32)
    values1 = tl.load(ptr1, eviction_policy="evict_last")

    row_sum = tl.sum(
        values0 + values1.to(tl.float32),
        axis=0,
    )
    row_square_sum = tl.sum(
        values0 * values0
        + values1.to(tl.float32) * values1.to(tl.float32),
        axis=0,
    )

    mean = row_sum * (1.0 / 8192.0)
    variance = row_square_sum * (1.0 / 8192.0) - mean * mean
    variance = tl.maximum(variance, 0.0)
    threshold = mean + tl.sqrt(variance) * std_multiplier

    tl.store(
        output + row_start + offsets,
        tl.maximum(values0 - threshold, 0.0),
    )
    tl.store(
        output + row_start + BLOCK_SIZE + offsets,
        tl.maximum(values1.to(tl.float32) - threshold, 0.0),
    )


@triton.jit
def _gaussian_sparse_12288_kernel(
    inputs,
    output,
    std_multiplier: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    row_start = row * 12288

    ptr0 = inputs + row_start + offsets
    ptr1 = ptr0 + BLOCK_SIZE
    ptr2 = ptr1 + BLOCK_SIZE

    values0 = tl.load(ptr0).to(tl.float32)
    values1 = tl.load(ptr1)
    values2 = tl.load(ptr2, eviction_policy="evict_last")

    row_sum = tl.sum(
        values0
        + values1.to(tl.float32)
        + values2.to(tl.float32),
        axis=0,
    )
    row_square_sum = tl.sum(
        values0 * values0
        + values1.to(tl.float32) * values1.to(tl.float32)
        + values2.to(tl.float32) * values2.to(tl.float32),
        axis=0,
    )

    mean = row_sum * (1.0 / 12288.0)
    variance = row_square_sum * (1.0 / 12288.0) - mean * mean
    variance = tl.maximum(variance, 0.0)
    threshold = mean + tl.sqrt(variance) * std_multiplier

    tl.store(
        output + row_start + offsets,
        tl.maximum(values0 - threshold, 0.0),
    )
    tl.store(
        output + row_start + BLOCK_SIZE + offsets,
        tl.maximum(values1.to(tl.float32) - threshold, 0.0),
    )
    tl.store(
        output + row_start + 2 * BLOCK_SIZE + offsets,
        tl.maximum(values2.to(tl.float32) - threshold, 0.0),
    )


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    if float(target_sparsity) == 0.0:
        return inputs

    feature_size = inputs.shape[-1]
    output = torch.empty_like(inputs)

    if inputs.numel() == 0:
        return output

    std_multiplier = _ndtri_scalar(target_sparsity)
    rows = inputs.numel() // feature_size

    if feature_size == 4096:
        _gaussian_sparse_4096_kernel[(rows,)](
            inputs,
            output,
            std_multiplier=std_multiplier,
            BLOCK_SIZE=2048,
            num_warps=4,
            num_stages=1,
        )
        return output

    if feature_size == 8192:
        _gaussian_sparse_8192_kernel[(rows,)](
            inputs,
            output,
            std_multiplier=std_multiplier,
            BLOCK_SIZE=4096,
            num_warps=8,
            num_stages=1,
        )
        return output

    if feature_size == 12288:
        _gaussian_sparse_12288_kernel[(rows,)](
            inputs,
            output,
            std_multiplier=std_multiplier,
            BLOCK_SIZE=4096,
            num_warps=16,
            num_stages=1,
        )
        return output

    block_size = triton.next_power_of_2(feature_size)

    if block_size <= 128:
        num_warps = 1
    elif block_size <= 512:
        num_warps = 2
    elif block_size <= 4096:
        num_warps = 4
    elif block_size <= 8192:
        num_warps = 8
    else:
        num_warps = 16

    reuse_values = block_size <= 32

    _gaussian_sparse_kernel[(rows,)](
        inputs,
        output,
        std_multiplier=std_multiplier,
        feature_size=feature_size,
        BLOCK_SIZE=block_size,
        REUSE_VALUES=reuse_values,
        num_warps=num_warps,
        num_stages=1,
    )
    return output