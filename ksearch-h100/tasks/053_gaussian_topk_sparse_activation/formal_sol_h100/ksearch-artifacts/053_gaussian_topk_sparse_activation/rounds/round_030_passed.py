# solution=GPT-5.6-Sol_053_gaussian_topk_sparse_activation_triton_optimized_r20 score=44.351701212316094 passed=True
import math

import torch
import triton
import triton.language as tl


def _ndtri_scalar(p: float) -> float:
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

    if p < 0.02425:
        if p == 0.0:
            return -math.inf
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / (
            (((d1 * q + d2) * q + d3) * q + d4) * q + 1.0
        )
    if p <= 0.97575:
        q = p - 0.5
        r = q * q
        return (
            (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
            / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
        )
    if p == 1.0:
        return math.inf
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(
        (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
        / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    )


@triton.jit
def _sum_pair_combine(sum_a, square_a, sum_b, square_b):
    return sum_a + sum_b, square_a + square_b


@triton.jit
def _gaussian_sparse_row_kernel(
    input_ptr,
    output_ptr,
    feature_size: tl.constexpr,
    std_multiplier: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < feature_size
    ptrs = input_ptr + row * feature_size + offsets
    values = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    inv_n = 1.0 / feature_size

    if std_multiplier == 0.0:
        cutoff = tl.sum(values, axis=0) * inv_n
    else:
        value_sum, square_sum = tl.reduce(
            (values, values * values),
            axis=0,
            combine_fn=_sum_pair_combine,
        )
        mean = value_sum * inv_n
        variance = tl.maximum(square_sum * inv_n - mean * mean, 0.0)
        cutoff = mean + tl.sqrt(variance) * std_multiplier

    tl.store(output_ptr + row * feature_size + offsets,
             tl.maximum(values - cutoff, 0.0), mask=mask)


@triton.jit
def _gaussian_sparse_row_grouped_kernel(
    input_ptr,
    output_ptr,
    row_count,
    feature_size: tl.constexpr,
    std_multiplier: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    feature_mask = offsets < feature_size
    inv_n = 1.0 / feature_size

    for ri in tl.static_range(0, ROWS_PER_PROGRAM):
        row = pid * ROWS_PER_PROGRAM + ri
        row_valid = row < row_count
        mask = feature_mask & row_valid
        ptrs = input_ptr + row * feature_size + offsets
        values = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

        if std_multiplier == 0.0:
            cutoff = tl.sum(values, axis=0) * inv_n
        else:
            value_sum, square_sum = tl.reduce(
                (values, values * values),
                axis=0,
                combine_fn=_sum_pair_combine,
            )
            mean = value_sum * inv_n
            variance = tl.maximum(square_sum * inv_n - mean * mean, 0.0)
            cutoff = mean + tl.sqrt(variance) * std_multiplier

        tl.store(
            output_ptr + row * feature_size + offsets,
            tl.maximum(values - cutoff, 0.0),
            mask=mask,
        )


@triton.jit
def _gaussian_sparse_row_12288_kernel(input_ptr, output_ptr, std_multiplier: tl.constexpr):
    row = tl.program_id(0)
    start = row * 12288
    o0 = tl.arange(0, 8192)
    o1 = tl.arange(0, 4096)
    v0 = tl.load(input_ptr + start + o0).to(tl.float32)
    v1 = tl.load(input_ptr + start + 8192 + o1).to(tl.float32)
    inv_n = 1.0 / 12288.0

    if std_multiplier == 0.0:
        cutoff = (tl.sum(v0, axis=0) + tl.sum(v1, axis=0)) * inv_n
    else:
        s0, q0 = tl.reduce((v0, v0 * v0), axis=0, combine_fn=_sum_pair_combine)
        s1, q1 = tl.reduce((v1, v1 * v1), axis=0, combine_fn=_sum_pair_combine)
        mean = (s0 + s1) * inv_n
        variance = tl.maximum((q0 + q1) * inv_n - mean * mean, 0.0)
        cutoff = mean + tl.sqrt(variance) * std_multiplier

    tl.store(output_ptr + start + o0, tl.maximum(v0 - cutoff, 0.0))
    tl.store(output_ptr + start + 8192 + o1, tl.maximum(v1 - cutoff, 0.0))


@triton.jit
def _gaussian_sparse_row_12288_grouped_kernel(
    input_ptr,
    output_ptr,
    row_count,
    std_multiplier: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    o0 = tl.arange(0, 8192)
    o1 = tl.arange(0, 4096)
    inv_n = 1.0 / 12288.0

    for ri in tl.static_range(0, ROWS_PER_PROGRAM):
        row = pid * ROWS_PER_PROGRAM + ri
        valid = row < row_count
        start = row * 12288
        v0 = tl.load(input_ptr + start + o0, mask=valid, other=0.0).to(tl.float32)
        v1 = tl.load(
            input_ptr + start + 8192 + o1, mask=valid, other=0.0
        ).to(tl.float32)

        if std_multiplier == 0.0:
            cutoff = (tl.sum(v0, axis=0) + tl.sum(v1, axis=0)) * inv_n
        else:
            s0, q0 = tl.reduce((v0, v0 * v0), axis=0, combine_fn=_sum_pair_combine)
            s1, q1 = tl.reduce((v1, v1 * v1), axis=0, combine_fn=_sum_pair_combine)
            mean = (s0 + s1) * inv_n
            variance = tl.maximum((q0 + q1) * inv_n - mean * mean, 0.0)
            cutoff = mean + tl.sqrt(variance) * std_multiplier

        tl.store(
            output_ptr + start + o0,
            tl.maximum(v0 - cutoff, 0.0),
            mask=valid,
        )
        tl.store(
            output_ptr + start + 8192 + o1,
            tl.maximum(v1 - cutoff, 0.0),
            mask=valid,
        )


@triton.jit
def _gaussian_sparse_row_16384_kernel(input_ptr, output_ptr, std_multiplier: tl.constexpr):
    row = tl.program_id(0)
    start = row * 16384
    offsets = tl.arange(0, 8192)
    v0 = tl.load(input_ptr + start + offsets).to(tl.float32)
    v1 = tl.load(input_ptr + start + 8192 + offsets).to(tl.float32)
    inv_n = 1.0 / 16384.0

    if std_multiplier == 0.0:
        cutoff = (tl.sum(v0, axis=0) + tl.sum(v1, axis=0)) * inv_n
    else:
        s0, q0 = tl.reduce((v0, v0 * v0), axis=0, combine_fn=_sum_pair_combine)
        s1, q1 = tl.reduce((v1, v1 * v1), axis=0, combine_fn=_sum_pair_combine)
        mean = (s0 + s1) * inv_n
        variance = tl.maximum((q0 + q1) * inv_n - mean * mean, 0.0)
        cutoff = mean + tl.sqrt(variance) * std_multiplier

    tl.store(output_ptr + start + offsets, tl.maximum(v0 - cutoff, 0.0))
    tl.store(
        output_ptr + start + 8192 + offsets,
        tl.maximum(v1 - cutoff, 0.0),
    )


@triton.jit
def _gaussian_sparse_row_16384_grouped_kernel(
    input_ptr,
    output_ptr,
    row_count,
    std_multiplier: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, 8192)
    inv_n = 1.0 / 16384.0

    for ri in tl.static_range(0, ROWS_PER_PROGRAM):
        row = pid * ROWS_PER_PROGRAM + ri
        valid = row < row_count
        start = row * 16384
        v0 = tl.load(input_ptr + start + offsets, mask=valid, other=0.0).to(tl.float32)
        v1 = tl.load(
            input_ptr + start + 8192 + offsets, mask=valid, other=0.0
        ).to(tl.float32)

        if std_multiplier == 0.0:
            cutoff = (tl.sum(v0, axis=0) + tl.sum(v1, axis=0)) * inv_n
        else:
            s0, q0 = tl.reduce((v0, v0 * v0), axis=0, combine_fn=_sum_pair_combine)
            s1, q1 = tl.reduce((v1, v1 * v1), axis=0, combine_fn=_sum_pair_combine)
            mean = (s0 + s1) * inv_n
            variance = tl.maximum((q0 + q1) * inv_n - mean * mean, 0.0)
            cutoff = mean + tl.sqrt(variance) * std_multiplier

        tl.store(
            output_ptr + start + offsets,
            tl.maximum(v0 - cutoff, 0.0),
            mask=valid,
        )
        tl.store(
            output_ptr + start + 8192 + offsets,
            tl.maximum(v1 - cutoff, 0.0),
            mask=valid,
        )


@torch.no_grad()
def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    if target_sparsity == 0.0:
        return inputs

    feature_size = inputs.shape[-1]
    row_count = inputs.numel() // feature_size
    output = torch.empty_like(inputs)
    std_multiplier = _ndtri_scalar(float(target_sparsity))

    if feature_size == 12288:
        num_warps = 8 if row_count < 768 else 4
        if row_count <= 512:
            _gaussian_sparse_row_12288_grouped_kernel[
                (triton.cdiv(row_count, 2),)
            ](
                inputs,
                output,
                row_count,
                std_multiplier=std_multiplier,
                ROWS_PER_PROGRAM=2,
                num_warps=num_warps,
                num_stages=1,
            )
        else:
            _gaussian_sparse_row_12288_kernel[(row_count,)](
                inputs,
                output,
                std_multiplier=std_multiplier,
                num_warps=num_warps,
                num_stages=1,
            )
        return output

    if feature_size == 16384:
        num_warps = 8 if row_count < 512 else 4
        if row_count <= 256:
            _gaussian_sparse_row_16384_grouped_kernel[
                (triton.cdiv(row_count, 2),)
            ](
                inputs,
                output,
                row_count,
                std_multiplier=std_multiplier,
                ROWS_PER_PROGRAM=2,
                num_warps=num_warps,
                num_stages=1,
            )
        else:
            _gaussian_sparse_row_16384_kernel[(row_count,)](
                inputs,
                output,
                std_multiplier=std_multiplier,
                num_warps=num_warps,
                num_stages=1,
            )
        return output

    block_size = triton.next_power_of_2(feature_size)

    if block_size == 4096:
        num_warps = 8 if row_count < 512 else 4
    elif block_size == 8192:
        num_warps = 8 if row_count < 1024 else 4
    elif block_size >= 16384:
        num_warps = 8 if row_count < 768 else 4
    elif block_size >= 1024:
        num_warps = 4
    else:
        num_warps = 2

    if block_size >= 4096 and row_count <= 512:
        _gaussian_sparse_row_grouped_kernel[
            (triton.cdiv(row_count, 2),)
        ](
            inputs,
            output,
            row_count,
            feature_size=feature_size,
            std_multiplier=std_multiplier,
            BLOCK_SIZE=block_size,
            ROWS_PER_PROGRAM=2,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        _gaussian_sparse_row_kernel[(row_count,)](
            inputs,
            output,
            feature_size=feature_size,
            std_multiplier=std_multiplier,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
            num_stages=1,
        )

    return output