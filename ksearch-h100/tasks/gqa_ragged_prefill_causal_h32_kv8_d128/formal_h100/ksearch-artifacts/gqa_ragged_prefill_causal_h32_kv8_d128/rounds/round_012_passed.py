# solution=GPT-5.6-Sol_gqa_ragged_prefill_causal_h32_kv8_d128_triton_optimized_r12 score=12.276477930791955 passed=True
import weakref

import torch
import triton
import triton.language as tl


@triton.jit
def _single_token_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    lse_ptr,
    log2_scale,
    stride_qh,
    stride_qd,
    stride_kh,
    stride_kd,
    stride_vh,
    stride_vd,
    HEAD_DIM: tl.constexpr,
    GQA_RATIO: tl.constexpr,
):
    kv_head_idx = tl.program_id(0)
    dimensions = tl.arange(0, HEAD_DIM)
    first_query_head = kv_head_idx * GQA_RATIO

    key_values = tl.load(
        k_ptr
        + kv_head_idx * stride_kh
        + dimensions * stride_kd
    ).to(tl.float32)
    value = tl.load(
        v_ptr
        + kv_head_idx * stride_vh
        + dimensions * stride_vd
    )

    q_base = (
        first_query_head * stride_qh
        + dimensions * stride_qd
    )
    q0 = tl.load(q_ptr + q_base).to(tl.float32)
    q1 = tl.load(q_ptr + q_base + stride_qh).to(tl.float32)
    q2 = tl.load(q_ptr + q_base + 2 * stride_qh).to(tl.float32)
    q3 = tl.load(q_ptr + q_base + 3 * stride_qh).to(tl.float32)

    lse0 = tl.sum(q0 * key_values, axis=0) * log2_scale
    lse1 = tl.sum(q1 * key_values, axis=0) * log2_scale
    lse2 = tl.sum(q2 * key_values, axis=0) * log2_scale
    lse3 = tl.sum(q3 * key_values, axis=0) * log2_scale

    output_base = first_query_head * HEAD_DIM + dimensions
    tl.store(output_ptr + output_base, value)
    tl.store(output_ptr + output_base + HEAD_DIM, value)
    tl.store(output_ptr + output_base + 2 * HEAD_DIM, value)
    tl.store(output_ptr + output_base + 3 * HEAD_DIM, value)

    tl.store(lse_ptr + first_query_head, lse0)
    tl.store(lse_ptr + first_query_head + 1, lse1)
    tl.store(lse_ptr + first_query_head + 2, lse2)
    tl.store(lse_ptr + first_query_head + 3, lse3)


@triton.jit
def _sequence_tiled_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    output_ptr,
    lse_ptr,
    log2_scale,
    total_q,
    total_kv,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_indptr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    SINGLE_SEQUENCE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    query_tile_idx = tl.program_id(2)

    if SINGLE_SEQUENCE:
        q_start = 0
        kv_start = 0
        q_length = total_q
        kv_length = total_kv
    else:
        q_start = tl.load(
            qo_indptr_ptr + batch_idx * stride_indptr
        )
        q_end = tl.load(
            qo_indptr_ptr + (batch_idx + 1) * stride_indptr
        )
        kv_start = tl.load(
            kv_indptr_ptr + batch_idx * stride_indptr
        )
        kv_end = tl.load(
            kv_indptr_ptr + (batch_idx + 1) * stride_indptr
        )
        q_length = q_end - q_start
        kv_length = kv_end - kv_start

    query_tile_start = query_tile_idx * BLOCK_Q
    if query_tile_start >= q_length:
        return

    rows = tl.arange(0, BLOCK_M)
    dimensions = tl.arange(0, HEAD_DIM)
    query_positions = query_tile_start + rows // GQA_RATIO
    query_heads = kv_head_idx * GQA_RATIO + rows % GQA_RATIO
    query_mask = query_positions < q_length
    global_query_positions = q_start + query_positions

    q_offsets = (
        global_query_positions[:, None] * stride_qt
        + query_heads[:, None] * stride_qh
        + dimensions[None, :] * stride_qd
    )
    q_values = tl.load(
        q_ptr + q_offsets,
        mask=query_mask[:, None],
        other=0.0,
    )

    accumulator = tl.zeros(
        (BLOCK_M, HEAD_DIM),
        dtype=tl.float32,
    )
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    running_max = tl.full(
        (BLOCK_M,),
        -float("inf"),
        dtype=tl.float32,
    )

    alignment_delta = kv_length - q_length
    kv_limit = tl.minimum(
        kv_length,
        query_tile_start + BLOCK_Q + alignment_delta,
    )
    kv_limit = tl.maximum(kv_limit, 0)

    fully_visible_limit = tl.minimum(
        kv_length,
        tl.maximum(query_tile_start + 1 + alignment_delta, 0),
    )
    fully_visible_limit = (
        fully_visible_limit // BLOCK_N
    ) * BLOCK_N

    for key_block_start in range(
        0,
        fully_visible_limit,
        BLOCK_N,
    ):
        key_offsets = key_block_start + tl.arange(0, BLOCK_N)
        global_key_offsets = kv_start + key_offsets

        k_offsets = (
            global_key_offsets[None, :] * stride_kt
            + kv_head_idx * stride_kh
            + dimensions[:, None] * stride_kd
        )
        key_values = tl.load(k_ptr + k_offsets)

        logits = tl.dot(q_values, key_values) * log2_scale

        block_max = tl.max(logits, axis=1)
        new_max = tl.maximum(running_max, block_max)
        correction = tl.exp2(running_max - new_max)
        probabilities = tl.exp2(logits - new_max[:, None])

        value_offsets = (
            global_key_offsets[:, None] * stride_vt
            + kv_head_idx * stride_vh
            + dimensions[None, :] * stride_vd
        )
        values = tl.load(v_ptr + value_offsets)

        accumulator *= correction[:, None]
        accumulator += tl.dot(
            probabilities.to(tl.bfloat16),
            values,
        )
        normalizer = (
            normalizer * correction
            + tl.sum(probabilities, axis=1)
        )
        running_max = new_max

    for key_block_start in range(
        fully_visible_limit,
        kv_limit,
        BLOCK_N,
    ):
        key_offsets = key_block_start + tl.arange(0, BLOCK_N)
        key_mask = key_offsets < kv_limit
        global_key_offsets = kv_start + key_offsets

        k_offsets = (
            global_key_offsets[None, :] * stride_kt
            + kv_head_idx * stride_kh
            + dimensions[:, None] * stride_kd
        )
        key_values = tl.load(
            k_ptr + k_offsets,
            mask=key_mask[None, :],
            other=0.0,
        )

        logits = tl.dot(q_values, key_values) * log2_scale

        causal_mask = (
            key_offsets[None, :]
            < query_positions[:, None] + 1 + alignment_delta
        )
        score_mask = (
            query_mask[:, None]
            & key_mask[None, :]
            & causal_mask
        )
        logits = tl.where(
            score_mask,
            logits,
            -float("inf"),
        )

        block_max = tl.max(logits, axis=1)
        new_max = tl.maximum(running_max, block_max)
        finite_max = new_max != -float("inf")

        correction = tl.where(
            finite_max,
            tl.exp2(running_max - new_max),
            0.0,
        )
        probabilities = tl.where(
            score_mask,
            tl.exp2(logits - new_max[:, None]),
            0.0,
        )

        value_offsets = (
            global_key_offsets[:, None] * stride_vt
            + kv_head_idx * stride_vh
            + dimensions[None, :] * stride_vd
        )
        values = tl.load(
            v_ptr + value_offsets,
            mask=key_mask[:, None],
            other=0.0,
        )

        accumulator *= correction[:, None]
        accumulator += tl.dot(
            probabilities.to(tl.bfloat16),
            values,
        )
        normalizer = (
            normalizer * correction
            + tl.sum(probabilities, axis=1)
        )
        running_max = new_max

    has_values = normalizer > 0.0
    safe_normalizer = tl.where(
        has_values,
        normalizer,
        1.0,
    )
    output_values = accumulator / safe_normalizer[:, None]
    output_values = tl.where(
        has_values[:, None],
        output_values,
        0.0,
    )

    output_offsets = (
        global_query_positions[:, None] * (32 * HEAD_DIM)
        + query_heads[:, None] * HEAD_DIM
        + dimensions[None, :]
    )
    tl.store(
        output_ptr + output_offsets,
        output_values,
        mask=query_mask[:, None],
    )

    lse_values = tl.where(
        has_values,
        running_max + tl.log2(safe_normalizer),
        -float("inf"),
    )
    lse_offsets = global_query_positions * 32 + query_heads
    tl.store(
        lse_ptr + lse_offsets,
        lse_values,
        mask=query_mask,
    )


@triton.jit
def _singleton_query_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    output_ptr,
    lse_ptr,
    log2_scale,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_indptr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA_RATIO: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    q_start = tl.load(
        qo_indptr_ptr + batch_idx * stride_indptr
    )
    q_end = tl.load(
        qo_indptr_ptr + (batch_idx + 1) * stride_indptr
    )
    if q_start >= q_end:
        return

    kv_start = tl.load(
        kv_indptr_ptr + batch_idx * stride_indptr
    )
    kv_end = tl.load(
        kv_indptr_ptr + (batch_idx + 1) * stride_indptr
    )
    kv_length = kv_end - kv_start

    dimensions = tl.arange(0, HEAD_DIM)
    first_query_head = kv_head_idx * GQA_RATIO

    q_base = (
        q_start * stride_qt
        + first_query_head * stride_qh
        + dimensions * stride_qd
    )
    q0 = tl.load(q_ptr + q_base)
    q1 = tl.load(q_ptr + q_base + stride_qh)
    q2 = tl.load(q_ptr + q_base + 2 * stride_qh)
    q3 = tl.load(q_ptr + q_base + 3 * stride_qh)

    accumulator0 = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    accumulator1 = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    accumulator2 = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    accumulator3 = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    normalizer0 = 0.0
    normalizer1 = 0.0
    normalizer2 = 0.0
    normalizer3 = 0.0

    running_max0 = -float("inf")
    running_max1 = -float("inf")
    running_max2 = -float("inf")
    running_max3 = -float("inf")

    for key_block_start in range(0, kv_length, BLOCK_N):
        key_offsets = key_block_start + tl.arange(0, BLOCK_N)
        key_mask = key_offsets < kv_length
        global_key_offsets = kv_start + key_offsets

        k_offsets = (
            global_key_offsets[:, None] * stride_kt
            + kv_head_idx * stride_kh
            + dimensions[None, :] * stride_kd
        )
        key_values = tl.load(
            k_ptr + k_offsets,
            mask=key_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        logits0 = tl.sum(
            key_values * q0[None, :],
            axis=1,
        )
        logits1 = tl.sum(
            key_values * q1[None, :],
            axis=1,
        )
        logits2 = tl.sum(
            key_values * q2[None, :],
            axis=1,
        )
        logits3 = tl.sum(
            key_values * q3[None, :],
            axis=1,
        )

        logits0 = tl.where(
            key_mask,
            logits0 * log2_scale,
            -float("inf"),
        )
        logits1 = tl.where(
            key_mask,
            logits1 * log2_scale,
            -float("inf"),
        )
        logits2 = tl.where(
            key_mask,
            logits2 * log2_scale,
            -float("inf"),
        )
        logits3 = tl.where(
            key_mask,
            logits3 * log2_scale,
            -float("inf"),
        )

        block_max0 = tl.max(logits0, axis=0)
        block_max1 = tl.max(logits1, axis=0)
        block_max2 = tl.max(logits2, axis=0)
        block_max3 = tl.max(logits3, axis=0)

        new_max0 = tl.maximum(running_max0, block_max0)
        new_max1 = tl.maximum(running_max1, block_max1)
        new_max2 = tl.maximum(running_max2, block_max2)
        new_max3 = tl.maximum(running_max3, block_max3)

        correction0 = tl.exp2(running_max0 - new_max0)
        correction1 = tl.exp2(running_max1 - new_max1)
        correction2 = tl.exp2(running_max2 - new_max2)
        correction3 = tl.exp2(running_max3 - new_max3)

        probabilities0 = tl.where(
            key_mask,
            tl.exp2(logits0 - new_max0),
            0.0,
        )
        probabilities1 = tl.where(
            key_mask,
            tl.exp2(logits1 - new_max1),
            0.0,
        )
        probabilities2 = tl.where(
            key_mask,
            tl.exp2(logits2 - new_max2),
            0.0,
        )
        probabilities3 = tl.where(
            key_mask,
            tl.exp2(logits3 - new_max3),
            0.0,
        )

        value_offsets = (
            global_key_offsets[:, None] * stride_vt
            + kv_head_idx * stride_vh
            + dimensions[None, :] * stride_vd
        )
        values = tl.load(
            v_ptr + value_offsets,
            mask=key_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        accumulator0 = (
            accumulator0 * correction0
            + tl.sum(
                probabilities0[:, None] * values,
                axis=0,
            )
        )
        accumulator1 = (
            accumulator1 * correction1
            + tl.sum(
                probabilities1[:, None] * values,
                axis=0,
            )
        )
        accumulator2 = (
            accumulator2 * correction2
            + tl.sum(
                probabilities2[:, None] * values,
                axis=0,
            )
        )
        accumulator3 = (
            accumulator3 * correction3
            + tl.sum(
                probabilities3[:, None] * values,
                axis=0,
            )
        )

        normalizer0 = (
            normalizer0 * correction0
            + tl.sum(probabilities0, axis=0)
        )
        normalizer1 = (
            normalizer1 * correction1
            + tl.sum(probabilities1, axis=0)
        )
        normalizer2 = (
            normalizer2 * correction2
            + tl.sum(probabilities2, axis=0)
        )
        normalizer3 = (
            normalizer3 * correction3
            + tl.sum(probabilities3, axis=0)
        )

        running_max0 = new_max0
        running_max1 = new_max1
        running_max2 = new_max2
        running_max3 = new_max3

    has_values0 = normalizer0 > 0.0
    has_values1 = normalizer1 > 0.0
    has_values2 = normalizer2 > 0.0
    has_values3 = normalizer3 > 0.0

    safe_normalizer0 = tl.where(has_values0, normalizer0, 1.0)
    safe_normalizer1 = tl.where(has_values1, normalizer1, 1.0)
    safe_normalizer2 = tl.where(has_values2, normalizer2, 1.0)
    safe_normalizer3 = tl.where(has_values3, normalizer3, 1.0)

    output0 = tl.where(
        has_values0,
        accumulator0 / safe_normalizer0,
        0.0,
    )
    output1 = tl.where(
        has_values1,
        accumulator1 / safe_normalizer1,
        0.0,
    )
    output2 = tl.where(
        has_values2,
        accumulator2 / safe_normalizer2,
        0.0,
    )
    output3 = tl.where(
        has_values3,
        accumulator3 / safe_normalizer3,
        0.0,
    )

    output_base = (
        q_start * (32 * HEAD_DIM)
        + first_query_head * HEAD_DIM
        + dimensions
    )
    tl.store(output_ptr + output_base, output0)
    tl.store(output_ptr + output_base + HEAD_DIM, output1)
    tl.store(output_ptr + output_base + 2 * HEAD_DIM, output2)
    tl.store(output_ptr + output_base + 3 * HEAD_DIM, output3)

    lse_base = q_start * 32 + first_query_head
    tl.store(
        lse_ptr + lse_base,
        tl.where(
            has_values0,
            running_max0 + tl.log2(safe_normalizer0),
            -float("inf"),
        ),
    )
    tl.store(
        lse_ptr + lse_base + 1,
        tl.where(
            has_values1,
            running_max1 + tl.log2(safe_normalizer1),
            -float("inf"),
        ),
    )
    tl.store(
        lse_ptr + lse_base + 2,
        tl.where(
            has_values2,
            running_max2 + tl.log2(safe_normalizer2),
            -float("inf"),
        ),
    )
    tl.store(
        lse_ptr + lse_base + 3,
        tl.where(
            has_values3,
            running_max3 + tl.log2(safe_normalizer3),
            -float("inf"),
        ),
    )


_MAX_Q_LENGTH_CACHE = {}


def _cached_max_q_length(indptr):
    tensor_id = id(indptr)
    version = getattr(indptr, "_version", None)
    cached = _MAX_Q_LENGTH_CACHE.get(tensor_id)

    if cached is not None:
        tensor_ref, cached_version, cached_value = cached
        if tensor_ref() is indptr and cached_version == version:
            return cached_value
        _MAX_Q_LENGTH_CACHE.pop(tensor_id, None)

    lengths = indptr[1:] - indptr[:-1]
    value = int(lengths.max().item())

    if len(_MAX_Q_LENGTH_CACHE) >= 64:
        stale_keys = [
            key
            for key, entry in _MAX_Q_LENGTH_CACHE.items()
            if entry[0]() is None
        ]
        for key in stale_keys:
            _MAX_Q_LENGTH_CACHE.pop(key, None)
        if len(_MAX_Q_LENGTH_CACHE) >= 64:
            _MAX_Q_LENGTH_CACHE.clear()

    _MAX_Q_LENGTH_CACHE[tensor_id] = (
        weakref.ref(indptr),
        version,
        value,
    )
    return value


def _select_cuda_device(tensors):
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            return tensor.device
    return torch.device(
        "cuda",
        torch.cuda.current_device(),
    )


def _move_to_cuda(tensor, device, name):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.is_cuda:
        if tensor.device == device:
            return tensor
        return tensor.to(device)
    return tensor.cuda(device=device)


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "gqa_ragged attention requires a CUDA-capable GPU"
        )

    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    if not isinstance(k, torch.Tensor):
        raise TypeError("k must be a torch.Tensor")
    if not isinstance(v, torch.Tensor):
        raise TypeError("v must be a torch.Tensor")
    if not isinstance(qo_indptr, torch.Tensor):
        raise TypeError("qo_indptr must be a torch.Tensor")
    if not isinstance(kv_indptr, torch.Tensor):
        raise TypeError("kv_indptr must be a torch.Tensor")

    if q.ndim != 3 or q.shape[1:] != (32, 128):
        raise ValueError(
            "q must have shape [total_q, 32, 128]"
        )
    if k.ndim != 3 or k.shape[1:] != (8, 128):
        raise ValueError(
            "k must have shape [total_kv, 8, 128]"
        )
    if v.shape != k.shape:
        raise ValueError("v must have the same shape as k")
    if qo_indptr.ndim != 1 or kv_indptr.ndim != 1:
        raise ValueError(
            "qo_indptr and kv_indptr must be one-dimensional"
        )
    if (
        qo_indptr.shape != kv_indptr.shape
        or qo_indptr.numel() < 2
    ):
        raise ValueError(
            "qo_indptr and kv_indptr must have the same length "
            "of at least 2"
        )
    if q.dtype != torch.bfloat16:
        raise TypeError("q must have dtype torch.bfloat16")
    if (
        k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
    ):
        raise TypeError(
            "k and v must have dtype torch.bfloat16"
        )
    if (
        qo_indptr.dtype != torch.int32
        or kv_indptr.dtype != torch.int32
    ):
        raise TypeError(
            "qo_indptr and kv_indptr must have dtype torch.int32"
        )

    original_device = q.device
    target_device = _select_cuda_device(
        (q, k, v, qo_indptr, kv_indptr)
    )

    batch_size = qo_indptr.numel() - 1
    total_q = q.shape[0]
    total_kv = k.shape[0]

    if total_q == 0:
        max_q_length = 0
    elif batch_size == 1:
        max_q_length = total_q
    elif qo_indptr.is_cuda:
        max_q_length = None
    else:
        max_q_length = _cached_max_q_length(qo_indptr)

    q_gpu = _move_to_cuda(q, target_device, "q")
    k_gpu = _move_to_cuda(k, target_device, "k")
    v_gpu = _move_to_cuda(v, target_device, "v")
    qo_indptr_gpu = _move_to_cuda(
        qo_indptr,
        target_device,
        "qo_indptr",
    )
    kv_indptr_gpu = _move_to_cuda(
        kv_indptr,
        target_device,
        "kv_indptr",
    )

    if isinstance(sm_scale, torch.Tensor):
        if sm_scale.numel() != 1:
            raise ValueError("sm_scale must be a scalar")
        scale_value = float(sm_scale.detach().item())
    else:
        scale_value = float(sm_scale)

    log2_scale = scale_value * 1.4426950408889634

    if max_q_length is None:
        max_q_length = _cached_max_q_length(qo_indptr_gpu)

    output_gpu = torch.empty(
        q_gpu.shape,
        dtype=torch.bfloat16,
        device=target_device,
    )
    lse_gpu = torch.empty(
        (total_q, 32),
        dtype=torch.float32,
        device=target_device,
    )

    if total_q > 0 and max_q_length > 0:
        with torch.cuda.device(target_device):
            if (
                batch_size == 1
                and total_q == 1
                and total_kv == 1
            ):
                _single_token_attention_kernel[(8,)](
                    q_gpu,
                    k_gpu,
                    v_gpu,
                    output_gpu,
                    lse_gpu,
                    log2_scale,
                    q_gpu.stride(1),
                    q_gpu.stride(2),
                    k_gpu.stride(1),
                    k_gpu.stride(2),
                    v_gpu.stride(1),
                    v_gpu.stride(2),
                    HEAD_DIM=128,
                    GQA_RATIO=4,
                    num_warps=4,
                    num_stages=1,
                )
            elif max_q_length == 1:
                _singleton_query_attention_kernel[
                    (batch_size, 8)
                ](
                    q_gpu,
                    k_gpu,
                    v_gpu,
                    qo_indptr_gpu,
                    kv_indptr_gpu,
                    output_gpu,
                    lse_gpu,
                    log2_scale,
                    q_gpu.stride(0),
                    q_gpu.stride(1),
                    q_gpu.stride(2),
                    k_gpu.stride(0),
                    k_gpu.stride(1),
                    k_gpu.stride(2),
                    v_gpu.stride(0),
                    v_gpu.stride(1),
                    v_gpu.stride(2),
                    qo_indptr_gpu.stride(0),
                    BLOCK_N=16,
                    HEAD_DIM=128,
                    GQA_RATIO=4,
                    num_warps=4,
                    num_stages=2,
                )
            else:
                if max_q_length <= 32:
                    block_q = 8
                    block_n = 16
                    num_warps = 4
                    num_stages = 3
                elif max_q_length <= 64:
                    block_q = 16
                    block_n = 32
                    num_warps = 8
                    num_stages = 3
                elif max_q_length <= 256:
                    block_q = 16
                    block_n = 64
                    num_warps = 8
                    num_stages = 4
                else:
                    block_q = 16
                    block_n = 128
                    num_warps = 8
                    num_stages = 3

                block_m = block_q * 4
                grid = (
                    batch_size,
                    8,
                    triton.cdiv(max_q_length, block_q),
                )

                _sequence_tiled_attention_kernel[grid](
                    q_gpu,
                    k_gpu,
                    v_gpu,
                    qo_indptr_gpu,
                    kv_indptr_gpu,
                    output_gpu,
                    lse_gpu,
                    log2_scale,
                    total_q,
                    total_kv,
                    q_gpu.stride(0),
                    q_gpu.stride(1),
                    q_gpu.stride(2),
                    k_gpu.stride(0),
                    k_gpu.stride(1),
                    k_gpu.stride(2),
                    v_gpu.stride(0),
                    v_gpu.stride(1),
                    v_gpu.stride(2),
                    qo_indptr_gpu.stride(0),
                    BLOCK_Q=block_q,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    HEAD_DIM=128,
                    GQA_RATIO=4,
                    SINGLE_SEQUENCE=batch_size == 1,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
    else:
        output_gpu.zero_()
        lse_gpu.fill_(-float("inf"))

    if (
        original_device.type == "cuda"
        and original_device == target_device
    ):
        return output_gpu, lse_gpu

    return (
        output_gpu.to(original_device),
        lse_gpu.to(original_device),
    )