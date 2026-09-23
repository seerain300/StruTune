# solution=GPT-5.6-Sol_gqa_ragged_prefill_causal_h32_kv8_d128_triton_optimized_r13 score=-1.0 passed=False
I’m keeping the proven sequence-tiled decomposition and focusing the change on the long-sequence launch configuration, where pipeline staging and register pressure are the most likely remaining cost. The short-sequence paths and wrapper behavior will remain unchanged.import weakref

import torch
import triton
import triton.language as tl


@triton.jit
def _sequence_tiled_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    output_ptr,
    lse_ptr,
    sm_scale,
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
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    query_tile_idx = tl.program_id(2)

    q_start = tl.load(qo_indptr_ptr + batch_idx * stride_indptr)
    q_end = tl.load(qo_indptr_ptr + (batch_idx + 1) * stride_indptr)
    kv_start = tl.load(kv_indptr_ptr + batch_idx * stride_indptr)
    kv_end = tl.load(kv_indptr_ptr + (batch_idx + 1) * stride_indptr)

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

    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    running_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)

    alignment_delta = kv_length - q_length
    kv_limit = tl.minimum(
        kv_length,
        query_tile_start + BLOCK_Q + alignment_delta,
    )
    kv_limit = tl.maximum(kv_limit, 0)

    for key_block_start in range(0, kv_limit, BLOCK_N):
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

        logits = tl.dot(q_values, key_values)
        logits *= sm_scale * 1.4426950408889634

        causal_mask = (
            key_offsets[None, :]
            < query_positions[:, None] + 1 + alignment_delta
        )
        score_mask = query_mask[:, None] & key_mask[None, :] & causal_mask
        logits = tl.where(score_mask, logits, -float("inf"))

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
        accumulator += tl.dot(probabilities.to(tl.bfloat16), values)
        normalizer = normalizer * correction + tl.sum(probabilities, axis=1)
        running_max = new_max

    has_values = normalizer > 0.0
    safe_normalizer = tl.where(has_values, normalizer, 1.0)
    output_values = accumulator / safe_normalizer[:, None]
    output_values = tl.where(has_values[:, None], output_values, 0.0)

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
    tl.store(lse_ptr + lse_offsets, lse_values, mask=query_mask)


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

    _MAX_Q_LENGTH_CACHE[tensor_id] = (weakref.ref(indptr), version, value)
    return value


def _select_cuda_device(tensors):
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            return tensor.device
    return torch.device("cuda", torch.cuda.current_device())


def _move_to_cuda(tensor, device, name):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.is_cuda:
        return tensor if tensor.device == device else tensor.to(device)
    return tensor.cuda(device=device)


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    if not torch.cuda.is_available():
        raise RuntimeError("gqa_ragged attention requires a CUDA-capable GPU")

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
        raise ValueError("q must have shape [total_q, 32, 128]")
    if k.ndim != 3 or k.shape[1:] != (8, 128):
        raise ValueError("k must have shape [total_kv, 8, 128]")
    if v.shape != k.shape:
        raise ValueError("v must have the same shape as k")
    if qo_indptr.ndim != 1 or kv_indptr.ndim != 1:
        raise ValueError("qo_indptr and kv_indptr must be one-dimensional")
    if qo_indptr.shape != kv_indptr.shape or qo_indptr.numel() < 2:
        raise ValueError(
            "qo_indptr and kv_indptr must have the same length of at least 2"
        )
    if q.dtype != torch.bfloat16:
        raise TypeError("q must have dtype torch.bfloat16")
    if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("k and v must have dtype torch.bfloat16")
    if qo_indptr.dtype != torch.int32 or kv_indptr.dtype != torch.int32:
        raise TypeError("qo_indptr and kv_indptr must have dtype torch.int32")

    original_device = q.device
    target_device = _select_cuda_device(
        (q, k, v, qo_indptr, kv_indptr)
    )

    if qo_indptr.is_cuda:
        max_q_length = None
    else:
        max_q_length = _cached_max_q_length(qo_indptr)

    q_gpu = _move_to_cuda(q, target_device, "q")
    k_gpu = _move_to_cuda(k, target_device, "k")
    v_gpu = _move_to_cuda(v, target_device, "v")
    qo_indptr_gpu = _move_to_cuda(
        qo_indptr, target_device, "qo_indptr"
    )
    kv_indptr_gpu = _move_to_cuda(
        kv_indptr, target_device, "kv_indptr"
    )

    if isinstance(sm_scale, torch.Tensor):
        if sm_scale.numel() != 1:
            raise ValueError("sm_scale must be a scalar")
        scale_value = float(sm_scale.detach().item())
    else:
        scale_value = float(sm_scale)

    batch_size = qo_indptr.numel() - 1
    total_q = q.shape[0]

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
        if max_q_length <= 4:
            block_q = 4
            block_n = 32
            num_warps = 4
            num_stages = 3
        elif max_q_length <= 32:
            block_q = 8
            block_n = 32
            num_warps = 4
            num_stages = 3
        elif max_q_length <= 64:
            block_q = 8
            block_n = 64
            num_warps = 4
            num_stages = 3
        elif max_q_length <= 128:
            block_q = 8
            block_n = 64
            num_warps = 4
            num_stages = 2
        else:
            block_q = 16
            block_n = 64
            num_warps = 8
            num_stages = 2

        block_m = block_q * 4
        grid = (
            batch_size,
            8,
            triton.cdiv(max_q_length, block_q),
        )

        with torch.cuda.device(target_device):
            _sequence_tiled_attention_kernel[grid](
                q_gpu,
                k_gpu,
                v_gpu,
                qo_indptr_gpu,
                kv_indptr_gpu,
                output_gpu,
                lse_gpu,
                scale_value,
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
                num_warps=num_warps,
                num_stages=num_stages,
            )
    else:
        output_gpu.zero_()
        lse_gpu.fill_(-float("inf"))

    if original_device.type == "cuda" and original_device == target_device:
        return output_gpu, lse_gpu

    return output_gpu.to(original_device), lse_gpu.to(original_device)