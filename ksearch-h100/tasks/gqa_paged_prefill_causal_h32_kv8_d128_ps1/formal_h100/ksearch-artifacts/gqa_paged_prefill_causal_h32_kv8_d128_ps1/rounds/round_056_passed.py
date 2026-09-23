# solution=GPT-5.6-Sol_gqa_paged_prefill_causal_h32_kv8_d128_ps1_triton_optimized_r2 score=1458.281967518392 passed=True
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _stream_single_query(
    q,
    k_cache,
    v_cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    output,
    lse,
    query_idx,
    sm_scale: tl.constexpr,
    num_sequences: tl.constexpr,
):
    kv_heads = tl.arange(0, 8)
    qo_head_offsets = tl.arange(0, 4)
    dim_offsets = tl.arange(0, 128)
    qo_heads = kv_heads[:, None] * 4 + qo_head_offsets[None, :]

    lower = 1
    upper = num_sequences + 1
    while lower < upper:
        middle = (lower + upper) // 2
        boundary = tl.load(qo_indptr + middle, cache_modifier=".ca")
        if boundary <= query_idx:
            lower = middle + 1
        else:
            upper = middle

    sequence_idx = lower - 1
    q_end = tl.load(qo_indptr + sequence_idx + 1, cache_modifier=".ca")
    kv_start = tl.load(kv_indptr + sequence_idx, cache_modifier=".ca")
    kv_end = tl.load(kv_indptr + sequence_idx + 1, cache_modifier=".ca")

    visible_kv_count = query_idx + 1 + kv_end - kv_start - q_end

    output_ptrs = (
        output
        + query_idx * 4096
        + qo_heads[:, :, None] * 128
        + dim_offsets[None, None, :]
    )
    lse_ptrs = lse + query_idx * 32 + qo_heads

    if visible_kv_count > 0:
        q_ptrs = (
            q
            + query_idx * 4096
            + qo_heads[:, :, None] * 128
            + dim_offsets[None, None, :]
        )
        q_values = tl.load(q_ptrs).to(tl.float32)

        running_max = tl.full((8, 4), -float("inf"), tl.float32)
        running_sum = tl.zeros((8, 4), tl.float32)
        accumulator = tl.zeros((8, 4, 128), tl.float32)

        kv_offset = 0
        while kv_offset < visible_kv_count:
            page_id = tl.load(
                kv_indices + kv_start + kv_offset,
                cache_modifier=".ca",
            )
            cache_base = page_id * 1024 + kv_heads[:, None] * 128

            k_values = tl.load(
                k_cache + cache_base + dim_offsets[None, :]
            ).to(tl.float32)
            v_values = tl.load(
                v_cache + cache_base + dim_offsets[None, :]
            ).to(tl.float32)

            logits = (
                tl.sum(q_values * k_values[:, None, :], axis=2)
                * sm_scale
                * 1.4426950408889634
            )

            new_max = tl.maximum(running_max, logits)
            old_weight = tl.exp2(running_max - new_max)
            new_weight = tl.exp2(logits - new_max)

            accumulator = (
                accumulator * old_weight[:, :, None]
                + new_weight[:, :, None] * v_values[:, None, :]
            )
            running_sum = running_sum * old_weight + new_weight
            running_max = new_max
            kv_offset += 1

        result = accumulator / running_sum[:, :, None]
        logsumexp_base2 = running_max + tl.log2(running_sum)

        tl.store(output_ptrs, result)
        tl.store(lse_ptrs, logsumexp_base2)
    else:
        tl.store(output_ptrs, 0.0)
        tl.store(lse_ptrs, -float("inf"))


@triton.jit
def _query_row_pair_streaming_kernel(
    q,
    k_cache,
    v_cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    output,
    lse,
    total_q,
    sm_scale: tl.constexpr,
    num_sequences: tl.constexpr,
):
    pair_idx = tl.program_id(0)
    query_idx0 = pair_idx * 2
    query_idx1 = query_idx0 + 1

    kv_heads = tl.arange(0, 8)
    qo_head_offsets = tl.arange(0, 4)
    dim_offsets = tl.arange(0, 128)
    qo_heads = kv_heads[:, None] * 4 + qo_head_offsets[None, :]

    lower = 1
    upper = num_sequences + 1
    while lower < upper:
        middle = (lower + upper) // 2
        boundary = tl.load(qo_indptr + middle, cache_modifier=".ca")
        if boundary <= query_idx0:
            lower = middle + 1
        else:
            upper = middle

    sequence_idx0 = lower - 1
    q_end0 = tl.load(qo_indptr + sequence_idx0 + 1, cache_modifier=".ca")
    kv_start0 = tl.load(kv_indptr + sequence_idx0, cache_modifier=".ca")
    kv_end0 = tl.load(kv_indptr + sequence_idx0 + 1, cache_modifier=".ca")

    same_sequence = (query_idx1 < total_q) & (query_idx1 < q_end0)

    if same_sequence:
        visible0 = query_idx0 + 1 + kv_end0 - kv_start0 - q_end0
        visible1 = visible0 + 1
        max_visible = tl.maximum(visible0, visible1)

        output_ptrs0 = (
            output
            + query_idx0 * 4096
            + qo_heads[:, :, None] * 128
            + dim_offsets[None, None, :]
        )
        output_ptrs1 = (
            output
            + query_idx1 * 4096
            + qo_heads[:, :, None] * 128
            + dim_offsets[None, None, :]
        )
        lse_ptrs0 = lse + query_idx0 * 32 + qo_heads
        lse_ptrs1 = lse + query_idx1 * 32 + qo_heads

        q_ptrs0 = (
            q
            + query_idx0 * 4096
            + qo_heads[:, :, None] * 128
            + dim_offsets[None, None, :]
        )
        q_ptrs1 = (
            q
            + query_idx1 * 4096
            + qo_heads[:, :, None] * 128
            + dim_offsets[None, None, :]
        )
        q_values0 = tl.load(q_ptrs0).to(tl.float32)
        q_values1 = tl.load(q_ptrs1).to(tl.float32)

        running_max0 = tl.full((8, 4), -float("inf"), tl.float32)
        running_max1 = tl.full((8, 4), -float("inf"), tl.float32)
        running_sum0 = tl.zeros((8, 4), tl.float32)
        running_sum1 = tl.zeros((8, 4), tl.float32)
        accumulator0 = tl.zeros((8, 4, 128), tl.float32)
        accumulator1 = tl.zeros((8, 4, 128), tl.float32)

        kv_offset = 0
        while kv_offset < max_visible:
            page_id = tl.load(
                kv_indices + kv_start0 + kv_offset,
                cache_modifier=".ca",
            )
            cache_base = page_id * 1024 + kv_heads[:, None] * 128

            k_values = tl.load(
                k_cache + cache_base + dim_offsets[None, :]
            ).to(tl.float32)
            v_values = tl.load(
                v_cache + cache_base + dim_offsets[None, :]
            ).to(tl.float32)

            logits0 = (
                tl.sum(q_values0 * k_values[:, None, :], axis=2)
                * sm_scale
                * 1.4426950408889634
            )
            logits1 = (
                tl.sum(q_values1 * k_values[:, None, :], axis=2)
                * sm_scale
                * 1.4426950408889634
            )

            valid0 = kv_offset < visible0
            valid1 = kv_offset < visible1
            logits0 = tl.where(valid0, logits0, -float("inf"))
            logits1 = tl.where(valid1, logits1, -float("inf"))

            new_max0 = tl.maximum(running_max0, logits0)
            new_max1 = tl.maximum(running_max1, logits1)
            old_weight0 = tl.exp2(running_max0 - new_max0)
            old_weight1 = tl.exp2(running_max1 - new_max1)
            new_weight0 = tl.exp2(logits0 - new_max0)
            new_weight1 = tl.exp2(logits1 - new_max1)

            accumulator0 = (
                accumulator0 * old_weight0[:, :, None]
                + new_weight0[:, :, None] * v_values[:, None, :]
            )
            accumulator1 = (
                accumulator1 * old_weight1[:, :, None]
                + new_weight1[:, :, None] * v_values[:, None, :]
            )
            running_sum0 = running_sum0 * old_weight0 + new_weight0
            running_sum1 = running_sum1 * old_weight1 + new_weight1
            running_max0 = new_max0
            running_max1 = new_max1
            kv_offset += 1

        if visible0 > 0:
            result0 = accumulator0 / running_sum0[:, :, None]
            lse0 = running_max0 + tl.log2(running_sum0)
            tl.store(output_ptrs0, result0)
            tl.store(lse_ptrs0, lse0)
        else:
            tl.store(output_ptrs0, 0.0)
            tl.store(lse_ptrs0, -float("inf"))

        if visible1 > 0:
            result1 = accumulator1 / running_sum1[:, :, None]
            lse1 = running_max1 + tl.log2(running_sum1)
            tl.store(output_ptrs1, result1)
            tl.store(lse_ptrs1, lse1)
        else:
            tl.store(output_ptrs1, 0.0)
            tl.store(lse_ptrs1, -float("inf"))
    else:
        _stream_single_query(
            q,
            k_cache,
            v_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            output,
            lse,
            query_idx0,
            sm_scale,
            num_sequences,
        )
        if query_idx1 < total_q:
            _stream_single_query(
                q,
                k_cache,
                v_cache,
                qo_indptr,
                kv_indptr,
                kv_indices,
                output,
                lse,
                query_idx1,
                sm_scale,
                num_sequences,
            )


def _validate_tensor(name, tensor, dtype=None):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if dtype is not None and tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")


def _move_to_execution_device(tensor, device):
    if tensor.device.type == "cpu":
        return tensor.cuda(device=device).contiguous()
    if tensor.device != device:
        return tensor.to(device=device).contiguous()
    return tensor.contiguous()


@torch.no_grad()
def run(
    q,
    k_cache,
    v_cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale=1.0 / math.sqrt(128.0),
):
    _validate_tensor("q", q, torch.bfloat16)
    _validate_tensor("k_cache", k_cache, torch.bfloat16)
    _validate_tensor("v_cache", v_cache, torch.bfloat16)
    _validate_tensor("qo_indptr", qo_indptr, torch.int32)
    _validate_tensor("kv_indptr", kv_indptr, torch.int32)
    _validate_tensor("kv_indices", kv_indices, torch.int32)

    if q.ndim != 3 or q.shape[1:] != (32, 128):
        raise ValueError("q must have shape [total_q, 32, 128]")
    if k_cache.ndim != 4 or k_cache.shape[1:] != (1, 8, 128):
        raise ValueError("k_cache must have shape [num_pages, 1, 8, 128]")
    if v_cache.shape != k_cache.shape:
        raise ValueError("v_cache must have the same shape as k_cache")
    if qo_indptr.ndim != 1 or kv_indptr.ndim != 1:
        raise ValueError("qo_indptr and kv_indptr must be one-dimensional")
    if qo_indptr.shape != kv_indptr.shape:
        raise ValueError("qo_indptr and kv_indptr must have the same shape")
    if kv_indices.ndim != 1:
        raise ValueError("kv_indices must be one-dimensional")
    if qo_indptr.numel() < 2:
        raise ValueError("indptr arrays must contain at least two elements")

    tensors = (
        ("q", q),
        ("k_cache", k_cache),
        ("v_cache", v_cache),
        ("qo_indptr", qo_indptr),
        ("kv_indptr", kv_indptr),
        ("kv_indices", kv_indices),
    )
    for name, tensor in tensors:
        if tensor.device.type not in ("cpu", "cuda"):
            raise ValueError(
                f"{name} must be on CPU or CUDA, got {tensor.device}"
            )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to execute the Triton attention kernel")

    original_device = q.device
    cuda_inputs = [
        tensor
        for _, tensor in tensors
        if tensor.device.type == "cuda"
    ]
    execution_device = (
        q.device
        if q.device.type == "cuda"
        else cuda_inputs[0].device
        if cuda_inputs
        else torch.device("cuda", torch.cuda.current_device())
    )

    if isinstance(sm_scale, torch.Tensor):
        if sm_scale.numel() != 1:
            raise ValueError("sm_scale must be a scalar")
        sm_scale_value = float(sm_scale.item())
    else:
        sm_scale_value = float(sm_scale)

    with torch.cuda.device(execution_device):
        q_gpu = _move_to_execution_device(q, execution_device)
        k_gpu = _move_to_execution_device(k_cache, execution_device)
        v_gpu = _move_to_execution_device(v_cache, execution_device)
        qo_gpu = _move_to_execution_device(qo_indptr, execution_device)
        kv_gpu = _move_to_execution_device(kv_indptr, execution_device)
        indices_gpu = _move_to_execution_device(kv_indices, execution_device)

        total_q = q_gpu.shape[0]
        output_gpu = torch.empty_like(q_gpu)
        lse_gpu = torch.empty(
            (total_q, 32),
            dtype=torch.float32,
            device=execution_device,
        )

        if total_q > 0:
            _query_row_pair_streaming_kernel[((total_q + 1) // 2,)](
                q_gpu,
                k_gpu,
                v_gpu,
                qo_gpu,
                kv_gpu,
                indices_gpu,
                output_gpu,
                lse_gpu,
                total_q,
                sm_scale_value,
                qo_gpu.numel() - 1,
                num_warps=8,
            )

    if original_device.type == "cuda":
        if output_gpu.device != original_device:
            output_gpu = output_gpu.to(original_device)
            lse_gpu = lse_gpu.to(original_device)
        return output_gpu, lse_gpu

    return output_gpu.cpu(), lse_gpu.cpu()