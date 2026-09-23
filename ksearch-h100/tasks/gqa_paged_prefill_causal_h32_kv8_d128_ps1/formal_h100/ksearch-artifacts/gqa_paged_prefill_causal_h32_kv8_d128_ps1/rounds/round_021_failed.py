# solution=GPT-5.6-Sol_gqa_paged_prefill_causal_h32_kv8_d128_ps1_triton_optimized_r21 score=-1.0 passed=False
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _query_row_streaming_kernel(
    q,
    k_cache,
    v_cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    output,
    lse,
    sm_scale: tl.constexpr,
    num_sequences: tl.constexpr,
):
    pid = tl.program_id(0)
    query_idx = pid // 4
    kv_head_group = pid & 3

    kv_head_offsets = tl.arange(0, 2)
    qo_head_offsets = tl.arange(0, 4)
    dim_offsets = tl.arange(0, 128)

    kv_heads = kv_head_group * 2 + kv_head_offsets
    qo_heads = kv_heads[:, None] * 4 + qo_head_offsets[None, :]

    lower = 1
    upper = num_sequences + 1
    while lower < upper:
        middle = (lower + upper) // 2
        boundary = tl.load(
            qo_indptr + middle,
            cache_modifier=".ca",
        )
        if boundary <= query_idx:
            lower = middle + 1
        else:
            upper = middle

    sequence_idx = lower - 1
    q_end = tl.load(
        qo_indptr + sequence_idx + 1,
        cache_modifier=".ca",
    )
    kv_start = tl.load(
        kv_indptr + sequence_idx,
        cache_modifier=".ca",
    )
    kv_end = tl.load(
        kv_indptr + sequence_idx + 1,
        cache_modifier=".ca",
    )

    visible_kv_count = query_idx + 1 + kv_end - kv_start - q_end

    output_base = output + query_idx * 4096
    output_ptrs = (
        output_base
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

        if visible_kv_count == 1:
            page_id = tl.load(
                kv_indices + kv_start,
                cache_modifier=".ca",
            )
            cache_base = page_id * 1024 + kv_heads[:, None] * 128

            k_values = tl.load(
                k_cache + cache_base + dim_offsets[None, :],
                cache_modifier=".ca",
            ).to(tl.float32)
            v_values = tl.load(
                v_cache + cache_base + dim_offsets[None, :],
                cache_modifier=".ca",
            )

            logits_base2 = (
                tl.sum(q_values * k_values[:, None, :], axis=2)
                * sm_scale
                * 1.4426950408889634
            )

            tl.store(
                output_ptrs,
                tl.broadcast_to(v_values[:, None, :], (2, 4, 128)),
            )
            tl.store(lse_ptrs, logits_base2)
        elif visible_kv_count == 3:
            page_id = tl.load(
                kv_indices + kv_start,
                cache_modifier=".ca",
            )
            cache_base = page_id * 1024 + kv_heads[:, None] * 128

            k_values = tl.load(
                k_cache + cache_base + dim_offsets[None, :],
                cache_modifier=".ca",
            ).to(tl.float32)
            v_values = tl.load(
                v_cache + cache_base + dim_offsets[None, :],
                cache_modifier=".ca",
            )

            running_max = (
                tl.sum(q_values * k_values[:, None, :], axis=2)
                * sm_scale
                * 1.4426950408889634
            )
            running_sum = tl.full((2, 4), 1.0, tl.float32)
            accumulator = tl.broadcast_to(
                v_values[:, None, :],
                (2, 4, 128),
            ).to(tl.float32)

            for kv_offset in tl.static_range(1, 3):
                page_id = tl.load(
                    kv_indices + kv_start + kv_offset,
                    cache_modifier=".ca",
                )
                cache_base = page_id * 1024 + kv_heads[:, None] * 128

                k_values = tl.load(
                    k_cache + cache_base + dim_offsets[None, :],
                    cache_modifier=".ca",
                ).to(tl.float32)
                v_values = tl.load(
                    v_cache + cache_base + dim_offsets[None, :],
                    cache_modifier=".ca",
                )

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

            result = accumulator / running_sum[:, :, None]
            logsumexp_base2 = running_max + tl.log2(running_sum)

            tl.store(output_ptrs, result)
            tl.store(lse_ptrs, logsumexp_base2)
        elif visible_kv_count == 4:
            page_id = tl.load(
                kv_indices + kv_start,
                cache_modifier=".ca",
            )
            cache_base = page_id * 1024 + kv_heads[:, None] * 128

            k_values = tl.load(
                k_cache + cache_base + dim_offsets[None, :],
                cache_modifier=".ca",
            ).to(tl.float32)
            v_values = tl.load(
                v_cache + cache_base + dim_offsets[None, :],
                cache_modifier=".ca",
            )

            running_max = (
                tl.sum(q_values * k_values[:, None, :], axis=2)
                * sm_scale
                * 1.4426950408889634
            )
            running_sum = tl.full((2, 4), 1.0, tl.float32)
            accumulator = tl.broadcast_to(
                v_values[:, None, :],
                (2, 4, 128),
            ).to(tl.float32)

            for kv_offset in tl.static_range(1, 4):
                page_id = tl.load(
                    kv_indices + kv_start + kv_offset,
                    cache_modifier=".ca",
                )
                cache_base = page_id * 1024 + kv_heads[:, None] * 128

                k_values = tl.load(
                    k_cache + cache_base + dim_offsets[None, :],
                    cache_modifier=".ca",
                ).to(tl.float32)
                v_values = tl.load(
                    v_cache + cache_base + dim_offsets[None, :],
                    cache_modifier=".ca",
                )

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

            result = accumulator / running_sum[:, :, None]
            logsumexp_base2 = running_max + tl.log2(running_sum)

            tl.store(output_ptrs, result)
            tl.store(lse_ptrs, logsumexp_base2)
        else:
            running_max = tl.full((2, 4), -float("inf"), tl.float32)
            running_sum = tl.zeros((2, 4), tl.float32)
            accumulator = tl.zeros((2, 4, 128), tl.float32)

            kv_offset = 0
            while kv_offset < visible_kv_count:
                page_id = tl.load(
                    kv_indices + kv_start + kv_offset,
                    cache_modifier=".ca",
                )
                cache_base = page_id * 1024 + kv_heads[:, None] * 128

                k_ptrs = k_cache + cache_base + dim_offsets[None, :]
                v_ptrs = v_cache + cache_base + dim_offsets[None, :]

                k_values = tl.load(
                    k_ptrs,
                    cache_modifier=".ca",
                ).to(tl.float32)
                v_values = tl.load(
                    v_ptrs,
                    cache_modifier=".ca",
                )

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
        raise RuntimeError(
            "CUDA is required to execute the Triton attention kernel"
        )

    original_device = q.device
    cuda_inputs = [
        tensor for _, tensor in tensors if tensor.device.type == "cuda"
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
            _query_row_streaming_kernel[(total_q * 4,)](
                q_gpu,
                k_gpu,
                v_gpu,
                qo_gpu,
                kv_gpu,
                indices_gpu,
                output_gpu,
                lse_gpu,
                sm_scale_value,
                qo_gpu.numel() - 1,
                num_warps=4,
            )

    if original_device.type == "cuda":
        if output_gpu.device != original_device:
            output_gpu = output_gpu.to(original_device)
            lse_gpu = lse_gpu.to(original_device)
        return output_gpu, lse_gpu

    return output_gpu.cpu(), lse_gpu.cpu()