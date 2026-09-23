import math

import torch
import triton
import triton.language as tl


@triton.jit
def _independent_row_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    output_ptr,
    lse_ptr,
    sm_scale,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA_RATIO: tl.constexpr,
    NUM_SEQS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    q_idx = pid // NUM_Q_HEADS
    qo_head = pid % NUM_Q_HEADS
    kv_head = qo_head // GQA_RATIO

    lo = 0
    hi = NUM_SEQS
    for _ in range(SEARCH_STEPS):
        active = lo < hi
        mid = (lo + hi) // 2
        safe_mid = tl.minimum(mid, NUM_SEQS - 1)
        q_end_mid = tl.load(qo_indptr_ptr + safe_mid + 1)
        move_right = active & (q_idx >= q_end_mid)

        new_lo = tl.where(move_right, mid + 1, lo)
        new_hi = tl.where(active & (~move_right), mid, hi)
        lo = new_lo
        hi = new_hi

    batch_idx = tl.minimum(lo, NUM_SEQS - 1)

    q_start = tl.load(qo_indptr_ptr + batch_idx)
    q_end = tl.load(qo_indptr_ptr + batch_idx + 1)
    kv_start = tl.load(kv_indptr_ptr + batch_idx)
    kv_end = tl.load(kv_indptr_ptr + batch_idx + 1)

    num_q_tokens = q_end - q_start
    num_kv_tokens = kv_end - kv_start
    local_q_idx = q_idx - q_start

    max_kv_idx = tl.minimum(
        local_q_idx + 1 + num_kv_tokens - num_q_tokens,
        num_kv_tokens,
    )

    if max_kv_idx > 0:
        offs_d = tl.arange(0, HEAD_DIM)
        q_offsets = (
            q_idx * NUM_Q_HEADS * HEAD_DIM
            + qo_head * HEAD_DIM
            + offs_d
        )
        q_vec = tl.load(q_ptr + q_offsets).to(tl.float32)

        running_max = -float("inf")
        running_sum = 0.0
        output_acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

        block_start = 0
        while block_start < max_kv_idx:
            offs_n = block_start + tl.arange(0, BLOCK_N)
            valid_n = offs_n < max_kv_idx

            page_ids = tl.load(
                kv_indices_ptr + kv_start + offs_n,
                mask=valid_n,
                other=0,
            )

            cache_offsets = (
                page_ids[:, None] * (NUM_KV_HEADS * HEAD_DIM)
                + kv_head * HEAD_DIM
                + offs_d[None, :]
            )

            k_block = tl.load(
                k_ptr + cache_offsets,
                mask=valid_n[:, None],
                other=0.0,
            ).to(tl.float32)

            logits = tl.sum(k_block * q_vec[None, :], axis=1)
            logits = logits * sm_scale
            logits = tl.where(valid_n, logits, -float("inf"))

            block_max = tl.max(logits, axis=0)
            new_max = tl.maximum(running_max, block_max)
            old_scale = tl.exp(running_max - new_max)

            probabilities = tl.exp(logits - new_max)
            probabilities = tl.where(valid_n, probabilities, 0.0)

            v_block = tl.load(
                v_ptr + cache_offsets,
                mask=valid_n[:, None],
                other=0.0,
            ).to(tl.float32)

            output_acc = (
                output_acc * old_scale
                + tl.sum(probabilities[:, None] * v_block, axis=0)
            )
            running_sum = (
                running_sum * old_scale
                + tl.sum(probabilities, axis=0)
            )
            running_max = new_max
            block_start += BLOCK_N

        output_acc = output_acc / running_sum

        output_offsets = (
            q_idx * NUM_Q_HEADS * HEAD_DIM
            + qo_head * HEAD_DIM
            + offs_d
        )
        tl.store(output_ptr + output_offsets, output_acc)

        lse_value = (
            running_max + tl.log(running_sum)
        ) * 1.4426950408889634
        tl.store(
            lse_ptr + q_idx * NUM_Q_HEADS + qo_head,
            lse_value,
        )


@torch.no_grad()
def run(
    q,
    k_cache,
    v_cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
):
    tensor_args = {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
    }

    for name, tensor in tensor_args.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    if q.ndim != 3 or q.shape[1:] != (32, 128):
        raise ValueError("q must have shape [total_q, 32, 128]")
    if k_cache.ndim != 4 or k_cache.shape[1:] != (1, 8, 128):
        raise ValueError(
            "k_cache must have shape [num_pages, 1, 8, 128]"
        )
    if v_cache.shape != k_cache.shape:
        raise ValueError("v_cache must have the same shape as k_cache")
    if qo_indptr.ndim != 1 or kv_indptr.ndim != 1:
        raise ValueError(
            "qo_indptr and kv_indptr must be one-dimensional"
        )
    if qo_indptr.numel() != kv_indptr.numel():
        raise ValueError(
            "qo_indptr and kv_indptr must have equal lengths"
        )
    if qo_indptr.numel() < 1:
        raise ValueError(
            "indptr arrays must contain at least one element"
        )
    if kv_indices.ndim != 1:
        raise ValueError("kv_indices must be one-dimensional")

    if q.dtype != torch.bfloat16:
        raise TypeError("q must have dtype torch.bfloat16")
    if k_cache.dtype != torch.bfloat16:
        raise TypeError("k_cache must have dtype torch.bfloat16")
    if v_cache.dtype != torch.bfloat16:
        raise TypeError("v_cache must have dtype torch.bfloat16")
    if qo_indptr.dtype != torch.int32:
        raise TypeError("qo_indptr must have dtype torch.int32")
    if kv_indptr.dtype != torch.int32:
        raise TypeError("kv_indptr must have dtype torch.int32")
    if kv_indices.dtype != torch.int32:
        raise TypeError("kv_indices must have dtype torch.int32")

    total_q = q.shape[0]
    num_sequences = qo_indptr.numel() - 1

    if int(qo_indptr[-1].item()) != total_q:
        raise ValueError("total_q must equal qo_indptr[-1]")
    if int(kv_indptr[-1].item()) != kv_indices.numel():
        raise ValueError(
            "num_kv_indices must equal kv_indptr[-1]"
        )

    if isinstance(sm_scale, torch.Tensor):
        if sm_scale.numel() != 1:
            raise ValueError("sm_scale must be a scalar")
        sm_scale_value = float(sm_scale.item())
    else:
        sm_scale_value = float(sm_scale)

    if not math.isfinite(sm_scale_value):
        raise ValueError("sm_scale must be finite")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to execute the Triton kernel"
        )

    original_output_device = q.device
    cuda_tensors = [
        tensor for tensor in tensor_args.values() if tensor.is_cuda
    ]

    if q.is_cuda:
        execution_device = q.device
    elif cuda_tensors:
        execution_device = cuda_tensors[0].device
    else:
        execution_device = torch.device(
            "cuda", torch.cuda.current_device()
        )

    def to_execution_device(tensor):
        if tensor.device == execution_device:
            return tensor.contiguous()
        if tensor.device.type == "cpu":
            return tensor.contiguous().cuda(device=execution_device)
        return tensor.to(device=execution_device).contiguous()

    q_gpu = to_execution_device(q)
    k_gpu = to_execution_device(k_cache)
    v_gpu = to_execution_device(v_cache)
    qo_indptr_gpu = to_execution_device(qo_indptr)
    kv_indptr_gpu = to_execution_device(kv_indptr)
    kv_indices_gpu = to_execution_device(kv_indices)

    with torch.cuda.device(execution_device):
        output_gpu = torch.zeros(
            (total_q, 32, 128),
            dtype=torch.bfloat16,
            device=execution_device,
        )
        lse_gpu = torch.full(
            (total_q, 32),
            -float("inf"),
            dtype=torch.float32,
            device=execution_device,
        )

        if total_q > 0:
            if num_sequences <= 0:
                raise ValueError(
                    "non-empty q requires at least one sequence"
                )

            search_steps = max(1, num_sequences.bit_length())
            grid = (total_q * 32,)

            _independent_row_attention_kernel[grid](
                q_gpu,
                k_gpu,
                v_gpu,
                qo_indptr_gpu,
                kv_indptr_gpu,
                kv_indices_gpu,
                output_gpu,
                lse_gpu,
                sm_scale_value,
                NUM_Q_HEADS=32,
                NUM_KV_HEADS=8,
                HEAD_DIM=128,
                GQA_RATIO=4,
                NUM_SEQS=num_sequences,
                SEARCH_STEPS=search_steps,
                BLOCK_N=16,
                num_warps=4,
                num_stages=1,
            )

    if original_output_device == execution_device:
        return output_gpu, lse_gpu

    return (
        output_gpu.to(device=original_output_device),
        lse_gpu.to(device=original_output_device),
    )