# solution=GPT-5.6-Sol_gqa_ragged_prefill_causal_h32_kv8_d128_triton_optimized_r5 score=-1.0 passed=False
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _query_centric_attention(
    q_ptr,
    k_ptr,
    v_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    output_ptr,
    lse_ptr,
    sm_scale,
    batch_size,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_ot,
    stride_oh,
    stride_od,
    stride_lset,
    stride_lseh,
    stride_qo_indptr,
    stride_kv_indptr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    MAX_LOG_BATCH: tl.constexpr,
):
    q_idx = tl.program_id(0)
    head_group = tl.program_id(1)

    dims = tl.arange(0, HEAD_DIM)
    group_offsets = tl.arange(0, GROUP_SIZE)
    qo_heads = head_group * GROUP_SIZE + group_offsets
    kv_head = (head_group * GROUP_SIZE) // 4

    lo = tl.zeros((), dtype=tl.int32)
    hi = batch_size

    for _ in tl.static_range(0, MAX_LOG_BATCH):
        mid = (lo + hi - 1) // 2
        q_end_probe = tl.load(
            qo_indptr_ptr + (mid + 1) * stride_qo_indptr,
            mask=mid < batch_size,
            other=2147483647,
        )
        move_right = q_end_probe <= q_idx
        lo = tl.where(move_right, mid + 1, lo)
        hi = tl.where(move_right, hi, mid)

    sequence = lo

    q_start = tl.load(
        qo_indptr_ptr + sequence * stride_qo_indptr
    )
    q_end = tl.load(
        qo_indptr_ptr + (sequence + 1) * stride_qo_indptr
    )
    kv_start = tl.load(
        kv_indptr_ptr + sequence * stride_kv_indptr
    )
    kv_end = tl.load(
        kv_indptr_ptr + (sequence + 1) * stride_kv_indptr
    )

    q_length = q_end - q_start
    kv_length = kv_end - kv_start
    q_position = q_idx - q_start

    causal_end = (
        kv_start + q_position + 1 + kv_length - q_length
    )
    causal_end = tl.minimum(causal_end, kv_end)
    loop_end = tl.maximum(causal_end, kv_start)

    q_offsets = (
        q_idx * stride_qt
        + qo_heads[:, None] * stride_qh
        + dims[None, :] * stride_qd
    )
    q_values = tl.load(q_ptr + q_offsets)

    running_max = tl.full(
        (GROUP_SIZE,), -float("inf"), tl.float32
    )
    running_sum = tl.zeros((GROUP_SIZE,), tl.float32)
    accumulator = tl.zeros(
        (GROUP_SIZE, HEAD_DIM), tl.float32
    )

    log2_scale = sm_scale * 1.4426950408889634

    for kv_block_start in tl.range(
        kv_start,
        loop_end,
        BLOCK_N,
        num_stages=1,
    ):
        kv_offsets = kv_block_start + tl.arange(0, BLOCK_N)
        valid_kv = kv_offsets < causal_end

        k_offsets = (
            kv_offsets[:, None] * stride_kt
            + kv_head * stride_kh
            + dims[None, :] * stride_kd
        )
        k_values = tl.load(
            k_ptr + k_offsets,
            mask=valid_kv[:, None],
            other=0.0,
        )

        logits = tl.sum(
            k_values[:, None, :] * q_values[None, :, :],
            axis=2,
        ) * log2_scale
        logits = tl.where(
            valid_kv[:, None],
            logits,
            -float("inf"),
        )

        block_max = tl.max(logits, axis=0)
        new_max = tl.maximum(running_max, block_max)
        previous_scale = tl.exp2(running_max - new_max)

        probabilities = tl.exp2(logits - new_max[None, :])
        probabilities = tl.where(
            valid_kv[:, None],
            probabilities,
            0.0,
        )

        v_offsets = (
            kv_offsets[:, None] * stride_vt
            + kv_head * stride_vh
            + dims[None, :] * stride_vd
        )
        v_values = tl.load(
            v_ptr + v_offsets,
            mask=valid_kv[:, None],
            other=0.0,
        )

        accumulator = (
            accumulator * previous_scale[:, None]
            + tl.sum(
                probabilities[:, :, None]
                * v_values[:, None, :],
                axis=0,
            )
        )
        running_sum = (
            running_sum * previous_scale
            + tl.sum(probabilities, axis=0)
        )
        running_max = new_max

    has_attention = running_sum > 0.0
    output_values = tl.where(
        has_attention[:, None],
        accumulator / running_sum[:, None],
        0.0,
    )
    lse_values = tl.where(
        has_attention,
        running_max + tl.log2(running_sum),
        -float("inf"),
    )

    output_offsets = (
        q_idx * stride_ot
        + qo_heads[:, None] * stride_oh
        + dims[None, :] * stride_od
    )
    tl.store(output_ptr + output_offsets, output_values)
    tl.store(
        lse_ptr
        + q_idx * stride_lset
        + qo_heads * stride_lseh,
        lse_values,
    )


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    if not all(
        isinstance(tensor, torch.Tensor)
        for tensor in (q, k, v, qo_indptr, kv_indptr)
    ):
        raise TypeError(
            "q, k, v, qo_indptr, and kv_indptr must be tensors"
        )

    if q.ndim != 3 or q.shape[1:] != (32, 128):
        raise ValueError("q must have shape [total_q, 32, 128]")
    if k.ndim != 3 or k.shape[1:] != (8, 128):
        raise ValueError("k must have shape [total_kv, 8, 128]")
    if v.shape != k.shape:
        raise ValueError("v must have the same shape as k")
    if qo_indptr.ndim != 1 or kv_indptr.ndim != 1:
        raise ValueError(
            "qo_indptr and kv_indptr must be one-dimensional"
        )
    if qo_indptr.shape != kv_indptr.shape:
        raise ValueError(
            "qo_indptr and kv_indptr must have the same shape"
        )
    if qo_indptr.numel() < 2:
        raise ValueError(
            "indptr arrays must contain at least two elements"
        )

    if q.dtype != torch.bfloat16:
        raise TypeError("q must have dtype torch.bfloat16")
    if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("k and v must have dtype torch.bfloat16")
    if (
        qo_indptr.dtype != torch.int32
        or kv_indptr.dtype != torch.int32
    ):
        raise TypeError(
            "indptr arrays must have dtype torch.int32"
        )

    input_tensors = (q, k, v, qo_indptr, kv_indptr)
    for tensor in input_tensors:
        if tensor.device.type not in ("cpu", "cuda"):
            raise ValueError(
                "all input tensors must be on CPU or CUDA devices"
            )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to execute the Triton attention kernel"
        )

    original_device = q.device

    if q.device.type == "cuda":
        execution_device = q.device
        q_gpu = q
    else:
        q_gpu = q.cuda()
        execution_device = q_gpu.device

    def move_to_execution_device(tensor):
        if tensor.device == execution_device:
            return tensor
        if tensor.device.type == "cpu":
            return tensor.cuda(execution_device)
        return tensor.to(execution_device)

    k_gpu = move_to_execution_device(k)
    v_gpu = move_to_execution_device(v)
    qo_indptr_gpu = move_to_execution_device(qo_indptr)
    kv_indptr_gpu = move_to_execution_device(kv_indptr)

    if isinstance(sm_scale, torch.Tensor):
        if sm_scale.numel() != 1:
            raise ValueError("sm_scale must be a scalar")
        sm_scale_value = float(sm_scale.detach().item())
    else:
        sm_scale_value = float(sm_scale)

    total_q = q.shape[0]
    total_kv = k.shape[0]
    batch_size = qo_indptr.numel() - 1

    qo_last = int(qo_indptr_gpu[-1].item())
    kv_last = int(kv_indptr_gpu[-1].item())

    if qo_last != total_q:
        raise ValueError(
            "total_q must equal the final value of qo_indptr"
        )
    if kv_last != total_kv:
        raise ValueError(
            "total_kv must equal the final value of kv_indptr"
        )

    output_gpu = torch.empty(
        (total_q, 32, 128),
        dtype=torch.bfloat16,
        device=execution_device,
    )
    lse_gpu = torch.empty(
        (total_q, 32),
        dtype=torch.float32,
        device=execution_device,
    )

    if total_q > 0:
        max_log_batch = max(
            1,
            int(math.ceil(math.log2(batch_size + 1))),
        )

        grid = (total_q, 16)
        _query_centric_attention[grid](
            q_gpu,
            k_gpu,
            v_gpu,
            qo_indptr_gpu,
            kv_indptr_gpu,
            output_gpu,
            lse_gpu,
            sm_scale_value,
            batch_size,
            q_gpu.stride(0),
            q_gpu.stride(1),
            q_gpu.stride(2),
            k_gpu.stride(0),
            k_gpu.stride(1),
            k_gpu.stride(2),
            v_gpu.stride(0),
            v_gpu.stride(1),
            v_gpu.stride(2),
            output_gpu.stride(0),
            output_gpu.stride(1),
            output_gpu.stride(2),
            lse_gpu.stride(0),
            lse_gpu.stride(1),
            qo_indptr_gpu.stride(0),
            kv_indptr_gpu.stride(0),
            HEAD_DIM=128,
            BLOCK_N=16,
            GROUP_SIZE=2,
            MAX_LOG_BATCH=max_log_batch,
            num_warps=4,
            num_stages=1,
        )

    output = output_gpu.to(original_device)
    lse = lse_gpu.to(original_device)
    return output, lse