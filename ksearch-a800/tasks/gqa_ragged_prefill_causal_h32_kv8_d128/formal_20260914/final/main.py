import torch
import triton
import triton.language as tl


@triton.jit
def _fused_query_tile_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    output_ptr,
    lse_ptr,
    sm_scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    query_tile = tl.program_id(0)
    kv_head = tl.program_id(1)
    batch_idx = tl.program_id(2)

    q_start = tl.load(qo_indptr_ptr + batch_idx)
    q_end = tl.load(qo_indptr_ptr + batch_idx + 1)
    kv_start = tl.load(kv_indptr_ptr + batch_idx)
    kv_end = tl.load(kv_indptr_ptr + batch_idx + 1)

    q_len = q_end - q_start
    kv_len = kv_end - kv_start
    query_tile_start = query_tile * BLOCK_Q

    if query_tile_start < q_len:
        row_offsets = tl.arange(0, BLOCK_Q * GROUP_SIZE)
        query_offsets = query_tile_start + row_offsets // GROUP_SIZE
        group_heads = row_offsets % GROUP_SIZE
        query_heads = kv_head * GROUP_SIZE + group_heads
        row_valid = query_offsets < q_len

        dim_offsets = tl.arange(0, HEAD_DIM)
        absolute_queries = q_start + query_offsets

        q_indices = (
            absolute_queries[:, None].to(tl.int64) * 4096
            + query_heads[:, None] * HEAD_DIM
            + dim_offsets[None, :]
        )
        q_values = tl.load(
            q_ptr + q_indices,
            mask=row_valid[:, None],
            other=0.0,
        )

        running_max = tl.full(
            (BLOCK_Q * GROUP_SIZE,),
            -float("inf"),
            tl.float32,
        )
        running_sum = tl.zeros(
            (BLOCK_Q * GROUP_SIZE,),
            tl.float32,
        )
        accumulator = tl.zeros(
            (BLOCK_Q * GROUP_SIZE, HEAD_DIM),
            tl.float32,
        )

        delta = kv_len - q_len
        query_tile_end = tl.minimum(
            q_len,
            query_tile_start + BLOCK_Q,
        )
        scan_end = tl.minimum(
            kv_len,
            query_tile_end + delta,
        )
        causal_limit = query_offsets[:, None] + 1 + delta
        log2_scale = sm_scale * 1.4426950408889634
        key_block_start = 0

        while key_block_start < scan_end:
            key_offsets = key_block_start + tl.arange(0, BLOCK_K)
            key_valid = key_offsets < scan_end
            absolute_keys = kv_start + key_offsets

            k_indices = (
                absolute_keys[None, :].to(tl.int64) * 1024
                + kv_head * HEAD_DIM
                + dim_offsets[:, None]
            )
            key_values = tl.load(
                k_ptr + k_indices,
                mask=key_valid[None, :],
                other=0.0,
            )

            scores = tl.dot(q_values, key_values) * log2_scale
            attention_mask = (
                row_valid[:, None]
                & key_valid[None, :]
                & (key_offsets[None, :] < causal_limit)
            )
            scores = tl.where(
                attention_mask,
                scores,
                -float("inf"),
            )

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, block_max)
            valid_new_max = new_max != -float("inf")

            correction = tl.where(
                valid_new_max,
                tl.exp2(running_max - new_max),
                1.0,
            )
            probabilities = tl.where(
                attention_mask,
                tl.exp2(scores - new_max[:, None]),
                0.0,
            )

            v_indices = (
                absolute_keys[:, None].to(tl.int64) * 1024
                + kv_head * HEAD_DIM
                + dim_offsets[None, :]
            )
            value_values = tl.load(
                v_ptr + v_indices,
                mask=key_valid[:, None],
                other=0.0,
            )

            accumulator = (
                accumulator * correction[:, None]
                + tl.dot(probabilities.to(tl.bfloat16), value_values)
            )
            running_sum = (
                running_sum * correction
                + tl.sum(probabilities, axis=1)
            )
            running_max = tl.where(
                valid_new_max,
                new_max,
                running_max,
            )

            key_block_start += BLOCK_K

        has_attention = running_sum > 0.0
        safe_sum = tl.where(has_attention, running_sum, 1.0)
        normalized_output = accumulator / safe_sum[:, None]

        empty_output = tl.where(
            kv_len == 0,
            0.0,
            float("nan"),
        )
        normalized_output = tl.where(
            has_attention[:, None],
            normalized_output,
            empty_output,
        )

        output_indices = (
            absolute_queries[:, None].to(tl.int64) * 4096
            + query_heads[:, None] * HEAD_DIM
            + dim_offsets[None, :]
        )
        tl.store(
            output_ptr + output_indices,
            normalized_output,
            mask=row_valid[:, None],
        )

        lse_values = tl.where(
            has_attention,
            running_max + tl.log2(safe_sum),
            -float("inf"),
        )
        lse_indices = absolute_queries.to(tl.int64) * 32 + query_heads
        tl.store(
            lse_ptr + lse_indices,
            lse_values,
            mask=row_valid,
        )


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to execute the Triton attention kernel."
        )

    if q.ndim != 3 or q.shape[1:] != (32, 128):
        raise ValueError("q must have shape [total_q, 32, 128].")
    if k.ndim != 3 or k.shape[1:] != (8, 128):
        raise ValueError("k must have shape [total_kv, 8, 128].")
    if v.shape != k.shape:
        raise ValueError("v must have the same shape as k.")
    if qo_indptr.ndim != 1 or kv_indptr.ndim != 1:
        raise ValueError(
            "qo_indptr and kv_indptr must be one-dimensional."
        )
    if qo_indptr.numel() != kv_indptr.numel():
        raise ValueError(
            "qo_indptr and kv_indptr must have the same length."
        )
    if qo_indptr.numel() < 1:
        raise ValueError(
            "indptr tensors must contain at least one element."
        )

    if q.dtype != torch.bfloat16:
        raise TypeError("q must have dtype torch.bfloat16.")
    if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("k and v must have dtype torch.bfloat16.")
    if qo_indptr.dtype != torch.int32:
        raise TypeError("qo_indptr must have dtype torch.int32.")
    if kv_indptr.dtype != torch.int32:
        raise TypeError("kv_indptr must have dtype torch.int32.")

    original_device = q.device
    tensor_inputs = (q, k, v, qo_indptr, kv_indptr)

    if q.is_cuda:
        execution_device = q.device
    else:
        execution_device = None
        for tensor in tensor_inputs:
            if tensor.is_cuda:
                execution_device = tensor.device
                break
        if execution_device is None:
            execution_device = torch.device(
                "cuda",
                torch.cuda.current_device(),
            )

    def move_to_execution_device(tensor):
        if tensor.device == execution_device:
            return tensor.contiguous()
        return tensor.cuda(
            device=execution_device,
            non_blocking=False,
        ).contiguous()

    batch_size = qo_indptr.numel() - 1
    total_q = q.shape[0]
    total_kv = k.shape[0]

    if batch_size == 1:
        max_q_length = total_q
    elif batch_size > 0 and total_q > 0 and not qo_indptr.is_cuda:
        max_q_length = int(
            (qo_indptr[1:] - qo_indptr[:-1]).max().item()
        )
    else:
        max_q_length = total_q

    q_gpu = move_to_execution_device(q)
    k_gpu = move_to_execution_device(k)
    v_gpu = move_to_execution_device(v)
    qo_indptr_gpu = move_to_execution_device(qo_indptr)
    kv_indptr_gpu = move_to_execution_device(kv_indptr)

    if isinstance(sm_scale, torch.Tensor):
        if sm_scale.numel() != 1:
            raise ValueError("sm_scale must be a scalar.")
        scale_value = float(sm_scale.detach().item())
    else:
        scale_value = float(sm_scale)

    output_gpu = torch.empty_like(q_gpu)
    lse_gpu = torch.empty(
        (total_q, 32),
        dtype=torch.float32,
        device=execution_device,
    )

    if total_q > 0 and batch_size > 0 and max_q_length > 0:
        average_kv_length = (
            total_kv + batch_size - 1
        ) // batch_size

        if max_q_length >= 8 and average_kv_length <= 256:
            block_q = 8
            block_k = 32
            num_warps = 4
            num_stages = 1
        elif average_kv_length >= 512:
            block_q = 4
            block_k = 64
            num_warps = 4
            num_stages = 2
        else:
            block_q = 4
            block_k = 32
            num_warps = 4
            num_stages = 2

        num_q_tiles = triton.cdiv(max_q_length, block_q)
        grid = (num_q_tiles, 8, batch_size)

        with torch.cuda.device(execution_device):
            _fused_query_tile_attention_kernel[grid](
                q_gpu,
                k_gpu,
                v_gpu,
                qo_indptr_gpu,
                kv_indptr_gpu,
                output_gpu,
                lse_gpu,
                scale_value,
                BLOCK_Q=block_q,
                BLOCK_K=block_k,
                GROUP_SIZE=4,
                HEAD_DIM=128,
                num_warps=num_warps,
                num_stages=num_stages,
            )

    if original_device == execution_device:
        return output_gpu, lse_gpu

    return (
        output_gpu.to(original_device),
        lse_gpu.to(original_device),
    )