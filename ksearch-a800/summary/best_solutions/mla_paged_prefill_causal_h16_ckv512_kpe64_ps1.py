# task: mla_paged_prefill_causal_h16_ckv512_kpe64_ps1
# bench: FlashInfer | batch: formal_20260914
# final eval (official evaluator, full workloads): valid=True pass=38/38 geomean=173.171x
# feedback best (5-workload sample during search): 342.446x
# torch fallback audit: 干净 (-)
# tokens: 1,820,164

import torch
import triton
import triton.language as tl


@triton.jit
def _build_query_batch_map(
    qo_indptr,
    query_batch,
    BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    q_start = tl.load(qo_indptr + batch_idx)
    q_end = tl.load(qo_indptr + batch_idx + 1)

    for block_start in tl.range(q_start, q_end, BLOCK):
        offsets = block_start + tl.arange(0, BLOCK)
        tl.store(
            query_batch + offsets,
            batch_idx,
            mask=offsets < q_end,
        )


@triton.jit
def _head_grouped_query_token_kernel(
    q_nope,
    q_pe,
    ckv_cache,
    kpe_cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    query_batch,
    output,
    lse,
    total_queries,
    sm_scale,
    BLOCK_K: tl.constexpr,
    GROUP_HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
    SINGLE_BATCH: tl.constexpr,
    SCALE_IS_PTR: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_group_idx = tl.program_id(1)

    if SINGLE_BATCH:
        batch_idx = 0
        q_end = total_queries
    else:
        batch_idx = tl.load(query_batch + query_idx)
        q_end = tl.load(qo_indptr + batch_idx + 1)

    kv_start = tl.load(kv_indptr + batch_idx)
    kv_end = tl.load(kv_indptr + batch_idx + 1)
    kv_length = kv_end - kv_start

    valid_length = kv_length - q_end + query_idx + 1
    valid_length = tl.maximum(valid_length, 0)
    valid_length = tl.minimum(valid_length, kv_length)

    head_offsets = (
        head_group_idx * GROUP_HEADS
        + tl.arange(0, GROUP_HEADS)
    )
    ckv_offsets = tl.arange(0, HEAD_DIM_CKV)
    kpe_offsets = tl.arange(0, HEAD_DIM_KPE)

    qn_ptrs = (
        q_nope
        + query_idx * TOTAL_HEADS * HEAD_DIM_CKV
        + head_offsets[:, None] * HEAD_DIM_CKV
        + ckv_offsets[None, :]
    )
    qp_ptrs = (
        q_pe
        + query_idx * TOTAL_HEADS * HEAD_DIM_KPE
        + head_offsets[:, None] * HEAD_DIM_KPE
        + kpe_offsets[None, :]
    )

    qn = tl.load(qn_ptrs)
    qp = tl.load(qp_ptrs)

    running_max = tl.full(
        (GROUP_HEADS,),
        -float("inf"),
        tl.float32,
    )
    running_sum = tl.zeros((GROUP_HEADS,), tl.float32)
    accumulator = tl.zeros(
        (GROUP_HEADS, HEAD_DIM_CKV),
        dtype=tl.float32,
    )

    if SCALE_IS_PTR:
        scale = tl.load(sm_scale).to(tl.float32)
    else:
        scale = sm_scale

    log2_scale = scale * 1.4426950408889634

    for kv_block_start in tl.range(0, valid_length, BLOCK_K):
        key_offsets = kv_block_start + tl.arange(0, BLOCK_K)
        key_mask = key_offsets < valid_length

        page_indices = tl.load(
            kv_indices + kv_start + key_offsets,
            mask=key_mask,
            other=0,
        )

        kc_ptrs = (
            ckv_cache
            + page_indices[:, None] * HEAD_DIM_CKV
            + ckv_offsets[None, :]
        )
        kp_ptrs = (
            kpe_cache
            + page_indices[:, None] * HEAD_DIM_KPE
            + kpe_offsets[None, :]
        )

        kc = tl.load(
            kc_ptrs,
            mask=key_mask[:, None],
            other=0.0,
        )
        kp = tl.load(
            kp_ptrs,
            mask=key_mask[:, None],
            other=0.0,
        )

        logits = tl.dot(qn, tl.trans(kc))
        logits += tl.dot(qp, tl.trans(kp))
        logits *= log2_scale
        logits = tl.where(
            key_mask[None, :],
            logits,
            -float("inf"),
        )

        block_max = tl.max(logits, axis=1)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp2(running_max - new_max)
        probabilities = tl.exp2(logits - new_max[:, None])

        accumulator *= old_scale[:, None]
        accumulator += tl.dot(
            probabilities.to(tl.bfloat16),
            kc,
        )

        running_sum = (
            running_sum * old_scale
            + tl.sum(probabilities, axis=1)
        )
        running_max = new_max

    inv_sum = tl.where(
        valid_length > 0,
        1.0 / running_sum,
        0.0,
    )
    normalized_output = accumulator * inv_sum[:, None]

    output_ptrs = (
        output
        + query_idx * TOTAL_HEADS * HEAD_DIM_CKV
        + head_offsets[:, None] * HEAD_DIM_CKV
        + ckv_offsets[None, :]
    )
    tl.store(output_ptrs, normalized_output)

    query_lse = running_max + tl.log2(running_sum)
    query_lse = tl.where(
        valid_length > 0,
        query_lse,
        -float("inf"),
    )
    tl.store(
        lse + query_idx * TOTAL_HEADS + head_offsets,
        query_lse,
    )


def _prepare_tensor(tensor, device, name):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")

    if tensor.device.type == "cpu":
        return tensor.cuda(device=device).contiguous()

    if tensor.device.type != "cuda":
        raise ValueError(
            f"{name} must be on CPU or CUDA, got {tensor.device}"
        )

    if tensor.device != device:
        return tensor.to(device=device).contiguous()

    return tensor.contiguous()


@torch.no_grad()
def run(
    q_nope,
    q_pe,
    ckv_cache,
    kpe_cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
):
    tensor_args = {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
    }

    for name, tensor in tensor_args.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required to execute the Triton attention kernel"
        )

    original_device = q_nope.device
    if original_device.type not in ("cpu", "cuda"):
        raise ValueError(
            f"q_nope must be on CPU or CUDA, got {original_device}"
        )

    if q_nope.ndim != 3 or q_nope.shape[1:] != (16, 512):
        raise ValueError("q_nope must have shape [total_q, 16, 512]")
    if q_pe.ndim != 3 or q_pe.shape != (q_nope.shape[0], 16, 64):
        raise ValueError("q_pe must have shape [total_q, 16, 64]")
    if ckv_cache.ndim != 3 or ckv_cache.shape[1:] != (1, 512):
        raise ValueError(
            "ckv_cache must have shape [num_pages, 1, 512]"
        )
    if kpe_cache.ndim != 3 or kpe_cache.shape[1:] != (1, 64):
        raise ValueError(
            "kpe_cache must have shape [num_pages, 1, 64]"
        )
    if kpe_cache.shape[0] != ckv_cache.shape[0]:
        raise ValueError(
            "ckv_cache and kpe_cache must have the same num_pages"
        )
    if qo_indptr.ndim != 1 or kv_indptr.ndim != 1:
        raise ValueError(
            "qo_indptr and kv_indptr must be one-dimensional"
        )
    if qo_indptr.numel() != kv_indptr.numel():
        raise ValueError(
            "qo_indptr and kv_indptr must have the same length"
        )
    if qo_indptr.numel() < 1:
        raise ValueError(
            "indptr arrays must contain at least one element"
        )
    if kv_indices.ndim != 1:
        raise ValueError("kv_indices must be one-dimensional")

    if q_nope.dtype != torch.bfloat16 or q_pe.dtype != torch.bfloat16:
        raise TypeError(
            "q_nope and q_pe must have dtype torch.bfloat16"
        )
    if (
        ckv_cache.dtype != torch.bfloat16
        or kpe_cache.dtype != torch.bfloat16
    ):
        raise TypeError(
            "ckv_cache and kpe_cache must have dtype torch.bfloat16"
        )
    if qo_indptr.dtype != torch.int32 or kv_indptr.dtype != torch.int32:
        raise TypeError(
            "qo_indptr and kv_indptr must have dtype torch.int32"
        )
    if kv_indices.dtype != torch.int32:
        raise TypeError("kv_indices must have dtype torch.int32")

    total_q = q_nope.shape[0]
    batch_size = qo_indptr.numel() - 1

    if qo_indptr.device.type == "cpu":
        if int(qo_indptr[-1].item()) != total_q:
            raise ValueError("total_q must equal qo_indptr[-1]")

    if kv_indptr.device.type == "cpu":
        if int(kv_indptr[-1].item()) != kv_indices.numel():
            raise ValueError(
                "num_kv_indices must equal kv_indptr[-1]"
            )

    if isinstance(sm_scale, torch.Tensor):
        if sm_scale.numel() != 1:
            raise ValueError("sm_scale must be a scalar")
        if sm_scale.device.type not in ("cpu", "cuda"):
            raise ValueError(
                f"sm_scale must be on CPU or CUDA, got {sm_scale.device}"
            )
    else:
        try:
            scale_value = float(sm_scale)
        except (TypeError, ValueError) as exc:
            raise TypeError("sm_scale must be a scalar") from exc

    if original_device.type == "cuda":
        execution_device = original_device
    else:
        execution_device = torch.device(
            "cuda",
            torch.cuda.current_device(),
        )

    average_kv_length = (
        kv_indices.numel() // batch_size
        if batch_size > 0
        else 0
    )
    block_k = 32 if average_kv_length >= 32 else 16
    single_batch = batch_size == 1

    if total_q < 128:
        group_heads = 8
        num_head_groups = 2
        kernel_num_warps = 4
    else:
        group_heads = 16
        num_head_groups = 1
        kernel_num_warps = 8

    kernel_num_stages = 3 if average_kv_length >= 256 else 2

    with torch.cuda.device(execution_device):
        q_nope_gpu = _prepare_tensor(
            q_nope, execution_device, "q_nope"
        )
        q_pe_gpu = _prepare_tensor(
            q_pe, execution_device, "q_pe"
        )
        ckv_cache_gpu = _prepare_tensor(
            ckv_cache, execution_device, "ckv_cache"
        )
        kpe_cache_gpu = _prepare_tensor(
            kpe_cache, execution_device, "kpe_cache"
        )
        qo_indptr_gpu = _prepare_tensor(
            qo_indptr, execution_device, "qo_indptr"
        )
        kv_indptr_gpu = _prepare_tensor(
            kv_indptr, execution_device, "kv_indptr"
        )
        kv_indices_gpu = _prepare_tensor(
            kv_indices, execution_device, "kv_indices"
        )

        if isinstance(sm_scale, torch.Tensor):
            if sm_scale.device.type == "cuda":
                kernel_scale = _prepare_tensor(
                    sm_scale, execution_device, "sm_scale"
                )
                scale_is_ptr = True
            else:
                kernel_scale = float(sm_scale.item())
                scale_is_ptr = False
        else:
            kernel_scale = scale_value
            scale_is_ptr = False

        output_gpu = torch.empty(
            (total_q, 16, 512),
            dtype=torch.bfloat16,
            device=execution_device,
        )
        lse_gpu = torch.empty(
            (total_q, 16),
            dtype=torch.float32,
            device=execution_device,
        )

        if total_q > 0:
            if single_batch:
                query_batch = qo_indptr_gpu
            else:
                query_batch = torch.empty(
                    (total_q,),
                    dtype=torch.int32,
                    device=execution_device,
                )
                if batch_size > 0:
                    _build_query_batch_map[(batch_size,)](
                        qo_indptr_gpu,
                        query_batch,
                        BLOCK=256,
                        num_warps=4,
                    )

            _head_grouped_query_token_kernel[
                (total_q, num_head_groups)
            ](
                q_nope_gpu,
                q_pe_gpu,
                ckv_cache_gpu,
                kpe_cache_gpu,
                qo_indptr_gpu,
                kv_indptr_gpu,
                kv_indices_gpu,
                query_batch,
                output_gpu,
                lse_gpu,
                total_q,
                kernel_scale,
                BLOCK_K=block_k,
                GROUP_HEADS=group_heads,
                TOTAL_HEADS=16,
                HEAD_DIM_CKV=512,
                HEAD_DIM_KPE=64,
                SINGLE_BATCH=single_batch,
                SCALE_IS_PTR=scale_is_ptr,
                num_warps=kernel_num_warps,
                num_stages=kernel_num_stages,
            )

    if original_device.type == "cuda":
        return output_gpu, lse_gpu

    return output_gpu.cpu(), lse_gpu.cpu()