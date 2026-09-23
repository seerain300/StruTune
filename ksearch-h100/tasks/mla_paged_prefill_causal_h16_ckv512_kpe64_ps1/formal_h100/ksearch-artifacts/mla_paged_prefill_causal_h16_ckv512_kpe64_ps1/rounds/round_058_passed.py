# solution=GPT-5.6-Sol_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1_triton_optimized_r4 score=268.41081736853243 passed=True
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _mla_paged_prefill_grouped_kernel(
    q_nope,
    q_pe,
    ckv_cache,
    kpe_cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    output,
    lse,
    sm_scale,
    batch_size,
    LOG2E: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    D_CKV: tl.constexpr,
    D_KPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SINGLE_BATCH: tl.constexpr,
):
    query_idx = tl.program_id(0)

    if SINGLE_BATCH:
        batch_idx = 0
    else:
        lo = 0
        hi = batch_size
        while lo < hi:
            mid = (lo + hi) // 2
            query_end_mid = tl.load(qo_indptr + mid + 1)
            go_right = query_idx >= query_end_mid
            lo = tl.where(go_right, mid + 1, lo)
            hi = tl.where(go_right, hi, mid)
        batch_idx = lo

    query_start = tl.load(qo_indptr + batch_idx)
    query_end = tl.load(qo_indptr + batch_idx + 1)
    kv_start = tl.load(kv_indptr + batch_idx)
    kv_end = tl.load(kv_indptr + batch_idx + 1)

    query_len = query_end - query_start
    kv_len = kv_end - kv_start
    query_pos = query_idx - query_start

    valid_len = kv_len - query_len + query_pos + 1
    valid_len = tl.maximum(0, tl.minimum(valid_len, kv_len))

    heads = tl.arange(0, NUM_HEADS)
    d_ckv = tl.arange(0, D_CKV)
    d_kpe = tl.arange(0, D_KPE)

    q_nope_offsets = (
        query_idx * NUM_HEADS * D_CKV
        + heads[:, None] * D_CKV
        + d_ckv[None, :]
    )
    q_pe_offsets = (
        query_idx * NUM_HEADS * D_KPE
        + heads[:, None] * D_KPE
        + d_kpe[None, :]
    )

    qn = tl.load(q_nope + q_nope_offsets)
    qp = tl.load(q_pe + q_pe_offsets)

    running_max = tl.full((NUM_HEADS,), -float("inf"), tl.float32)
    running_sum = tl.zeros((NUM_HEADS,), dtype=tl.float32)
    output_acc = tl.zeros((NUM_HEADS, D_CKV), dtype=tl.float32)

    block_start = 0
    while block_start < valid_len:
        n = block_start + tl.arange(0, BLOCK_N)
        n_mask = n < valid_len

        pages = tl.load(
            kv_indices + kv_start + n,
            mask=n_mask,
            other=0,
        )

        ckv_offsets = pages[:, None] * D_CKV + d_ckv[None, :]
        kpe_offsets = pages[:, None] * D_KPE + d_kpe[None, :]

        ckv = tl.load(
            ckv_cache + ckv_offsets,
            mask=n_mask[:, None],
            other=0.0,
        )
        kpe = tl.load(
            kpe_cache + kpe_offsets,
            mask=n_mask[:, None],
            other=0.0,
        )

        score = tl.dot(qn, tl.trans(ckv))
        score += tl.dot(qp, tl.trans(kpe))
        score *= sm_scale * LOG2E
        score = tl.where(n_mask[None, :], score, -float("inf"))

        block_max = tl.max(score, axis=1)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp2(running_max - new_max)
        probabilities = tl.exp2(score - new_max[:, None])

        output_acc *= old_scale[:, None]
        output_acc += tl.dot(probabilities.to(tl.bfloat16), ckv)
        running_sum = (
            running_sum * old_scale
            + tl.sum(probabilities, axis=1)
        )
        running_max = new_max
        block_start += BLOCK_N

    has_values = running_sum > 0.0
    denominator = tl.where(has_values, running_sum, 1.0)

    result = output_acc / denominator[:, None]
    result = tl.where(has_values[:, None], result, 0.0)
    lse_value = tl.where(
        has_values,
        running_max + tl.log2(denominator),
        -float("inf"),
    )

    output_offsets = (
        query_idx * NUM_HEADS * D_CKV
        + heads[:, None] * D_CKV
        + d_ckv[None, :]
    )
    lse_offsets = query_idx * NUM_HEADS + heads

    tl.store(output + output_offsets, result)
    tl.store(lse + lse_offsets, lse_value)


@triton.jit
def _mla_paged_prefill_kernel(
    q_nope,
    q_pe,
    ckv_cache,
    kpe_cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    output,
    lse,
    sm_scale,
    batch_size,
    LOG2E: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    D_CKV: tl.constexpr,
    D_KPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SINGLE_BATCH: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if SINGLE_BATCH:
        batch_idx = 0
    else:
        lo = 0
        hi = batch_size
        while lo < hi:
            mid = (lo + hi) // 2
            query_end_mid = tl.load(qo_indptr + mid + 1)
            go_right = query_idx >= query_end_mid
            lo = tl.where(go_right, mid + 1, lo)
            hi = tl.where(go_right, hi, mid)
        batch_idx = lo

    query_start = tl.load(qo_indptr + batch_idx)
    query_end = tl.load(qo_indptr + batch_idx + 1)
    kv_start = tl.load(kv_indptr + batch_idx)
    kv_end = tl.load(kv_indptr + batch_idx + 1)

    query_len = query_end - query_start
    kv_len = kv_end - kv_start
    query_pos = query_idx - query_start

    valid_len = kv_len - query_len + query_pos + 1
    valid_len = tl.maximum(0, tl.minimum(valid_len, kv_len))

    d_ckv = tl.arange(0, D_CKV)
    d_kpe = tl.arange(0, D_KPE)

    q_nope_offsets = (
        query_idx * NUM_HEADS * D_CKV
        + head_idx * D_CKV
        + d_ckv
    )
    q_pe_offsets = (
        query_idx * NUM_HEADS * D_KPE
        + head_idx * D_KPE
        + d_kpe
    )

    qn = tl.load(q_nope + q_nope_offsets).to(tl.float32)
    qp = tl.load(q_pe + q_pe_offsets).to(tl.float32)

    running_max = -float("inf")
    running_sum = 0.0
    output_acc = tl.zeros((D_CKV,), dtype=tl.float32)

    block_start = 0
    while block_start < valid_len:
        n = block_start + tl.arange(0, BLOCK_N)
        n_mask = n < valid_len

        pages = tl.load(
            kv_indices + kv_start + n,
            mask=n_mask,
            other=0,
        )

        ckv_offsets = pages[:, None] * D_CKV + d_ckv[None, :]
        kpe_offsets = pages[:, None] * D_KPE + d_kpe[None, :]

        ckv = tl.load(
            ckv_cache + ckv_offsets,
            mask=n_mask[:, None],
            other=0.0,
        )
        kpe = tl.load(
            kpe_cache + kpe_offsets,
            mask=n_mask[:, None],
            other=0.0,
        )

        ckv_f32 = ckv.to(tl.float32)
        kpe_f32 = kpe.to(tl.float32)

        score = tl.sum(ckv_f32 * qn[None, :], axis=1)
        score += tl.sum(kpe_f32 * qp[None, :], axis=1)
        score *= sm_scale * LOG2E
        score = tl.where(n_mask, score, -float("inf"))

        block_max = tl.max(score, axis=0)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp2(running_max - new_max)
        probabilities = tl.exp2(score - new_max)

        output_acc *= old_scale
        output_acc += tl.sum(
            probabilities[:, None] * ckv_f32,
            axis=0,
        )
        running_sum = (
            running_sum * old_scale
            + tl.sum(probabilities, axis=0)
        )
        running_max = new_max
        block_start += BLOCK_N

    has_values = running_sum > 0.0
    denominator = tl.where(has_values, running_sum, 1.0)

    result = output_acc / denominator
    result = tl.where(has_values, result, 0.0)
    lse_value = tl.where(
        has_values,
        running_max + tl.log2(denominator),
        -float("inf"),
    )

    output_offsets = (
        query_idx * NUM_HEADS * D_CKV
        + head_idx * D_CKV
        + d_ckv
    )
    lse_offset = query_idx * NUM_HEADS + head_idx

    tl.store(output + output_offsets, result)
    tl.store(lse + lse_offset, lse_value)


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
        raise RuntimeError("CUDA is required to execute the Triton kernel")

    if q_nope.ndim != 3:
        raise ValueError("q_nope must have shape [total_q, 16, 512]")

    total_q, num_qo_heads, head_dim_ckv = q_nope.shape

    if num_qo_heads != 16:
        raise ValueError(
            f"num_qo_heads must be 16, got {num_qo_heads}"
        )
    if head_dim_ckv != 512:
        raise ValueError(
            f"head_dim_ckv must be 512, got {head_dim_ckv}"
        )
    if q_pe.shape != (total_q, 16, 64):
        raise ValueError("q_pe must have shape [total_q, 16, 64]")
    if ckv_cache.ndim != 3 or ckv_cache.shape[1:] != (1, 512):
        raise ValueError(
            "ckv_cache must have shape [num_pages, 1, 512]"
        )
    if kpe_cache.ndim != 3 or kpe_cache.shape[1:] != (1, 64):
        raise ValueError(
            "kpe_cache must have shape [num_pages, 1, 64]"
        )
    if ckv_cache.shape[0] != kpe_cache.shape[0]:
        raise ValueError(
            "ckv_cache and kpe_cache must have the same num_pages"
        )
    if qo_indptr.ndim != 1:
        raise ValueError("qo_indptr must be one-dimensional")
    if kv_indptr.ndim != 1:
        raise ValueError("kv_indptr must be one-dimensional")
    if kv_indices.ndim != 1:
        raise ValueError("kv_indices must be one-dimensional")
    if qo_indptr.shape != kv_indptr.shape:
        raise ValueError(
            "qo_indptr and kv_indptr must have the same shape"
        )
    if qo_indptr.numel() < 2:
        raise ValueError(
            "indptr tensors must contain at least two elements"
        )

    if q_nope.dtype != torch.bfloat16:
        raise TypeError("q_nope must have dtype torch.bfloat16")
    if q_pe.dtype != torch.bfloat16:
        raise TypeError("q_pe must have dtype torch.bfloat16")
    if ckv_cache.dtype != torch.bfloat16:
        raise TypeError("ckv_cache must have dtype torch.bfloat16")
    if kpe_cache.dtype != torch.bfloat16:
        raise TypeError("kpe_cache must have dtype torch.bfloat16")
    if qo_indptr.dtype != torch.int32:
        raise TypeError("qo_indptr must have dtype torch.int32")
    if kv_indptr.dtype != torch.int32:
        raise TypeError("kv_indptr must have dtype torch.int32")
    if kv_indices.dtype != torch.int32:
        raise TypeError("kv_indices must have dtype torch.int32")

    original_device = q_nope.device

    cuda_devices = {
        tensor.device
        for tensor in tensor_args.values()
        if tensor.is_cuda
    }
    if len(cuda_devices) > 1:
        raise ValueError(
            "all CUDA input tensors must be on the same device"
        )

    if cuda_devices:
        execution_device = next(iter(cuda_devices))
    else:
        execution_device = torch.device(
            "cuda", torch.cuda.current_device()
        )

    def prepare(tensor):
        if tensor.device.type == "cpu":
            return tensor.cuda(
                device=execution_device,
                non_blocking=False,
            ).contiguous()
        if tensor.device != execution_device:
            return tensor.to(
                device=execution_device,
                non_blocking=False,
            ).contiguous()
        return tensor.contiguous()

    with torch.cuda.device(execution_device):
        q_nope_cuda = prepare(q_nope)
        q_pe_cuda = prepare(q_pe)
        ckv_cache_cuda = prepare(ckv_cache)
        kpe_cache_cuda = prepare(kpe_cache)
        qo_indptr_cuda = prepare(qo_indptr)
        kv_indptr_cuda = prepare(kv_indptr)
        kv_indices_cuda = prepare(kv_indices)

        if isinstance(sm_scale, torch.Tensor):
            if sm_scale.numel() != 1:
                raise ValueError("sm_scale must be a scalar")
            sm_scale_value = float(sm_scale.detach().cpu().item())
        else:
            sm_scale_value = float(sm_scale)

        num_kv_indices = kv_indices_cuda.numel()

        output_cuda = torch.empty_like(q_nope_cuda)
        lse_cuda = torch.empty(
            (total_q, 16),
            dtype=torch.float32,
            device=execution_device,
        )

        if total_q > 0:
            batch_size = qo_indptr_cuda.numel() - 1
            single_batch = batch_size == 1

            common_args = (
                q_nope_cuda,
                q_pe_cuda,
                ckv_cache_cuda,
                kpe_cache_cuda,
                qo_indptr_cuda,
                kv_indptr_cuda,
                kv_indices_cuda,
                output_cuda,
                lse_cuda,
                sm_scale_value,
                batch_size,
            )

            if total_q >= 16:
                average_query_len = total_q / batch_size
                average_prefix_len = max(
                    0.0,
                    (num_kv_indices - total_q) / batch_size,
                )
                estimated_valid_len = (
                    average_prefix_len + 0.5 * average_query_len
                )

                if estimated_valid_len <= 16.0:
                    block_n = 16
                elif estimated_valid_len <= 64.0:
                    block_n = 32
                else:
                    block_n = 64

                _mla_paged_prefill_grouped_kernel[(total_q,)](
                    *common_args,
                    LOG2E=1.0 / math.log(2.0),
                    NUM_HEADS=16,
                    D_CKV=512,
                    D_KPE=64,
                    BLOCK_N=block_n,
                    SINGLE_BATCH=single_batch,
                    num_warps=8,
                    num_stages=1,
                )
            else:
                _mla_paged_prefill_kernel[(total_q, 16)](
                    *common_args,
                    LOG2E=1.0 / math.log(2.0),
                    NUM_HEADS=16,
                    D_CKV=512,
                    D_KPE=64,
                    BLOCK_N=8,
                    SINGLE_BATCH=single_batch,
                    num_warps=4,
                    num_stages=2,
                )

    if original_device.type == "cuda":
        output = output_cuda.to(original_device)
        lse_result = lse_cuda.to(original_device)
    else:
        output = output_cuda.cpu()
        lse_result = lse_cuda.cpu()

    return output, lse_result