# solution=GPT-5.6-Sol_mla_paged_decode_h16_ckv512_kpe64_ps1_triton_optimized_r3 score=-1.0 passed=False
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _head_grouped_streaming_decode(
    q_nope_ptr,
    q_pe_ptr,
    ckv_cache_ptr,
    kpe_cache_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    output_ptr,
    lse_ptr,
    sm_scale_log2: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_group = tl.program_id(1)

    ckv_offsets = tl.arange(0, HEAD_DIM_CKV)
    kpe_offsets = tl.arange(0, HEAD_DIM_KPE)
    heads = head_group * HEADS_PER_GROUP + tl.arange(0, HEADS_PER_GROUP)

    page_begin = tl.load(kv_indptr_ptr + batch_idx)
    page_end = tl.load(kv_indptr_ptr + batch_idx + 1)

    q_nope_offsets = (
        batch_idx * 16 * HEAD_DIM_CKV
        + heads[:, None] * HEAD_DIM_CKV
        + ckv_offsets[None, :]
    )
    q_pe_offsets = (
        batch_idx * 16 * HEAD_DIM_KPE
        + heads[:, None] * HEAD_DIM_KPE
        + kpe_offsets[None, :]
    )

    q_nope = tl.load(q_nope_ptr + q_nope_offsets).to(tl.float32)
    q_pe = tl.load(q_pe_ptr + q_pe_offsets).to(tl.float32)

    running_max = tl.full((HEADS_PER_GROUP,), -float("inf"), tl.float32)
    running_sum = tl.zeros((HEADS_PER_GROUP,), tl.float32)
    output_acc = tl.zeros((HEADS_PER_GROUP, HEAD_DIM_CKV), tl.float32)

    for position in range(page_begin, page_end):
        page_idx = tl.load(kv_indices_ptr + position)

        ckv = tl.load(
            ckv_cache_ptr + page_idx * HEAD_DIM_CKV + ckv_offsets
        ).to(tl.float32)
        kpe = tl.load(
            kpe_cache_ptr + page_idx * HEAD_DIM_KPE + kpe_offsets
        ).to(tl.float32)

        logits = (
            tl.sum(q_nope * ckv[None, :], axis=1)
            + tl.sum(q_pe * kpe[None, :], axis=1)
        ) * sm_scale_log2

        new_max = tl.maximum(running_max, logits)
        previous_scale = tl.exp2(running_max - new_max)
        token_scale = tl.exp2(logits - new_max)

        output_acc = (
            output_acc * previous_scale[:, None]
            + token_scale[:, None] * ckv[None, :]
        )
        running_sum = running_sum * previous_scale + token_scale
        running_max = new_max

    nonempty = running_sum > 0.0
    denominator = tl.where(nonempty, running_sum, 1.0)
    output_values = tl.where(
        nonempty[:, None],
        output_acc / denominator[:, None],
        0.0,
    )

    output_offsets = (
        batch_idx * 16 * HEAD_DIM_CKV
        + heads[:, None] * HEAD_DIM_CKV
        + ckv_offsets[None, :]
    )
    tl.store(output_ptr + output_offsets, output_values)

    lse_values = running_max + tl.log2(denominator)
    lse_values = tl.where(nonempty, lse_values, -float("inf"))
    tl.store(lse_ptr + batch_idx * 16 + heads, lse_values)


@torch.no_grad()
def run(
    q_nope,
    q_pe,
    ckv_cache,
    kpe_cache,
    kv_indptr,
    kv_indices,
    sm_scale,
):
    named_tensors = (
        ("q_nope", q_nope),
        ("q_pe", q_pe),
        ("ckv_cache", ckv_cache),
        ("kpe_cache", kpe_cache),
        ("kv_indptr", kv_indptr),
        ("kv_indices", kv_indices),
    )

    for name, tensor in named_tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    if q_nope.ndim != 3 or q_nope.shape[1:] != (16, 512):
        raise ValueError("q_nope must have shape [batch_size, 16, 512]")

    batch_size = q_nope.shape[0]
    if q_pe.ndim != 3 or q_pe.shape != (batch_size, 16, 64):
        raise ValueError("q_pe must have shape [batch_size, 16, 64]")
    if ckv_cache.ndim != 3 or ckv_cache.shape[1:] != (1, 512):
        raise ValueError("ckv_cache must have shape [num_pages, 1, 512]")
    if kpe_cache.ndim != 3 or kpe_cache.shape[1:] != (1, 64):
        raise ValueError("kpe_cache must have shape [num_pages, 1, 64]")
    if ckv_cache.shape[0] != kpe_cache.shape[0]:
        raise ValueError("ckv_cache and kpe_cache must have equal page counts")
    if kv_indptr.ndim != 1 or kv_indptr.numel() != batch_size + 1:
        raise ValueError("kv_indptr length must equal batch_size + 1")
    if kv_indices.ndim != 1:
        raise ValueError("kv_indices must be one-dimensional")

    for name, tensor in named_tensors[:4]:
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must have dtype torch.bfloat16")
    if kv_indptr.dtype != torch.int32:
        raise TypeError("kv_indptr must have dtype torch.int32")
    if kv_indices.dtype != torch.int32:
        raise TypeError("kv_indices must have dtype torch.int32")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to execute the Triton kernel")

    original_device = q_nope.device
    execution_device = (
        original_device
        if original_device.type == "cuda"
        else torch.device("cuda", torch.cuda.current_device())
    )

    for name, tensor in named_tensors:
        if tensor.is_cuda and tensor.device != execution_device:
            raise ValueError(f"{name} must be on {execution_device}, or on CPU")

    if kv_indptr.device.type == "cpu":
        if int(kv_indptr[-1].item()) != kv_indices.numel():
            raise ValueError(
                "kv_indices length must equal the final kv_indptr value"
            )

    def on_execution_device(tensor):
        if tensor.device == execution_device:
            return tensor.contiguous()
        return tensor.to(execution_device).contiguous()

    q_nope_gpu = on_execution_device(q_nope)
    q_pe_gpu = on_execution_device(q_pe)
    ckv_cache_gpu = on_execution_device(ckv_cache)
    kpe_cache_gpu = on_execution_device(kpe_cache)
    kv_indptr_gpu = on_execution_device(kv_indptr)
    kv_indices_gpu = on_execution_device(kv_indices)

    if isinstance(sm_scale, torch.Tensor):
        if sm_scale.numel() != 1:
            raise ValueError("sm_scale must be a scalar")
        sm_scale_value = float(sm_scale.item())
    else:
        sm_scale_value = float(sm_scale)

    output_gpu = torch.empty_like(q_nope_gpu)
    lse_gpu = torch.empty(
        (batch_size, 16), dtype=torch.float32, device=execution_device
    )

    _head_grouped_streaming_decode[(batch_size, 2)](
        q_nope_gpu,
        q_pe_gpu,
        ckv_cache_gpu,
        kpe_cache_gpu,
        kv_indptr_gpu,
        kv_indices_gpu,
        output_gpu,
        lse_gpu,
        sm_scale_value / math.log(2.0),
        HEADS_PER_GROUP=8,
        HEAD_DIM_CKV=512,
        HEAD_DIM_KPE=64,
        num_warps=8,
        num_stages=1,
    )

    if original_device.type == "cuda":
        return output_gpu, lse_gpu
    return output_gpu.to(original_device), lse_gpu.to(original_device)