# solution=GPT-5.6-Sol_mla_paged_decode_h16_ckv512_kpe64_ps1_triton_optimized_r5 score=-1.0 passed=False
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
    sm_scale_log2,
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

    q_nope = tl.load(q_nope_ptr + q_nope_offsets)
    q_pe = tl.load(q_pe_ptr + q_pe_offsets)

    running_max = tl.full(
        (HEADS_PER_GROUP,),
        -float("inf"),
        tl.float32,
    )
    running_sum = tl.zeros((HEADS_PER_GROUP,), tl.float32)
    output_acc = tl.zeros(
        (HEADS_PER_GROUP, HEAD_DIM_CKV),
        tl.float32,
    )

    for position in range(page_begin, page_end):
        page_idx = tl.load(kv_indices_ptr + position)

        ckv = tl.load(
            ckv_cache_ptr + page_idx * HEAD_DIM_CKV + ckv_offsets
        )
        kpe = tl.load(
            kpe_cache_ptr + page_idx * HEAD_DIM_KPE + kpe_offsets
        )

        nope_logits = tl.dot(
            q_nope,
            ckv[:, None],
            out_dtype=tl.float32,
        )[:, 0]
        pe_logits = tl.dot(
            q_pe,
            kpe[:, None],
            out_dtype=tl.float32,
        )[:, 0]
        logits = (nope_logits + pe_logits) * sm_scale_log2

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

    output_values = output_acc / denominator[:, None]
    output_values = tl.where(
        nonempty[:, None],
        output_values,
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


@triton.jit
def _blocked_decode(
    q_nope_ptr,
    q_pe_ptr,
    ckv_cache_ptr,
    kpe_cache_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    output_ptr,
    lse_ptr,
    sm_scale_log2,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    value_block = tl.program_id(1)

    heads = tl.arange(0, 16)
    ckv_dims = tl.arange(0, 512