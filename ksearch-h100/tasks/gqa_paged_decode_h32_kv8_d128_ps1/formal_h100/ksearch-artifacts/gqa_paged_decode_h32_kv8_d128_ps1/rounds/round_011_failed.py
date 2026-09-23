# solution=GPT-5.6-Sol_gqa_paged_decode_h32_kv8_d128_ps1_triton_optimized_r11 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


@triton.jit
def _gqa_paged_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    output_ptr,
    lse_ptr,
    sm_scale,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_page: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_page: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_d: tl.constexpr,
    out_stride_b: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    lse_stride_b: tl.constexpr,
    lse_stride_h: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PREFETCH_KV: tl.constexpr,
):
    program_id = tl.program_id(0)
    batch_idx = program_id // 32
    query_head = program_id % 32
    kv_head = query_head // 4

    dims = tl.arange(0, BLOCK_D)
    q_offsets = (
        batch_idx * q_stride_b
        + query_head * q_stride_h
        + dims * q_stride_d
    )
    q = tl.load(q_ptr + q_offsets).to(tl.float32)

    page_start = tl.load(kv_indptr_ptr + batch_idx)
    page_end = tl.load(kv_indptr_ptr + batch_idx + 1)

    max_logit = -float("inf")
    denominator = 0.0
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    token_offsets = tl.arange(0, BLOCK_N)
    page_offset = page_start

    if PREFETCH_KV:
        current_positions = page_offset + token_offsets
        current_mask = current_positions < page_end
        current_page_ids = tl.load(
            kv_indices_ptr + current_positions,
            mask=current_mask,
            other=0,
        )

        current_k_offsets = (
            current_page_ids[:, None] * k_stride_page
            + kv_head * k_stride_h
            + dims[None, :] * k_stride_d
        )
        current_v_offsets = (
            current_page_ids[:, None] * v_stride_page
            + kv_head * v_stride_h
            + dims[None, :] * v_stride_d
        )
        current_k = tl.load(
            k_ptr + current_k_offsets,
            mask=current_mask[:, None],
            other=0.0,
        )
        current_v = tl.load(
            v_ptr + current_v_offsets,
            mask=current_mask[:, None],
            other=0.0,
        )

        while page_offset < page_end:
            k = current_k.to(tl.float32)
            logits = tl.sum(k * q[None, :], axis=1)
            logits = logits * sm_scale * 1.4426950408889634
            logits = tl.where(current_mask, logits, -float("inf"))

            block_max = tl.max(logits, axis=0)
            new_max = tl.maximum(max_logit, block_max)
            old_scale = tl.exp2(max_logit - new_max)
            weights = tl.exp2(logits - new_max)

            v = current_v.to(tl.float32)
            accumulator = (
                accumulator * old_scale
                + tl.sum(weights[:, None] * v, axis=0)
            )
            denominator = (
                denominator * old_scale + tl.sum(weights, axis=0)
            )
            max_logit = new_max

            page_offset += BLOCK_N
            next_positions = page_offset + token_offsets
            next_mask = next_positions < page_end
            next_page_ids = tl.load(
                kv_indices_ptr + next_positions,
                mask=next_mask,
                other=0,
            )

            next_k_offsets = (
                next_page_ids[:, None] * k_stride_page
                + kv_head * k_stride_h
                + dims[None, :] * k_stride_d
            )
            next_v_offsets = (
                next_page_ids[:, None] * v_stride_page
                + kv_head * v_stride_h
                + dims[None, :] * v_stride_d
            )
            current_k = tl.load(
                k_ptr + next_k_offsets,
                mask=next_mask[:, None],
                other=0.0,
            )
            current_v = tl.load(
                v_ptr + next_v_offsets,
                mask=next_mask[:, None],
                other=0.0,
            )
            current_mask = next_mask
    else:
        while page_offset < page_end:
            positions = page_offset + token_offsets
            token_mask = positions < page_end
            page_ids = tl.load(
                kv_indices_ptr + positions,
                mask=token_mask,
                other=0,
            )

            k_offsets = (
                page_ids[:, None] * k_stride_page
                + kv_head * k_stride_h
                + dims[None, :] * k_stride_d
            )
            k = tl.load(
                k_ptr + k_offsets,
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)

            logits = tl.sum(k * q[None, :], axis=1)
            logits = logits * sm_scale * 1.4426950408889634
            logits = tl.where(token_mask, logits, -float("inf"))

            block_max = tl.max(logits, axis=0)
            new_max = tl.maximum(max_logit, block_max)
            old_scale = tl.exp2(max_logit - new_max)
            weights = tl.exp2(logits - new_max)

            v_offsets = (
                page_ids[:, None] * v_stride_page
                + kv_head * v_stride_h
                + dims[None, :] * v_stride_d
            )
            v = tl.load(
                v_ptr + v_offsets,
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)

            accumulator = (
                accumulator * old_scale
                + tl.sum(weights[:, None] * v, axis=0)
            )
            denominator = (
                denominator * old_scale + tl.sum(weights, axis=0)
            )
            max_logit = new_max
            page_offset += BLOCK_N

    output_offsets = (
        batch_idx * out_stride_b
        + query_head * out_stride_h
        + dims * out_stride_d
    )
    lse_offset =