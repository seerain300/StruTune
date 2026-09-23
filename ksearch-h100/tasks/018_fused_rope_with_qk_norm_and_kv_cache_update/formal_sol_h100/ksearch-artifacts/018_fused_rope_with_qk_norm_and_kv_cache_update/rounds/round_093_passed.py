# solution=GPT-5.6-Sol_018_fused_rope_with_qk_norm_and_kv_cache_update_triton_optimized_r13 score=24.016403105254234 passed=True
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _fused_qk_norm_rope_cache_kernel(
    query,
    key,
    value,
    position_ids,
    cache_position,
    q_norm_weight,
    k_norm_weight,
    inv_freq,
    query_rotated,
    key_rotated,
    key_cache,
    value_cache,
    q_stride_b: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_s: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_s: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_s: tl.constexpr,
    pos_stride_b: tl.constexpr,
    pos_stride_s: tl.constexpr,
    qo_stride_b: tl.constexpr,
    qo_stride_h: tl.constexpr,
    qo_stride_s: tl.constexpr,
    ko_stride_b: tl.constexpr,
    ko_stride_h: tl.constexpr,
    ko_stride_s: tl.constexpr,
    kc_stride_b: tl.constexpr,
    kc_stride_h: tl.constexpr,
    kc_stride_p: tl.constexpr,
    vc_stride_b: tl.constexpr,
    vc_stride_h: tl.constexpr,
    vc_stride_p: tl.constexpr,
    rms_norm_eps: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_HEAD_DIM: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    KV_HEADS_PER_PROGRAM: tl.constexpr,
):
    seq = tl.program_id(0)
    head_group = tl.program_id(1)
    batch = tl.program_id(2)

    half_dims = tl.arange(0, HALF_HEAD_DIM)
    head_offsets = tl.arange(0, HEADS_PER_PROGRAM)
    heads = head_group * HEADS_PER_PROGRAM + head_offsets

    position_id = tl.load(
        position_ids
        + batch * pos_stride_b
        + seq * pos_stride_s,
        cache_modifier=".ca",
    )
    position = position_id.to(tl.float32)

    frequencies = tl.load(
        inv_freq + half_dims,
        cache_modifier=".ca",
    )
    angle = position * frequencies
    cos = libdevice.cos(angle).to(tl.bfloat16)
    sin = libdevice.sin(angle).to(tl.bfloat16)

    q_base = (
        query
        + batch * q_stride_b
        + heads[:, None] * q_stride_h
        + seq * q_stride_s
    )
    q_lo = tl.load(
        q_base + half_dims[None, :]
    ).to(tl.float32)
    q_hi = tl.load(
        q_base + (half_dims + HALF_HEAD_DIM)[None, :]
    ).to(tl.float32)

    q_variance = tl.sum(q_lo * q_lo + q_hi * q_hi, axis=1) * (
        1.0 / HEAD_DIM
    )
    q_inv_rms = tl.rsqrt(q_variance + rms_norm_eps)

    q_norm_lo = (q_lo * q_inv_rms[:, None]).to(tl.bfloat16)
    q_norm_hi = (q_hi * q_inv_rms[:, None]).to(tl.bfloat16)

    q_rotated_lo = (
        q_norm_lo * cos[None, :]
        - q_norm_hi * sin[None, :]
    ).to(tl.bfloat16)
    q_rotated_hi = (
        q_norm_hi * cos[None, :]
        + q_norm_lo * sin[None, :]
    ).to(tl.bfloat16)

    qo_base = (
        query_rotated
        + batch * qo_stride_b
        + heads[:, None] * qo_stride_h
        + seq * qo_stride_s
    )
    tl.store(
        qo_base + half_dims[None, :],
        q_rotated_lo,
    )
    tl.store(
        qo_base + (half_dims + HALF_HEAD_DIM)[None, :],
        q_rotated_hi,
    )

    if head_group < NUM_KV_HEADS // KV_HEADS_PER_PROGRAM:
        cache_pos = tl.load(
            cache_position + seq,
            cache_modifier=".ca",
        )

        kv_heads = (
            head_group * KV_HEADS_PER_PROGRAM
            + tl.arange(0, KV_HEADS_PER_PROGRAM)
        )

        k_base = (
            key
            + batch * k_stride_b
            + kv_heads[:, None] * k_stride_h
            + seq * k_stride_s
        )
        k_lo = tl.load(
            k_base + half_dims[None, :]
        ).to(tl.float32)
        k_hi = tl.load(
            k_base + (half_dims + HALF_HEAD_DIM)[None, :]
        ).to(tl.float32)

        k_variance = tl.sum(k_lo * k_lo + k_hi * k_hi, axis=1) * (
            1.0 / HEAD_DIM
        )
        k_inv_rms = tl.rsqrt(k_variance + rms_norm_eps)

        k_norm_lo = (k_lo * k_inv_rms[:, None]).to(tl.bfloat16)
        k_norm_hi = (k_hi * k_inv_rms[:, None]).to(tl.bfloat16)

        k_rotated_lo = (
            k_norm_lo * cos[None, :]
            - k_norm_hi * sin[None, :]
        ).to(tl.bfloat16)
        k_rotated_hi = (
            k_norm_hi * cos[None, :]
            + k_norm_lo * sin[None, :]
        ).to(tl.bfloat16)

        ko_base = (
            key_rotated
            + batch * ko_stride_b
            + kv_heads[:, None] * ko_stride_h
            + seq * ko_stride_s
        )
        tl.store(
            ko_base + half_dims[None, :],
            k_rotated_lo,
        )
        tl.store(
            ko_base + (half_dims + HALF_HEAD_DIM)[None, :],
            k_rotated_hi,
        )

        kc_base = (
            key_cache
            + batch * kc_stride_b
            + kv_heads[:, None] * kc_stride_h
            + cache_pos * kc_stride_p
        )
        tl.store(
            kc_base + half_dims[None, :],
            k_rotated_lo,
            cache_modifier=".cg",
        )
        tl.store(
            kc_base + (half_dims + HALF_HEAD_DIM)[None, :],
            k_rotated_hi,
            cache_modifier=".cg",
        )

        v_base = (
            value
            + batch * v_stride_b
            + kv_heads[:, None] * v_stride_h
            + seq * v_stride_s
        )
        value_lo = tl.load(
            v_base + half_dims[None, :]
        )
        value_hi = tl.load(
            v_base + (half_dims + HALF_HEAD_DIM)[None, :]
        )

        vc_base = (
            value_cache
            + batch * vc_stride_b
            + kv_heads[:, None] * vc_stride_h
            + cache_pos * vc_stride_p
        )
        tl.store(
            vc_base + half_dims[None, :],
            value_lo,
            cache_modifier=".cg",
        )
        tl.store(
            vc_base + (half_dims + HALF_HEAD_DIM)[None, :],
            value_hi,
            cache_modifier=".cg",
        )


@torch.no_grad()
def run(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position_ids: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    inv_freq: torch.Tensor,
    rms_norm_eps: float,
):
    batch_size, num_q_heads, seq_len, head_dim = query.shape
    num_kv_heads = key.shape[1]

    if seq_len == 1:
        heads_per_program = 8
        kv_heads_per_program = 2
    else:
        heads_per_program = 4
        kv_heads_per_program = 1

    query_rotated = torch.empty_like(query)
    key_rotated = torch.empty_like(key)

    grid = (
        seq_len,
        triton.cdiv(num_q_heads, heads_per_program),
        batch_size,
    )

    _fused_qk_norm_rope_cache_kernel[grid](
        query,
        key,
        value,
        position_ids,
        cache_position,
        q_norm_weight,
        k_norm_weight,
        inv_freq,
        query_rotated,
        key_rotated,
        key_cache,
        value_cache,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        position_ids.stride(0),
        position_ids.stride(1),
        query_rotated.stride(0),
        query_rotated.stride(1),
        query_rotated.stride(2),
        key_rotated.stride(0),
        key_rotated.stride(1),
        key_rotated.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        rms_norm_eps,
        HEAD_DIM=head_dim,
        HALF_HEAD_DIM=head_dim // 2,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEADS_PER_PROGRAM=heads_per_program,
        KV_HEADS_PER_PROGRAM=kv_heads_per_program,
        num_warps=2,
    )

    return query_rotated, key_rotated, key_cache, value_cache