# solution=GPT-5.6-Sol_018_fused_rope_with_qk_norm_and_kv_cache_update_triton_optimized_r1 score=9.283676848694308 passed=True
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
    seq_len,
    rms_norm_eps,
    HEAD_DIM: tl.constexpr,
    HALF_HEAD_DIM: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
):
    token = tl.program_id(0)
    q_head = tl.program_id(1)

    batch = token // seq_len
    seq = token - batch * seq_len

    dims = tl.arange(0, HEAD_DIM)
    pair_dims = (dims + HALF_HEAD_DIM) % HEAD_DIM
    freq_dims = dims % HALF_HEAD_DIM

    position = tl.load(
        position_ids + batch * pos_stride_b + seq * pos_stride_s
    ).to(tl.float32)
    frequencies = tl.load(inv_freq + freq_dims)
    angle = position * frequencies
    cos = libdevice.cos(angle).to(tl.bfloat16)
    sin = libdevice.sin(angle).to(tl.bfloat16)

    q_base = (
        query
        + batch * q_stride_b
        + q_head * q_stride_h
        + seq * q_stride_s
    )
    q = tl.load(q_base + dims).to(tl.float32)
    q_variance = tl.sum(q * q, axis=0) * (1.0 / HEAD_DIM)
    q_inv_rms = tl.rsqrt(q_variance + rms_norm_eps)
    q_weight = tl.load(q_norm_weight + dims).to(tl.float32)
    q_norm = (q * q_inv_rms * q_weight).to(tl.bfloat16)

    q_pair = tl.load(q_base + pair_dims).to(tl.float32)
    q_pair_norm = (
        q_pair * q_inv_rms * tl.load(q_norm_weight + pair_dims).to(tl.float32)
    ).to(tl.bfloat16)

    q_main_term = (q_norm * cos).to(tl.bfloat16)
    q_pair_term = (q_pair_norm * sin).to(tl.bfloat16)
    q_rotated = tl.where(
        dims < HALF_HEAD_DIM,
        q_main_term - q_pair_term,
        q_main_term + q_pair_term,
    )

    qo_base = (
        query_rotated
        + batch * qo_stride_b
        + q_head * qo_stride_h
        + seq * qo_stride_s
    )
    tl.store(qo_base + dims, q_rotated)

    if q_head < NUM_KV_HEADS:
        kv_head = q_head
        k_base = (
            key
            + batch * k_stride_b
            + kv_head * k_stride_h
            + seq * k_stride_s
        )
        k = tl.load(k_base + dims).to(tl.float32)
        k_variance = tl.sum(k * k, axis=0) * (1.0 / HEAD_DIM)
        k_inv_rms = tl.rsqrt(k_variance + rms_norm_eps)
        k_weight = tl.load(k_norm_weight + dims).to(tl.float32)
        k_norm = (k * k_inv_rms * k_weight).to(tl.bfloat16)

        k_pair = tl.load(k_base + pair_dims).to(tl.float32)
        k_pair_norm = (
            k_pair
            * k_inv_rms
            * tl.load(k_norm_weight + pair_dims).to(tl.float32)
        ).to(tl.bfloat16)

        k_main_term = (k_norm * cos).to(tl.bfloat16)
        k_pair_term = (k_pair_norm * sin).to(tl.bfloat16)
        k_rotated = tl.where(
            dims < HALF_HEAD_DIM,
            k_main_term - k_pair_term,
            k_main_term + k_pair_term,
        )

        ko_base = (
            key_rotated
            + batch * ko_stride_b
            + kv_head * ko_stride_h
            + seq * ko_stride_s
        )
        tl.store(ko_base + dims, k_rotated)

        cache_pos = tl.load(cache_position + seq)
        kc_base = (
            key_cache
            + batch * kc_stride_b
            + kv_head * kc_stride_h
            + cache_pos * kc_stride_p
        )
        tl.store(kc_base + dims, k_rotated)

        v_base = (
            value
            + batch * v_stride_b
            + kv_head * v_stride_h
            + seq * v_stride_s
        )
        value_row = tl.load(v_base + dims)
        vc_base = (
            value_cache
            + batch * vc_stride_b
            + kv_head * vc_stride_h
            + cache_pos * vc_stride_p
        )
        tl.store(vc_base + dims, value_row)


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

    query_rotated = torch.empty_like(query)
    key_rotated = torch.empty_like(key)

    grid = (batch_size * seq_len, num_q_heads)
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
        seq_len,
        rms_norm_eps,
        HEAD_DIM=head_dim,
        HALF_HEAD_DIM=head_dim // 2,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        num_warps=4,
    )

    return query_rotated, key_rotated, key_cache, value_cache