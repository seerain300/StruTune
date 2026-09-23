# solution=GPT-5.6-Sol_018_fused_rope_with_qk_norm_and_kv_cache_update_triton_optimized_r10 score=19.891080456436182 passed=True
import torch
import triton
import triton.language as tl


_NUM_Q_HEADS = 96
_NUM_KV_HEADS = 8
_HEAD_DIM = 128
_HALF_HEAD_DIM = 64
_Q_HEADS_PER_BLOCK = 8
_KV_HEADS_PER_BLOCK = 8
_SHORT_PREFILL_THRESHOLD = 128
_SHORT_Q_HEADS_PER_BLOCK = 4
_SHORT_KV_HEADS_PER_BLOCK = 4


@triton.jit
def _q_norm_rope_kernel(
    query_ptr,
    norm_weight_ptr,
    rope_phase_ptr,
    output_ptr,
    seq_len: tl.constexpr,
    rms_norm_eps,
    HEADS_PER_BLOCK: tl.constexpr,
):
    head_block = tl.program_id(0)
    token = tl.program_id(1)
    batch = tl.program_id(2)

    heads = head_block * HEADS_PER_BLOCK + tl.arange(0, HEADS_PER_BLOCK)
    dims = tl.arange(0, 64)

    row_base = ((batch * 96 + heads[:, None]) * seq_len + token) * 128

    x_low = tl.load(query_ptr + row_base + dims[None, :]).to(tl.float32)
    x_high = tl.load(
        query_ptr + row_base + dims[None, :] + 64
    ).to(tl.float32)

    square_sum = tl.sum(x_low * x_low + x_high * x_high, axis=1)
    inverse_rms = tl.rsqrt(square_sum * (1.0 / 128.0) + rms_norm_eps)

    weight_low = tl.load(norm_weight_ptr + dims).to(tl.float32)
    weight_high = tl.load(norm_weight_ptr + dims + 64).to(tl.float32)

    norm_low = (
        x_low * inverse_rms[:, None] * weight_low[None, :]
    ).to(tl.bfloat16)
    norm_high = (
        x_high * inverse_rms[:, None] * weight_high[None, :]
    ).to(tl.bfloat16)

    phase_base = (batch * seq_len + token) * 128
    cosine = tl.load(rope_phase_ptr + phase_base + dims)
    sine = tl.load(rope_phase_ptr + phase_base + dims + 64)

    low_cos = (norm_low * cosine[None, :]).to(tl.bfloat16)
    high_sin = (norm_high * sine[None, :]).to(tl.bfloat16)
    high_cos = (norm_high * cosine[None, :]).to(tl.bfloat16)
    low_sin = (norm_low * sine[None, :]).to(tl.bfloat16)

    rotated_low = (low_cos - high_sin).to(tl.bfloat16)
    rotated_high = (high_cos + low_sin).to(tl.bfloat16)

    tl.store(output_ptr + row_base + dims[None, :], rotated_low)
    tl.store(
        output_ptr + row_base + dims[None, :] + 64,
        rotated_high,
    )


@triton.jit
def _kv_norm_rope_cache_kernel(
    key_ptr,
    value_ptr,
    position_ids_ptr,
    cache_position_ptr,
    norm_weight_ptr,
    inv_freq_ptr,
    key_output_ptr,
    key_cache_ptr,
    value_cache_ptr,
    rope_phase_ptr,
    seq_len: tl.constexpr,
    max_position_embeddings: tl.constexpr,
    position_batch_stride,
    position_seq_stride,
    rms_norm_eps,
    HEADS_PER_BLOCK: tl.constexpr,
):
    head_block = tl.program_id(0)
    token = tl.program_id(1)
    batch = tl.program_id(2)

    heads = head_block * HEADS_PER_BLOCK + tl.arange(0, HEADS_PER_BLOCK)
    dims = tl.arange(0, 64)

    row_base = ((batch * 8 + heads[:, None]) * seq_len + token) * 128

    key_low = tl.load(key_ptr + row_base + dims[None, :]).to(tl.float32)
    key_high = tl.load(
        key_ptr + row_base + dims[None, :] + 64
    ).to(tl.float32)

    square_sum = tl.sum(
        key_low * key_low + key_high * key_high,
        axis=1,
    )
    inverse_rms = tl.rsqrt(square_sum * (1.0 / 128.0) + rms_norm_eps)

    weight_low = tl.load(norm_weight_ptr + dims).to(tl.float32)
    weight_high = tl.load(norm_weight_ptr + dims + 64).to(tl.float32)

    norm_low = (
        key_low * inverse_rms[:, None] * weight_low[None, :]
    ).to(tl.bfloat16)
    norm_high = (
        key_high * inverse_rms[:, None] * weight_high[None, :]
    ).to(tl.bfloat16)

    position = tl.load(
        position_ids_ptr
        + batch * position_batch_stride
        + token * position_seq_stride
    ).to(tl.float32)
    frequency = tl.load(inv_freq_ptr + dims).to(tl.float32)
    angle = position * frequency

    cosine = tl.cos(angle).to(tl.bfloat16)
    sine = tl.sin(angle).to(tl.bfloat16)

    phase_base = (batch * seq_len + token) * 128
    tl.store(
        rope_phase_ptr + phase_base + dims,
        cosine,
    )
    tl.store(
        rope_phase_ptr + phase_base + dims + 64,
        sine,
    )

    low_cos = (norm_low * cosine[None, :]).to(tl.bfloat16)
    high_sin = (norm_high * sine[None, :]).to(tl.bfloat16)
    high_cos = (norm_high * cosine[None, :]).to(tl.bfloat16)
    low_sin = (norm_low * sine[None, :]).to(tl.bfloat16)

    rotated_low = (low_cos - high_sin).to(tl.bfloat16)
    rotated_high = (high_cos + low_sin).to(tl.bfloat16)

    tl.store(
        key_output_ptr + row_base + dims[None, :],
        rotated_low,
    )
    tl.store(
        key_output_ptr + row_base + dims[None, :] + 64,
        rotated_high,
    )

    cache_position = tl.load(cache_position_ptr + token)
    cache_row_base = (
        (batch * 8 + heads[:, None]) * max_position_embeddings
        + cache_position
    ) * 128

    tl.store(
        key_cache_ptr + cache_row_base + dims[None, :],
        rotated_low,
    )
    tl.store(
        key_cache_ptr + cache_row_base + dims[None, :] + 64,
        rotated_high,
    )

    value_low = tl.load(value_ptr + row_base + dims[None, :])
    value_high = tl.load(value_ptr + row_base + dims[None, :] + 64)

    tl.store(
        value_cache_ptr + cache_row_base + dims[None, :],
        value_low,
    )
    tl.store(
        value_cache_ptr + cache_row_base + dims[None, :] + 64,
        value_high,
    )


@triton.jit
def _fused_decode_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    position_ids_ptr,
    cache_position_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    inv_freq_ptr,
    query_output_ptr,
    key_output_ptr,
    key_cache_ptr,
    value_cache_ptr,
    max_position_embeddings: tl.constexpr,
    position_batch_stride,
    position_seq_stride,
    rms_norm_eps,
):
    q_head_block = tl.program_id(0)
    batch = tl.program_id(1)

    dims = tl.arange(0, 64)
    q_heads = q_head_block * 4 + tl.arange(0, 4)
    q_row_base = (batch * 96 + q_heads[:, None]) * 128

    position = tl.load(
        position_ids_ptr
        + batch * position_batch_stride
    ).to(tl.float32)
    frequency = tl.load(inv_freq_ptr + dims).to(tl.float32)
    angle = position * frequency
    cosine = tl.cos(angle).to(tl.bfloat16)
    sine = tl.sin(angle).to(tl.bfloat16)

    q_low = tl.load(
        query_ptr + q_row_base + dims[None, :]
    ).to(tl.float32)
    q_high = tl.load(
        query_ptr + q_row_base + dims[None, :] + 64
    ).to(tl.float32)

    q_square_sum = tl.sum(
        q_low * q_low + q_high * q_high,
        axis=1,
    )
    q_inverse_rms = tl.rsqrt(
        q_square_sum * (1.0 / 128.0) + rms_norm_eps
    )

    q_weight_low = tl.load(
        q_norm_weight_ptr + dims
    ).to(tl.float32)
    q_weight_high = tl.load(
        q_norm_weight_ptr + dims + 64
    ).to(tl.float32)

    q_norm_low = (
        q_low * q_inverse_rms[:, None] * q_weight_low[None, :]
    ).to(tl.bfloat16)
    q_norm_high = (
        q_high * q_inverse_rms[:, None] * q_weight_high[None, :]
    ).to(tl.bfloat16)

    q_low_cos = (q_norm_low * cosine[None, :]).to(tl.bfloat16)
    q_high_sin = (q_norm_high * sine[None, :]).to(tl.bfloat16)
    q_high_cos = (q_norm_high * cosine[None, :]).to(tl.bfloat16)
    q_low_sin = (q_norm_low * sine[None, :]).to(tl.bfloat16)

    q_rotated_low = (q_low_cos - q_high_sin).to(tl.bfloat16)
    q_rotated_high = (q_high_cos + q_low_sin).to(tl.bfloat16)

    tl.store(
        query_output_ptr + q_row_base + dims[None, :],
        q_rotated_low,
    )
    tl.store(
        query_output_ptr + q_row_base + dims[None, :] + 64,
        q_rotated_high,
    )

    if q_head_block < 2:
        kv_heads = q_head_block * 4 + tl.arange(0, 4)
        kv_row_base = (batch * 8 + kv_heads[:, None]) * 128

        key_low = tl.load(
            key_ptr + kv_row_base + dims[None, :]
        ).to(tl.float32)
        key_high = tl.load(
            key_ptr + kv_row_base + dims[None, :] + 64
        ).to(tl.float32)

        key_square_sum = tl.sum(
            key_low * key_low + key_high * key_high,
            axis=1,
        )
        key_inverse_rms = tl.rsqrt(
            key_square_sum * (1.0 / 128.0) + rms_norm_eps
        )

        k_weight_low = tl.load(
            k_norm_weight_ptr + dims
        ).to(tl.float32)
        k_weight_high = tl.load(
            k_norm_weight_ptr + dims + 64
        ).to(tl.float32)

        key_norm_low = (
            key_low
            * key_inverse_rms[:, None]
            * k_weight_low[None, :]
        ).to(tl.bfloat16)
        key_norm_high = (
            key_high
            * key_inverse_rms[:, None]
            * k_weight_high[None, :]
        ).to(tl.bfloat16)

        key_low_cos = (
            key_norm_low * cosine[None, :]
        ).to(tl.bfloat16)
        key_high_sin = (
            key_norm_high * sine[None, :]
        ).to(tl.bfloat16)
        key_high_cos = (
            key_norm_high * cosine[None, :]
        ).to(tl.bfloat16)
        key_low_sin = (
            key_norm_low * sine[None, :]
        ).to(tl.bfloat16)

        key_rotated_low = (
            key_low_cos - key_high_sin
        ).to(tl.bfloat16)
        key_rotated_high = (
            key_high_cos + key_low_sin
        ).to(tl.bfloat16)

        tl.store(
            key_output_ptr + kv_row_base + dims[None, :],
            key_rotated_low,
        )
        tl.store(
            key_output_ptr + kv_row_base + dims[None, :] + 64,
            key_rotated_high,
        )

        cache_position = tl.load(cache_position_ptr)
        cache_row_base = (
            (batch * 8 + kv_heads[:, None])
            * max_position_embeddings
            + cache_position
        ) * 128

        tl.store(
            key_cache_ptr + cache_row_base + dims[None, :],
            key_rotated_low,
        )
        tl.store(
            key_cache_ptr + cache_row_base + dims[None, :] + 64,
            key_rotated_high,
        )

        value_low = tl.load(
            value_ptr + kv_row_base + dims[None, :]
        )
        value_high = tl.load(
            value_ptr + kv_row_base + dims[None, :] + 64
        )

        tl.store(
            value_cache_ptr + cache_row_base + dims[None, :],
            value_low,
        )
        tl.store(
            value_cache_ptr + cache_row_base + dims[None, :] + 64,
            value_high,
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
    batch_size, _, seq_len, _ = query.shape
    max_position_embeddings = key_cache.shape[2]

    query_rotated = torch.empty_like(query)
    key_rotated = torch.empty_like(key)

    if seq_len == 1:
        _fused_decode_kernel[(24, batch_size)](
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
            max_position_embeddings=max_position_embeddings,
            position_batch_stride=position_ids.stride(0),
            position_seq_stride=position_ids.stride(1),
            rms_norm_eps=rms_norm_eps,
            num_warps=4,
        )
        return query_rotated, key_rotated, key_cache, value_cache

    rope_phase = torch.empty(
        (batch_size, seq_len, 128),
        dtype=torch.bfloat16,
        device=query.device,
    )

    if seq_len <= _SHORT_PREFILL_THRESHOLD:
        q_heads_per_block = _SHORT_Q_HEADS_PER_BLOCK
        kv_heads_per_block = _SHORT_KV_HEADS_PER_BLOCK
        q_num_warps = 4
        kv_num_warps = 4
    else:
        q_heads_per_block = _Q_HEADS_PER_BLOCK
        kv_heads_per_block = _KV_HEADS_PER_BLOCK
        q_num_warps = 8
        kv_num_warps = 8

    kv_grid = (
        triton.cdiv(_NUM_KV_HEADS, kv_heads_per_block),
        seq_len,
        batch_size,
    )
    _kv_norm_rope_cache_kernel[kv_grid](
        key,
        value,
        position_ids,
        cache_position,
        k_norm_weight,
        inv_freq,
        key_rotated,
        key_cache,
        value_cache,
        rope_phase,
        seq_len=seq_len,
        max_position_embeddings=max_position_embeddings,
        position_batch_stride=position_ids.stride(0),
        position_seq_stride=position_ids.stride(1),
        rms_norm_eps=rms_norm_eps,
        HEADS_PER_BLOCK=kv_heads_per_block,
        num_warps=kv_num_warps,
    )

    q_grid = (
        triton.cdiv(_NUM_Q_HEADS, q_heads_per_block),
        seq_len,
        batch_size,
    )
    _q_norm_rope_kernel[q_grid](
        query,
        q_norm_weight,
        rope_phase,
        query_rotated,
        seq_len=seq_len,
        rms_norm_eps=rms_norm_eps,
        HEADS_PER_BLOCK=q_heads_per_block,
        num_warps=q_num_warps,
    )

    return query_rotated, key_rotated, key_cache, value_cache