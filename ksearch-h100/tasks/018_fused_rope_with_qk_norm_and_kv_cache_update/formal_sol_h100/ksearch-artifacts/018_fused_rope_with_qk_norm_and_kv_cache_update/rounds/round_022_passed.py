# solution=GPT-5.6-Sol_018_fused_rope_with_qk_norm_and_kv_cache_update_triton_optimized_r22 score=20.163842206154563 passed=True
import torch
import triton
import triton.language as tl


HEAD_DIM = 128
HALF_HEAD_DIM = 64
NUM_Q_HEADS = 96
NUM_KV_HEADS = 8
ROWS_PER_PROGRAM = 8
LONG_Q_ROWS_PER_PROGRAM = 16
VERY_LONG_Q_ROWS_PER_PROGRAM = 8
LONG_PREFILL_THRESHOLD = 512
VERY_LONG_PREFILL_THRESHOLD = 2048


@triton.jit
def _build_rope_phase_kernel(
    position_ids,
    inv_freq,
    phase,
    stride_pos_b: tl.constexpr,
    stride_pos_s: tl.constexpr,
    seq_len: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    batch = token // seq_len
    seq = token - batch * seq_len

    dims = tl.arange(0, BLOCK)
    position = tl.load(
        position_ids + batch * stride_pos_b + seq * stride_pos_s
    ).to(tl.float32)
    frequency = position * tl.load(inv_freq + dims).to(tl.float32)

    phase_base = token * (2 * BLOCK)
    tl.store(
        phase + phase_base + dims,
        tl.cos(frequency).to(tl.bfloat16),
    )
    tl.store(
        phase + phase_base + BLOCK + dims,
        tl.sin(frequency).to(tl.bfloat16),
    )


@triton.jit
def _query_norm_rope_kernel(
    query,
    q_norm_weight,
    phase,
    query_rotated,
    rms_norm_eps,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    PHASE_BATCH_STRIDE: tl.constexpr,
    TOKEN_MAJOR: tl.constexpr,
):
    block = tl.program_id(0)
    row = tl.arange(0, BLOCK_ROWS)[:, None]
    dims = tl.arange(0, HALF_DIM)[None, :]

    if TOKEN_MAJOR:
        groups_per_token: tl.constexpr = num_heads // BLOCK_ROWS
        token = block // groups_per_token
        group = block - token * groups_per_token
        batch = token // seq_len
        seq = token - batch * seq_len
        head = group * BLOCK_ROWS + row
        rows = (batch * num_heads + head) * seq_len + seq
    else:
        rows = block * BLOCK_ROWS + row
        seq = rows % seq_len
        batch_head = rows // seq_len
        batch = batch_head // num_heads

    row_base = rows * (2 * HALF_DIM)

    x0 = tl.load(query + row_base + dims).to(tl.float32)
    x1 = tl.load(query + row_base + HALF_DIM + dims).to(tl.float32)

    square_sum = tl.sum(x0 * x0 + x1 * x1, axis=1)
    inv_rms = tl.expand_dims(
        tl.rsqrt(
            square_sum * (1.0 / (2 * HALF_DIM)) + rms_norm_eps
        ),
        1,
    )

    w0 = tl.load(q_norm_weight + dims).to(tl.float32)
    w1 = tl.load(q_norm_weight + HALF_DIM + dims).to(tl.float32)
    n0 = (x0 * inv_rms * w0).to(tl.bfloat16)
    n1 = (x1 * inv_rms * w1).to(tl.bfloat16)

    phase_base = batch * PHASE_BATCH_STRIDE + seq * (2 * HALF_DIM)
    cosine = tl.load(phase + phase_base + dims)
    sine = tl.load(phase + phase_base + HALF_DIM + dims)

    p0 = (n0 * cosine).to(tl.bfloat16)
    p1 = (n1 * sine).to(tl.bfloat16)
    p2 = (n1 * cosine).to(tl.bfloat16)
    p3 = (n0 * sine).to(tl.bfloat16)

    out0 = (p0 - p1).to(tl.bfloat16)
    out1 = (p2 + p3).to(tl.bfloat16)

    tl.store(query_rotated + row_base + dims, out0)
    tl.store(query_rotated + row_base + HALF_DIM + dims, out1)


@triton.jit
def _kv_norm_rope_cache_kernel(
    key,
    value,
    k_norm_weight,
    phase,
    cache_position,
    key_rotated,
    key_cache,
    value_cache,
    rms_norm_eps,
    stride_kc_b,
    stride_kc_h,
    stride_kc_p,
    stride_kc_d,
    stride_vc_b,
    stride_vc_h,
    stride_vc_p,
    stride_vc_d,
    seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    PHASE_BATCH_STRIDE: tl.constexpr,
):
    block = tl.program_id(0)
    row = tl.arange(0, BLOCK_ROWS)[:, None]
    dims = tl.arange(0, HALF_DIM)[None, :]

    rows = block * BLOCK_ROWS + row
    seq = rows % seq_len
    batch_head = rows // seq_len
    head = batch_head % num_heads
    batch = batch_head // num_heads

    row_base = rows * (2 * HALF_DIM)

    k0 = tl.load(key + row_base + dims).to(tl.float32)
    k1 = tl.load(key + row_base + HALF_DIM + dims).to(tl.float32)

    square_sum = tl.sum(k0 * k0 + k1 * k1, axis=1)
    inv_rms = tl.expand_dims(
        tl.rsqrt(
            square_sum * (1.0 / (2 * HALF_DIM)) + rms_norm_eps
        ),
        1,
    )

    w0 = tl.load(k_norm_weight + dims).to(tl.float32)
    w1 = tl.load(k_norm_weight + HALF_DIM + dims).to(tl.float32)
    n0 = (k0 * inv_rms * w0).to(tl.bfloat16)
    n1 = (k1 * inv_rms * w1).to(tl.bfloat16)

    phase_base = batch * PHASE_BATCH_STRIDE + seq * (2 * HALF_DIM)
    cosine = tl.load(phase + phase_base + dims)
    sine = tl.load(phase + phase_base + HALF_DIM + dims)

    p0 = (n0 * cosine).to(tl.bfloat16)
    p1 = (n1 * sine).to(tl.bfloat16)
    p2 = (n1 * cosine).to(tl.bfloat16)
    p3 = (n0 * sine).to(tl.bfloat16)

    out0 = (p0 - p1).to(tl.bfloat16)
    out1 = (p2 + p3).to(tl.bfloat16)

    tl.store(key_rotated + row_base + dims, out0)
    tl.store(key_rotated + row_base + HALF_DIM + dims, out1)

    cache_pos = tl.load(cache_position + seq)
    key_cache_base = (
        batch * stride_kc_b
        + head * stride_kc_h
        + cache_pos * stride_kc_p
    )
    value_cache_base = (
        batch * stride_vc_b
        + head * stride_vc_h
        + cache_pos * stride_vc_p
    )

    tl.store(
        key_cache + key_cache_base + dims * stride_kc_d,
        out0,
    )
    tl.store(
        key_cache
        + key_cache_base
        + (HALF_DIM + dims) * stride_kc_d,
        out1,
    )

    value0 = tl.load(value + row_base + dims)
    value1 = tl.load(value + row_base + HALF_DIM + dims)
    tl.store(
        value_cache + value_cache_base + dims * stride_vc_d,
        value0,
    )
    tl.store(
        value_cache
        + value_cache_base
        + (HALF_DIM + dims) * stride_vc_d,
        value1,
    )


@triton.jit
def _long_kv_phase_norm_rope_cache_kernel(
    key,
    value,
    position_ids,
    inv_freq,
    k_norm_weight,
    phase,
    cache_position,
    key_rotated,
    key_cache,
    value_cache,
    rms_norm_eps,
    stride_pos_b: tl.constexpr,
    stride_pos_s: tl.constexpr,
    stride_kc_b,
    stride_kc_h,
    stride_kc_p,
    stride_kc_d,
    stride_vc_b,
    stride_vc_h,
    stride_vc_p,
    stride_vc_d,
    seq_len: tl.constexpr,
    HALF_DIM: tl.constexpr,
    PHASE_BATCH_STRIDE: tl.constexpr,
    SHARED_PHASE: tl.constexpr,
):
    token = tl.program_id(0)
    batch = token // seq_len
    seq = token - batch * seq_len

    row = tl.arange(0, 8)[:, None]
    dims = tl.arange(0, HALF_DIM)[None, :]
    phase_dims = tl.arange(0, HALF_DIM)

    position = tl.load(
        position_ids + batch * stride_pos_b + seq * stride_pos_s
    ).to(tl.float32)
    frequency = position * tl.load(inv_freq + phase_dims).to(tl.float32)
    cosine_1d = tl.cos(frequency).to(tl.bfloat16)
    sine_1d = tl.sin(frequency).to(tl.bfloat16)

    phase_base = batch * PHASE_BATCH_STRIDE + seq * (2 * HALF_DIM)
    if SHARED_PHASE:
        if batch == 0:
            tl.store(phase + phase_base + phase_dims, cosine_1d)
            tl.store(
                phase + phase_base + HALF_DIM + phase_dims,
                sine_1d,
            )
    else:
        tl.store(phase + phase_base + phase_dims, cosine_1d)
        tl.store(
            phase + phase_base + HALF_DIM + phase_dims,
            sine_1d,
        )

    cosine = cosine_1d[None, :]
    sine = sine_1d[None, :]
    head = row
    rows = (batch * 8 + head) * seq_len + seq
    row_base = rows * (2 * HALF_DIM)

    k0 = tl.load(key + row_base + dims).to(tl.float32)
    k1 = tl.load(key + row_base + HALF_DIM + dims).to(tl.float32)

    square_sum = tl.sum(k0 * k0 + k1 * k1, axis=1)
    inv_rms = tl.expand_dims(
        tl.rsqrt(
            square_sum * (1.0 / (2 * HALF_DIM)) + rms_norm_eps
        ),
        1,
    )

    w0 = tl.load(k_norm_weight + dims).to(tl.float32)
    w1 = tl.load(k_norm_weight + HALF_DIM + dims).to(tl.float32)
    n0 = (k0 * inv_rms * w0).to(tl.bfloat16)
    n1 = (k1 * inv_rms * w1).to(tl.bfloat16)

    p0 = (n0 * cosine).to(tl.bfloat16)
    p1 = (n1 * sine).to(tl.bfloat16)
    p2 = (n1 * cosine).to(tl.bfloat16)
    p3 = (n0 * sine).to(tl.bfloat16)

    out0 = (p0 - p1).to(tl.bfloat16)
    out1 = (p2 + p3).to(tl.bfloat16)

    tl.store(key_rotated + row_base + dims, out0)
    tl.store(key_rotated + row_base + HALF_DIM + dims, out1)

    cache_pos = tl.load(cache_position + seq)
    key_cache_base = (
        batch * stride_kc_b
        + head * stride_kc_h
        + cache_pos * stride_kc_p
    )
    value_cache_base = (
        batch * stride_vc_b
        + head * stride_vc_h
        + cache_pos * stride_vc_p
    )

    tl.store(
        key_cache + key_cache_base + dims * stride_kc_d,
        out0,
    )
    tl.store(
        key_cache
        + key_cache_base
        + (HALF_DIM + dims) * stride_kc_d,
        out1,
    )

    value0 = tl.load(value + row_base + dims)
    value1 = tl.load(value + row_base + HALF_DIM + dims)
    tl.store(
        value_cache + value_cache_base + dims * stride_vc_d,
        value0,
    )
    tl.store(
        value_cache
        + value_cache_base
        + (HALF_DIM + dims) * stride_vc_d,
        value1,
    )


@triton.jit
def _decode_norm_rope_cache_kernel(
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
    rms_norm_eps,
    stride_pos_b: tl.constexpr,
    stride_kc_b,
    stride_kc_h,
    stride_kc_p,
    stride_kc_d,
    stride_vc_b,
    stride_vc_h,
    stride_vc_p,
    stride_vc_d,
    HALF_DIM: tl.constexpr,
):
    program = tl.program_id(0)
    group = program % 13
    batch = program // 13

    row = tl.arange(0, 8)[:, None]
    dims = tl.arange(0, HALF_DIM)[None, :]
    phase_dims = tl.arange(0, HALF_DIM)

    position = tl.load(position_ids + batch * stride_pos_b).to(tl.float32)
    frequency = position * tl.load(inv_freq + phase_dims).to(tl.float32)
    cosine = tl.cos(frequency).to(tl.bfloat16)[None, :]
    sine = tl.sin(frequency).to(tl.bfloat16)[None, :]

    if group < 12:
        head = group * 8 + row
        row_base = (batch * 96 + head) * (2 * HALF_DIM)

        x0 = tl.load(query + row_base + dims).to(tl.float32)
        x1 = tl.load(query + row_base + HALF_DIM + dims).to(tl.float32)

        square_sum = tl.sum(x0 * x0 + x1 * x1, axis=1)
        inv_rms = tl.expand_dims(
            tl.rsqrt(
                square_sum * (1.0 / (2 * HALF_DIM)) + rms_norm_eps
            ),
            1,
        )

        w0 = tl.load(q_norm_weight + dims).to(tl.float32)
        w1 = tl.load(q_norm_weight + HALF_DIM + dims).to(tl.float32)
        n0 = (x0 * inv_rms * w0).to(tl.bfloat16)
        n1 = (x1 * inv_rms * w1).to(tl.bfloat16)

        p0 = (n0 * cosine).to(tl.bfloat16)
        p1 = (n1 * sine).to(tl.bfloat16)
        p2 = (n1 * cosine).to(tl.bfloat16)
        p3 = (n0 * sine).to(tl.bfloat16)

        out0 = (p0 - p1).to(tl.bfloat16)
        out1 = (p2 + p3).to(tl.bfloat16)

        tl.store(query_rotated + row_base + dims, out0)
        tl.store(
            query_rotated + row_base + HALF_DIM + dims,
            out1,
        )
    else:
        head = row
        row_base = (batch * 8 + head) * (2 * HALF_DIM)

        k0 = tl.load(key + row_base + dims).to(tl.float32)
        k1 = tl.load(key + row_base + HALF_DIM + dims).to(tl.float32)

        square_sum = tl.sum(k0 * k0 + k1 * k1, axis=1)
        inv_rms = tl.expand_dims(
            tl.rsqrt(
                square_sum * (1.0 / (2 * HALF_DIM)) + rms_norm_eps
            ),
            1,
        )

        w0 = tl.load(k_norm_weight + dims).to(tl.float32)
        w1 = tl.load(k_norm_weight + HALF_DIM + dims).to(tl.float32)
        n0 = (k0 * inv_rms * w0).to(tl.bfloat16)
        n1 = (k1 * inv_rms * w1).to(tl.bfloat16)

        p0 = (n0 * cosine).to(tl.bfloat16)
        p1 = (n1 * sine).to(tl.bfloat16)
        p2 = (n1 * cosine).to(tl.bfloat16)
        p3 = (n0 * sine).to(tl.bfloat16)

        out0 = (p0 - p1).to(tl.bfloat16)
        out1 = (p2 + p3).to(tl.bfloat16)

        tl.store(key_rotated + row_base + dims, out0)
        tl.store(
            key_rotated + row_base + HALF_DIM + dims,
            out1,
        )

        cache_pos = tl.load(cache_position)
        key_cache_base = (
            batch * stride_kc_b
            + head * stride_kc_h
            + cache_pos * stride_kc_p
        )
        value_cache_base = (
            batch * stride_vc_b
            + head * stride_vc_h
            + cache_pos * stride_vc_p
        )

        tl.store(
            key_cache + key_cache_base + dims * stride_kc_d,
            out0,
        )
        tl.store(
            key_cache
            + key_cache_base
            + (HALF_DIM + dims) * stride_kc_d,
            out1,
        )

        value0 = tl.load(value + row_base + dims)
        value1 = tl.load(value + row_base + HALF_DIM + dims)
        tl.store(
            value_cache + value_cache_base + dims * stride_vc_d,
            value0,
        )
        tl.store(
            value_cache
            + value_cache_base
            + (HALF_DIM + dims) * stride_vc_d,
            value1,
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

    query_rotated = torch.empty_like(query)
    key_rotated = torch.empty_like(key)

    if seq_len == 1:
        _decode_norm_rope_cache_kernel[(batch_size * 13,)](
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
            rms_norm_eps,
            position_ids.stride(0),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            value_cache.stride(3),
            HALF_DIM=HALF_HEAD_DIM,
            num_warps=8,
        )
        return query_rotated, key_rotated, key_cache, value_cache

    if seq_len >= LONG_PREFILL_THRESHOLD:
        shared_phase = position_ids.stride(0) == 0
        phase_batches = 1 if shared_phase else batch_size
        phase_batch_stride = 0 if shared_phase else seq_len * head_dim

        phase = torch.empty(
            (phase_batches, seq_len, head_dim),
            dtype=torch.bfloat16,
            device=query.device,
        )

        _long_kv_phase_norm_rope_cache_kernel[
            (batch_size * seq_len,)
        ](
            key,
            value,
            position_ids,
            inv_freq,
            k_norm_weight,
            phase,
            cache_position,
            key_rotated,
            key_cache,
            value_cache,
            rms_norm_eps,
            position_ids.stride(0),
            position_ids.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            value_cache.stride(3),
            seq_len=seq_len,
            HALF_DIM=HALF_HEAD_DIM,
            PHASE_BATCH_STRIDE=phase_batch_stride,
            SHARED_PHASE=shared_phase,
            num_warps=8,
        )

        long_q_rows = (
            VERY_LONG_Q_ROWS_PER_PROGRAM
            if seq_len >= VERY_LONG_PREFILL_THRESHOLD
            else LONG_Q_ROWS_PER_PROGRAM
        )
        query_blocks = (
            batch_size
            * seq_len
            * (num_q_heads // long_q_rows)
        )

        _query_norm_rope_kernel[(query_blocks,)](
            query,
            q_norm_weight,
            phase,
            query_rotated,
            rms_norm_eps,
            seq_len=seq_len,
            num_heads=num_q_heads,
            HALF_DIM=HALF_HEAD_DIM,
            BLOCK_ROWS=long_q_rows,
            PHASE_BATCH_STRIDE=phase_batch_stride,
            TOKEN_MAJOR=True,
            num_warps=8,
        )

        return query_rotated, key_rotated, key_cache, value_cache

    shared_phase = position_ids.stride(0) == 0
    phase_batches = 1 if shared_phase else batch_size
    phase_batch_stride = 0 if shared_phase else seq_len * head_dim

    phase = torch.empty(
        (phase_batches, seq_len, head_dim),
        dtype=torch.bfloat16,
        device=query.device,
    )

    _build_rope_phase_kernel[(phase_batches * seq_len,)](
        position_ids,
        inv_freq,
        phase,
        position_ids.stride(0),
        position_ids.stride(1),
        seq_len=seq_len,
        BLOCK=HALF_HEAD_DIM,
        num_warps=2,
    )

    query_rows = batch_size * num_q_heads * seq_len
    query_blocks = triton.cdiv(query_rows, ROWS_PER_PROGRAM)

    _query_norm_rope_kernel[(query_blocks,)](
        query,
        q_norm_weight,
        phase,
        query_rotated,
        rms_norm_eps,
        seq_len=seq_len,
        num_heads=num_q_heads,
        HALF_DIM=HALF_HEAD_DIM,
        BLOCK_ROWS=ROWS_PER_PROGRAM,
        PHASE_BATCH_STRIDE=phase_batch_stride,
        TOKEN_MAJOR=False,
        num_warps=8,
    )

    kv_rows = batch_size * num_kv_heads * seq_len
    kv_blocks = triton.cdiv(kv_rows, ROWS_PER_PROGRAM)

    _kv_norm_rope_cache_kernel[(kv_blocks,)](
        key,
        value,
        k_norm_weight,
        phase,
        cache_position,
        key_rotated,
        key_cache,
        value_cache,
        rms_norm_eps,
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        seq_len=seq_len,
        num_heads=num_kv_heads,
        HALF_DIM=HALF_HEAD_DIM,
        BLOCK_ROWS=ROWS_PER_PROGRAM,
        PHASE_BATCH_STRIDE=phase_batch_stride,
        num_warps=8,
    )

    return query_rotated, key_rotated, key_cache, value_cache