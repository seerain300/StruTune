# solution=GPT-5.6-Sol_092_gqa_attention_with_qk_norm_triton_optimized_r17 score=3.3775083322837167 passed=True
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _norm_rope_kernel(
    X,
    W,
    COS,
    SIN,
    OUT,
    S: tl.constexpr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, 64)

    b = tl.program_id(1)
    token = rows // H
    head = rows % H
    valid = token < S

    base = (
        b * S * H * 128
        + token[:, None] * H * 128
        + head[:, None] * 128
    )

    x0 = tl.load(
        X + base + d[None, :],
        valid[:, None],
        0.0,
    ).to(tl.float32)
    x1 = tl.load(
        X + base + 64 + d[None, :],
        valid[:, None],
        0.0,
    ).to(tl.float32)

    w0 = tl.load(W + d).to(tl.float32)
    w1 = tl.load(W + 64 + d).to(tl.float32)

    variance = tl.sum(x0 * x0 + x1 * x1, axis=1) * (1.0 / 128.0)
    inv_rms = tl.rsqrt(variance + EPS)

    y0 = (
        x0 * inv_rms[:, None] * w0[None, :]
    ).to(tl.bfloat16)
    y1 = (
        x1 * inv_rms[:, None] * w1[None, :]
    ).to(tl.bfloat16)

    if H == 96:
        rope_token = (tl.program_id(0) * ROWS) // H
        rope_valid = rope_token < S
        rope_base = b * S * 128 + rope_token * 128

        cos0_row = tl.load(
            COS + rope_base + d,
            rope_valid,
            0.0,
        ).to(tl.float32)
        cos1_row = tl.load(
            COS + rope_base + 64 + d,
            rope_valid,
            0.0,
        ).to(tl.float32)
        sin0_row = tl.load(
            SIN + rope_base + d,
            rope_valid,
            0.0,
        ).to(tl.float32)
        sin1_row = tl.load(
            SIN + rope_base + 64 + d,
            rope_valid,
            0.0,
        ).to(tl.float32)

        cos0 = tl.broadcast_to(cos0_row[None, :], (ROWS, 64))
        cos1 = tl.broadcast_to(cos1_row[None, :], (ROWS, 64))
        sin0 = tl.broadcast_to(sin0_row[None, :], (ROWS, 64))
        sin1 = tl.broadcast_to(sin1_row[None, :], (ROWS, 64))
    elif H == 8:
        rope_token0 = (tl.program_id(0) * ROWS) // H
        rope_token1 = rope_token0 + 1
        rope_base0 = b * S * 128 + rope_token0 * 128
        rope_base1 = b * S * 128 + rope_token1 * 128

        cos00 = tl.load(
            COS + rope_base0 + d,
            rope_token0 < S,
            0.0,
        ).to(tl.float32)
        cos01 = tl.load(
            COS + rope_base0 + 64 + d,
            rope_token0 < S,
            0.0,
        ).to(tl.float32)
        sin00 = tl.load(
            SIN + rope_base0 + d,
            rope_token0 < S,
            0.0,
        ).to(tl.float32)
        sin01 = tl.load(
            SIN + rope_base0 + 64 + d,
            rope_token0 < S,
            0.0,
        ).to(tl.float32)

        cos10 = tl.load(
            COS + rope_base1 + d,
            rope_token1 < S,
            0.0,
        ).to(tl.float32)
        cos11 = tl.load(
            COS + rope_base1 + 64 + d,
            rope_token1 < S,
            0.0,
        ).to(tl.float32)
        sin10 = tl.load(
            SIN + rope_base1 + d,
            rope_token1 < S,
            0.0,
        ).to(tl.float32)
        sin11 = tl.load(
            SIN + rope_base1 + 64 + d,
            rope_token1 < S,
            0.0,
        ).to(tl.float32)

        first_token = token[:, None] == rope_token0

        cos0 = tl.where(
            first_token,
            tl.broadcast_to(cos00[None, :], (ROWS, 64)),
            tl.broadcast_to(cos10[None, :], (ROWS, 64)),
        )
        cos1 = tl.where(
            first_token,
            tl.broadcast_to(cos01[None, :], (ROWS, 64)),
            tl.broadcast_to(cos11[None, :], (ROWS, 64)),
        )
        sin0 = tl.where(
            first_token,
            tl.broadcast_to(sin00[None, :], (ROWS, 64)),
            tl.broadcast_to(sin10[None, :], (ROWS, 64)),
        )
        sin1 = tl.where(
            first_token,
            tl.broadcast_to(sin01[None, :], (ROWS, 64)),
            tl.broadcast_to(sin11[None, :], (ROWS, 64)),
        )
    else:
        rope_base = (
            b * S * 128
            + token[:, None] * 128
        )

        cos0 = tl.load(
            COS + rope_base + d[None, :],
            valid[:, None],
            0.0,
        ).to(tl.float32)
        cos1 = tl.load(
            COS + rope_base + 64 + d[None, :],
            valid[:, None],
            0.0,
        ).to(tl.float32)
        sin0 = tl.load(
            SIN + rope_base + d[None, :],
            valid[:, None],
            0.0,
        ).to(tl.float32)
        sin1 = tl.load(
            SIN + rope_base + 64 + d[None, :],
            valid[:, None],
            0.0,
        ).to(tl.float32)

    a0 = (y0.to(tl.float32) * cos0).to(tl.bfloat16)
    c0 = ((-y1).to(tl.float32) * sin0).to(tl.bfloat16)
    result0 = (
        a0.to(tl.float32) + c0.to(tl.float32)
    ).to(tl.bfloat16)

    a1 = (y1.to(tl.float32) * cos1).to(tl.bfloat16)
    c1 = (y0.to(tl.float32) * sin1).to(tl.bfloat16)
    result1 = (
        a1.to(tl.float32) + c1.to(tl.float32)
    ).to(tl.bfloat16)

    if H == 96:
        out_base = (
            b * S * H * 128
            + token[:, None] * H * 128
            + head[:, None] * 128
        )

        tl.store(
            OUT + out_base + d[None, :],
            result0,
            valid[:, None],
        )
        tl.store(
            OUT + out_base + 64 + d[None, :],
            result1,
            valid[:, None],
        )
    else:
        out_base = (
            b * H * S * 128
            + head[:, None] * S * 128
            + token[:, None] * 128
        )

        tl.store(
            OUT + out_base + d[None, :],
            result0,
            valid[:, None],
        )
        tl.store(
            OUT + out_base + 64 + d[None, :],
            result1,
            valid[:, None],
        )


@triton.jit
def _group_attention_kernel(
    Q,
    K,
    V,
    O,
    S: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    TOKENS: tl.constexpr,
):
    block = tl.program_id(0)
    kv_head = tl.program_id(1)
    batch = tl.program_id(2)

    row = tl.arange(0, BM)
    col = tl.arange(0, BN)
    d = tl.arange(0, 128)

    query_token = block * TOKENS + row // 12
    local_head = row % 12
    valid_query = (row < TOKENS * 12) & (query_token < S)
    query_head = kv_head * 12 + local_head

    q = tl.load(
        Q
        + batch * S * 96 * 128
        + query_token[:, None] * 96 * 128
        + query_head[:, None] * 128
        + d[None, :],
        valid_query[:, None],
        0,
    )

    m = tl.full((BM,), float("-inf"), tl.float32)
    l = tl.full((BM,), 0.0, tl.float32)
    acc = tl.full((BM, 128), 0.0, tl.float32)

    last_token = tl.minimum((block + 1) * TOKENS, S)

    for start in tl.range(
        0,
        tl.cdiv(last_token, BN),
        num_stages=2,
    ):
        key_token = start * BN + col
        key_valid = key_token < last_token

        k = tl.load(
            K
            + batch * 8 * S * 128
            + kv_head * S * 128
            + key_token[:, None] * 128
            + d[None, :],
            key_valid[:, None],
            0,
        )

        scores = tl.dot(q, tl.trans(k)).to(tl.bfloat16).to(tl.float32)
        scores = (
            scores * 0.08838834764831845
        ).to(tl.bfloat16).to(tl.float32)

        scores = tl.where(
            (key_token[None, :] <= query_token[:, None])
            & key_valid[None, :]
            & valid_query[:, None],
            scores,
            float("-inf"),
        )

        block_max = tl.max(scores, axis=1)
        new_m = tl.maximum(m, block_max)
        new_m = tl.where(valid_query, new_m, 0.0)

        correction = tl.exp(m - new_m)
        p = tl.exp(scores - new_m[:, None])
        new_l = l * correction + tl.sum(p, axis=1)

        v = tl.load(
            V
            + batch * S * 8 * 128
            + key_token[:, None] * 8 * 128
            + kv_head * 128
            + d[None, :],
            key_valid[:, None],
            0,
        )

        acc = (
            acc * correction[:, None]
            + tl.dot(p.to(tl.bfloat16), v)
        )
        m = new_m
        l = new_l

    result = (
        acc / tl.maximum(l[:, None], 1.0e-20)
    ).to(tl.bfloat16)

    tl.store(
        O
        + batch * S * 96 * 128
        + query_token[:, None] * 96 * 128
        + query_head[:, None] * 128
        + d[None, :],
        result,
        valid_query[:, None],
    )


@torch.no_grad()
def run(
    hidden_states,
    q_proj_weight,
    q_proj_bias,
    k_proj_weight,
    k_proj_bias,
    v_proj_weight,
    v_proj_bias,
    o_proj_weight,
    q_norm_weight,
    k_norm_weight,
    cos,
    sin,
    rms_norm_eps,
):
    batch, seq_len, _ = hidden_states.shape

    q_projected = F.linear(
        hidden_states,
        q_proj_weight,
        q_proj_bias,
    )
    k_projected = F.linear(
        hidden_states,
        k_proj_weight,
        k_proj_bias,
    )
    v_projected = F.linear(
        hidden_states,
        v_proj_weight,
        v_proj_bias,
    )

    q = q_projected
    k = torch.empty(
        (batch, 8, seq_len, 128),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )

    _norm_rope_kernel[
        (triton.cdiv(seq_len * 96, 32), batch)
    ](
        q_projected,
        q_norm_weight,
        cos,
        sin,
        q,
        seq_len,
        96,
        rms_norm_eps,
        32,
        num_warps=8,
    )

    _norm_rope_kernel[
        (triton.cdiv(seq_len * 8, 16), batch)
    ](
        k_projected,
        k_norm_weight,
        cos,
        sin,
        k,
        seq_len,
        8,
        rms_norm_eps,
        16,
        num_warps=4,
    )

    attention_output = torch.empty(
        (batch, seq_len, 96, 128),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )

    if seq_len >= 1024:
        tokens = 10
        bm = 128
        bn = 128
        warps = 8
    elif seq_len >= 256:
        tokens = 10
        bm = 128
        bn = 64
        warps = 8
    else:
        tokens = 5
        bm = 64
        bn = 64
        warps = 4

    _group_attention_kernel[
        (triton.cdiv(seq_len, tokens), 8, batch)
    ](
        q,
        k,
        v_projected,
        attention_output,
        seq_len,
        bm,
        bn,
        tokens,
        num_warps=warps,
    )

    return F.linear(
        attention_output.reshape(
            batch,
            seq_len,
            96 * 128,
        ),
        o_proj_weight,
    )