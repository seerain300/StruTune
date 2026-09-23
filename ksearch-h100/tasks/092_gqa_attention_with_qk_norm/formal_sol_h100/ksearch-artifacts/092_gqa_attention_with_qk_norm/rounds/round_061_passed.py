# solution=GPT-5.6-Sol_092_gqa_attention_with_qk_norm_triton_optimized_r3 score=3.34508867753132 passed=True
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
    V_IN,
    V_OUT,
    S: tl.constexpr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, 128)

    b = tl.program_id(1)
    token = rows // H
    head = rows % H
    valid = token < S

    x = tl.load(
        X
        + b * S * H * 128
        + token[:, None] * H * 128
        + head[:, None] * 128
        + d[None, :],
        valid[:, None],
        0,
    ).to(tl.float32)

    weight = tl.load(W + d).to(tl.float32)
    variance = tl.sum(x * x, axis=1) * (1.0 / 128.0)
    x = (
        x
        * tl.rsqrt(variance[:, None] + EPS)
        * weight[None, :]
    ).to(tl.bfloat16)

    other_d = tl.where(d < 64, d + 64, d - 64)
    other = tl.gather(
        x,
        tl.broadcast_to(other_d[None, :], (ROWS, 128)),
        axis=1,
    )
    rotated = tl.where(d[None, :] < 64, -other, other)

    cos = tl.load(
        COS
        + b * S * 128
        + token[:, None] * 128
        + d[None, :],
        valid[:, None],
        0,
    ).to(tl.float32)
    sin = tl.load(
        SIN
        + b * S * 128
        + token[:, None] * 128
        + d[None, :],
        valid[:, None],
        0,
    ).to(tl.float32)

    a = (x * cos).to(tl.bfloat16)
    c = (rotated.to(tl.float32) * sin).to(tl.bfloat16)
    result = (a.to(tl.float32) + c.to(tl.float32)).to(tl.bfloat16)

    tl.store(
        OUT
        + b * H * S * 128
        + head[:, None] * S * 128
        + token[:, None] * 128
        + d[None, :],
        result,
        valid[:, None],
    )

    v = tl.load(
        V_IN
        + b * S * H * 128
        + token[:, None] * H * 128
        + head[:, None] * 128
        + d[None, :],
        valid[:, None],
        0,
    )
    tl.store(
        V_OUT
        + b * H * S * 128
        + head[:, None] * S * 128
        + token[:, None] * 128
        + d[None, :],
        v,
        valid[:, None],
    )


@triton.jit
def _group_attention_kernel(
    Q,
    Q_WEIGHT,
    COS,
    SIN,
    K,
    V,
    O,
    S: tl.constexpr,
    EPS: tl.constexpr,
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

    q_fp32 = tl.load(
        Q
        + batch * S * 96 * 128
        + query_token[:, None] * 96 * 128
        + query_head[:, None] * 128
        + d[None, :],
        valid_query[:, None],
        0,
    ).to(tl.float32)

    q_weight = tl.load(Q_WEIGHT + d).to(tl.float32)
    q_variance = tl.sum(q_fp32 * q_fp32, axis=1) * (1.0 / 128.0)
    q_norm = (
        q_fp32
        * tl.rsqrt(q_variance[:, None] + EPS)
        * q_weight[None, :]
    ).to(tl.bfloat16)

    other_d = tl.where(d < 64, d + 64, d - 64)
    q_other = tl.gather(
        q_norm,
        tl.broadcast_to(other_d[None, :], (BM, 128)),
        axis=1,
    )
    q_rotated = tl.where(d[None, :] < 64, -q_other, q_other)

    q_cos = tl.load(
        COS
        + batch * S * 128
        + query_token[:, None] * 128
        + d[None, :],
        valid_query[:, None],
        0,
    ).to(tl.float32)
    q_sin = tl.load(
        SIN
        + batch * S * 128
        + query_token[:, None] * 128
        + d[None, :],
        valid_query[:, None],
        0,
    ).to(tl.float32)

    q_a = (q_norm * q_cos).to(tl.bfloat16)
    q_c = (q_rotated.to(tl.float32) * q_sin).to(tl.bfloat16)
    q = (
        q_a.to(tl.float32) + q_c.to(tl.float32)
    ).to(tl.bfloat16)

    m = tl.full((BM,), float("-inf"), tl.float32)
    l = tl.full((BM,), 0.0, tl.float32)
    acc = tl.full((BM, 128), 0.0, tl.float32)

    last_token = tl.minimum((block + 1) * TOKENS, S)

    for start in range(tl.cdiv(last_token, BN)):
        key_token = start * BN + col
        key_valid = key_token < last_token

        k = tl.load(
            K
            + batch * 8 * S * 128
            + kv_head * S * 128
            + key_token[None, :] * 128
            + d[:, None],
            key_valid[None, :],
            0,
        )

        scores = tl.dot(q, k).to(tl.bfloat16).to(tl.float32)
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
            + batch * 8 * S * 128
            + kv_head * S * 128
            + key_token[:, None] * 128
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

    k = torch.empty(
        (batch, 8, seq_len, 128),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    v = torch.empty(
        (batch, 8, seq_len, 128),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )

    _norm_rope_kernel[
        (triton.cdiv(seq_len * 8, 16), batch)
    ](
        k_projected,
        k_norm_weight,
        cos,
        sin,
        k,
        v_projected,
        v,
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

    if batch * seq_len >= 512:
        tokens = 10
        bm = 128
        warps = 8
    else:
        tokens = 5
        bm = 64
        warps = 4

    _group_attention_kernel[
        (triton.cdiv(seq_len, tokens), 8, batch)
    ](
        q_projected,
        q_norm_weight,
        cos,
        sin,
        k,
        v,
        attention_output,
        seq_len,
        rms_norm_eps,
        bm,
        64,
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