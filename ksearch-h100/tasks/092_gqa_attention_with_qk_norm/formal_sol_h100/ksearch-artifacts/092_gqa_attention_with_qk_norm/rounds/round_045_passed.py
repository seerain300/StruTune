# solution=GPT-5.6-Sol_092_gqa_attention_with_qk_norm_triton_optimized_r6 score=3.191507831400335 passed=True
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
    d = tl.arange(0, 128)
    valid = rows < tl.program_id(1) * 0 + S * H * tl.num_programs(1)

    b = tl.program_id(1)
    token = rows // H
    head = rows % H
    valid = token < S

    x = tl.load(
        X + b * S * H * 128 + token[:, None] * H * 128
        + head[:, None] * 128 + d[None, :],
        valid[:, None],
        0,
    ).to(tl.float32)
    weight = tl.load(W + d).to(tl.float32)

    variance = tl.sum(x * x, 1) * (1.0 / 128.0)
    x = (x * tl.rsqrt(variance[:, None] + EPS) * weight[None, :]).to(tl.bfloat16)

    other_d = tl.where(d < 64, d + 64, d - 64)
    other = tl.gather(x, tl.broadcast_to(other_d[None, :], (ROWS, 128)), 1)
    rotated = tl.where(d[None, :] < 64, -other, other)

    cos = tl.load(
        COS + b * S * 128 + token[:, None] * 128 + d[None, :],
        valid[:, None],
        0,
    ).to(tl.float32)
    sin = tl.load(
        SIN + b * S * 128 + token[:, None] * 128 + d[None, :],
        valid[:, None],
        0,
    ).to(tl.float32)

    a = (x.to(tl.float32) * cos).to(tl.bfloat16)
    c = (rotated.to(tl.float32) * sin).to(tl.bfloat16)
    result = (a.to(tl.float32) + c.to(tl.float32)).to(tl.bfloat16)

    tl.store(
        OUT + b * H * S * 128 + head[:, None] * S * 128
        + token[:, None] * 128 + d[None, :],
        result,
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
        Q + batch * 96 * S * 128
        + query_head[:, None] * S * 128
        + query_token[:, None] * 128
        + d[None, :],
        valid_query[:, None],
        0,
    )

    m = tl.full((BM,), float("-inf"), tl.float32)
    l = tl.full((BM,), 0.0, tl.float32)
    acc = tl.full((BM, 128), 0.0, tl.float32)

    last_token = tl.minimum((block + 1) * TOKENS, S)
    for start in range(tl.cdiv(last_token, BN)):
        key_token = start * BN + col

        k = tl.load(
            K + batch * 8 * S * 128
            + kv_head * S * 128
            + key_token[None, :] * 128
            + d[:, None],
            key_token[None, :] < S,
            0,
        )

        scores = tl.dot(q, k).to(tl.bfloat16).to(tl.float32)
        scores = (scores * 0.08838834764831845).to(tl.bfloat16).to(tl.float32)
        scores = tl.where(
            (key_token[None, :] <= query_token[:, None])
            & (key_token[None, :] < S)
            & valid_query[:, None],
            scores,
            float("-inf"),
        )

        block_max = tl.max(scores, 1)
        new_m = tl.maximum(m, block_max)
        new_m = tl.where(valid_query, new_m, 0.0)

        correction = tl.exp(m - new_m)
        p = tl.exp(scores - new_m[:, None])
        new_l = l * correction + tl.sum(p, 1)

        v = tl.load(
            V + batch * S * 8 * 128
            + key_token[:, None] * 8 * 128
            + kv_head * 128
            + d[None, :],
            key_token[:, None] < S,
            0,
        )

        acc = acc * correction[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m = new_m
        l = new_l

    result = (acc / tl.maximum(l[:, None], 1.0e-20)).to(tl.bfloat16)

    tl.store(
        O + batch * S * 96 * 128
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

    q_projected = F.linear(hidden_states, q_proj_weight, q_proj_bias)
    k_projected = F.linear(hidden_states, k_proj_weight, k_proj_bias)
    v_projected = F.linear(hidden_states, v_proj_weight, v_proj_bias)

    q = torch.empty(
        (batch, 96, seq_len, 128),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    k = torch.empty(
        (batch, 8, seq_len, 128),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )

    _norm_rope_kernel[(triton.cdiv(seq_len * 96, 4), batch)](
        q_projected,
        q_norm_weight,
        cos,
        sin,
        q,
        seq_len,
        96,
        rms_norm_eps,
        4,
        num_warps=4,
    )
    _norm_rope_kernel[(triton.cdiv(seq_len * 8, 4), batch)](
        k_projected,
        k_norm_weight,
        cos,
        sin,
        k,
        seq_len,
        8,
        rms_norm_eps,
        4,
        num_warps=4,
    )

    attention_output = torch.empty(
        (batch, seq_len, 96, 128),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )

    tokens = 8 if seq_len >= 512 else 4
    _group_attention_kernel[(triton.cdiv(seq_len, tokens), 8, batch)](
        q,
        k,
        v_projected,
        attention_output,
        seq_len,
        tokens * 16,
        64,
        tokens,
        num_warps=8 if tokens == 8 else 4,
    )

    return F.linear(
        attention_output.reshape(batch, seq_len, 96 * 128),
        o_proj_weight,
    )