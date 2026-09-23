# solution=GPT-5.6-Sol_092_gqa_attention_with_qk_norm_triton_optimized_r3 score=2.492054713694955 passed=True
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _qk_norm_rope_kernel(
    x,
    norm_weight,
    cos,
    sin,
    seq_len,
    rms_eps,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)

    head_id = pid % HEADS
    tmp = pid // HEADS
    seq_id = tmp % seq_len
    batch_id = tmp // seq_len

    offs = tl.arange(0, HEAD_DIM)
    half_dim = HEAD_DIM // 2

    x_base = ((batch_id * seq_len + seq_id) * HEADS + head_id) * HEAD_DIM
    x_ptrs = x + x_base + offs

    x_vals = tl.load(x_ptrs)
    x_fp32 = x_vals.to(tl.float32)

    variance = tl.sum(x_fp32 * x_fp32, axis=0) / HEAD_DIM
    inv_rms = tl.rsqrt(variance + rms_eps)

    weight_vals = tl.load(norm_weight + offs).to(tl.float32)
    norm_vals = (x_fp32 * inv_rms * weight_vals).to(tl.bfloat16)

    partner_offs = tl.where(
        offs < half_dim,
        offs + half_dim,
        offs - half_dim,
    )
    partner_x = tl.load(x + x_base + partner_offs).to(tl.float32)
    partner_weight = tl.load(norm_weight + partner_offs).to(tl.float32)
    partner_norm = (
        partner_x * inv_rms * partner_weight
    ).to(tl.bfloat16)

    rotated = tl.where(offs < half_dim, -partner_norm, partner_norm)

    rope_base = (batch_id * seq_len + seq_id) * HEAD_DIM
    cos_vals = tl.load(cos + rope_base + offs)
    sin_vals = tl.load(sin + rope_base + offs)

    result = (
        (norm_vals * cos_vals).to(tl.bfloat16)
        + (rotated * sin_vals).to(tl.bfloat16)
    ).to(tl.bfloat16)

    tl.store(x_ptrs, result)


@triton.jit
def _gqa_attention_kernel(
    q,
    k,
    v,
    out,
    seq_len,
    scale,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    kv_head = pid % NUM_KV_HEADS
    tmp = pid // NUM_KV_HEADS
    query_pos = tmp % seq_len
    batch_id = tmp // seq_len

    group_ids = tl.arange(0, GROUP_BLOCK)
    group_mask = group_ids < GROUP_SIZE
    head_ids = kv_head * GROUP_SIZE + group_ids
    d_offsets = tl.arange(0, HEAD_DIM)

    q_base = (
        (batch_id * seq_len + query_pos) * NUM_HEADS * HEAD_DIM
    )
    q_ptrs = (
        q
        + q_base
        + head_ids[:, None] * HEAD_DIM
        + d_offsets[None, :]
    )
    q_vals = tl.load(
        q_ptrs,
        mask=group_mask[:, None],
        other=0.0,
    )

    m_i = tl.full((GROUP_BLOCK,), -float("inf"), tl.float32)
    l_i = tl.zeros((GROUP_BLOCK,), tl.float32)
    acc = tl.zeros((GROUP_BLOCK, HEAD_DIM), tl.float32)

    scale_fp32 = scale.to(tl.float32)

    for start_n in tl.range(0, query_pos + 1, BLOCK_N):
        key_offsets = start_n + tl.arange(0, BLOCK_N)
        key_mask = key_offsets <= query_pos

        k_ptrs = (
            k
            + (
                (
                    (batch_id * seq_len + key_offsets[None, :])
                    * NUM_KV_HEADS
                    + kv_head
                )
                * HEAD_DIM
                + d_offsets[:, None]
            )
        )
        k_vals = tl.load(
            k_ptrs,
            mask=key_mask[None, :],
            other=0.0,
        )

        qk = tl.dot(q_vals, k_vals)
        scores = qk.to(tl.float32) * scale_fp32
        scores = tl.where(
            key_mask[None, :],
            scores,
            -float("inf"),
        )

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = (
            v
            + (
                (
                    (batch_id * seq_len + key_offsets[:, None])
                    * NUM_KV_HEADS
                    + kv_head
                )
                * HEAD_DIM
                + d_offsets[None, :]
            )
        )
        v_vals = tl.load(
            v_ptrs,
            mask=key_mask[:, None],
            other=0.0,
        )

        acc = tl.dot(
            p.to(tl.bfloat16),
            v_vals,
            acc=acc * alpha[:, None],
        )
        m_i = m_new

    acc = acc / l_i[:, None]

    out_base = (
        (batch_id * seq_len + query_pos) * NUM_HEADS * HEAD_DIM
    )
    out_ptrs = (
        out
        + out_base
        + head_ids[:, None] * HEAD_DIM
        + d_offsets[None, :]
    )
    tl.store(
        out_ptrs,
        acc.to(tl.bfloat16),
        mask=group_mask[:, None],
    )


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_proj_bias: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_norm_eps: float,
):
    batch_size, seq_length, _ = hidden_states.shape

    num_attention_heads = 96
    num_key_value_heads = 8
    num_key_value_groups = 12
    head_dim = 128

    query_states = F.linear(
        hidden_states,
        q_proj_weight,
        q_proj_bias,
    )
    key_states = F.linear(
        hidden_states,
        k_proj_weight,
        k_proj_bias,
    )
    value_states = F.linear(
        hidden_states,
        v_proj_weight,
        v_proj_bias,
    )

    _qk_norm_rope_kernel[
        (batch_size * seq_length * num_attention_heads,)
    ](
        query_states,
        q_norm_weight,
        cos,
        sin,
        seq_length,
        rms_norm_eps,
        HEADS=num_attention_heads,
        HEAD_DIM=head_dim,
        num_warps=4,
    )

    _qk_norm_rope_kernel[
        (batch_size * seq_length * num_key_value_heads,)
    ](
        key_states,
        k_norm_weight,
        cos,
        sin,
        seq_length,
        rms_norm_eps,
        HEADS=num_key_value_heads,
        HEAD_DIM=head_dim,
        num_warps=4,
    )

    attention_output = torch.empty_like(query_states)

    _gqa_attention_kernel[
        (batch_size * seq_length * num_key_value_heads,)
    ](
        query_states,
        key_states,
        value_states,
        attention_output,
        seq_length,
        head_dim ** -0.5,
        NUM_HEADS=num_attention_heads,
        NUM_KV_HEADS=num_key_value_heads,
        GROUP_SIZE=num_key_value_groups,
        GROUP_BLOCK=16,
        HEAD_DIM=head_dim,
        BLOCK_N=64,
        num_warps=8,
    )

    output = F.linear(
        attention_output.reshape(
            batch_size * seq_length,
            num_attention_heads * head_dim,
        ),
        o_proj_weight,
        None,
    )

    return output.reshape(batch_size, seq_length, -1)