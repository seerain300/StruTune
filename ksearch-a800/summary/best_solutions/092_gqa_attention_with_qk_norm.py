# task: 092_gqa_attention_with_qk_norm
# bench: SOL-L1 | batch: formal_20260914
# final eval (official evaluator, full workloads): valid=True pass=16/16 geomean=2.843x
# feedback best (5-workload sample during search): 3.151x
# torch fallback audit: B·自研为主 (linear×4)
# tokens: 1,550,189

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _qk_rmsnorm_rope_inplace_kernel(
    q_ptr,
    k_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    num_tokens,
    seq_len,
    eps,
    stride_cos_b,
    stride_cos_s,
    stride_sin_b,
    stride_sin_s,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    token_block = tl.program_id(0)
    combined_head = tl.program_id(1)

    is_q = combined_head < NUM_Q_HEADS
    local_head = tl.where(
        is_q,
        combined_head,
        combined_head - NUM_Q_HEADS,
    )

    offs_m = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_half = tl.arange(0, HEAD_DIM // 2)
    valid_m = offs_m < num_tokens

    q_base = (
        offs_m[:, None] * NUM_Q_HEADS * HEAD_DIM
        + combined_head * HEAD_DIM
        + offs_half[None, :]
    )
    k_base = (
        offs_m[:, None] * NUM_KV_HEADS * HEAD_DIM
        + local_head * HEAD_DIM
        + offs_half[None, :]
    )

    q_mask = valid_m[:, None] & is_q
    k_mask = valid_m[:, None] & (~is_q)

    projected1 = (
        tl.load(q_ptr + q_base, mask=q_mask, other=0.0)
        + tl.load(k_ptr + k_base, mask=k_mask, other=0.0)
    )
    projected2 = (
        tl.load(
            q_ptr + q_base + HEAD_DIM // 2,
            mask=q_mask,
            other=0.0,
        )
        + tl.load(
            k_ptr + k_base + HEAD_DIM // 2,
            mask=k_mask,
            other=0.0,
        )
    )

    projected1_fp32 = projected1.to(tl.float32)
    projected2_fp32 = projected2.to(tl.float32)

    variance = tl.sum(
        projected1_fp32 * projected1_fp32
        + projected2_fp32 * projected2_fp32,
        axis=1,
    ) * (1.0 / HEAD_DIM)
    inv_rms = tl.rsqrt(variance + eps)

    q_weight1 = tl.load(
        q_norm_weight_ptr + offs_half,
        mask=is_q,
        other=0.0,
    )
    q_weight2 = tl.load(
        q_norm_weight_ptr + HEAD_DIM // 2 + offs_half,
        mask=is_q,
        other=0.0,
    )
    k_weight1 = tl.load(
        k_norm_weight_ptr + offs_half,
        mask=~is_q,
        other=0.0,
    )
    k_weight2 = tl.load(
        k_norm_weight_ptr + HEAD_DIM // 2 + offs_half,
        mask=~is_q,
        other=0.0,
    )

    norm_weight1 = (q_weight1 + k_weight1).to(tl.float32)
    norm_weight2 = (q_weight2 + k_weight2).to(tl.float32)

    normalized1 = (
        projected1_fp32
        * inv_rms[:, None]
        * norm_weight1[None, :]
    ).to(tl.bfloat16)
    normalized2 = (
        projected2_fp32
        * inv_rms[:, None]
        * norm_weight2[None, :]
    ).to(tl.bfloat16)

    batch_idx = offs_m // seq_len
    seq_idx = offs_m - batch_idx * seq_len

    cos_base = (
        batch_idx[:, None] * stride_cos_b
        + seq_idx[:, None] * stride_cos_s
    )
    sin_base = (
        batch_idx[:, None] * stride_sin_b
        + seq_idx[:, None] * stride_sin_s
    )

    cos1 = tl.load(
        cos_ptr + cos_base + offs_half[None, :],
        mask=valid_m[:, None],
        other=0.0,
    )
    cos2 = tl.load(
        cos_ptr + cos_base + HEAD_DIM // 2 + offs_half[None, :],
        mask=valid_m[:, None],
        other=0.0,
    )
    sin1 = tl.load(
        sin_ptr + sin_base + offs_half[None, :],
        mask=valid_m[:, None],
        other=0.0,
    )
    sin2 = tl.load(
        sin_ptr + sin_base + HEAD_DIM // 2 + offs_half[None, :],
        mask=valid_m[:, None],
        other=0.0,
    )

    result1 = (
        (normalized1 * cos1).to(tl.bfloat16)
        + (-normalized2 * sin1).to(tl.bfloat16)
    ).to(tl.bfloat16)
    result2 = (
        (normalized2 * cos2).to(tl.bfloat16)
        + (normalized1 * sin2).to(tl.bfloat16)
    ).to(tl.bfloat16)

    tl.store(q_ptr + q_base, result1, mask=q_mask)
    tl.store(
        q_ptr + q_base + HEAD_DIM // 2,
        result2,
        mask=q_mask,
    )
    tl.store(k_ptr + k_base, result1, mask=k_mask)
    tl.store(
        k_ptr + k_base + HEAD_DIM // 2,
        result2,
        mask=k_mask,
    )


@triton.jit
def _causal_gqa_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    seq_len: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    GROUP_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    block_m = tl.program_id(0)
    batch_query_group = tl.program_id(1)

    query_groups_per_batch = num_query_heads // GROUP_Q
    batch_idx = batch_query_group // query_groups_per_batch
    query_group = batch_query_group - batch_idx * query_groups_per_batch
    query_head = query_group * GROUP_Q
    kv_head = query_head // (num_query_heads // num_kv_heads)

    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, head_dim)
    valid_m = offs_m < seq_len

    q_offsets = (
        batch_idx * seq_len * num_query_heads * head_dim
        + offs_m[:, None] * num_query_heads * head_dim
        + query_head * head_dim
        + offs_d[None, :]
    )
    q = tl.load(
        q_ptr + q_offsets,
        mask=valid_m[:, None],
        other=0.0,
    )

    acc = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)
    row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    if GROUP_Q == 2:
        q_next = tl.load(
            q_ptr + q_offsets + head_dim,
            mask=valid_m[:, None],
            other=0.0,
        )
        acc_next = tl.zeros((BLOCK_M, head_dim), dtype=tl.float32)
        row_max_next = tl.full(
            (BLOCK_M,),
            -float("inf"),
            dtype=tl.float32,
        )
        row_sum_next = tl.zeros((BLOCK_M,), dtype=tl.float32)

    loop_end = tl.minimum((block_m + 1) * BLOCK_M, seq_len)

    for start_n in tl.range(0, loop_end, BLOCK_N):
        offs_n = start_n + offs_n_base
        valid_n = offs_n < seq_len

        kv_offsets = (
            batch_idx * seq_len * num_kv_heads * head_dim
            + offs_n[:, None] * num_kv_heads * head_dim
            + kv_head * head_dim
            + offs_d[None, :]
        )

        k = tl.load(
            k_ptr + kv_offsets,
            mask=valid_n[:, None],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k))
        scores = scores.to(tl.bfloat16).to(tl.float32)
        scores = (
            scores * 0.1275174302827162
        ).to(tl.bfloat16).to(tl.float32)

        causal_mask = (
            valid_m[:, None]
            & valid_n[None, :]
            & (offs_n[None, :] <= offs_m[:, None])
        )
        scores = tl.where(causal_mask, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, block_max)
        correction = tl.exp2(row_max - new_max)
        probabilities = tl.exp2(scores - new_max[:, None])

        row_sum = (
            row_sum * correction
            + tl.sum(probabilities, axis=1)
        )

        v = tl.load(
            v_ptr + kv_offsets,
            mask=valid_n[:, None],
            other=0.0,
        )

        acc = acc * correction[:, None]
        acc += tl.dot(probabilities.to(tl.bfloat16), v)
        row_max = new_max

        if GROUP_Q == 2:
            scores_next = tl.dot(q_next, tl.trans(k))
            scores_next = scores_next.to(tl.bfloat16).to(tl.float32)
            scores_next = (
                scores_next * 0.1275174302827162
            ).to(tl.bfloat16).to(tl.float32)
            scores_next = tl.where(
                causal_mask,
                scores_next,
                -float("inf"),
            )

            block_max_next = tl.max(scores_next, axis=1)
            new_max_next = tl.maximum(
                row_max_next,
                block_max_next,
            )
            correction_next = tl.exp2(
                row_max_next - new_max_next
            )
            probabilities_next = tl.exp2(
                scores_next - new_max_next[:, None]
            )

            row_sum_next = (
                row_sum_next * correction_next
                + tl.sum(probabilities_next, axis=1)
            )
            acc_next = acc_next * correction_next[:, None]
            acc_next += tl.dot(
                probabilities_next.to(tl.bfloat16),
                v,
            )
            row_max_next = new_max_next

    result = acc / row_sum[:, None]

    out_offsets = (
        batch_idx * seq_len * num_query_heads * head_dim
        + offs_m[:, None] * num_query_heads * head_dim
        + query_head * head_dim
        + offs_d[None, :]
    )
    tl.store(
        out_ptr + out_offsets,
        result.to(tl.bfloat16),
        mask=valid_m[:, None],
    )

    if GROUP_Q == 2:
        result_next = acc_next / row_sum_next[:, None]
        tl.store(
            out_ptr + out_offsets + head_dim,
            result_next.to(tl.bfloat16),
            mask=valid_m[:, None],
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
    batch_size, seq_len, _ = hidden_states.shape

    num_query_heads = 96
    num_kv_heads = 8
    head_dim = 128
    num_tokens = batch_size * seq_len

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

    postprocess_block_m = 16

    with torch.cuda.device(hidden_states.device):
        postprocess_grid = (
            triton.cdiv(num_tokens, postprocess_block_m),
            num_query_heads + num_kv_heads,
        )
        _qk_rmsnorm_rope_inplace_kernel[postprocess_grid](
            query_states,
            key_states,
            q_norm_weight,
            k_norm_weight,
            cos,
            sin,
            num_tokens,
            seq_len,
            rms_norm_eps,
            cos.stride(0),
            cos.stride(1),
            sin.stride(0),
            sin.stride(1),
            NUM_Q_HEADS=num_query_heads,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            BLOCK_M=postprocess_block_m,
            num_warps=8,
            num_stages=1,
        )

        attention_output = query_states

        if 768 <= seq_len <= 2048:
            group_q = 2
            block_m = 16
            block_n = 64
            num_warps = 4
        elif seq_len > 2048:
            group_q = 1
            block_m = 64
            block_n = 64
            num_warps = 4
        else:
            group_q = 1
            block_m = 32
            block_n = 64
            num_warps = 4

        attention_grid = (
            triton.cdiv(seq_len, block_m),
            batch_size * (num_query_heads // group_q),
        )

        _causal_gqa_attention_kernel[attention_grid](
            query_states,
            key_states,
            value_states,
            attention_output,
            seq_len=seq_len,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            GROUP_Q=group_q,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=3,
        )

    return F.linear(attention_output, o_proj_weight, None)