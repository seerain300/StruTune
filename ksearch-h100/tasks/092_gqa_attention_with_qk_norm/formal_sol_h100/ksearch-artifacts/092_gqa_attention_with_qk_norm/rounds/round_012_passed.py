# solution=GPT-5.6-Sol_092_gqa_attention_with_qk_norm_triton_optimized_r12 score=3.1278203521627352 passed=True
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _qk_projection_rmsnorm_rope_twohead_kernel(
    hidden_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    q_output_ptr,
    k_output_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    num_tokens,
    rms_norm_eps,
    HIDDEN_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HALF_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_block = tl.program_id(0)
    head_tile = tl.program_id(1)

    num_q_head_tiles = NUM_Q_HEADS // 2
    is_query = head_tile < num_q_head_tiles
    head_pair = tl.where(
        is_query,
        head_tile,
        head_tile - num_q_head_tiles,
    )
    head = head_pair * 2

    token_offsets = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    half_offsets = tl.arange(0, HALF_SIZE)
    token_mask = token_offsets < num_tokens

    weight_ptr_0 = tl.where(
        is_query,
        q_weight_ptr + head * HEAD_SIZE * HIDDEN_SIZE,
        k_weight_ptr + head * HEAD_SIZE * HIDDEN_SIZE,
    )
    weight_ptr_1 = weight_ptr_0 + HEAD_SIZE * HIDDEN_SIZE

    bias_ptr_0 = tl.where(
        is_query,
        q_bias_ptr + head * HEAD_SIZE,
        k_bias_ptr + head * HEAD_SIZE,
    )
    bias_ptr_1 = bias_ptr_0 + HEAD_SIZE

    norm_weight_ptr = tl.where(
        is_query,
        q_norm_weight_ptr,
        k_norm_weight_ptr,
    )

    accumulator_0_lo = tl.zeros(
        (BLOCK_M, HALF_SIZE),
        dtype=tl.float32,
    )
    accumulator_0_hi = tl.zeros(
        (BLOCK_M, HALF_SIZE),
        dtype=tl.float32,
    )
    accumulator_1_lo = tl.zeros(
        (BLOCK_M, HALF_SIZE),
        dtype=tl.float32,
    )
    accumulator_1_hi = tl.zeros(
        (BLOCK_M, HALF_SIZE),
        dtype=tl.float32,
    )

    for k_start in range(0, HIDDEN_SIZE, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        hidden = tl.load(
            hidden_ptr
            + token_offsets[:, None] * HIDDEN_SIZE
            + k_offsets[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )

        weights_0_lo = tl.load(
            weight_ptr_0
            + k_offsets[:, None]
            + half_offsets[None, :] * HIDDEN_SIZE,
        )
        weights_0_hi = tl.load(
            weight_ptr_0
            + k_offsets[:, None]
            + (HALF_SIZE + half_offsets[None, :]) * HIDDEN_SIZE,
        )
        weights_1_lo = tl.load(
            weight_ptr_1
            + k_offsets[:, None]
            + half_offsets[None, :] * HIDDEN_SIZE,
        )
        weights_1_hi = tl.load(
            weight_ptr_1
            + k_offsets[:, None]
            + (HALF_SIZE + half_offsets[None, :]) * HIDDEN_SIZE,
        )

        accumulator_0_lo += tl.dot(hidden, weights_0_lo)
        accumulator_0_hi += tl.dot(hidden, weights_0_hi)
        accumulator_1_lo += tl.dot(hidden, weights_1_lo)
        accumulator_1_hi += tl.dot(hidden, weights_1_hi)

    norm_weight_lo = tl.load(
        norm_weight_ptr + half_offsets,
    ).to(tl.float32)
    norm_weight_hi = tl.load(
        norm_weight_ptr + HALF_SIZE + half_offsets,
    ).to(tl.float32)

    rope_base = token_offsets[:, None] * HEAD_SIZE

    output_ptr_0 = tl.where(
        is_query,
        q_output_ptr
        + (
            token_offsets[:, None] * NUM_Q_HEADS
            + head
        )
        * HEAD_SIZE,
        k_output_ptr
        + (
            token_offsets[:, None] * NUM_K_HEADS
            + head
        )
        * HEAD_SIZE,
    )
    output_ptr_1 = output_ptr_0 + HEAD_SIZE

    bias_0_lo = tl.load(
        bias_ptr_0 + half_offsets,
    ).to(tl.float32)
    bias_0_hi = tl.load(
        bias_ptr_0 + HALF_SIZE + half_offsets,
    ).to(tl.float32)

    projected_0_lo = (
        accumulator_0_lo + bias_0_lo[None, :]
    ).to(tl.bfloat16).to(tl.float32)
    projected_0_hi = (
        accumulator_0_hi + bias_0_hi[None, :]
    ).to(tl.bfloat16).to(tl.float32)

    squared_sum_0 = tl.sum(
        projected_0_lo * projected_0_lo
        + projected_0_hi * projected_0_hi,
        axis=1,
    )
    inv_rms_0 = tl.rsqrt(
        squared_sum_0 / HEAD_SIZE + rms_norm_eps,
    )

    normalized_0_lo = (
        projected_0_lo
        * inv_rms_0[:, None]
        * norm_weight_lo[None, :]
    )
    normalized_0_hi = (
        projected_0_hi
        * inv_rms_0[:, None]
        * norm_weight_hi[None, :]
    )

    cos_0_lo = tl.load(
        cos_ptr + rope_base + half_offsets[None, :],
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    cos_0_hi = tl.load(
        cos_ptr
        + rope_base
        + HALF_SIZE
        + half_offsets[None, :],
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    sin_0_lo = tl.load(
        sin_ptr + rope_base + half_offsets[None, :],
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    sin_0_hi = tl.load(
        sin_ptr
        + rope_base
        + HALF_SIZE
        + half_offsets[None, :],
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    result_0_lo = (
        normalized_0_lo * cos_0_lo
        - normalized_0_hi * sin_0_lo
    )
    result_0_hi = (
        normalized_0_hi * cos_0_hi
        + normalized_0_lo * sin_0_hi
    )

    tl.store(
        output_ptr_0 + half_offsets[None, :],
        result_0_lo,
        mask=token_mask[:, None],
    )
    tl.store(
        output_ptr_0
        + HALF_SIZE
        + half_offsets[None, :],
        result_0_hi,
        mask=token_mask[:, None],
    )

    bias_1_lo = tl.load(
        bias_ptr_1 + half_offsets,
    ).to(tl.float32)
    bias_1_hi = tl.load(
        bias_ptr_1 + HALF_SIZE + half_offsets,
    ).to(tl.float32)

    projected_1_lo = (
        accumulator_1_lo + bias_1_lo[None, :]
    ).to(tl.bfloat16).to(tl.float32)
    projected_1_hi = (
        accumulator_1_hi + bias_1_hi[None, :]
    ).to(tl.bfloat16).to(tl.float32)

    squared_sum_1 = tl.sum(
        projected_1_lo * projected_1_lo
        + projected_1_hi * projected_1_hi,
        axis=1,
    )
    inv_rms_1 = tl.rsqrt(
        squared_sum_1 / HEAD_SIZE + rms_norm_eps,
    )

    normalized_1_lo = (
        projected_1_lo
        * inv_rms_1[:, None]
        * norm_weight_lo[None, :]
    )
    normalized_1_hi = (
        projected_1_hi
        * inv_rms_1[:, None]
        * norm_weight_hi[None, :]
    )

    cos_1_lo = tl.load(
        cos_ptr + rope_base + half_offsets[None, :],
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    cos_1_hi = tl.load(
        cos_ptr
        + rope_base
        + HALF_SIZE
        + half_offsets[None, :],
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    sin_1_lo = tl.load(
        sin_ptr + rope_base + half_offsets[None, :],
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    sin_1_hi = tl.load(
        sin_ptr
        + rope_base
        + HALF_SIZE
        + half_offsets[None, :],
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    result_1_lo = (
        normalized_1_lo * cos_1_lo
        - normalized_1_hi * sin_1_lo
    )
    result_1_hi = (
        normalized_1_hi * cos_1_hi
        + normalized_1_lo * sin_1_hi
    )

    tl.store(
        output_ptr_1 + half_offsets[None, :],
        result_1_lo,
        mask=token_mask[:, None],
    )
    tl.store(
        output_ptr_1
        + HALF_SIZE
        + half_offsets[None, :],
        result_1_hi,
        mask=token_mask[:, None],
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
    batch_size, seq_len, hidden_size = hidden_states.shape
    num_attention_heads = 96
    num_key_value_heads = 8
    head_dim = 128
    num_tokens = batch_size * seq_len

    hidden_states = hidden_states.contiguous()
    q_proj_weight = q_proj_weight.contiguous()
    q_proj_bias = q_proj_bias.contiguous()
    k_proj_weight = k_proj_weight.contiguous()
    k_proj_bias = k_proj_bias.contiguous()
    v_proj_weight = v_proj_weight.contiguous()
    v_proj_bias = v_proj_bias.contiguous()
    q_norm_weight = q_norm_weight.contiguous()
    k_norm_weight = k_norm_weight.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    query_states = torch.empty(
        (
            batch_size,
            seq_len,
            num_attention_heads,
            head_dim,
        ),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    key_states = torch.empty(
        (
            batch_size,
            seq_len,
            num_key_value_heads,
            head_dim,
        ),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    block_m = 128 if num_tokens >= 2048 else 64
    num_head_tiles = (
        num_attention_heads // 2
        + num_key_value_heads // 2
    )

    _qk_projection_rmsnorm_rope_twohead_kernel[
        (
            triton.cdiv(num_tokens, block_m),
            num_head_tiles,
        )
    ](
        hidden_states,
        q_proj_weight,
        q_proj_bias,
        k_proj_weight,
        k_proj_bias,
        query_states,
        key_states,
        q_norm_weight,
        k_norm_weight,
        cos,
        sin,
        num_tokens,
        rms_norm_eps,
        HIDDEN_SIZE=hidden_size,
        NUM_Q_HEADS=num_attention_heads,
        NUM_K_HEADS=num_key_value_heads,
        HEAD_SIZE=head_dim,
        HALF_SIZE=head_dim // 2,
        BLOCK_M=block_m,
        BLOCK_K=128,
        num_warps=8,
        num_stages=2,
    )

    value_states = F.linear(
        hidden_states,
        v_proj_weight,
        v_proj_bias,
    ).reshape(
        batch_size,
        seq_len,
        num_key_value_heads,
        head_dim,
    )

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        scale=head_dim ** -0.5,
        enable_gqa=True,
    )

    attn_output = (
        attn_output.transpose(1, 2)
        .contiguous()
        .reshape(
            batch_size,
            seq_len,
            num_attention_heads * head_dim,
        )
    )

    return F.linear(
        attn_output,
        o_proj_weight,
        None,
    )