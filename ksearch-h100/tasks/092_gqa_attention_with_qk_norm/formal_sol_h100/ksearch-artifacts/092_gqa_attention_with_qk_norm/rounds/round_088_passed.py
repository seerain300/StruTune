# solution=GPT-5.6-Sol_092_gqa_attention_with_qk_norm_triton_optimized_r7 score=2.9670324498983764 passed=True
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _qk_projection_rmsnorm_rope_singlehead_kernel(
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

    is_query = head_tile < NUM_Q_HEADS
    head = tl.where(
        is_query,
        head_tile,
        head_tile - NUM_Q_HEADS,
    )

    token_offsets = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    half_offsets = tl.arange(0, HALF_SIZE)
    token_mask = token_offsets < num_tokens

    weight_ptr = tl.where(
        is_query,
        q_weight_ptr + head * HEAD_SIZE * HIDDEN_SIZE,
        k_weight_ptr + head * HEAD_SIZE * HIDDEN_SIZE,
    )
    bias_ptr = tl.where(
        is_query,
        q_bias_ptr + head * HEAD_SIZE,
        k_bias_ptr + head * HEAD_SIZE,
    )
    norm_weight_ptr = tl.where(
        is_query,
        q_norm_weight_ptr,
        k_norm_weight_ptr,
    )

    accumulator_lo = tl.zeros(
        (BLOCK_M, HALF_SIZE),
        dtype=tl.float32,
    )
    accumulator_hi = tl.zeros(
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

        weights_lo = tl.load(
            weight_ptr
            + k_offsets[:, None]
            + half_offsets[None, :] * HIDDEN_SIZE,
        )
        weights_hi = tl.load(
            weight_ptr
            + k_offsets[:, None]
            + (HALF_SIZE + half_offsets[None, :]) * HIDDEN_SIZE,
        )

        accumulator_lo += tl.dot(hidden, weights_lo)
        accumulator_hi += tl.dot(hidden, weights_hi)

    bias_lo = tl.load(
        bias_ptr + half_offsets,
    ).to(tl.float32)
    bias_hi = tl.load(
        bias_ptr + HALF_SIZE + half_offsets,
    ).to(tl.float32)

    projected_lo = (
        accumulator_lo + bias_lo[None, :]
    ).to(tl.bfloat16).to(tl.float32)
    projected_hi = (
        accumulator_hi + bias_hi[None, :]
    ).to(tl.bfloat16).to(tl.float32)

    squared_sum = tl.sum(
        projected_lo * projected_lo
        + projected_hi * projected_hi,
        axis=1,
    )
    inv_rms = tl.rsqrt(
        squared_sum / HEAD_SIZE + rms_norm_eps,
    )

    norm_weight_lo = tl.load(
        norm_weight_ptr + half_offsets,
    ).to(tl.float32)
    norm_weight_hi = tl.load(
        norm_weight_ptr + HALF_SIZE + half_offsets,
    ).to(tl.float32)

    normalized_lo = (
        projected_lo
        * inv_rms[:, None]
        * norm_weight_lo[None, :]
    )
    normalized_hi = (
        projected_hi
        * inv_rms[:, None]
        * norm_weight_hi[None, :]
    )

    rope_base = token_offsets[:, None] * HEAD_SIZE

    cos_lo = tl.load(
        cos_ptr + rope_base + half_offsets[None, :],
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    cos_hi = tl.load(
        cos_ptr
        + rope_base
        + HALF_SIZE
        + half_offsets[None, :],
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    sin_lo = tl.load(
        sin_ptr + rope_base + half_offsets[None, :],
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    sin_hi = tl.load(
        sin_ptr
        + rope_base
        + HALF_SIZE
        + half_offsets[None, :],
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    result_lo = (
        normalized_lo * cos_lo
        - normalized_hi * sin_lo
    )
    result_hi = (
        normalized_hi * cos_hi
        + normalized_lo * sin_hi
    )

    output_ptr = tl.where(
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

    tl.store(
        output_ptr + half_offsets[None, :],
        result_lo,
        mask=token_mask[:, None],
    )
    tl.store(
        output_ptr
        + HALF_SIZE
        + half_offsets[None, :],
        result_hi,
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

    block_m = 64
    num_head_tiles = num_attention_heads + num_key_value_heads

    _qk_projection_rmsnorm_rope_singlehead_kernel[
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
        num_stages=4,
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