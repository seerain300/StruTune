# solution=GPT-5.6-Sol_092_gqa_attention_with_qk_norm_triton_optimized_r5 score=3.8401937521525333 passed=True
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _qk_rmsnorm_rope_kernel(
    q_ptr,
    k_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    cos_ptr,
    sin_ptr,
    num_tokens,
    rms_norm_eps,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    NUM_Q_HEAD_BLOCKS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HALF_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_block = tl.program_id(0)
    combined_head_block = tl.program_id(1)

    is_query = combined_head_block < NUM_Q_HEAD_BLOCKS
    local_head_block = tl.where(
        is_query,
        combined_head_block,
        combined_head_block - NUM_Q_HEAD_BLOCKS,
    )

    num_heads = tl.where(is_query, NUM_Q_HEADS, NUM_K_HEADS)
    states_ptr = tl.where(is_query, q_ptr, k_ptr)
    norm_weight_ptr = tl.where(
        is_query,
        q_norm_weight_ptr,
        k_norm_weight_ptr,
    )

    token_offsets = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    head_offsets = local_head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    half_offsets = tl.arange(0, HALF_SIZE)

    token_mask = token_offsets < num_tokens
    head_mask = head_offsets < num_heads
    state_mask = token_mask[:, None, None] & head_mask[None, :, None]

    state_base = (
        token_offsets[:, None] * num_heads + head_offsets[None, :]
    ) * HEAD_SIZE

    projected_lo = tl.load(
        states_ptr
        + state_base[:, :, None]
        + half_offsets[None, None, :],
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    projected_hi = tl.load(
        states_ptr
        + state_base[:, :, None]
        + HALF_SIZE
        + half_offsets[None, None, :],
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)

    squared_sum = tl.sum(
        projected_lo * projected_lo + projected_hi * projected_hi,
        axis=2,
    )
    inv_rms = tl.rsqrt(squared_sum / HEAD_SIZE + rms_norm_eps)

    norm_weight_lo = tl.load(
        norm_weight_ptr + half_offsets
    ).to(tl.float32)
    norm_weight_hi = tl.load(
        norm_weight_ptr + HALF_SIZE + half_offsets
    ).to(tl.float32)

    normalized_lo = (
        projected_lo
        * inv_rms[:, :, None]
        * norm_weight_lo[None, None, :]
    )
    normalized_hi = (
        projected_hi
        * inv_rms[:, :, None]
        * norm_weight_hi[None, None, :]
    )

    rope_base = token_offsets[:, None] * HEAD_SIZE
    rope_mask = token_mask[:, None]

    cos_lo = tl.load(
        cos_ptr + rope_base + half_offsets[None, :],
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    cos_hi = tl.load(
        cos_ptr + rope_base + HALF_SIZE + half_offsets[None, :],
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    sin_lo = tl.load(
        sin_ptr + rope_base + half_offsets[None, :],
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    sin_hi = tl.load(
        sin_ptr + rope_base + HALF_SIZE + half_offsets[None, :],
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)

    result_lo = (
        normalized_lo * cos_lo[:, None, :]
        - normalized_hi * sin_lo[:, None, :]
    )
    result_hi = (
        normalized_hi * cos_hi[:, None, :]
        + normalized_lo * sin_hi[:, None, :]
    )

    tl.store(
        states_ptr
        + state_base[:, :, None]
        + half_offsets[None, None, :],
        result_lo,
        mask=state_mask,
    )
    tl.store(
        states_ptr
        + state_base[:, :, None]
        + HALF_SIZE
        + half_offsets[None, None, :],
        result_hi,
        mask=state_mask,
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
    num_attention_heads = 96
    num_key_value_heads = 8
    head_dim = 128
    num_tokens = batch_size * seq_len

    query_states = F.linear(
        hidden_states,
        q_proj_weight,
        q_proj_bias,
    ).reshape(
        batch_size,
        seq_len,
        num_attention_heads,
        head_dim,
    )
    key_states = F.linear(
        hidden_states,
        k_proj_weight,
        k_proj_bias,
    ).reshape(
        batch_size,
        seq_len,
        num_key_value_heads,
        head_dim,
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

    block_m = 2
    block_h = 8
    num_q_head_blocks = triton.cdiv(num_attention_heads, block_h)
    num_k_head_blocks = triton.cdiv(num_key_value_heads, block_h)

    _qk_rmsnorm_rope_kernel[
        (
            triton.cdiv(num_tokens, block_m),
            num_q_head_blocks + num_k_head_blocks,
        )
    ](
        query_states,
        key_states,
        q_norm_weight,
        k_norm_weight,
        cos,
        sin,
        num_tokens,
        rms_norm_eps,
        NUM_Q_HEADS=num_attention_heads,
        NUM_K_HEADS=num_key_value_heads,
        NUM_Q_HEAD_BLOCKS=num_q_head_blocks,
        HEAD_SIZE=head_dim,
        HALF_SIZE=head_dim // 2,
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        num_warps=8,
        num_stages=2,
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

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(
        batch_size,
        seq_len,
        num_attention_heads * head_dim,
    )

    return F.linear(
        attn_output,
        o_proj_weight,
        None,
    )