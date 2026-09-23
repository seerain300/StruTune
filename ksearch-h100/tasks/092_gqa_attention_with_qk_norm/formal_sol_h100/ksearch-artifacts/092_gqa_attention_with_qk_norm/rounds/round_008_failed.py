# solution=GPT-5.6-Sol_092_gqa_attention_with_qk_norm_triton_optimized_r8 score=-1.0 passed=False
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
    NUM_Q_TILES: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HALF_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_block = tl.program_id(0)
    head_tile = tl.program_id(1)

    is_query = head_tile < NUM_Q_TILES
    head_base = tl.where(
        is_query,
        head_tile * 2,
        (head_tile - NUM_Q_TILES) * 2,
    )

    token_offsets = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    half_offsets = tl.arange(0, HALF_SIZE)
    token_mask = token_offsets < num_tokens

    weight_base0 = tl.where(
        is_query,
        q_weight_ptr + head_base * HEAD_SIZE * HIDDEN_SIZE,
        k_weight_ptr + head_base * HEAD_SIZE * HIDDEN_SIZE,
    )
    weight_base1 = tl.where(
        is_query,
        q_weight_ptr + (head_base + 1) * HEAD_SIZE * HIDDEN_SIZE,
        k_weight_ptr + (head_base + 1) * HEAD_SIZE * HIDDEN_SIZE,
    )

    bias_base0 = tl.where(
        is_query,
        q_bias_ptr + head_base * HEAD_SIZE,
        k_bias_ptr + head_base * HEAD_SIZE,
    )
    bias_base1 = tl.where(
        is_query,
        q_bias_ptr + (head_base + 1) * HEAD_SIZE,
        k_bias_ptr + (head_base + 1) * HEAD_SIZE,
    )

    norm_weight_ptr = tl.where(
        is_query,
        q_norm_weight_ptr,
        k_norm_weight_ptr,
    )

    accumulator0_lo = tl.zeros((BLOCK_M, HALF_SIZE), dtype=tl.float32)
    accumulator0_hi = tl.zeros((BLOCK_M, HALF_SIZE), dtype=tl.float32)
    accumulator1_lo = tl.zeros((BLOCK_M, HALF_SIZE), dtype=tl.float32)
    accumulator1_hi = tl.zeros((BLOCK_M, HALF_SIZE), dtype=tl.float32)

    for k_start in range(0, HIDDEN_SIZE, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        hidden = tl.load(
            hidden_ptr
            + token_offsets[:, None] * HIDDEN_SIZE
            + k_offsets[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )

        weights0_lo = tl.load(
            weight_base0
            + k_offsets[:, None]
            + half_offsets[None, :] * HIDDEN_SIZE,
        )
        weights0_hi = tl.load(
            weight_base0
            + k_offsets[:, None]
            + (HALF_SIZE + half_offsets)[None, :] * HIDDEN_SIZE,
        )
        weights1_lo = tl.load(
            weight_base1
            + k_offsets[:, None]
            + half_offsets[None, :] * HIDDEN_SIZE,
        )
        weights1_hi = tl.load(
            weight_base1
            + k_offsets[:, None]
            + (HALF_SIZE + half_offsets)[None, :] * HIDDEN_SIZE,
        )

        accumulator0_lo += tl.dot(hidden, weights0_lo)
        accumulator0_hi += tl.dot(hidden, weights0_hi)
        accumulator1_lo += tl.dot(hidden, weights1_lo)
        accumulator1_hi += tl.dot(hidden, weights1_hi)

    bias0_lo = tl.load(bias_base0 + half_offsets).to(tl.float32)
    bias0_hi = tl.load(bias_base0 + HALF_SIZE + half_offsets).to(tl.float32)
    bias1_lo = tl.load(bias_base1 + half_offsets).to(tl.float32)
    bias1_hi = tl.load(bias_base1 + HALF_SIZE + half_offsets).to(tl.float32)

    projected0_lo = (
        accumulator0_lo + bias0_lo[None, :]
    ).to(tl.bfloat16).to(tl.float32)
    projected0_hi = (
        accumulator0_hi + bias0_hi[None, :]
    ).to(tl.bfloat16).to(tl.float32)
    projected1_lo = (
        accumulator1_lo + bias1_lo[None, :]
    ).to(tl.bfloat16).to(tl.float32)
    projected1_hi = (
        accumulator1_hi + bias1_hi[None, :]
    ).to(tl.bfloat16).to(tl.float32)

    squared_sum0 = tl.sum(
        projected0_lo * projected0_lo
        + projected0_hi * projected0_hi,
        axis=1,
    )
    squared_sum1 = tl.sum(
        projected1_lo * projected1_lo
        + projected1_hi * projected1_hi,
        axis=1,
    )

    inv_rms0 = tl.rsqrt(squared_sum0 / HEAD_SIZE + rms_norm_eps)
    inv_rms1 = tl.rsqrt(squared_sum1 / HEAD_SIZE + rms_norm_eps)

    norm_weight_lo = tl.load(
        norm_weight_ptr + half_offsets,
    ).to(tl.float32)
    norm_weight_hi = tl.load(
        norm_weight_ptr + HALF_SIZE + half_offsets,
    ).to(tl.float32)

    normalized0_lo = (
        projected0_lo
        * inv_rms0[:, None]
        * norm_weight_lo[None, :]
    )
    normalized0_hi = (
        projected0_hi
        * inv_rms0[:, None]
        * norm_weight_hi[None, :]
    )
    normalized1_lo = (
        projected1_lo
        * inv_rms1[:, None]
        * norm_weight_lo[None, :]
    )
    normalized1_hi = (
        projected1_hi
        * inv_rms1[:, None]
        * norm_weight_hi[None, :]
    )

    rope_base = token_offsets[:, None] * HEAD_SIZE

    cos_lo = tl.load(
        cos_ptr + rope_base + half_offsets[None, :],
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    cos_hi = tl.load(
        cos_ptr + rope_base + HALF_SIZE + half_offsets[None, :],
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    sin_lo = tl.load(
        sin_ptr + rope_base + half_offsets[None, :],
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    sin_hi = tl.load(
        sin_ptr + rope_base + HALF_SIZE + half_offsets[None, :],
        mask=token_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    result0_lo = normalized0_lo * cos_lo - normalized0_hi * sin_lo
    result0_hi = normalized0_hi * cos_hi + normalized0_lo * sin_hi
    result1_lo = normalized1_lo * cos_lo - normalized1_hi * sin_lo
    result1_hi = normalized1_hi * cos_hi + normalized1_lo * sin_hi

    output_base0 = tl.where(
        is_query,
        q_output_ptr
        + (token_offsets[:, None] * NUM_Q_HEADS + head_base) * HEAD_SIZE,
        k_output_ptr
        + (token_offsets[:, None] * NUM_K_HEADS + head_base) * HEAD_SIZE,
    )
    output_base1 = output_base0 + HEAD_SIZE

    tl.store(
        output_base0 + half_offsets[None, :],
        result0_lo,
        mask=token_mask[:, None],
    )
    tl.store(
        output_base0 + HALF_SIZE + half_offsets[None, :],
        result0_hi,
        mask=token_mask[:, None],
    )
    tl.store(
        output_base1 + half_offsets[None, :],
        result1_lo,
        mask=token_mask[:, None],
    )
    tl.store(
        output_base1 + HALF_SIZE + half_offsets[None, :],
        result1_hi,
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
    num_q_tiles = num_attention_heads // 2
    num_k_tiles = num_key_value_heads // 2

    _qk_projection_rmsnorm_rope_twohead_kernel[
        (
            triton.cdiv(num_tokens, block_m),
            num_q_tiles + num_k_tiles,
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
        NUM_Q_TILES=num_q_tiles,
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