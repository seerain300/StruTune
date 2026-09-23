# solution=GPT-5.6-Sol_092_gqa_attention_with_qk_norm_triton_optimized_r5 score=-1.0 passed=False
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _qkv_projection_rmsnorm_rope_multihead_kernel(
    hidden_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    v_weight_ptr,
    v_bias_ptr,
    q_output_ptr,
    k_output_ptr,
    v_output_ptr,
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
    HEADS_PER_Q_TILE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_block = tl.program_id(0)
    head_tile = tl.program_id(1)

    num_q_tiles = NUM_Q_HEADS // HEADS_PER_Q_TILE
    is_query = head_tile < num_q_tiles
    family_tile = tl.where(
        is_query,
        head_tile,
        head_tile - num_q_tiles,
    )

    head_0 = tl.where(
        is_query,
        family_tile * HEADS_PER_Q_TILE,
        family_tile,
    )
    head_1 = head_0 + 1

    token_offsets = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    half_offsets = tl.arange(0, HALF_SIZE)
    token_mask = token_offsets < num_tokens

    weight_ptr_0 = tl.where(
        is_query,
        q_weight_ptr + head_0 * HEAD_SIZE * HIDDEN_SIZE,
        k_weight_ptr + head_0 * HEAD_SIZE * HIDDEN_SIZE,
    )
    weight_ptr_1 = tl.where(
        is_query,
        q_weight_ptr + head_1 * HEAD_SIZE * HIDDEN_SIZE,
        v_weight_ptr + head_0 * HEAD_SIZE * HIDDEN_SIZE,
    )

    bias_ptr_0 = tl.where(
        is_query,
        q_bias_ptr + head_0 * HEAD_SIZE,
        k_bias_ptr + head_0 * HEAD_SIZE,
    )
    bias_ptr_1 = tl.where(
        is_query,
        q_bias_ptr + head_1 * HEAD_SIZE,
        v_bias_ptr + head_0 * HEAD_SIZE,
    )

    norm_weight_ptr = tl.where(
        is_query,
        q_norm_weight_ptr,
        k_norm_weight_ptr,
    )

    accumulator_0_lo = tl.zeros(
        (BLOCK_M, HALF_SIZE), dtype=tl.float32
    )
    accumulator_0_hi = tl.zeros(
        (BLOCK_M, HALF_SIZE), dtype=tl.float32
    )
    accumulator_1_lo = tl.zeros(
        (BLOCK_M, HALF_SIZE), dtype=tl.float32
    )
    accumulator_1_hi = tl.zeros(
        (BLOCK_M, HALF_SIZE), dtype=tl.float32
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

    bias_0_lo = tl.load(
        bias_ptr_0 + half_offsets
    ).to(tl.float32)
    bias_0_hi = tl.load(
        bias_ptr_0 + HALF_SIZE + half_offsets
    ).to(tl.float32)
    bias_1_lo = tl.load(
        bias_ptr_1 + half_offsets
    ).to(tl.float32)
    bias_1_hi = tl.load(
        bias_ptr_1 + HALF_SIZE + half_offsets
    ).to(tl.float32)

    projected_0_lo = (
        accumulator_0_lo + bias_0_lo[None, :]
    ).to(tl.bfloat16).to(tl.float32)
    projected_0_hi = (
        accumulator_0_hi + bias_0_hi[None, :]
    ).to(tl.bfloat16).to(tl.float32)
    projected_1_lo = (
        accumulator_1_lo + bias_1_lo[None, :]
    ).to(tl.bfloat16).to(tl.float32)
    projected_1_hi = (
        accumulator_1_hi + bias_1_hi[None, :]
    ).to(tl.bfloat16).to(tl.float32)

    squared_sum_0 = tl.sum(
        projected_0_lo * projected_0_lo
        + projected_0_hi * projected_0_hi,
        axis=1,
    )
    squared_sum_1 = tl.sum(
        projected_1_lo * projected_1_lo
        + projected_1_hi * projected_1_hi,
        axis=1,
    )

    inv_rms_0 = tl.rsqrt(
        squared_sum_0 / HEAD_SIZE + rms_norm_eps
    )
    inv_rms_1 = tl.rsqrt(
        squared_sum_1 / HEAD_SIZE + rms_norm_eps
    )

    norm_weight_lo = tl.load(
        norm_weight_ptr + half_offsets
    ).to(tl.float32)
    norm_weight_hi = tl.load(
        norm_weight_ptr + HALF_SIZE + half_offsets
    ).to(tl.float32)

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

    rotated_0_lo = (
        normalized_0_lo * cos_lo
        - normalized_0_hi * sin_lo
    )
    rotated_0_hi = (
        normalized_0_hi * cos_hi
        + normalized_0_lo * sin_hi
    )
    rotated_1_lo = (
        normalized_1_lo * cos_lo
        - normalized_1_hi * sin_lo
    )
    rotated_1_hi = (
        normalized_1_hi * cos_hi
        + normalized_1_lo * sin_hi
    )

    result_1_lo = tl.where(
        is_query,
        rotated_1_lo,
        projected_1_lo,
    )
    result_1_hi = tl.where(
        is_query,
        rotated_1_hi,
        projected_1_hi,
    )

    output_ptr_0 = tl.where(
        is_query,
        q_output_ptr
        + (
            token_offsets[:, None] * NUM_Q_HEADS
            + head_0
        )
        * HEAD_SIZE,
        k_output_ptr
        + (
            token_offsets[:, None] * NUM_K_HEADS
            + head_0
        )
        * HEAD_SIZE,
    )
    output_ptr_1 = tl.where(
        is_query,
        q_output_ptr
        + (
            token_offsets[:, None] * NUM_Q_HEADS
            + head_1
        )
        * HEAD_SIZE,
        v_output_ptr
        + (
            token_offsets