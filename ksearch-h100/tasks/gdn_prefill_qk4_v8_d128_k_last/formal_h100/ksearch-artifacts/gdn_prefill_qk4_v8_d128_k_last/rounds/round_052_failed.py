# solution=GPT-5.6-Sol_gdn_prefill_qk4_v8_d128_k_last_triton_optimized_r10 score=-1.0 passed=False
import math

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 1}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_V": 1}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_V": 2}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_V": 2}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_V": 4}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_V": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 8}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_V": 8}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 16}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 16}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_V": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 32}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_V": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_V": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_V": 128}, num_warps=8, num_stages=1),
    ],
    key=["SEQ_LEN_BUCKET", "NUM_SEQS_BUCKET", "HAS_STATE"],
)
@triton.jit
def _gdn_prefill_direct_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    state_ptr,
    A_log_ptr,
    a_ptr,
    dt_bias_ptr,
    b_ptr,
    cu_seqlens_ptr,
    output_ptr,
    new_state_ptr,
    scale,
    SEQ_LEN_BUCKET,
    NUM_SEQS_BUCKET,
    HEAD_SIZE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    HAS_STATE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    value_pair = tl.program_id(1)
    value_block = tl.program_id(2)

    value_head0 = value_pair * 2
    value_head1 = value_head0 + 1

    value_offsets = value_block * BLOCK_V + tl.arange(0, BLOCK_V)
    key_offsets = tl.arange(0, HEAD_SIZE)

    query_head = value_pair
    key_head = value_pair

    seq_start = tl.load(cu_seqlens_ptr + seq_idx).to(tl.int32)
    seq_end = tl.load(cu_seqlens_ptr + seq_idx + 1).to(tl.int32)

    state_offsets0 = (
        (
            (seq_idx * NUM_V_HEADS + value_head0) * HEAD_SIZE
            + value_offsets[:, None]
        )
        * HEAD_SIZE
        + key_offsets[None, :]
    )
    state_offsets1 = (
        (
            (seq_idx * NUM_V_HEADS + value_head1) * HEAD_SIZE
            + value_offsets[:, None]
        )
        * HEAD_SIZE
        + key_offsets[None, :]
    )

    state_values0 = tl.zeros((BLOCK_V, HEAD_SIZE), dtype=tl.float32)
    state_values1 = tl.zeros((BLOCK_V, HEAD_SIZE), dtype=tl.float32)

    if HAS_STATE:
        if seq_start < seq_end:
            state_values0 = tl.load(
                state_ptr + state_offsets0,
                cache_modifier=".cg",
            ).to(tl.float32)
            state_values1 = tl.load(
                state_ptr + state_offsets1,
                cache_modifier=".cg",
            ).to(tl.float32)

    decay_rate0 = tl.exp(
        tl.load(A_log_ptr + value_head0).to(tl.float32)
    )
    decay_rate1 = tl.exp(
        tl.load(A_log_ptr + value_head1).to(tl.float32)
    )

    decay_bias0 = tl.load(
        dt_bias_ptr + value_head0
    ).to(tl.float32)
    decay_bias1 = tl.load(
        dt_bias_ptr + value_head1
    ).to(tl.float32)

    token_idx = seq_start
    while token_idx < seq_end:
        key_base = (
            (token_idx * NUM_K_HEADS + key_head) * HEAD_SIZE
        )
        query_base = (
            (token_idx * NUM_Q_HEADS + query_head) * HEAD_SIZE
        )
        value_base0 = (
            (token_idx * NUM_V_HEADS + value_head0) * HEAD_SIZE
        )
        value_base1 = (
            (token_idx * NUM_V_HEADS + value_head1) * HEAD_SIZE
        )

        key_values = tl.load(
            k_ptr + key_base + key_offsets
        ).to(tl.float32)

        query_values = tl.load(
            q_ptr + query_base + key_offsets
        ).to(tl.float32)

        input_values0 = tl.load(
            v_ptr + value_base0 + value_offsets
        ).to(tl.float32)
        input_values1 = tl.load(
            v_ptr + value_base1 + value_offsets
        ).to(tl.float32)

        gate_offset0 = token_idx * NUM_V_HEADS + value_head0
        gate_offset1 = token_idx * NUM_V_HEADS + value_head1

        decay_input0 = (
            tl.load(a_ptr + gate_offset0).to(tl.float32)
            + decay_bias0
        )
        decay_input1 = (
            tl.load(a_ptr + gate_offset1).to(tl.float32)
            + decay_bias1
        )

        softplus0 = (
            tl.maximum(decay_input0, 0.0)
            + tl.log(1.0 + tl.exp(-tl.abs(decay_input0)))
        )
        softplus1 = (
            tl.maximum(decay_input1, 0.0)
            + tl.log(1.0 + tl.exp(-tl.abs(decay_input1)))
        )

        decay0 = tl.exp(-decay_rate0 * softplus0)
        decay1 = tl.exp(-decay_rate1 * softplus1)

        beta0 = tl.sigmoid(
            tl.load(b_ptr + gate_offset0).to(tl.float32)
        )
        beta1 = tl.sigmoid(
            tl.load(b_ptr + gate_offset1).to(tl.float32)
        )

        state_values0 *= decay0
        predicted_values0 = tl.sum(
            state_values0 * key_values[None, :],
            axis=1,
        )
        correction0 = beta0 * (input_values0 - predicted_values0)
        state_values0 += correction0[:, None] * key_values[None, :]

        state_values1 *= decay1
        predicted_values1 = tl.sum(
            state_values1 * key_values[None, :],
            axis=1,
        )
        correction1 = beta1 * (input_values1 - predicted_values1)
        state_values1 += correction1[:, None] * key_values[None, :]

        output_values0 = tl.sum(
            state_values0 * query_values[None, :],
            axis=1,
        ) * scale
        output_values1 = tl.sum(
            state_values1 * query_values[None, :],
            axis=1,
        ) * scale

        output_offsets0 = (
            (token_idx * NUM_V_HEADS + value_head0) * HEAD_SIZE
            + value_offsets
        )
        output_offsets1 = (
            (token_idx * NUM_V_HEADS + value_head1) * HEAD_SIZE
            + value_offsets
        )

        tl.store(output_ptr + output_offsets0, output_values0)
        tl.store(output_ptr + output_offsets1, output_values1)

        token_idx += 1

    tl.store(
        new_state_ptr + state_offsets0,
        state_values0,
        cache_modifier=".cs",
    )
    tl.store(
        new_state_ptr + state_offsets1,
        state_values1,
        cache_modifier=".cs",
    )


def _validate_tensor(name, tensor, dtype, ndim):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.dtype != dtype:
        raise TypeError(
            f"{name} must have dtype {dtype}, got {tensor.dtype}"
        )
    if tensor.ndim != ndim:
        raise ValueError(
            f"{name} must have {ndim} dimensions, got {tensor.ndim}"
        )


def _next_power_of_two(value):
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def run(
    q,
    k,
    v,
    state,
    A_log,
    a,
    dt_bias,
    b,
    cu_seqlens,
    scale,
):
    _validate_tensor("q", q, torch.bfloat16, 3)
    _validate_tensor("k", k, torch.bfloat16, 3)
    _validate_tensor("v", v, torch.bfloat16, 3)
    _validate_tensor("A_log", A_log, torch.float32, 1)
    _validate_tensor("a", a, torch.bfloat16, 2)
    _validate_tensor("dt_bias", dt_bias, torch.float32, 1)
    _validate_tensor("b", b, torch.bfloat16, 2)
    _validate_tensor("cu_seqlens", cu_seqlens, torch.int64, 1)

    if state is not None:
        _validate_tensor("state", state, torch.float32, 4)

    total_seq_len = q.shape[0]
    num_seqs = cu_seqlens.numel() - 1

    if num_seqs < 0:
        raise ValueError(
            "cu_seqlens must contain at least one element"
        )

    if q.shape != (total_seq_len, 4, 128):
        raise ValueError(
            f"q must have shape [{total_seq_len}, 4, 128]"
        )
    if k.shape != (total_seq_len, 4, 128):
        raise ValueError(
            f"k must have shape [{total_seq_len}, 4, 128]"
        )
    if v.shape != (total_seq_len, 8, 128):
        raise ValueError(
            f"v must have shape [{total_seq_len}, 8, 128]"
        )
    if a.shape != (total_seq_len, 8):
        raise ValueError(
            f"a must have shape [{total_seq_len}, 8]"
        )
    if b.shape != (total_seq_len, 8):
        raise ValueError(