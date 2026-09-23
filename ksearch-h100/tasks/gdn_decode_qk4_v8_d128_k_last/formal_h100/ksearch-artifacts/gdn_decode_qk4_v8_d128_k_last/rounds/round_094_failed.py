# solution=GPT-5.6-Sol_gdn_decode_qk4_v8_d128_k_last_triton_optimized_r7 score=-1.0 passed=False
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_grouped_row_streaming_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    state_ptr,
    A_log_ptr,
    a_ptr,
    dt_bias_ptr,
    beta_input_ptr,
    output_ptr,
    new_state_ptr,
    scale,
    HAS_STATE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)

    groups_per_head = 128 // BLOCK_ROWS
    row_group = pid % groups_per_head
    bh = pid // groups_per_head

    value_head = bh % 8
    batch = bh // 8
    query_head = value_head // 2

    rows = row_group * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    k_offsets = tl.arange(0, BLOCK_K)

    qk_offset = (batch * 4 + query_head) * 128 + k_offsets
    state_offset = (
        ((batch * 8 + value_head) * 128 + rows[:, None]) * 128
        + k_offsets[None, :]
    )
    gate_offset = batch * 8 + value_head
    value_offset = (batch * 8 + value_head) * 128 + rows

    q = tl.load(q_ptr + q