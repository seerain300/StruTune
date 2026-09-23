# solution=GPT-5.6-Sol_gdn_prefill_qk4_v8_d128_k_last_triton_optimized_r5 score=-1.0 passed=False
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_precompute_gates_kernel(
    q_ptr,
    k_ptr,
    A_log_ptr,
    a_ptr,
    dt_bias_ptr,
    b_ptr,
    gates_ptr,
    qk_ptr,
    total_seq_len,
    BLOCK_T: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    v_heads = tl.arange(0, 8)

    gate_offsets = tokens[:, None] * 8 + v_heads[None, :]
    gate_mask = tokens[:, None] < total_seq_len

    decay_input = (
        tl.load(a_ptr + gate_offsets, mask=gate_mask, other=0.0).to(tl.float32)
        + tl.load(dt_bias_ptr + v_heads)[None, :]
    )
    decay_rate = tl.exp(tl.load(A_log_ptr + v_heads