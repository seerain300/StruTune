# KDA A800 best solution: gqa_ragged_prefill_causal_h32_kv8_d128
# candidate: c001  |  feedback: 13.89x  |  final (authoritative): 12.92x
# campaign formal-kda-20260916 (A800, g0056)  |  evaluations: 1
# source: tasks/formal-kda-20260916--flashinfer--gqa_ragged_prefill_causal_h32_kv8_d128/control/candidates/c001/solution.py (sha256-frozen snapshot)

"""KDA candidate c001 — gqa_ragged_prefill_causal_h32_kv8_d128 (A800 / sm_80).

Fused ragged (varlen) causal Grouped-Query-Attention prefill in a single Triton
launch. Numerics follow docs/draft.md §3 (base-2 online-softmax flash):
  s2 = (q·k) * sm_scale * log2(e)   →   exp(s) = exp2(s2)
  output = softmax(logits) @ v      (fp32 accumulate)
  lse    = m + log2(l)              (base-2 log-sum-exp, matches reference /ln2)

Design (plan §1):
  * grid = (cdiv(total_q, BLOCK_M), num_qo_heads=32, batch)   [path A, zero host sync]
  * one program = one BLOCK_M query tile, one q-head, one sequence
  * kv head = q_head // gqa_ratio (gqa_ratio = 4)
  * bottom-right causal: query local i attends kv local j iff j <= i + delta,
    delta = Nk - Nq
  * bf16 inputs, fp32 accumulate (Ampere HMMA)
  * degenerate/all-masked rows guarded → output 0, lse -inf (no NaN)

Triton only; no Torch/CPU/NumPy computational fallback.
"""

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _gqa_ragged_causal_fwd(
    Q, K, V, O, L,
    qo_indptr, kv_indptr,
    qk_scale,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    stride_lt, stride_lh,
    GQA_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    # --- sequence extents (ragged) ---
    q_start = tl.load(qo_indptr + pid_b)
    q_end = tl.load(qo_indptr + pid_b + 1)
    Nq = q_end - q_start
    m_start = pid_m * BLOCK_M
    if m_start >= Nq:
        # This M-block lies past the end of sequence pid_b → nothing to do.
        return

    kv_start = tl.load(kv_indptr + pid_b)
    kv_end = tl.load(kv_indptr + pid_b + 1)
    Nk = kv_end - kv_start
    delta = Nk - Nq
    kv_head = pid_h // GQA_RATIO

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_mask = offs_m < Nq

    q_ptrs = (
        Q
        + (q_start + offs_m)[:, None] * stride_qt
        + pid_h * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q_tile = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    # Causal upper bound on kv columns for this block: last query row can see up
    # to (local_i + delta). Clamp to [0, Nk).
    max_i = tl.minimum(m_start + BLOCK_M - 1, Nq - 1)
    end_n = tl.minimum(max_i + delta + 1, Nk)

    for start_n in range(0, end_n, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < Nk

        k_ptrs = (
            K
            + (kv_start + offs_n)[:, None] * stride_kt
            + kv_head * stride_kh
            + offs_d[None, :] * stride_kd
        )
        k_tile = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        qk = tl.dot(q_tile, tl.trans(k_tile))  # [BLOCK_M, BLOCK_N] fp32
        s2 = qk * qk_scale

        causal = (offs_n[None, :] <= (offs_m[:, None] + delta)) & n_mask[None, :]
        s2 = tl.where(causal, s2, -float("inf"))

        m_ij = tl.max(s2, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        # Guard fully-masked rows (m_new == -inf) so exp2 does not produce NaN.
        m_new_safe = tl.where(m_new == -float("inf"), 0.0, m_new)

        p = tl.math.exp2(s2 - m_new_safe[:, None])
        alpha = tl.where(m_i == -float("inf"), 0.0, tl.math.exp2(m_i - m_new_safe))

        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = (
            V
            + (kv_start + offs_n)[:, None] * stride_vt
            + kv_head * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v_tile = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile)

        m_i = m_new

    # --- finalize (guard rows with no valid kv → 0 / -inf) ---
    valid = l_i > 0.0
    l_safe = tl.where(valid, l_i, 1.0)
    acc = acc / l_safe[:, None]
    acc = tl.where(valid[:, None], acc, 0.0)
    lse = tl.where(valid, m_i + tl.math.log2(l_safe), -float("inf"))

    o_ptrs = (
        O
        + (q_start + offs_m)[:, None] * stride_ot
        + pid_h * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=q_mask[:, None])

    l_ptrs = L + (q_start + offs_m) * stride_lt + pid_h * stride_lh
    tl.store(l_ptrs, lse, mask=q_mask)


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    total_kv, num_kv_heads, _ = k.shape
    batch = qo_indptr.shape[0] - 1
    gqa_ratio = num_qo_heads // num_kv_heads

    output = torch.empty(
        (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device
    )
    lse = torch.empty(
        (total_q, num_qo_heads), dtype=torch.float32, device=q.device
    )

    if total_q == 0 or batch <= 0:
        return output, lse

    BLOCK_M = 32
    BLOCK_N = 64
    qk_scale = float(sm_scale) * LOG2E

    grid = (triton.cdiv(total_q, BLOCK_M), num_qo_heads, batch)
    _gqa_ragged_causal_fwd[grid](
        q, k, v, output, lse,
        qo_indptr, kv_indptr,
        qk_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        GQA_RATIO=gqa_ratio,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return output, lse
