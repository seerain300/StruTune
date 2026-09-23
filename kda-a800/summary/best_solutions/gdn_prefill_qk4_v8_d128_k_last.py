# KDA A800 best solution: gdn_prefill_qk4_v8_d128_k_last
# candidate: c002  |  feedback: 174.75x  |  final (authoritative): 161.69x
# campaign formal-kda-20260916 (A800, g0056)  |  evaluations: 2
# source: tasks/formal-kda-20260916--flashinfer--gdn_prefill_qk4_v8_d128_k_last/control/candidates/c002/solution.py (sha256-frozen snapshot)

"""
KDA candidate c002 — gdn_prefill_qk4_v8_d128_k_last

Occupancy-tuning step over c001 (plan Phase D — V/K blocking):
identical order-preserving, pure-float32 per-token recurrence, but the V axis
is split into finer blocks (BV=32 -> 4 V-blocks per (seq, v_head) instead of 2).
The V columns are fully independent in this recurrence (old_v, delta_v, the S
rank-1 update, and the output reduction are all computed per-v), so halving BV
is semantics-preserving and simply doubles the program count while halving the
BV-scaled per-token compute inside each program. This directly targets the
occupancy floor: 4 of 5 feedback workloads launch far fewer programs than the
A800 has SMs (e.g. d3dc3577 N=1 -> 16 programs at BV=64), so more, smaller
programs should improve utilization and reduce per-iteration latency without
regressing correctness. No tensor cores are used; every reduction is an explicit
float32 elementwise-multiply + tl.sum, matching the reference's per-token
float32 summation order/precision.

Semantics reproduced exactly (k-last state layout [N, H, V, K]):
    x    = a.f32() + dt_bias.f32()
    g    = exp(-exp(A_log) * softplus(x))          # scalar per (token, v_head)
    beta = sigmoid(b.f32())
    old  = g * S                                    # S internal is [K, V]
    old_v = k^T @ old                               # [V]
    new_v = beta*v + (1-beta)*old_v
    S    = old + k ⊗ (new_v - old_v)                # rank-1 update
    o    = scale * q^T @ S                          # inclusive (post-update) state

GVA: v_head h uses q/k head h // (Hv // Hq).
Empty sequences (seq_len <= 0) leave new_state = 0 (zero-initialized allocation),
matching the reference which `continue`s before writing.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_prefill_kernel(
    q_ptr, k_ptr, v_ptr, s_ptr, ns_ptr, o_ptr,
    a_ptr, b_ptr, alog_ptr, dtb_ptr, cu_ptr,
    scale,
    stride_qt, stride_qh,
    stride_kt, stride_kh,
    stride_vt, stride_vh,
    stride_ot, stride_oh,
    stride_sn, stride_sh, stride_sv, stride_sk,
    stride_at, stride_bt,
    GVA: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    seq_id = tl.program_id(0)
    h = tl.program_id(1)          # v-head, 0..Hv-1
    vb = tl.program_id(2)         # V-block index

    seq_start = tl.load(cu_ptr + seq_id).to(tl.int32)
    seq_end = tl.load(cu_ptr + seq_id + 1).to(tl.int32)
    seq_len = seq_end - seq_start
    if seq_len <= 0:
        # new_state already zero-initialized in run(); nothing to write.
        return

    qkh = h // GVA               # shared q/k head for this v-head (GVA)

    offs_k = tl.arange(0, BK)                     # K axis, full head (128)
    offs_v = vb * BV + tl.arange(0, BV)           # V axis block

    # Load entry state block: internal S[k, v] = state[seq, h, v, k]  (k-last -> [K,V]).
    s_base = s_ptr + seq_id * stride_sn + h * stride_sh
    s_ptrs = s_base + offs_k[:, None] * stride_sk + offs_v[None, :] * stride_sv
    S = tl.load(s_ptrs).to(tl.float32)           # [BK, BV]

    alog = tl.load(alog_ptr + h).to(tl.float32)
    dtb = tl.load(dtb_ptr + h).to(tl.float32)
    ea = tl.exp(alog)                            # exp(A_log)

    for i in range(0, seq_len):
        t = seq_start + i
        q_vec = tl.load(q_ptr + t * stride_qt + qkh * stride_qh + offs_k).to(tl.float32)  # [BK]
        k_vec = tl.load(k_ptr + t * stride_kt + qkh * stride_kh + offs_k).to(tl.float32)  # [BK]
        v_vec = tl.load(v_ptr + t * stride_vt + h * stride_vh + offs_v).to(tl.float32)    # [BV]

        a_s = tl.load(a_ptr + t * stride_at + h).to(tl.float32)
        b_s = tl.load(b_ptr + t * stride_bt + h).to(tl.float32)

        x = a_s + dtb
        # numerically safe softplus
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        g = tl.exp(-ea * sp)
        beta = 1.0 / (1.0 + tl.exp(-b_s))

        old = g * S                                          # [BK, BV]
        old_v = tl.sum(k_vec[:, None] * old, axis=0)         # [BV]
        delta_v = beta * (v_vec - old_v)                     # [BV]
        S = old + k_vec[:, None] * delta_v[None, :]          # [BK, BV]
        o = tl.sum(q_vec[:, None] * S, axis=0) * scale       # [BV]

        tl.store(o_ptr + t * stride_ot + h * stride_oh + offs_v, o.to(tl.bfloat16))

    # Store new state block back in k-last layout: new_state[seq, h, v, k] = S[k, v].
    ns_base = ns_ptr + seq_id * stride_sn + h * stride_sh
    ns_ptrs = ns_base + offs_k[:, None] * stride_sk + offs_v[None, :] * stride_sv
    tl.store(ns_ptrs, S)


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """
    Gated Delta Net prefill (k-last layout). Triton-only compute.

    Shapes:
      q [T, Hq=4, K=128] bf16, k [T, Hk=4, 128] bf16, v [T, Hv=8, 128] bf16
      state [N, Hv=8, V=128, K=128] f32 (k-last), optional
      A_log [Hv] f32, a [T, Hv] bf16, dt_bias [Hv] f32, b [T, Hv] bf16
      cu_seqlens [N+1] int64, scale scalar
    Returns:
      output [T, Hv=8, 128] bf16, new_state [N, Hv=8, 128, 128] f32 (k-last)
    """
    T, Hq, K = q.shape
    Hv = v.shape[1]
    V = v.shape[2]
    N = cu_seqlens.numel() - 1
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)
    scale = float(scale)

    # Ensure contiguous layouts so the assumed unit strides on the last axis hold.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    A_log = A_log.contiguous()
    dt_bias = dt_bias.contiguous()
    cu_seqlens = cu_seqlens.contiguous()

    output = torch.zeros((T, Hv, V), dtype=torch.bfloat16, device=device)
    new_state = torch.zeros((N, Hv, V, K), dtype=torch.float32, device=device)

    if state is None:
        state = torch.zeros((N, Hv, V, K), dtype=torch.float32, device=device)
    else:
        state = state.contiguous()

    if T == 0 or N == 0:
        return output, new_state

    GVA = Hv // Hq                 # 2
    BK = K                         # 128 (full head)
    BV = 32                        # finer V-blocking than c001 (was 64): 4 V-blocks/head
    num_vblocks = V // BV
    grid = (N, Hv, num_vblocks)

    _gdn_prefill_kernel[grid](
        q, k, v, state, new_state, output,
        a, b, A_log, dt_bias, cu_seqlens,
        scale,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        output.stride(0), output.stride(1),
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        a.stride(0), b.stride(0),
        GVA=GVA,
        BK=BK,
        BV=BV,
        num_warps=4,
    )

    return output, new_state
