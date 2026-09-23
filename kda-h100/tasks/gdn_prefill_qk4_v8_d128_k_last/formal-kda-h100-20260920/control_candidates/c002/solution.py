"""
Gated Delta Net prefill (k-last layout), Triton solution for H100 / sm_90.

Task: gdn_prefill_qk4_v8_d128_k_last
Candidate: c002 -- adaptive V-column splitting over the c001 token-recurrent kernel.

Algorithm (matches definition.json reference exactly, per (seq, v-head)):
    internal state S is [K, V] (fp32).
    For each token t in a sequence:
        S      = g_t * S
        old_v  = k_t @ S                 # reduction over K
        u      = beta_t * (v_t - old_v)
        S      = S + outer(k_t, u)       # rank-1 update
        o_t    = scale * (q_t @ S)       # uses POST-update state

c002 change vs c001 (exact, no numerical reordering):
    The V (=128) columns of S are mutually independent given the per-token
    reduction over K (old_v[v] and o[v] each depend only on column v; the
    rank-1 update touches column v only through u[v]).  So we split V into
    NSPLIT blocks of BV = 128 / NSPLIT columns and give each block its own
    program.  Grid becomes (num_seqs, num_v_heads, NSPLIT).  This
      (a) multiplies CTA count by NSPLIT to fill the 132 SMs on the
          occupancy-starved low-num_seqs / long-seq workloads, and
      (b) shrinks each program's fp32 state footprint from [128,128] to
          [128,BV], reducing register pressure -> higher per-SM occupancy.
    NSPLIT is chosen on the host: workloads whose base program count
    (num_seqs*num_v_heads) already >= ~SM count are left unsplit (NSPLIT=1)
    so the already-strong high-num_seqs cases are never penalised by the
    small redundant q/k/gate reloads that splitting introduces.

Gates (fp32, computed in-kernel to keep all math in Triton):
    x    = a + dt_bias
    g    = exp(-exp(A_log) * softplus(x))     # softplus stable form
    beta = sigmoid(b)

Layout notes:
    q,k: [T, 4, 128]  (K-space)   v: [T, 8, 128]  (V-space)
    v-head hv uses q/k head hv // 2  (GVA, 8/4 = 2).
    state / new_state: [N, 8, V, K] (k-last). Loading with offset
    (k*sK + v*sV) yields S[k,v] = state[v,k] == reference transpose;
    storing S the same way writes new_state[v,k] = S[k,v]. No extra kernels.

Primary implementation is Triton. PyTorch is used only for tensor metadata
and launch plumbing (dtype/shape, contiguity, output allocation). There is no
Torch/CPU/NumPy computational fallback.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_prefill_vsplit_kernel(
    Q_ptr, K_ptr, V_ptr,
    A_ptr, B_ptr, ALOG_ptr, DTB_ptr,
    STATE_ptr, OUT_ptr, NEWSTATE_ptr,
    CU_ptr,
    scale,
    sQt, sQh, sKt, sKh, sVt, sVh,
    sAt, sBt,
    sSn, sSh, sSv, sSk,
    sOt, sOh,
    sNn, sNh, sNv, sNk,
    D: tl.constexpr,             # head_size = 128 (both K and V)
    BV: tl.constexpr,            # V columns handled by this program
    HEADS_PER_QK: tl.constexpr,  # v-heads per q/k head = 2
):
    pid_n = tl.program_id(0)     # sequence index
    pid_h = tl.program_id(1)     # v-head index 0..7
    pid_s = tl.program_id(2)     # V-column block index
    qk_h = pid_h // HEADS_PER_QK

    start = tl.load(CU_ptr + pid_n).to(tl.int32)
    end = tl.load(CU_ptr + pid_n + 1).to(tl.int32)

    k_idx = tl.arange(0, D)                    # K axis (rows of S)
    v_idx = pid_s * BV + tl.arange(0, BV)      # this program's V columns

    # Load initial state slab S[k, v] = state[n, h, v, k]  (k-last transpose).
    state_base = STATE_ptr + pid_n * sSn + pid_h * sSh
    S = tl.load(state_base + k_idx[:, None] * sSk + v_idx[None, :] * sSv).to(tl.float32)

    a_log = tl.load(ALOG_ptr + pid_h).to(tl.float32)
    dtb = tl.load(DTB_ptr + pid_h).to(tl.float32)
    exp_alog = tl.exp(a_log)

    for t in range(start, end):
        q_vec = tl.load(Q_ptr + t * sQt + qk_h * sQh + k_idx).to(tl.float32)
        k_vec = tl.load(K_ptr + t * sKt + qk_h * sKh + k_idx).to(tl.float32)
        v_vec = tl.load(V_ptr + t * sVt + pid_h * sVh + v_idx).to(tl.float32)
        a_t = tl.load(A_ptr + t * sAt + pid_h).to(tl.float32)
        b_t = tl.load(B_ptr + t * sBt + pid_h).to(tl.float32)

        # gates in fp32 (scalar per head; recomputed per V-block, cheap)
        x = a_t + dtb
        sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))  # softplus
        g = tl.exp(-exp_alog * sp)
        beta = 1.0 / (1.0 + tl.exp(-b_t))

        S = S * g
        old_v = tl.sum(k_vec[:, None] * S, axis=0)      # [BV]
        u = beta * (v_vec - old_v)                       # [BV]
        S = S + k_vec[:, None] * u[None, :]              # rank-1 update
        o = scale * tl.sum(q_vec[:, None] * S, axis=0)   # [BV]

        tl.store(OUT_ptr + t * sOt + pid_h * sOh + v_idx, o.to(OUT_ptr.dtype.element_ty))

    # Store new_state[n, h, v, k] = S[k, v]  (transpose back to k-last).
    # Reference leaves empty sequences (seq_len <= 0) as zeros, so mask S.
    S = S * (end > start).to(tl.float32)
    ns_base = NEWSTATE_ptr + pid_n * sNn + pid_h * sNh
    tl.store(ns_base + k_idx[:, None] * sNk + v_idx[None, :] * sNv, S)


def _choose_nsplit(num_seqs, num_v_heads, head_size, sm_target=128, max_nsplit=8):
    """Pick a power-of-2 V-split factor.

    Leave workloads whose base program count already fills the machine
    (base >= sm_target) unsplit so their (already strong) performance is not
    diluted by redundant per-block q/k/gate reloads. Otherwise grow the split
    toward ~2x sm_target CTAs, capped so BV stays >= 16 for sane tile shapes.
    """
    base = num_seqs * num_v_heads
    if base >= sm_target:
        return 1
    nsplit = 1
    goal = 2 * sm_target
    while nsplit < max_nsplit and base * nsplit < goal:
        nsplit *= 2
    # keep BV >= 16 (head_size // nsplit)
    while nsplit > 1 and head_size % nsplit != 0:
        nsplit //= 2
    return nsplit


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    total_seq_len, num_q_heads, head_size = q.shape
    num_v_heads = v.shape[1]
    num_seqs = cu_seqlens.shape[0] - 1
    device = q.device

    assert num_q_heads == 4
    assert num_v_heads == 8
    assert head_size == 128

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_size)
    scale = float(scale)

    heads_per_qk = num_v_heads // num_q_heads  # 2

    # Plumbing only: ensure contiguity / device / dtypes for stride math.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    A_log = A_log.contiguous().float()
    dt_bias = dt_bias.contiguous().float()
    cu_seqlens = cu_seqlens.contiguous()

    if state is None:
        state = torch.zeros(
            (num_seqs, num_v_heads, head_size, head_size),
            dtype=torch.float32, device=device,
        )
    else:
        state = state.contiguous().float()

    output = torch.zeros(
        (total_seq_len, num_v_heads, head_size), dtype=torch.bfloat16, device=device
    )
    new_state = torch.zeros(
        (num_seqs, num_v_heads, head_size, head_size), dtype=torch.float32, device=device
    )

    if total_seq_len == 0 or num_seqs == 0:
        return output, new_state

    nsplit = _choose_nsplit(num_seqs, num_v_heads, head_size)
    bv = head_size // nsplit

    grid = (num_seqs, num_v_heads, nsplit)
    _gdn_prefill_vsplit_kernel[grid](
        q, k, v,
        a, b, A_log, dt_bias,
        state, output, new_state,
        cu_seqlens,
        scale,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        a.stride(0), b.stride(0),
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        output.stride(0), output.stride(1),
        new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3),
        D=head_size,
        BV=bv,
        HEADS_PER_QK=heads_per_qk,
        num_warps=4,
    )
    return output, new_state
