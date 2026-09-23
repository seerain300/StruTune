"""Gated Delta Net decode (GVA, k-last state layout) — Triton implementation.

Task: gdn_decode_qk4_v8_d128_k_last (A800 / sm_80).

Per (batch b, v-head h) the reference recurrence, re-derived directly in the stored
[V, K] state layout (see docs/draft.md §2), collapses to a scalar-scaled state plus a
single rank-1 update:

    g            = exp(-exp(A_log[h]) * softplus(a[b,0,h] + dt_bias[h]))
    beta         = sigmoid(b[b,0,h])
    old_v[v]     = g * dot_K(S[v, :], k_h)          # S = state[b, h]  in [V, K]
    delta_v[v]   = beta * (v_h[v] - old_v[v])
    new_state[v, k] = g * S[v, k] + delta_v[v] * k_h[k]
    output[v]    = scale * dot_K(new_state[v, :], q_h)

with q_h/k_h = q/k of head (h // G), G = num_v_heads // num_q_heads.

Everything is done in f32 (no tl.dot / TF32 / tensor cores). The op is memory-bound;
the goal is a single streaming pass over state/new_state with coalesced 128-wide f32
accesses.

c002 (vs c001): keep the identical algebra but replace the fixed BLOCK_V=16 /
num_warps=4 launch with @triton.autotune over BLOCK_V in {8,16,32,64} and
num_warps in {1,2,4}, keyed on batch_size B. Hypothesis: small B is
launch/occupancy-bound and benefits from finer v-splitting (small BLOCK_V, more
programs to fill 108 SMs), while B=64 — the only workload showing GPU-execution
time above the fixed launch floor in c001 — benefits from fatter tiles (fewer
programs, more state reuse per program). All candidate BLOCK_V divide V=128, so no
masking is needed.
"""

import math

import torch
import triton
import triton.language as tl


def _autotune_configs():
    configs = []
    for block_v in (8, 16, 32, 64):
        for num_warps in (1, 2, 4):
            configs.append(
                triton.Config({"BLOCK_V": block_v}, num_warps=num_warps, num_stages=1)
            )
    return configs


@triton.autotune(configs=_autotune_configs(), key=["B"])
@triton.jit
def _gdn_decode_kernel(
    q_ptr, k_ptr, v_ptr, state_ptr, a_ptr, b_ptr, A_log_ptr, dt_bias_ptr,
    out_ptr, new_state_ptr,
    scale,
    # q strides (batch, head, k)
    q_sb, q_sh, q_sk,
    # k strides (batch, head, k)
    k_sb, k_sh, k_sk,
    # v strides (batch, head, v)
    v_sb, v_sh, v_sv,
    # state strides (batch, head, v, k)
    s_sb, s_sh, s_sv, s_sk,
    # a strides (batch, head)
    a_sb, a_sh,
    # b strides (batch, head)
    b_sb, b_sh,
    # output strides (batch, head, v)
    o_sb, o_sh, o_sv,
    # new_state strides (batch, head, v, k)
    ns_sb, ns_sh, ns_sv, ns_sk,
    B, G,
    K: tl.constexpr, V: tl.constexpr, BLOCK_V: tl.constexpr,
):
    b_idx = tl.program_id(0)
    h_idx = tl.program_id(1)
    pid_v = tl.program_id(2)
    qk_head = h_idx // G

    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)      # [BLOCK_V]
    k_offs = tl.arange(0, K)                              # [K]

    # --- gates (scalars, all f32) ---
    a_val = tl.load(a_ptr + b_idx * a_sb + h_idx * a_sh).to(tl.float32)
    b_val = tl.load(b_ptr + b_idx * b_sb + h_idx * b_sh).to(tl.float32)
    A_log_val = tl.load(A_log_ptr + h_idx).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + h_idx).to(tl.float32)

    x = a_val + dt_bias_val
    # softplus matching F.softplus(beta=1, threshold=20)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    g = tl.exp(-tl.exp(A_log_val) * sp)
    beta = 1.0 / (1.0 + tl.exp(-b_val))

    # --- head-shared vectors q_h, k_h  [K]  (bf16 -> f32) ---
    q_h = tl.load(q_ptr + b_idx * q_sb + qk_head * q_sh + k_offs * q_sk).to(tl.float32)
    k_h = tl.load(k_ptr + b_idx * k_sb + qk_head * k_sh + k_offs * k_sk).to(tl.float32)

    # --- value tile  [BLOCK_V]  (bf16 -> f32) ---
    v_tile = tl.load(v_ptr + b_idx * v_sb + h_idx * v_sh + v_offs * v_sv).to(tl.float32)

    # --- state tile  [BLOCK_V, K]  (contiguous over K, coalesced) ---
    s_ptrs = (state_ptr + b_idx * s_sb + h_idx * s_sh
              + v_offs[:, None] * s_sv + k_offs[None, :] * s_sk)
    S_tile = tl.load(s_ptrs).to(tl.float32)

    # --- delta-rule update ---
    old_v = g * tl.sum(S_tile * k_h[None, :], axis=1)          # [BLOCK_V]
    delta_v = beta * (v_tile - old_v)                          # [BLOCK_V]
    new_state_tile = g * S_tile + delta_v[:, None] * k_h[None, :]  # [BLOCK_V, K]

    ns_ptrs = (new_state_ptr + b_idx * ns_sb + h_idx * ns_sh
               + v_offs[:, None] * ns_sv + k_offs[None, :] * ns_sk)
    tl.store(ns_ptrs, new_state_tile)

    # --- read-out ---
    out = scale * tl.sum(new_state_tile * q_h[None, :], axis=1)   # [BLOCK_V]
    tl.store(out_ptr + b_idx * o_sb + h_idx * o_sh + v_offs * o_sv, out.to(tl.bfloat16))


def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    B, T, num_q_heads, K = q.shape
    _, _, num_k_heads, _ = k.shape
    _, _, num_v_heads, V = v.shape
    H = num_v_heads
    device = q.device

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)
    scale = float(scale)

    G = num_v_heads // num_q_heads

    if state is None:
        state = torch.zeros((B, H, V, K), dtype=torch.float32, device=device)
    elif state.dtype != torch.float32:
        state = state.to(torch.float32)

    output = torch.empty((B, T, H, V), dtype=torch.bfloat16, device=device)
    new_state = torch.empty((B, H, V, K), dtype=torch.float32, device=device)

    # All autotuned BLOCK_V divide V=128 -> exact v-block count, no masking.
    grid = lambda META: (B, H, V // META["BLOCK_V"])

    _gdn_decode_kernel[grid](
        q, k, v, state, a, b, A_log, dt_bias,
        output, new_state,
        scale,
        q.stride(0), q.stride(2), q.stride(3),
        k.stride(0), k.stride(2), k.stride(3),
        v.stride(0), v.stride(2), v.stride(3),
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        a.stride(0), a.stride(2),
        b.stride(0), b.stride(2),
        output.stride(0), output.stride(2), output.stride(3),
        new_state.stride(0), new_state.stride(1), new_state.stride(2), new_state.stride(3),
        B, G,
        K=K, V=V,
    )
    return output, new_state
