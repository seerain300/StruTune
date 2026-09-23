"""KDA candidate c001 — fused RMSNorm(Q,K) + RoPE + KV-cache update.

Task: L1/018_fused_rope_with_qk_norm_and_kv_cache_update  (H100 / sm_90)

Design (see docs/plan.md Phase 0):
  * Two Triton kernels, both memory-bandwidth oriented, structured grid A:
      grid = (B, H, ceil(S / BLOCK_S)); each program owns a [BLOCK_S, D] tile.
    - _q_kernel : per-head RMSNorm(q_norm_weight) -> RoPE -> query_rotated.
    - _kv_kernel: per-head RMSNorm(k_norm_weight) -> RoPE -> key_rotated and
                  key_cache slice; raw value copy -> value_cache slice.
  * All compute in fp32; cast to bf16 only at the store.
  * cos/sin recomputed in-register from inv_freq + position_ids (no [B,S,128]
    materialization); the cat([freqs,freqs]) structure reduces one row to
        o[0:64]   = x1*cos - x2*sin
        o[64:128] = x2*cos + x1*sin
  * value is copied verbatim (no norm/rope).
  * KV-cache addressing uses int64 offsets (262144-length axis, draft §2).

Triton is the sole compute path; PyTorch is used only for allocation and launch.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _q_kernel(
    q_ptr, o_ptr, pos_ptr, w_ptr, invf_ptr, eps,
    sqb, sqh, sqs, sqd,
    sob, soh, sos, sod,
    spb, sps,
    S,
    D: tl.constexpr, HALF: tl.constexpr, BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    sblk = tl.program_id(2)

    s_off = sblk * BLOCK_S + tl.arange(0, BLOCK_S)      # [BLOCK_S]
    s_mask = s_off < S
    dh = tl.arange(0, HALF)                             # [HALF]
    m = s_mask[:, None]

    q_base = q_ptr + b * sqb + h * sqh
    off1 = s_off[:, None] * sqs + dh[None, :] * sqd
    off2 = s_off[:, None] * sqs + (HALF + dh)[None, :] * sqd
    x1 = tl.load(q_base + off1, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(q_base + off2, mask=m, other=0.0).to(tl.float32)

    ssq = tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)   # [BLOCK_S]
    inv_rms = tl.rsqrt(ssq / D + eps)                         # [BLOCK_S]

    w1 = tl.load(w_ptr + dh).to(tl.float32)
    w2 = tl.load(w_ptr + HALF + dh).to(tl.float32)
    x1n = x1 * inv_rms[:, None] * w1[None, :]
    x2n = x2 * inv_rms[:, None] * w2[None, :]

    pos = tl.load(pos_ptr + b * spb + s_off * sps, mask=s_mask, other=0).to(tl.float32)
    invf = tl.load(invf_ptr + dh)                             # [HALF] fp32
    fr = pos[:, None] * invf[None, :]
    c = tl.cos(fr)
    s = tl.sin(fr)
    o1 = x1n * c - x2n * s
    o2 = x2n * c + x1n * s

    o_base = o_ptr + b * sob + h * soh
    oo1 = s_off[:, None] * sos + dh[None, :] * sod
    oo2 = s_off[:, None] * sos + (HALF + dh)[None, :] * sod
    tl.store(o_base + oo1, o1.to(tl.bfloat16), mask=m)
    tl.store(o_base + oo2, o2.to(tl.bfloat16), mask=m)


@triton.jit
def _kv_kernel(
    k_ptr, v_ptr, ko_ptr, kc_ptr, vc_ptr, pos_ptr, cpos_ptr, w_ptr, invf_ptr, eps,
    skb, skh, sks, skd,
    svb, svh, svs, svd,
    skob, skoh, skos, skod,
    skcb, skch, skcs, skcd,
    svcb, svch, svcs, svcd,
    spb, sps,
    S,
    D: tl.constexpr, HALF: tl.constexpr, BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    sblk = tl.program_id(2)

    s_off = sblk * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < S
    dh = tl.arange(0, HALF)
    m = s_mask[:, None]

    # ---- Key: RMSNorm + RoPE ----
    k_base = k_ptr + b * skb + h * skh
    off1 = s_off[:, None] * sks + dh[None, :] * skd
    off2 = s_off[:, None] * sks + (HALF + dh)[None, :] * skd
    x1 = tl.load(k_base + off1, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(k_base + off2, mask=m, other=0.0).to(tl.float32)

    ssq = tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)
    inv_rms = tl.rsqrt(ssq / D + eps)

    w1 = tl.load(w_ptr + dh).to(tl.float32)
    w2 = tl.load(w_ptr + HALF + dh).to(tl.float32)
    x1n = x1 * inv_rms[:, None] * w1[None, :]
    x2n = x2 * inv_rms[:, None] * w2[None, :]

    pos = tl.load(pos_ptr + b * spb + s_off * sps, mask=s_mask, other=0).to(tl.float32)
    invf = tl.load(invf_ptr + dh)
    fr = pos[:, None] * invf[None, :]
    c = tl.cos(fr)
    s = tl.sin(fr)
    o1 = (x1n * c - x2n * s).to(tl.bfloat16)
    o2 = (x2n * c + x1n * s).to(tl.bfloat16)

    # store key_rotated
    ko_base = ko_ptr + b * skob + h * skoh
    tl.store(ko_base + s_off[:, None] * skos + dh[None, :] * skod, o1, mask=m)
    tl.store(ko_base + s_off[:, None] * skos + (HALF + dh)[None, :] * skod, o2, mask=m)

    # store into key_cache at cache_position rows (int64 offsets)
    crow = tl.load(cpos_ptr + s_off, mask=s_mask, other=0).to(tl.int64)   # [BLOCK_S]
    kc_base = b.to(tl.int64) * skcb + h.to(tl.int64) * skch
    kc1 = kc_base + crow[:, None] * skcs + dh[None, :] * skcd
    kc2 = kc_base + crow[:, None] * skcs + (HALF + dh)[None, :] * skcd
    tl.store(kc_ptr + kc1, o1, mask=m)
    tl.store(kc_ptr + kc2, o2, mask=m)

    # ---- Value: raw copy into value_cache ----
    dfull = tl.arange(0, D)
    v_base = v_ptr + b * svb + h * svh
    v = tl.load(v_base + s_off[:, None] * svs + dfull[None, :] * svd, mask=m)   # bf16
    vc_base = b.to(tl.int64) * svcb + h.to(tl.int64) * svch
    vc = vc_base + crow[:, None] * svcs + dfull[None, :] * svcd
    tl.store(vc_ptr + vc, v, mask=m)


@torch.no_grad()
def run(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    position_ids: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_position: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    inv_freq: torch.Tensor,
    rms_norm_eps: float,
):
    B, Hq, S, D = query.shape
    Hkv = key.shape[1]
    HALF = D // 2
    eps = float(rms_norm_eps)

    query_rotated = torch.empty_like(query)
    key_rotated = torch.empty_like(key)

    BLOCK_S = 32

    grid_q = (B, Hq, triton.cdiv(S, BLOCK_S))
    _q_kernel[grid_q](
        query, query_rotated, position_ids, q_norm_weight, inv_freq, eps,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        query_rotated.stride(0), query_rotated.stride(1), query_rotated.stride(2), query_rotated.stride(3),
        position_ids.stride(0), position_ids.stride(1),
        S,
        D=D, HALF=HALF, BLOCK_S=BLOCK_S,
        num_warps=4, num_stages=2,
    )

    grid_kv = (B, Hkv, triton.cdiv(S, BLOCK_S))
    _kv_kernel[grid_kv](
        key, value, key_rotated, key_cache, value_cache,
        position_ids, cache_position, k_norm_weight, inv_freq, eps,
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        key_rotated.stride(0), key_rotated.stride(1), key_rotated.stride(2), key_rotated.stride(3),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        position_ids.stride(0), position_ids.stride(1),
        S,
        D=D, HALF=HALF, BLOCK_S=BLOCK_S,
        num_warps=4, num_stages=2,
    )

    return query_rotated, key_rotated, key_cache, value_cache
