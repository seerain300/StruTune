import torch
import triton
import triton.language as tl


# =============================================================================
# L1/018 Fused RoPE + QK-Norm + KV-Cache update  (candidate c002)
#
# Option B (see docs/plan.md §3): two Triton kernels.
#   * Q kernel   : per-head RMS-norm + RoPE -> query_rotated
#   * K+V kernel : per-head RMS-norm + RoPE on K -> key_rotated + key_cache
#                  scatter; copy V verbatim -> value_cache scatter.
#
# c002 vs c001: single lever changed -> @triton.autotune over
#   BLOCK_S in {1,2,4,8,16,32} x num_warps in {1,2,4}, keyed on seq_len S.
# Everything else (math, layout, invariants) is identical to c001.
#
# Each program handles one block of BLOCK_S consecutive seq tokens of one
# (b, head). Row is loaded as two halves [BLOCK_S, HALF] so rotate_half is
# free (halves-paired RoPE, dim j pairs with j+HALF, shared angle theta_j).
#
# Correctness invariants (unchanged from c001):
#   - RMS variance accumulated in fp32, divided by HEAD_DIM, +eps inside rsqrt.
#   - weight applied as weight.float() * xn, rounded to bf16 on store.
#   - RoPE halves-paired (j, j+64); position_ids drives the angle.
#   - V copied verbatim into value_cache (never normed/roped).
#   - cache_position drives the cache row; cache offsets computed in int64.
#   - in-place scatter of only S rows; caches returned as the passed objects.
# =============================================================================


def _autotune_configs():
    configs = []
    for bs in (1, 2, 4, 8, 16, 32):
        for nw in (1, 2, 4):
            configs.append(
                triton.Config({"BLOCK_S": bs}, num_warps=nw, num_stages=1)
            )
    return configs


_CONFIGS = _autotune_configs()


@triton.autotune(configs=_CONFIGS, key=["S"])
@triton.jit
def _q_norm_rope_kernel(
    q_ptr, out_ptr,
    pos_ptr,
    w_ptr, inv_freq_ptr,
    eps,
    S,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_pb, stride_ps,
    NH: tl.constexpr,
    BLOCK_S: tl.constexpr,
    HALF: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    b = pid_bh // NH
    h = pid_bh % NH

    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < S
    d = tl.arange(0, HALF)  # [HALF]

    q_base = q_ptr + b * stride_qb + h * stride_qh
    lo_ptr = q_base + s_off[:, None] * stride_qs + d[None, :] * stride_qd
    hi_ptr = q_base + s_off[:, None] * stride_qs + (d + HALF)[None, :] * stride_qd

    a = tl.load(lo_ptr, mask=s_mask[:, None], other=0.0).to(tl.float32)
    c = tl.load(hi_ptr, mask=s_mask[:, None], other=0.0).to(tl.float32)

    ssq = tl.sum(a * a, axis=1) + tl.sum(c * c, axis=1)          # [BLOCK_S]
    inv = tl.rsqrt(ssq / HEAD_DIM + eps)                         # [BLOCK_S]

    w_lo = tl.load(w_ptr + d).to(tl.float32)                    # [HALF]
    w_hi = tl.load(w_ptr + d + HALF).to(tl.float32)

    an = a * inv[:, None] * w_lo[None, :]
    cn = c * inv[:, None] * w_hi[None, :]

    pos = tl.load(pos_ptr + b * stride_pb + s_off * stride_ps,
                  mask=s_mask, other=0).to(tl.float32)          # [BLOCK_S]
    invf = tl.load(inv_freq_ptr + d)                            # [HALF] fp32
    theta = pos[:, None] * invf[None, :]                        # [BLOCK_S, HALF]
    cos = tl.cos(theta)
    sin = tl.sin(theta)

    out_lo = an * cos - cn * sin
    out_hi = cn * cos + an * sin

    o_base = out_ptr + b * stride_ob + h * stride_oh
    o_lo_ptr = o_base + s_off[:, None] * stride_os + d[None, :] * stride_od
    o_hi_ptr = o_base + s_off[:, None] * stride_os + (d + HALF)[None, :] * stride_od
    tl.store(o_lo_ptr, out_lo.to(tl.bfloat16), mask=s_mask[:, None])
    tl.store(o_hi_ptr, out_hi.to(tl.bfloat16), mask=s_mask[:, None])


@triton.autotune(configs=_CONFIGS, key=["S"])
@triton.jit
def _kv_norm_rope_kernel(
    k_ptr, v_ptr, out_ptr,
    kcache_ptr, vcache_ptr,
    pos_ptr, cache_pos_ptr,
    w_ptr, inv_freq_ptr,
    eps,
    S,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_kcb, stride_kch, stride_kcp, stride_kcd,
    stride_vcb, stride_vch, stride_vcp, stride_vcd,
    stride_pb, stride_ps,
    stride_cp,
    NH: tl.constexpr,
    BLOCK_S: tl.constexpr,
    HALF: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    b = pid_bh // NH
    h = pid_bh % NH

    s_off = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_off < S
    d = tl.arange(0, HALF)  # [HALF]

    # --- load K row as two halves ---
    k_base = k_ptr + b * stride_kb + h * stride_kh
    k_lo_ptr = k_base + s_off[:, None] * stride_ks + d[None, :] * stride_kd
    k_hi_ptr = k_base + s_off[:, None] * stride_ks + (d + HALF)[None, :] * stride_kd
    a = tl.load(k_lo_ptr, mask=s_mask[:, None], other=0.0).to(tl.float32)
    c = tl.load(k_hi_ptr, mask=s_mask[:, None], other=0.0).to(tl.float32)

    ssq = tl.sum(a * a, axis=1) + tl.sum(c * c, axis=1)
    inv = tl.rsqrt(ssq / HEAD_DIM + eps)

    w_lo = tl.load(w_ptr + d).to(tl.float32)
    w_hi = tl.load(w_ptr + d + HALF).to(tl.float32)

    an = a * inv[:, None] * w_lo[None, :]
    cn = c * inv[:, None] * w_hi[None, :]

    pos = tl.load(pos_ptr + b * stride_pb + s_off * stride_ps,
                  mask=s_mask, other=0).to(tl.float32)
    invf = tl.load(inv_freq_ptr + d)
    theta = pos[:, None] * invf[None, :]
    cos = tl.cos(theta)
    sin = tl.sin(theta)

    out_lo = an * cos - cn * sin
    out_hi = cn * cos + an * sin
    out_lo_bf = out_lo.to(tl.bfloat16)
    out_hi_bf = out_hi.to(tl.bfloat16)

    # --- write key_rotated ---
    o_base = out_ptr + b * stride_ob + h * stride_oh
    o_lo_ptr = o_base + s_off[:, None] * stride_os + d[None, :] * stride_od
    o_hi_ptr = o_base + s_off[:, None] * stride_os + (d + HALF)[None, :] * stride_od
    tl.store(o_lo_ptr, out_lo_bf, mask=s_mask[:, None])
    tl.store(o_hi_ptr, out_hi_bf, mask=s_mask[:, None])

    # --- cache row index (int64) ---
    crow = tl.load(cache_pos_ptr + s_off * stride_cp, mask=s_mask, other=0).to(tl.int64)

    # --- scatter rotated K into key_cache ---
    kc_base = kcache_ptr + b.to(tl.int64) * stride_kcb + h.to(tl.int64) * stride_kch
    kc_lo_ptr = kc_base + crow[:, None] * stride_kcp + d[None, :] * stride_kcd
    kc_hi_ptr = kc_base + crow[:, None] * stride_kcp + (d + HALF)[None, :] * stride_kcd
    tl.store(kc_lo_ptr, out_lo_bf, mask=s_mask[:, None])
    tl.store(kc_hi_ptr, out_hi_bf, mask=s_mask[:, None])

    # --- copy V verbatim into value_cache ---
    v_base = v_ptr + b * stride_vb + h * stride_vh
    v_lo_ptr = v_base + s_off[:, None] * stride_vs + d[None, :] * stride_vd
    v_hi_ptr = v_base + s_off[:, None] * stride_vs + (d + HALF)[None, :] * stride_vd
    v_lo = tl.load(v_lo_ptr, mask=s_mask[:, None], other=0.0)
    v_hi = tl.load(v_hi_ptr, mask=s_mask[:, None], other=0.0)

    vc_base = vcache_ptr + b.to(tl.int64) * stride_vcb + h.to(tl.int64) * stride_vch
    vc_lo_ptr = vc_base + crow[:, None] * stride_vcp + d[None, :] * stride_vcd
    vc_hi_ptr = vc_base + crow[:, None] * stride_vcp + (d + HALF)[None, :] * stride_vcd
    tl.store(vc_lo_ptr, v_lo, mask=s_mask[:, None])
    tl.store(vc_hi_ptr, v_hi, mask=s_mask[:, None])


HALF = 64
HEAD_DIM = 128


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
    batch_size, num_q_heads, seq_len, head_dim = query.shape
    num_kv_heads = key.shape[1]

    query_rotated = torch.empty_like(query)
    key_rotated = torch.empty_like(key)

    eps = float(rms_norm_eps)

    grid_q = lambda META: (batch_size * num_q_heads, triton.cdiv(seq_len, META["BLOCK_S"]))
    _q_norm_rope_kernel[grid_q](
        query, query_rotated,
        position_ids,
        q_norm_weight, inv_freq,
        eps,
        seq_len,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        query_rotated.stride(0), query_rotated.stride(1), query_rotated.stride(2), query_rotated.stride(3),
        position_ids.stride(0), position_ids.stride(1),
        NH=num_q_heads,
        HALF=HALF,
        HEAD_DIM=HEAD_DIM,
    )

    grid_kv = lambda META: (batch_size * num_kv_heads, triton.cdiv(seq_len, META["BLOCK_S"]))
    _kv_norm_rope_kernel[grid_kv](
        key, value, key_rotated,
        key_cache, value_cache,
        position_ids, cache_position,
        k_norm_weight, inv_freq,
        eps,
        seq_len,
        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        key_rotated.stride(0), key_rotated.stride(1), key_rotated.stride(2), key_rotated.stride(3),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        position_ids.stride(0), position_ids.stride(1),
        cache_position.stride(0),
        NH=num_kv_heads,
        HALF=HALF,
        HEAD_DIM=HEAD_DIM,
    )

    return query_rotated, key_rotated, key_cache, value_cache
