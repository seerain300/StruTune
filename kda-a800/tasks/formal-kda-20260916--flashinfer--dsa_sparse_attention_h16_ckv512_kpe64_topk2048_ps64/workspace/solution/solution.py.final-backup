"""
c005 — DSA sparse paged MLA attention (DeepSeek-V3.2), Triton split-K flash-decode.

Target: NVIDIA A800 (sm_80, Ampere).
Entry point: run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)

Design (split-K flash-decode; see docs/draft.md §4.1-B, docs/plan.md Phase 0/2):
  * Grid (num_tokens, num_kv_splits). Each program owns ONE token and one contiguous
    chunk of the topk index axis, processing ALL 16 heads together (max K-row reuse).
  * Online (base-2) softmax over gathered rows using exp2; masked/-1 indices -> -inf.
  * Emits partial (acc[16,512], m[16], l[16]) per split to fp32 scratch.
  * A small combine kernel reduces across splits -> bf16 output + base-2 lse.

Change vs c002 (parent, current BEST at geomean 25.45x): num_warps 4 -> 8 in the
split kernel only. Everything else identical (splits=8, BLOCK_N=64, PV bf16, combine
unchanged).

Rationale (Phase 2, H4): c002/c003/c004 evidence shows sol_ms sits on a ~0.128 ms
floor that did NOT move with split count (8->32 flat) and got WORSE when fused
(c004). The kernel is bound by *scattered* 1 KiB row gathers across a 541,568-row
cache -> memory-LATENCY bound with poor effective bandwidth. With only
num_tokens*8 = 16-64 CTAs on 108 SMs, per-SM occupancy is NOT the limit, so raising
warps/CTA (4 -> 8) is pure memory-level-parallelism gain: 256 threads issue the
[64,512] gather instead of 128, doubling concurrent in-flight memory requests to
hide gather latency. smem is unchanged (warps/registers only), so no OOM risk;
num_stages stays 2 because the double-buffered [64,512] bf16 kc tile already nearly
fills smem (num_stages=3 would OOM, as c001 showed with an fp32 kc copy).

Compute is Triton-only. PyTorch is used solely for metadata/allocation/launch.
No Torch/CPU/NumPy computational fallback.
"""

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634  # log2(e)

# Guaranteed constants for this task definition.
H = 16          # num_qo_heads
D_CKV = 512     # head_dim_ckv (also the V dim)
D_KPE = 64      # head_dim_kpe
TOPK = 2048     # sparse_indices.shape[-1]


@triton.jit
def _dsa_split_kernel(
    q_nope_ptr, q_pe_ptr,
    ckv_ptr, kpe_ptr,
    sparse_ptr,
    accp_ptr, mp_ptr, lp_ptr,
    qk_scale,
    stride_qn_t, stride_qn_h, stride_qn_d,
    stride_qp_t, stride_qp_h, stride_qp_d,
    stride_ckv_row, stride_ckv_d,
    stride_kpe_row, stride_kpe_d,
    stride_si_t, stride_si_n,
    stride_accp_ts, stride_accp_h, stride_accp_d,
    stride_ml_ts, stride_ml_h,
    topk,
    H: tl.constexpr, D_CKV: tl.constexpr, D_KPE: tl.constexpr,
    SPLIT_SIZE: tl.constexpr, BLOCK_N: tl.constexpr, NUM_SPLITS: tl.constexpr,
):
    t = tl.program_id(0)
    split_id = tl.program_id(1)

    offs_h = tl.arange(0, H)
    offs_ckv = tl.arange(0, D_CKV)
    offs_kpe = tl.arange(0, D_KPE)
    offs_n = tl.arange(0, BLOCK_N)

    # Load this token's queries (all heads) once.
    qn = tl.load(
        q_nope_ptr + t * stride_qn_t
        + offs_h[:, None] * stride_qn_h + offs_ckv[None, :] * stride_qn_d
    )  # [H, D_CKV] bf16
    qp = tl.load(
        q_pe_ptr + t * stride_qp_t
        + offs_h[:, None] * stride_qp_h + offs_kpe[None, :] * stride_qp_d
    )  # [H, D_KPE] bf16

    m_i = tl.full([H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([H], dtype=tl.float32)
    acc = tl.zeros([H, D_CKV], dtype=tl.float32)

    split_start = split_id * SPLIT_SIZE
    for n0 in range(0, SPLIT_SIZE, BLOCK_N):
        idx_pos = split_start + n0 + offs_n          # position along the topk axis
        in_range = idx_pos < topk
        idx = tl.load(sparse_ptr + t * stride_si_t + idx_pos * stride_si_n,
                      mask=in_range, other=-1)        # int32
        valid = (idx != -1) & in_range
        safe_idx = tl.where(valid, idx, 0).to(tl.int64)

        kc = tl.load(
            ckv_ptr + safe_idx[:, None] * stride_ckv_row + offs_ckv[None, :] * stride_ckv_d,
            mask=valid[:, None], other=0.0,
        )  # [BLOCK_N, D_CKV] bf16
        kp = tl.load(
            kpe_ptr + safe_idx[:, None] * stride_kpe_row + offs_kpe[None, :] * stride_kpe_d,
            mask=valid[:, None], other=0.0,
        )  # [BLOCK_N, D_KPE] bf16

        # QK: [H, BLOCK_N] in fp32. bf16 x bf16 -> fp32 accumulate matches fp32-widened ref.
        s = tl.dot(qn, tl.trans(kc))
        s += tl.dot(qp, tl.trans(kp))
        s = s * qk_scale                              # base-2 logit domain
        s = tl.where(valid[None, :], s, float("-inf"))

        m_ij = tl.max(s, axis=1)                      # [H]
        m_new = tl.maximum(m_i, m_ij)
        # Guard fully-invalid tiles (m stays -inf) against -inf - -inf = NaN.
        m_new_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_new_safe)             # [H]
        p = tl.exp2(s - m_new_safe[:, None])          # [H, BLOCK_N]; masked -> 0

        l_i = l_i * alpha + tl.sum(p, axis=1)
        # PV in bf16 (fp32 accumulate). Keeping K in bf16 avoids the 128 KiB fp32
        # K operand that overflowed smem in c001. Loose tolerance accommodates bf16 P.
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kc)
        m_i = m_new

    ts = t * NUM_SPLITS + split_id
    tl.store(accp_ptr + ts * stride_accp_ts
             + offs_h[:, None] * stride_accp_h + offs_ckv[None, :] * stride_accp_d, acc)
    tl.store(mp_ptr + ts * stride_ml_ts + offs_h * stride_ml_h, m_i)
    tl.store(lp_ptr + ts * stride_ml_ts + offs_h * stride_ml_h, l_i)


@triton.jit
def _dsa_combine_kernel(
    accp_ptr, mp_ptr, lp_ptr,
    out_ptr, lse_ptr,
    stride_accp_ts, stride_accp_h, stride_accp_d,
    stride_ml_ts, stride_ml_h,
    stride_o_t, stride_o_h, stride_o_d,
    stride_lse_t, stride_lse_h,
    H: tl.constexpr, D_CKV: tl.constexpr, NUM_SPLITS: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid // H
    h = pid % H

    offs_s = tl.arange(0, NUM_SPLITS)
    offs_d = tl.arange(0, D_CKV)

    base_ts = t * NUM_SPLITS + offs_s               # [NUM_SPLITS]
    m_s = tl.load(mp_ptr + base_ts * stride_ml_ts + h * stride_ml_h)   # [NUM_SPLITS]
    l_s = tl.load(lp_ptr + base_ts * stride_ml_ts + h * stride_ml_h)   # [NUM_SPLITS]

    gm = tl.max(m_s, axis=0)                          # scalar
    gm_safe = tl.where(gm == float("-inf"), 0.0, gm)
    scale_s = tl.exp2(m_s - gm_safe)                 # [NUM_SPLITS]; -inf splits -> 0
    L = tl.sum(l_s * scale_s, axis=0)                # scalar

    acc_s = tl.load(
        accp_ptr + base_ts[:, None] * stride_accp_ts
        + h * stride_accp_h + offs_d[None, :] * stride_accp_d
    )  # [NUM_SPLITS, D_CKV]
    acc_tot = tl.sum(acc_s * scale_s[:, None], axis=0)  # [D_CKV]

    is_empty = gm == float("-inf")
    out = tl.where(is_empty, 0.0, acc_tot / L)
    lse = tl.where(is_empty, float("-inf"), gm + tl.log2(L))

    tl.store(out_ptr + t * stride_o_t + h * stride_o_h + offs_d * stride_o_d,
             out.to(tl.bfloat16))
    tl.store(lse_ptr + t * stride_lse_t + h * stride_lse_h, lse)


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    topk = sparse_indices.shape[-1]

    device = q_nope.device

    output = torch.zeros((num_tokens, num_qo_heads, head_dim_ckv),
                         dtype=torch.bfloat16, device=device)
    lse = torch.full((num_tokens, num_qo_heads), float("-inf"),
                     dtype=torch.float32, device=device)

    if num_tokens == 0:
        return output, lse

    # Flatten paged caches to token-level rows (contiguous view, no copy).
    ckv_flat = ckv_cache.reshape(num_pages * page_size, head_dim_ckv)
    kpe_flat = kpe_cache.reshape(num_pages * page_size, head_dim_kpe)

    # --- launch config (c005: c002 split-K with num_warps 4 -> 8) ---
    NUM_SPLITS = 8
    BLOCK_N = 64
    SPLIT_SIZE = triton.cdiv(topk, NUM_SPLITS)
    # Round SPLIT_SIZE up to a multiple of BLOCK_N so the inner loop is clean.
    SPLIT_SIZE = triton.cdiv(SPLIT_SIZE, BLOCK_N) * BLOCK_N

    qk_scale = float(sm_scale) * LOG2E

    acc_partial = torch.empty((num_tokens * NUM_SPLITS, num_qo_heads, head_dim_ckv),
                              dtype=torch.float32, device=device)
    m_partial = torch.empty((num_tokens * NUM_SPLITS, num_qo_heads),
                            dtype=torch.float32, device=device)
    l_partial = torch.empty((num_tokens * NUM_SPLITS, num_qo_heads),
                            dtype=torch.float32, device=device)

    grid_split = (num_tokens, NUM_SPLITS)
    _dsa_split_kernel[grid_split](
        q_nope, q_pe,
        ckv_flat, kpe_flat,
        sparse_indices,
        acc_partial, m_partial, l_partial,
        qk_scale,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_flat.stride(0), ckv_flat.stride(1),
        kpe_flat.stride(0), kpe_flat.stride(1),
        sparse_indices.stride(0), sparse_indices.stride(1),
        acc_partial.stride(0), acc_partial.stride(1), acc_partial.stride(2),
        m_partial.stride(0), m_partial.stride(1),
        topk,
        H=num_qo_heads, D_CKV=head_dim_ckv, D_KPE=head_dim_kpe,
        SPLIT_SIZE=SPLIT_SIZE, BLOCK_N=BLOCK_N, NUM_SPLITS=NUM_SPLITS,
        num_warps=8, num_stages=2,
    )

    grid_combine = (num_tokens * num_qo_heads,)
    _dsa_combine_kernel[grid_combine](
        acc_partial, m_partial, l_partial,
        output, lse,
        acc_partial.stride(0), acc_partial.stride(1), acc_partial.stride(2),
        m_partial.stride(0), m_partial.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        H=num_qo_heads, D_CKV=head_dim_ckv, NUM_SPLITS=NUM_SPLITS,
        num_warps=4, num_stages=1,
    )

    return output, lse
