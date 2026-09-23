"""
c004 — DSA sparse paged MLA attention (DeepSeek-V3.2), Triton FUSED single-pass.

Target: NVIDIA A800 (sm_80, Ampere).
Entry point: run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)

Design (design A / Phase 4 fused single-pass, see docs/draft.md §4.1-A, docs/plan.md):
  * Grid (num_tokens,). ONE program per token processes ALL 16 heads and loops the
    ENTIRE topk=2048 index axis with online (base-2) softmax (exp2), then normalizes
    and writes bf16 output + base-2 lse DIRECTLY. No split-K scratch, no combine kernel.

Why (evidence from c002/c003):
  c002 (splits=8) geomean 25.45x, c003 (splits=32) geomean 25.20x. sol_ms was FLAT at
  ~0.128 ms across BOTH num_tokens (2->8) AND split counts (8->32). That falsifies the
  occupancy hypothesis (H2/H3): the split kernel's compute is fully hidden; the ~0.128 ms
  is FIXED per-call overhead dominated by two kernel launches + three fp32 scratch
  allocations (acc/m/l partials). This candidate removes that overhead: a single fused
  kernel with no scratch and no combine launch (tests H6). PV stays bf16 (fp32 accumulate).

Compute is Triton-only. PyTorch is used solely for metadata/allocation/launch.
No Torch/CPU/NumPy computational fallback.
"""

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634  # log2(e)


@triton.jit
def _dsa_fused_kernel(
    q_nope_ptr, q_pe_ptr,
    ckv_ptr, kpe_ptr,
    sparse_ptr,
    out_ptr, lse_ptr,
    qk_scale,
    stride_qn_t, stride_qn_h, stride_qn_d,
    stride_qp_t, stride_qp_h, stride_qp_d,
    stride_ckv_row, stride_ckv_d,
    stride_kpe_row, stride_kpe_d,
    stride_si_t, stride_si_n,
    stride_o_t, stride_o_h, stride_o_d,
    stride_lse_t, stride_lse_h,
    H: tl.constexpr, D_CKV: tl.constexpr, D_KPE: tl.constexpr,
    BLOCK_N: tl.constexpr, TOPK: tl.constexpr,
):
    t = tl.program_id(0)

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

    # Loop the whole topk axis. TOPK is a multiple of BLOCK_N, so the index load is
    # always in-range along the topk axis; only the -1 padding needs masking.
    for n0 in range(0, TOPK, BLOCK_N):
        idx_pos = n0 + offs_n
        idx = tl.load(sparse_ptr + t * stride_si_t + idx_pos * stride_si_n)  # int32
        valid = idx != -1
        safe_idx = tl.where(valid, idx, 0).to(tl.int64)

        kc = tl.load(
            ckv_ptr + safe_idx[:, None] * stride_ckv_row + offs_ckv[None, :] * stride_ckv_d,
            mask=valid[:, None], other=0.0,
        )  # [BLOCK_N, D_CKV] bf16
        kp = tl.load(
            kpe_ptr + safe_idx[:, None] * stride_kpe_row + offs_kpe[None, :] * stride_kpe_d,
            mask=valid[:, None], other=0.0,
        )  # [BLOCK_N, D_KPE] bf16

        # QK: [H, BLOCK_N] fp32. bf16 x bf16 -> fp32 accumulate matches fp32-widened ref.
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
        # PV in bf16 (fp32 accumulate); loose tolerance accommodates bf16 P.
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kc)
        m_i = m_new

    is_empty = m_i == float("-inf")                   # [H]; no valid index for this token
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)           # avoid 0/0
    out = tl.where(is_empty[:, None], 0.0, acc / l_safe[:, None])
    lse = tl.where(is_empty, float("-inf"), m_i + tl.log2(l_i))

    tl.store(
        out_ptr + t * stride_o_t
        + offs_h[:, None] * stride_o_h + offs_ckv[None, :] * stride_o_d,
        out.to(tl.bfloat16),
    )
    tl.store(lse_ptr + t * stride_lse_t + offs_h * stride_lse_h, lse)


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

    # --- launch config (c004: fused single-pass, no split-K, no combine) ---
    BLOCK_N = 64
    # TOPK must be a multiple of BLOCK_N for the maskless index-position load.
    TOPK = triton.cdiv(topk, BLOCK_N) * BLOCK_N

    qk_scale = float(sm_scale) * LOG2E

    grid = (num_tokens,)
    _dsa_fused_kernel[grid](
        q_nope, q_pe,
        ckv_flat, kpe_flat,
        sparse_indices,
        output, lse,
        qk_scale,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_flat.stride(0), ckv_flat.stride(1),
        kpe_flat.stride(0), kpe_flat.stride(1),
        sparse_indices.stride(0), sparse_indices.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        H=num_qo_heads, D_CKV=head_dim_ckv, D_KPE=head_dim_kpe,
        BLOCK_N=BLOCK_N, TOPK=TOPK,
        num_warps=4, num_stages=2,
    )

    return output, lse
