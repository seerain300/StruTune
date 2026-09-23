import math

import torch
import triton
import triton.language as tl

# =============================================================================
# Candidate c004 (M1: split-K flash-decoding + adaptive split heuristic) for
#   dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64  (DeepSeek-V3.2 sparse MLA)
#
# c003 evidence: two-kernel split-K lifted geomean 18.35x -> 21.66x, but with a
# fixed 16 splits the small-num_tokens cases underfill the GPU (num_tokens=1 =>
# 1*16 = 16 stage-1 CTAs on 132 SMs, still the slowest at 12.57x). sol_ms was
# nearly flat across shapes => latency/occupancy bound.
#
# c004 fix: choose the KV tile width BLOCK_N (a power of 2 in [16,128]) on the
# host from num_tokens so that the stage-1 grid num_tokens * (TOPK/BLOCK_N)
# targets ~128 CTAs (~1 wave of H100's 132 SMs):
#     BLOCK_N = clamp(prev_pow2(TOPK * num_tokens / TARGET_GRID), 16, 128)
#   num_tokens=1 -> BLOCK_N=16  -> 128 splits, grid 128
#   num_tokens=2 -> BLOCK_N=32  ->  64 splits, grid 128
#   num_tokens=6 -> BLOCK_N=64  ->  32 splits, grid 192
#   num_tokens=7 -> BLOCK_N=64  ->  32 splits, grid 224
#   num_tokens=8 -> BLOCK_N=128 ->  16 splits, grid 128  (identical to c003)
# So the strong large-token cases are unchanged while the small ones get many
# more concurrent CTAs to hide gather latency. Each stage-1 CTA handles exactly
# one BLOCK_N tile; a fp32 stage-2 kernel combines the partials per (token,head).
#
# Correctness contract (unchanged from c003):
#   * logits = q_nope @ Kc^T + q_pe @ Kp^T, scaled by sm_scale
#   * lse    = base-2 log-sum-exp of the scaled logits (log2(sum exp(scaled)))
#   * -1 sparse indices are padding: excluded from softmax and output
#   * fully-empty token -> output row = 0 AND lse = -inf
# No Torch/CPU/NumPy/CUDA-extension computational fallback anywhere.
# =============================================================================

LOG2E_HOST = 1.4426950408889634  # log2(e); passed to the kernel as a constexpr


@triton.jit
def _dsa_sparse_mla_partial_kernel(
    q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr, idx_ptr,
    m_buf_ptr, l_buf_ptr, acc_buf_ptr,
    sm_scale,
    stride_qn_t, stride_qn_h, stride_qn_d,
    stride_qp_t, stride_qp_h, stride_qp_d,
    stride_ckv_r, stride_ckv_d,
    stride_kpe_r, stride_kpe_d,
    stride_idx_t, stride_idx_k,
    stride_m_t, stride_m_s, stride_m_h,
    stride_a_t, stride_a_s, stride_a_h, stride_a_d,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    DPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOG2E: tl.constexpr,
):
    t = tl.program_id(0)
    s = tl.program_id(1)

    h_range = tl.arange(0, H)
    d_range = tl.arange(0, D)
    dpe_range = tl.arange(0, DPE)

    qn = tl.load(
        q_nope_ptr + t * stride_qn_t
        + h_range[:, None] * stride_qn_h + d_range[None, :] * stride_qn_d
    )  # [H, D]
    qp = tl.load(
        q_pe_ptr + t * stride_qp_t
        + h_range[:, None] * stride_qp_h + dpe_range[None, :] * stride_qp_d
    )  # [H, DPE]

    qk_scale = sm_scale * LOG2E

    # Each stage-1 CTA processes exactly one BLOCK_N-wide KV tile.
    start = s * BLOCK_N
    n_range = start + tl.arange(0, BLOCK_N)
    kv_mask = n_range < TOPK
    idx = tl.load(idx_ptr + t * stride_idx_t + n_range * stride_idx_k,
                  mask=kv_mask, other=-1)
    valid = (idx != -1) & kv_mask
    safe_idx = tl.where(valid, idx, 0).to(tl.int64)

    kc = tl.load(
        ckv_ptr + safe_idx[:, None] * stride_ckv_r + d_range[None, :] * stride_ckv_d,
        mask=valid[:, None], other=0.0,
    )  # [BLOCK_N, D]
    kp = tl.load(
        kpe_ptr + safe_idx[:, None] * stride_kpe_r + dpe_range[None, :] * stride_kpe_d,
        mask=valid[:, None], other=0.0,
    )  # [BLOCK_N, DPE]

    sc = tl.dot(qn, tl.trans(kc))
    sc += tl.dot(qp, tl.trans(kp))
    sc = sc * qk_scale
    sc = tl.where(valid[None, :], sc, float("-inf"))

    m_i = tl.max(sc, axis=1)                                   # [H]
    m_safe = tl.where(m_i == float("-inf"), 0.0, m_i)          # guard all-(-inf)
    p = tl.where(valid[None, :], tl.exp2(sc - m_safe[:, None]), 0.0)  # [H, BLOCK_N]
    l_i = tl.sum(p, axis=1)                                    # [H]
    acc = tl.dot(p.to(tl.bfloat16), kc)                        # [H, D]

    tl.store(m_buf_ptr + t * stride_m_t + s * stride_m_s + h_range * stride_m_h, m_i)
    tl.store(l_buf_ptr + t * stride_m_t + s * stride_m_s + h_range * stride_m_h, l_i)
    tl.store(
        acc_buf_ptr + t * stride_a_t + s * stride_a_s
        + h_range[:, None] * stride_a_h + d_range[None, :] * stride_a_d,
        acc,
    )


@triton.jit
def _dsa_sparse_mla_combine_kernel(
    m_buf_ptr, l_buf_ptr, acc_buf_ptr,
    out_ptr, lse_ptr,
    stride_m_t, stride_m_s, stride_m_h,
    stride_a_t, stride_a_s, stride_a_h, stride_a_d,
    stride_o_t, stride_o_h, stride_o_d,
    stride_lse_t, stride_lse_h,
    H: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    s_range = tl.arange(0, NUM_SPLITS)
    d_range = tl.arange(0, D)

    m_s = tl.load(m_buf_ptr + t * stride_m_t + s_range * stride_m_s + h * stride_m_h)  # [S]
    l_s = tl.load(l_buf_ptr + t * stride_m_t + s_range * stride_m_s + h * stride_m_h)  # [S]
    acc = tl.load(
        acc_buf_ptr + t * stride_a_t + s_range[:, None] * stride_a_s
        + h * stride_a_h + d_range[None, :] * stride_a_d
    )  # [S, D]

    m = tl.max(m_s, axis=0)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    scale = tl.exp2(m_s - m_safe)               # [S]
    l = tl.sum(l_s * scale, axis=0)             # scalar
    acc = tl.sum(acc * scale[:, None], axis=0)  # [D]

    empty = l == 0.0
    out = tl.where(empty, 0.0, acc / l)
    lse = tl.where(empty, float("-inf"), m + tl.log2(l))

    tl.store(out_ptr + t * stride_o_t + h * stride_o_h + d_range * stride_o_d,
             out.to(tl.bfloat16))
    tl.store(lse_ptr + t * stride_lse_t + h * stride_lse_h, lse)


def _pick_block_n(num_tokens, topk, target_grid=128, bn_min=16, bn_max=128):
    """Choose the KV tile width so num_tokens * (topk/BLOCK_N) ~ target_grid.

    Returns a power of 2 in [bn_min, bn_max] that divides topk (topk is a power
    of 2 for this task, so any pow2 <= topk divides it).
    """
    raw = max(1.0, topk * num_tokens / float(target_grid))
    bn = 1 << int(math.floor(math.log2(raw)))
    bn = max(bn_min, min(bn_max, bn))
    bn = min(bn, topk)
    return bn


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Sparse paged MLA attention (DeepSeek-V3.2 DSA), Triton flash-decoding.

    q_nope:        [num_tokens, num_qo_heads, head_dim_ckv]  bf16
    q_pe:          [num_tokens, num_qo_heads, head_dim_kpe]   bf16
    ckv_cache:     [num_pages, page_size, head_dim_ckv]       bf16
    kpe_cache:     [num_pages, page_size, head_dim_kpe]       bf16
    sparse_indices:[num_tokens, topk]                         int32  (-1 = pad)
    sm_scale:      scalar fp32
    returns (output [num_tokens, H, ckv] bf16, lse [num_tokens, H] fp32)
    """
    num_tokens, H, D = q_nope.shape
    DPE = q_pe.shape[-1]
    num_pages, page_size, _ = ckv_cache.shape
    TOPK = sparse_indices.shape[-1]
    device = q_nope.device

    ckv_flat = ckv_cache.reshape(num_pages * page_size, D)
    kpe_flat = kpe_cache.reshape(num_pages * page_size, DPE)

    output = torch.empty((num_tokens, H, D), dtype=torch.bfloat16, device=device)
    lse = torch.empty((num_tokens, H), dtype=torch.float32, device=device)

    if num_tokens == 0:
        return output, lse

    BLOCK_N = _pick_block_n(num_tokens, TOPK)
    num_kv_splits = (TOPK + BLOCK_N - 1) // BLOCK_N

    m_buf = torch.empty((num_tokens, num_kv_splits, H), dtype=torch.float32, device=device)
    l_buf = torch.empty((num_tokens, num_kv_splits, H), dtype=torch.float32, device=device)
    acc_buf = torch.empty((num_tokens, num_kv_splits, H, D), dtype=torch.float32, device=device)

    _dsa_sparse_mla_partial_kernel[(num_tokens, num_kv_splits)](
        q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
        m_buf, l_buf, acc_buf,
        float(sm_scale),
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_flat.stride(0), ckv_flat.stride(1),
        kpe_flat.stride(0), kpe_flat.stride(1),
        sparse_indices.stride(0), sparse_indices.stride(1),
        m_buf.stride(0), m_buf.stride(1), m_buf.stride(2),
        acc_buf.stride(0), acc_buf.stride(1), acc_buf.stride(2), acc_buf.stride(3),
        TOPK=TOPK, H=H, D=D, DPE=DPE, BLOCK_N=BLOCK_N, LOG2E=LOG2E_HOST,
        num_warps=4, num_stages=2,
    )

    _dsa_sparse_mla_combine_kernel[(num_tokens, H)](
        m_buf, l_buf, acc_buf,
        output, lse,
        m_buf.stride(0), m_buf.stride(1), m_buf.stride(2),
        acc_buf.stride(0), acc_buf.stride(1), acc_buf.stride(2), acc_buf.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        H=H, D=D, NUM_SPLITS=num_kv_splits,
        num_warps=4, num_stages=1,
    )
    return output, lse
