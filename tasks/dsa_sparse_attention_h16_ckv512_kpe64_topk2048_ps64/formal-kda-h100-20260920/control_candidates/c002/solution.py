import torch
import triton
import triton.language as tl

# =============================================================================
# Candidate c002 (M0 baseline, compile fix of c001) for
#   dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64  (DeepSeek-V3.2 sparse MLA)
#
# One fused Triton kernel per token (grid = (num_tokens,)), all 16 heads in one
# program (BLOCK_M = 16). Online base-2 flash softmax over the topk=2048 sparse
# KV entries. The compressed latent Kc is BOTH the key-nope operand and the
# value operand, so each gathered KV tile is loaded once and reused for QK and
# PV. Accumulation in fp32; P@V uses bf16 inputs with fp32 accumulate.
#
# Change vs c001: the log2(e) constant is passed as a tl.constexpr kernel arg
# (LOG2E) instead of read from a module-level Python global, which Triton
# forbids inside @jit functions. No other logic change; this re-tests the M0
# correctness hypothesis.
#
# Correctness contract reproduced:
#   * logits = q_nope @ Kc^T + q_pe @ Kp^T, scaled by sm_scale
#   * lse    = base-2 log-sum-exp of the scaled logits  (log2(sum exp(scaled)))
#   * -1 sparse indices are padding: excluded from softmax and output
#   * fully-empty token  -> output row = 0  AND  lse = -inf
# No Torch/CPU/NumPy/CUDA-extension computational fallback anywhere.
# =============================================================================

# log2(e); folds natural exp into exp2(). Passed to the kernel as a constexpr.
LOG2E_HOST = 1.4426950408889634


@triton.jit
def _dsa_sparse_mla_kernel(
    q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr, idx_ptr,
    out_ptr, lse_ptr,
    sm_scale,
    stride_qn_t, stride_qn_h, stride_qn_d,
    stride_qp_t, stride_qp_h, stride_qp_d,
    stride_ckv_r, stride_ckv_d,
    stride_kpe_r, stride_kpe_d,
    stride_idx_t, stride_idx_k,
    stride_o_t, stride_o_h, stride_o_d,
    stride_lse_t, stride_lse_h,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    DPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOG2E: tl.constexpr,
):
    t = tl.program_id(0)

    h_range = tl.arange(0, H)      # 16 query heads
    d_range = tl.arange(0, D)      # 512 latent / value dim
    dpe_range = tl.arange(0, DPE)  # 64 positional-encoding dim

    # ---- load this token's query (bf16), reused across all KV tiles ----------
    qn = tl.load(
        q_nope_ptr + t * stride_qn_t
        + h_range[:, None] * stride_qn_h + d_range[None, :] * stride_qn_d
    )  # [H, D]
    qp = tl.load(
        q_pe_ptr + t * stride_qp_t
        + h_range[:, None] * stride_qp_h + dpe_range[None, :] * stride_qp_d
    )  # [H, DPE]

    qk_scale = sm_scale * LOG2E  # fold sm_scale and log2(e) so we can use exp2

    m_i = tl.full([H], float("-inf"), tl.float32)
    l_i = tl.zeros([H], tl.float32)
    acc = tl.zeros([H, D], tl.float32)

    for start in range(0, TOPK, BLOCK_N):
        n_range = start + tl.arange(0, BLOCK_N)
        idx = tl.load(idx_ptr + t * stride_idx_t + n_range * stride_idx_k)  # int32
        valid = idx != -1
        safe_idx = tl.where(valid, idx, 0).to(tl.int64)

        # Gather the KV tile once; reuse for QK (Kc, Kp) and PV (Kc == V).
        kc = tl.load(
            ckv_ptr + safe_idx[:, None] * stride_ckv_r + d_range[None, :] * stride_ckv_d,
            mask=valid[:, None], other=0.0,
        )  # [BLOCK_N, D] bf16
        kp = tl.load(
            kpe_ptr + safe_idx[:, None] * stride_kpe_r + dpe_range[None, :] * stride_kpe_d,
            mask=valid[:, None], other=0.0,
        )  # [BLOCK_N, DPE] bf16

        # scores [H, BLOCK_N] in fp32
        s = tl.dot(qn, tl.trans(kc))
        s += tl.dot(qp, tl.trans(kp))
        s = s * qk_scale
        s = tl.where(valid[None, :], s, float("-inf"))

        m_ij = tl.max(s, axis=1)              # [H]
        m_new = tl.maximum(m_i, m_ij)
        # Guard the all-(-inf) case so 0*inf / inf-inf never produce NaN.
        alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp2(m_i - m_new))
        p = tl.where(valid[None, :], tl.exp2(s - m_new[:, None]), 0.0)  # [H, BLOCK_N]

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kc)
        m_i = m_new

    empty = l_i == 0.0
    out = tl.where(empty[:, None], 0.0, acc / l_i[:, None])
    lse = tl.where(empty, float("-inf"), m_i + tl.log2(l_i))

    tl.store(
        out_ptr + t * stride_o_t
        + h_range[:, None] * stride_o_h + d_range[None, :] * stride_o_d,
        out.to(tl.bfloat16),
    )
    tl.store(lse_ptr + t * stride_lse_t + h_range * stride_lse_h, lse)


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """Sparse paged MLA attention (DeepSeek-V3.2 DSA), Triton implementation.

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

    # Flatten paged caches to token-level rows; sparse_indices address these
    # rows directly (index = page_idx * page_size + offset). Contiguous -> view.
    ckv_flat = ckv_cache.reshape(num_pages * page_size, D)
    kpe_flat = kpe_cache.reshape(num_pages * page_size, DPE)

    output = torch.empty((num_tokens, H, D), dtype=torch.bfloat16, device=device)
    lse = torch.empty((num_tokens, H), dtype=torch.float32, device=device)

    if num_tokens == 0:
        return output, lse

    BLOCK_N = 128
    grid = (num_tokens,)
    _dsa_sparse_mla_kernel[grid](
        q_nope, q_pe, ckv_flat, kpe_flat, sparse_indices,
        output, lse,
        float(sm_scale),
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_flat.stride(0), ckv_flat.stride(1),
        kpe_flat.stride(0), kpe_flat.stride(1),
        sparse_indices.stride(0), sparse_indices.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        TOPK=TOPK, H=H, D=D, DPE=DPE, BLOCK_N=BLOCK_N, LOG2E=LOG2E_HOST,
        num_warps=4, num_stages=2,
    )
    return output, lse
