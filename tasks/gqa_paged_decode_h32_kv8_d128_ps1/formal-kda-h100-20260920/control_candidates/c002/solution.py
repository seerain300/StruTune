import torch
import triton
import triton.language as tl


@triton.jit
def _gqa_paged_decode_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
    indptr_ptr, indices_ptr,
    qk_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kp, stride_kh, stride_kd,
    stride_vp, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    stride_lb, stride_lh,
    GQA: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # One program owns (batch b, kv_head). It processes the GQA group of
    # query heads [pid_kv*GQA : pid_kv*GQA + GQA] together so the K/V tiles
    # are read from HBM exactly once for the whole group.
    pid_b = tl.program_id(0)
    pid_kv = tl.program_id(1)

    kv_start = tl.load(indptr_ptr + pid_b)
    kv_end = tl.load(indptr_ptr + pid_b + 1)
    seq_len = kv_end - kv_start

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)

    # Query-head rows: first GQA rows are real, the rest are padding to reach
    # a tensor-core-friendly M (>=16) and are masked on load/store.
    h_idx = pid_kv * GQA + offs_h
    h_mask = offs_h < GQA

    q_ptrs = (
        q_ptr
        + pid_b * stride_qb
        + h_idx[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)  # [BLOCK_H, BLOCK_D] bf16

    m_i = tl.full([BLOCK_H], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        tok = start_n + offs_n
        n_mask = tok < seq_len
        # page id == token row id because page_size == 1
        page = tl.load(indices_ptr + kv_start + tok, mask=n_mask, other=0).to(tl.int64)

        k_ptrs = (
            k_ptr
            + page[:, None] * stride_kp
            + pid_kv * stride_kh
            + offs_d[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)  # [BLOCK_N, BLOCK_D] bf16

        # logits = q_group @ K^T ; bf16 inputs, fp32 accumulate (no tf32 path
        # since inputs are bf16). Matches the fp32 reference dot exactly.
        qk = tl.dot(q, tl.trans(k))  # [BLOCK_H, BLOCK_N] fp32
        qk = qk * qk_scale
        qk = tl.where(n_mask[None, :], qk, -float("inf"))

        m_curr = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_curr)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])  # [BLOCK_H, BLOCK_N] fp32

        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = (
            v_ptr
            + page[:, None] * stride_vp
            + pid_kv * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)  # [BLOCK_N, BLOCK_D] bf16

        # bf16 x bf16 tensor-core second dot with fp32 accumulate. p in [0,1]
        # rounds to bf16 with ~2^-8 relative error; masked lanes are exact 0.
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    l_safe = l_i > 0.0
    denom = tl.where(l_safe, l_i, 1.0)
    out = acc / denom[:, None]
    out = tl.where(l_safe[:, None], out, 0.0)

    o_ptrs = (
        o_ptr
        + pid_b * stride_ob
        + h_idx[:, None] * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, out.to(tl.bfloat16), mask=h_mask[:, None])

    # base-2 log-sum-exp: qk_scale already folds log2(e), so m_i + log2(l_i)
    # == log2(sum exp(scaled logits)) == reference lse.
    lse_val = tl.where(l_safe, m_i + tl.log2(denom), -float("inf"))
    lse_ptrs = lse_ptr + pid_b * stride_lb + h_idx * stride_lh
    tl.store(lse_ptrs, lse_val, mask=h_mask)


def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape
    gqa = num_qo_heads // num_kv_heads

    output = torch.empty(
        (batch, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device
    )
    lse = torch.empty((batch, num_qo_heads), dtype=torch.float32, device=q.device)

    if batch == 0:
        return output, lse

    LOG2E = 1.4426950408889634
    qk_scale = float(sm_scale) * LOG2E

    BLOCK_H = 16
    BLOCK_N = 64
    BLOCK_D = head_dim

    grid = (batch, num_kv_heads)
    _gqa_paged_decode_kernel[grid](
        q, k_cache, v_cache, output, lse,
        kv_indptr, kv_indices,
        qk_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(2), v_cache.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        GQA=gqa,
        BLOCK_H=BLOCK_H,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return output, lse
