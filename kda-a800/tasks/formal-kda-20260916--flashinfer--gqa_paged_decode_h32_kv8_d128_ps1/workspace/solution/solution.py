"""
KDA candidate c001 — gqa_paged_decode_h32_kv8_d128_ps1 (A800 / sm_80).

Correctness-first single-pass Triton baseline for batched Grouped-Query-Attention
decode over a paged KV cache (page_size == 1, so page ids are token ids).

Design (see docs/plan.md §2, c001):
  * Grid = (batch_size, num_kv_heads).  Each program owns one (b, kv) pair and
    computes ALL G = num_qo_heads // num_kv_heads = 4 query heads that share this
    KV head, so each (token, kv_head) K/V slice is loaded from HBM exactly once
    (the decisive optimization for this HBM-bandwidth-bound kernel).
  * Online (flash) softmax carried in the base-2 domain: with
    qk_scale = sm_scale * log2(e), the running quantities give the required
    base-2 LSE directly as  lse = m + log2(l), and output = acc / l.
  * QK and PV use tensor-core tl.dot with bf16 operands and fp32 accumulation.
    bf16*bf16 products are exact in fp32, so QK matches the fp32 reference dot;
    softmax probabilities are cast to bf16 for the PV dot (standard flash-decode
    practice, within bf16 output tolerance).
  * fp32 accumulation everywhere; bf16 only at the output store; lse stored f32.
  * Empty sequence (start >= end) and l == 0 guard -> output = 0, lse = -inf.

Triton carries all computation; PyTorch is used only for allocation / strides /
grid plumbing.  No Torch/CPU/NumPy/CUDA-extension fallback.
"""

import math

import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634  # log2(e)


@triton.jit
def _gqa_paged_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    out_ptr,
    lse_ptr,
    qk_scale,  # fp32: sm_scale * log2(e)
    stride_qb, stride_qh, stride_qd,
    stride_kp, stride_kh, stride_kd,
    stride_vp, stride_vh, stride_vd,
    stride_ob, stride_oh, stride_od,
    stride_lb, stride_lh,
    GQA: tl.constexpr,        # query heads per kv head (= 4)
    HEAD_DIM: tl.constexpr,   # = 128
    BLOCK_N: tl.constexpr,    # KV tokens per tile
):
    b = tl.program_id(0)
    kv = tl.program_id(1)

    h0 = kv * GQA
    offs_h = h0 + tl.arange(0, GQA)          # [GQA]
    offs_d = tl.arange(0, HEAD_DIM)          # [HEAD_DIM]

    # Load the GQA query rows for this (b, kv): [GQA, HEAD_DIM] bf16.
    q_ptrs = (
        q_ptr
        + b * stride_qb
        + offs_h[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q_rows = tl.load(q_ptrs)  # bf16

    start = tl.load(kv_indptr_ptr + b)
    end = tl.load(kv_indptr_ptr + b + 1)
    seqlen = end - start

    m_i = tl.full([GQA], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([GQA], dtype=tl.float32)
    acc = tl.zeros([GQA, HEAD_DIM], dtype=tl.float32)

    for n0 in range(0, seqlen, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seqlen

        page = tl.load(kv_indices_ptr + start + offs_n, mask=mask_n, other=0)
        page = page.to(tl.int64)

        # K loaded transposed: kT[d, n] = k_cache[page[n], kv, d]  -> [HEAD_DIM, BLOCK_N]
        k_ptrs = (
            k_ptr
            + page[None, :] * stride_kp
            + kv * stride_kh
            + offs_d[:, None] * stride_kd
        )
        kT = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)  # bf16

        # V loaded normally: v_tile[n, d] = v_cache[page[n], kv, d]  -> [BLOCK_N, HEAD_DIM]
        v_ptrs = (
            v_ptr
            + page[:, None] * stride_vp
            + kv * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v_tile = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)  # bf16

        # QK: exact fp32 accumulation from bf16 operands -> [GQA, BLOCK_N]
        qk = tl.dot(q_rows, kT, out_dtype=tl.float32)
        qk = qk * qk_scale
        qk = tl.where(mask_n[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])       # masked lanes -> 0

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v_tile, out_dtype=tl.float32)
        m_i = m_new

    is_empty = l_i == 0.0
    l_safe = tl.where(is_empty, 1.0, l_i)
    out = acc / l_safe[:, None]
    out = tl.where(is_empty[:, None], 0.0, out)
    lse_val = tl.where(is_empty, -float("inf"), m_i + tl.log2(l_i))

    out_ptrs = (
        out_ptr
        + b * stride_ob
        + offs_h[:, None] * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, out.to(tl.bfloat16))

    lse_ptrs = lse_ptr + b * stride_lb + offs_h * stride_lh
    tl.store(lse_ptrs, lse_val)


@torch.no_grad()
def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape

    assert page_size == 1
    gqa = num_qo_heads // num_kv_heads  # 4

    device = q.device
    output = torch.empty(
        (batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

    qk_scale = float(sm_scale) * _LOG2E

    BLOCK_N = 64
    grid = (batch_size, num_kv_heads)

    _gqa_paged_decode_kernel[grid](
        q,
        k_cache,
        v_cache,
        kv_indptr,
        kv_indices,
        output,
        lse,
        qk_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(2), v_cache.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        GQA=gqa,
        HEAD_DIM=head_dim,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    return output, lse
