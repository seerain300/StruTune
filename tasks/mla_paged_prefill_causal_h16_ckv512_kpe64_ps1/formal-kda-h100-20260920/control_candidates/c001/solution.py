"""
Triton MLA paged prefill (causal) — candidate c001.

Task: mla_paged_prefill_causal_h16_ckv512_kpe64_ps1  (H100 / sm_90)

Fused Multi-head Latent Attention prefill with a paged KV cache (page_size = 1),
causal mask. Constants: num_qo_heads=16, head_dim_ckv=512, head_dim_kpe=64.

Design (c001 — correct fused baseline, per docs/plan.md §3):
  * grid = (batch, num_heads=16, cdiv(max_q_len, BLOCK_M))
  * each program handles BLOCK_M query rows of one (seq, head); loops KV in BLOCK_N chunks
  * splits: BLOCK_DMODEL=512 (ckv/value), BLOCK_DPE=64 (kpe)
  * page_size=1 => kv_indices entries are direct token indices into the cache
  * value == ckv content (V == Kc); kpe contributes to logits only
  * causal mask: token j kept iff j <= prefix_len + i, prefix_len = kv_len - q_len
  * causal early-exit on the KV loop
  * base-2 online softmax (exp2) so lse is natively base-2 (matches reference /ln2)
  * fp32 accumulators; only tl.dot operands are bf16
  * empty-seq / padding guards (no NaN)

Feature (last) dims are contiguous (stride 1) for every input, so feature offsets are
added directly. Triton does all attention math; no Torch/CPU/NumPy computational fallback.
"""

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634  # 1/ln(2)


@triton.jit
def _mla_prefill_kernel(
    Q_nope,        # [total_q, H, Dc]  bf16
    Q_pe,          # [total_q, H, Dp]  bf16
    Ckv,           # [num_pages, 1, Dc] bf16  (key-content AND value)
    Kpe,           # [num_pages, 1, Dp] bf16  (positional key)
    qo_indptr,     # [batch+1] int32
    kv_indptr,     # [batch+1] int32
    kv_indices,    # [num_kv_indices] int32
    Out,           # [total_q, H, Dc] bf16
    Lse,           # [total_q, H] fp32
    sm_scale,      # fp32 scalar
    stride_qn_t, stride_qn_h,
    stride_qp_t, stride_qp_h,
    stride_ckv_p,
    stride_kpe_p,
    stride_o_t, stride_o_h,
    stride_lse_t, stride_lse_h,
    H: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,   # 512
    BLOCK_DPE: tl.constexpr,      # 64
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)

    q_start = tl.load(qo_indptr + cur_seq)
    q_end = tl.load(qo_indptr + cur_seq + 1)
    q_len = q_end - q_start

    # Whole query block is out of range -> nothing to do (output/lse pre-initialized).
    if cur_block_m * BLOCK_M >= q_len:
        return

    kv_start = tl.load(kv_indptr + cur_seq)
    kv_end = tl.load(kv_indptr + cur_seq + 1)
    kv_len = kv_end - kv_start

    # No KV for this sequence -> leave output=0, lse=-inf (matches reference `continue`).
    if kv_len <= 0:
        return

    prefix_len = kv_len - q_len  # number of cached tokens before the extend region

    offs_m = cur_block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dpe = tl.arange(0, BLOCK_DPE)
    offs_n = tl.arange(0, BLOCK_N)

    mask_m = offs_m < q_len

    # Load Q (nope + pe) for these rows / this head. Feature dims contiguous (stride 1).
    q_rows = q_start + offs_m
    qn_ptrs = (
        Q_nope + q_rows[:, None] * stride_qn_t + cur_head * stride_qn_h + offs_d[None, :]
    )
    q_nope = tl.load(qn_ptrs, mask=mask_m[:, None], other=0.0)  # [BM, Dc]

    qp_ptrs = (
        Q_pe + q_rows[:, None] * stride_qp_t + cur_head * stride_qp_h + offs_dpe[None, :]
    )
    q_pe = tl.load(qp_ptrs, mask=mask_m[:, None], other=0.0)  # [BM, Dp]

    qk_scale = sm_scale * LOG2E

    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Causal early-exit: only need KV up to prefix_len + (max query row index in block).
    kv_upper = tl.minimum(kv_len, prefix_len + cur_block_m * BLOCK_M + BLOCK_M)

    for start_n in range(0, kv_upper, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cur_n = start_n + offs_n
        mask_n = cur_n < kv_len

        # page_size == 1  =>  kv_indices entry is the token/page index directly.
        tok = tl.load(kv_indices + kv_start + cur_n, mask=mask_n, other=0).to(tl.int64)

        # K content, transposed for QK: [Dc, BN]
        kc_ptrs = Ckv + tok[None, :] * stride_ckv_p + offs_d[:, None]
        kc_t = tl.load(kc_ptrs, mask=mask_n[None, :], other=0.0)

        qk = tl.dot(q_nope, kc_t)  # [BM, BN]

        # K positional, transposed: [Dp, BN]
        kp_ptrs = Kpe + tok[None, :] * stride_kpe_p + offs_dpe[:, None]
        kp_t = tl.load(kp_ptrs, mask=mask_n[None, :], other=0.0)
        qk += tl.dot(q_pe, kp_t)

        s2 = qk * qk_scale  # base-2 exponent domain

        # Causal + range mask: keep iff j <= prefix_len + i, j < kv_len, row valid.
        causal = cur_n[None, :] <= (prefix_len + offs_m[:, None])
        valid = mask_m[:, None] & mask_n[None, :] & causal
        s2 = tl.where(valid, s2, -float("inf"))

        row_max = tl.max(s2, 1)
        row_max = tl.where(row_max == -float("inf"), -1e30, row_max)
        n_e_max = tl.maximum(e_max, row_max)

        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(s2 - n_e_max[:, None])            # [BM, BN]
        deno = deno * re_scale + tl.sum(p, 1)

        # Value == K content: [BN, Dc]
        v_ptrs = Ckv + tok[:, None] * stride_ckv_p + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        acc = acc * re_scale[:, None] + tl.dot(p.to(v.dtype), v)
        e_max = n_e_max

    deno_safe = tl.where(deno == 0.0, 1.0, deno)
    out = acc / deno_safe[:, None]
    lse = tl.where(deno == 0.0, -float("inf"), e_max + tl.log2(deno))

    o_ptrs = (
        Out + q_rows[:, None] * stride_o_t + cur_head * stride_o_h + offs_d[None, :]
    )
    tl.store(o_ptrs, out.to(Out.dtype.element_ty), mask=mask_m[:, None])

    lse_ptrs = Lse + q_rows * stride_lse_t + cur_head * stride_lse_h
    tl.store(lse_ptrs, lse, mask=mask_m)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    device = q_nope.device

    output = torch.zeros(
        (total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    batch = qo_indptr.shape[0] - 1
    if batch <= 0 or total_q == 0:
        return output, lse

    # Launch metadata only (allowed torch): max extend length across sequences.
    q_lens = qo_indptr[1:] - qo_indptr[:-1]
    max_q_len = int(q_lens.max().item())
    if max_q_len == 0:
        return output, lse

    sm_scale = float(sm_scale)

    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_DMODEL = head_dim_ckv   # 512
    BLOCK_DPE = head_dim_kpe      # 64

    grid = (batch, num_qo_heads, triton.cdiv(max_q_len, BLOCK_M))

    _mla_prefill_kernel[grid](
        q_nope,
        q_pe,
        ckv_cache,
        kpe_cache,
        qo_indptr,
        kv_indptr,
        kv_indices,
        output,
        lse,
        sm_scale,
        q_nope.stride(0), q_nope.stride(1),
        q_pe.stride(0), q_pe.stride(1),
        ckv_cache.stride(0),
        kpe_cache.stride(0),
        output.stride(0), output.stride(1),
        lse.stride(0), lse.stride(1),
        H=num_qo_heads,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=1,
    )

    return output, lse
