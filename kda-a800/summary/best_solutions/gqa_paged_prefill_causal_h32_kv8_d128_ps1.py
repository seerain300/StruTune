# KDA A800 best solution: gqa_paged_prefill_causal_h32_kv8_d128_ps1
# candidate: c001  |  feedback: 92.36x  |  final (authoritative): 63.06x
# campaign formal-kda-20260916 (A800, g0056)  |  evaluations: 1
# source: tasks/formal-kda-20260916--flashinfer--gqa_paged_prefill_causal_h32_kv8_d128_ps1/control/candidates/c001/solution.py (sha256-frozen snapshot)

"""
Triton GQA paged causal prefill — candidate c001 (correctness baseline).

Task: gqa_paged_prefill_causal_h32_kv8_d128_ps1 on NVIDIA A800 (sm_80).

Constants (from task/definition.json):
    num_qo_heads = 32, num_kv_heads = 8, head_dim = 128, page_size = 1,
    gqa_ratio = 4.

Design (see docs/draft.md, docs/plan.md):
  * The feedback regime is num_kv << num_q, so `output` is overwhelmingly zeros
    and the problem is memset-bound. We allocate `output`=zeros / `lse`=-inf
    (plumbing) and let the Triton kernel write ONLY the active query rows.
  * Fused, flash-attention style kernel with an online-softmax loop over KV
    blocks (correct for any num_kv, not just the tiny feedback case).
  * Base-2 LSE via exp2/log2 with qk_scale = sm_scale * log2(e):
        exp2(S_j) = e^(sm_scale * qk_j),  lse = M + log2(L) = log2(sum e^logit).
  * All dot products in fp32 (input_precision="ieee") to mirror the reference,
    which upcasts q,k,v to float32 before matmul.
  * GQA packing: one program handles one kv_head and a tile of BLOCK_Q queries,
    packing the gqa_ratio=4 qo heads into the row (M) dimension. All 4 heads of a
    query share the same KV set and the same causal mask.
  * Grid (batch, max_q_tiles, num_kv_heads); dense over query tiles with an
    in-kernel early-exit for tiles outside the active tail. Dead blocks are cheap
    (a couple of int loads + return).

Only active query rows are written; inactive rows / empty sequences keep the
zeros / -inf supplied by the allocation, exactly matching the reference.
"""

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit
def _gqa_paged_prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    kv_indices_ptr,
    qk_scale,
    # strides
    stride_q_tok,
    stride_q_head,
    stride_q_dim,
    stride_k_page,
    stride_k_head,
    stride_k_dim,
    stride_v_page,
    stride_v_head,
    stride_v_dim,
    stride_o_tok,
    stride_o_head,
    stride_o_dim,
    stride_lse_tok,
    stride_lse_head,
    # constexprs
    GQA: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    pid_b = tl.program_id(0)       # batch (sequence) index
    pid_t = tl.program_id(1)       # query tile index within the sequence
    pid_kv = tl.program_id(2)      # kv head index

    q_start = tl.load(qo_indptr_ptr + pid_b)
    q_end = tl.load(qo_indptr_ptr + pid_b + 1)
    num_q = q_end - q_start

    tile_q_start = pid_t * BLOCK_Q
    # Tile lies entirely past this sequence's queries -> nothing to do.
    if tile_q_start >= num_q:
        return

    kv_start = tl.load(kv_indptr_ptr + pid_b)
    kv_end = tl.load(kv_indptr_ptr + pid_b + 1)
    num_kv = kv_end - kv_start
    # Empty KV for this sequence -> all rows stay zeros / -inf.
    if num_kv <= 0:
        return

    delta = num_kv - num_q

    # Skip tiles that lie entirely below the active tail. A local query qi is
    # active iff qi + delta >= 0. The largest qi in this tile is
    # tile_q_start + BLOCK_Q - 1, so if that + delta < 0 no row here is active.
    if (tile_q_start + BLOCK_Q - 1 + delta) < 0:
        return

    # Row layout: BLOCK_M = BLOCK_Q * GQA. row -> (local query, head-in-group).
    row = tl.arange(0, BLOCK_M)
    qi_local = row // GQA            # 0..BLOCK_Q-1
    g = row % GQA                    # 0..GQA-1
    h = pid_kv * GQA + g             # qo head index
    qi_seq = tile_q_start + qi_local  # local query position within the sequence
    q_row_global = q_start + qi_seq   # global query row index
    row_in_range = qi_seq < num_q     # within this sequence's query span

    d = tl.arange(0, HEAD_DIM)

    # Load Q tile [BLOCK_M, HEAD_DIM] (bf16 -> fp32).
    q_ptrs = (
        q_ptr
        + q_row_global[:, None] * stride_q_tok
        + h[:, None] * stride_q_head
        + d[None, :] * stride_q_dim
    )
    q_tile = tl.load(q_ptrs, mask=row_in_range[:, None], other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, num_kv, BLOCK_KV):
        kv_off = start_n + tl.arange(0, BLOCK_KV)          # local kv token index
        kv_mask = kv_off < num_kv
        page_id = tl.load(
            kv_indices_ptr + kv_start + kv_off, mask=kv_mask, other=0
        ).to(tl.int64)

        # k_cache / v_cache: [num_pages, 1(page_size), num_kv_heads, head_dim].
        k_ptrs = (
            k_ptr
            + page_id[:, None] * stride_k_page
            + pid_kv * stride_k_head
            + d[None, :] * stride_k_dim
        )
        v_ptrs = (
            v_ptr
            + page_id[:, None] * stride_v_page
            + pid_kv * stride_v_head
            + d[None, :] * stride_v_dim
        )
        k_tile = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        v_tile = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

        # QK^T in fp32.
        qk = tl.dot(q_tile, tl.trans(k_tile), input_precision="ieee") * qk_scale

        # Causal prefix mask: key j valid for query qi_seq iff j <= qi_seq + delta
        # (and j < num_kv). This is the end-anchored causal mask.
        causal_valid = (kv_off[None, :] <= (qi_seq[:, None] + delta)) & kv_mask[None, :]
        qk = tl.where(causal_valid, qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v_tile, input_precision="ieee")
        m_i = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / l_safe[:, None]
    out_bf16 = out.to(tl.bfloat16)
    lse_val = m_i + tl.log2(l_i)

    # A query row is written iff it is in range AND has >= 1 valid key, i.e.
    # max_kv_idx = min(qi_seq + 1 + delta, num_kv) > 0  <=>  qi_seq + delta >= 0.
    store_mask = row_in_range & ((qi_seq + delta) >= 0)

    o_ptrs = (
        o_ptr
        + q_row_global[:, None] * stride_o_tok
        + h[:, None] * stride_o_head
        + d[None, :] * stride_o_dim
    )
    tl.store(o_ptrs, out_bf16, mask=store_mask[:, None])

    lse_ptrs = lse_ptr + q_row_global * stride_lse_tok + h * stride_lse_head
    tl.store(lse_ptrs, lse_val, mask=store_mask)


@torch.no_grad()
def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape
    device = q.device

    gqa_ratio = num_qo_heads // num_kv_heads
    batch = qo_indptr.shape[0] - 1

    output = torch.zeros(
        (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    if batch <= 0 or total_q == 0:
        return output, lse

    # Host-side: max queries in any single sequence -> number of query tiles.
    seq_q = qo_indptr[1:] - qo_indptr[:-1]
    max_num_q = int(seq_q.max().item())
    if max_num_q <= 0:
        return output, lse

    BLOCK_Q = 16
    BLOCK_KV = 64
    BLOCK_M = BLOCK_Q * gqa_ratio
    max_q_tiles = (max_num_q + BLOCK_Q - 1) // BLOCK_Q

    qk_scale = float(sm_scale) * LOG2E

    grid = (batch, max_q_tiles, num_kv_heads)

    _gqa_paged_prefill_kernel[grid](
        q,
        k_cache,
        v_cache,
        output,
        lse,
        qo_indptr,
        kv_indptr,
        kv_indices,
        qk_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(2),
        v_cache.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        GQA=gqa_ratio,
        HEAD_DIM=head_dim,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        BLOCK_KV=BLOCK_KV,
        num_warps=4,
        num_stages=2,
    )

    return output, lse
