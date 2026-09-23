# gqa_ragged_prefill_causal_h32_kv8_d128 — best candidate c004 (batch formal-kda-h100-20260920)
# feedback (w2/i10 coarse): 11.93x | final (100 iters, paired timing): 12.45x, valid
# source: kda-control/formal-kda-h100-20260920--flashinfer--gqa_ragged_prefill_causal_h32_kv8_d128/candidates/c004/solution.py (sha256-locked snapshot)
"""Triton fused GQA ragged causal prefill attention (H100 / sm_90).

Candidate c004 — deepen the KV-loop software pipeline (num_stages 2 -> 3).

Parent: c003 (geomean 11.71x, 21/21 correct). The Triton kernel body and the host
launch logic (sync-free for total_q<=BLOCK_M, torch.empty allocations) are byte-for-byte
identical to c003. The ONLY change is the kernel launch parameter num_stages: 2 -> 3.

Rationale (Perf branch A, KernelWiki technique-pipeline-stages): the 6 compute-bound
workloads (#4=81, #17=982, #21=92, and the 3 large ~12.5-13.5k tokens) spend their time
in the KV loop doing wgmma (QK^T and PV) with TMA/global loads of K and V tiles. A
3-stage pipeline lets the compiler prefetch two K/V tiles ahead, overlapping the next
tile's loads with the current tile's matmuls, which raises throughput on the long causal
KV loops of the large cases. The 15 launch-bound micro/tiny cases (total_q<=71, KV loop
= 1 tile) have no loop to pipeline, so num_stages should be neutral for them (guard:
watch for any regression from higher smem/register pressure at compile time).

Definition: gqa_ragged_prefill_causal_h32_kv8_d128
  q: [total_q, 32, 128] bf16, k/v: [total_kv, 8, 128] bf16,
  qo_indptr/kv_indptr: [B+1] int32, sm_scale: fp32 scalar.
  output: [total_q, 32, 128] bf16, lse: [total_q, 32] fp32 (base-2 logsumexp).

GQA: query head h uses kv head h // 4 (gqa_ratio = 32/8 = 4).
Causal: query row i (0-based in-seq) attends kv col j iff j < i + 1 + delta,
        delta = kv_len - q_len. All observed data has delta == 0 (j <= i).

Only Triton performs the attention math. PyTorch is used solely for output
allocation and (for large cases only) a single host read of max query sequence length.
"""

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634  # log2(e); fold into scale so exp2 matches natural softmax


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    qo_indptr,
    kv_indptr,
    qk_scale,  # sm_scale * log2(e), applied to raw QK^T before exp2
    stride_qm,
    stride_qh,
    stride_km,
    stride_kh,
    stride_vm,
    stride_vh,
    stride_om,
    stride_oh,
    stride_lm,
    stride_lh,
    GQA_RATIO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // GQA_RATIO

    q_start = tl.load(qo_indptr + cur_seq)
    q_end = tl.load(qo_indptr + cur_seq + 1)
    q_len = q_end - q_start

    # Blocks past this sequence's query length do nothing (rows belong to other seqs).
    if cur_block_m * BLOCK_M >= q_len:
        return

    kv_start = tl.load(kv_indptr + cur_seq)
    kv_end = tl.load(kv_indptr + cur_seq + 1)
    kv_len = kv_end - kv_start
    delta = kv_len - q_len

    offs_m = cur_block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < q_len

    # Load Q tile [BLOCK_M, BLOCK_D] (bf16).
    q_ptrs = (
        Q + (q_start + offs_m)[:, None] * stride_qm + cur_head * stride_qh + offs_d[None, :]
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    # Online-softmax running state (fp32).
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Causal: highest query index in this block is (cur_block_m+1)*BLOCK_M - 1,
    # which may attend up to kv col (that + delta). Stop the KV loop there.
    if IS_CAUSAL:
        kv_hi = tl.minimum(kv_len, (cur_block_m + 1) * BLOCK_M + delta)
    else:
        kv_hi = kv_len

    for start_n in range(0, kv_hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cols = start_n + offs_n
        mask_n = cols < kv_len

        # K tile loaded transposed [BLOCK_D, BLOCK_N] for dot(q, k).
        k_ptrs = (
            K
            + (kv_start + cols)[None, :] * stride_km
            + cur_kv_head * stride_kh
            + offs_d[:, None]
        )
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

        qk = tl.dot(q, k, out_dtype=tl.float32) * qk_scale

        final_mask = mask_m[:, None] & mask_n[None, :]
        if IS_CAUSAL:
            final_mask &= (offs_m[:, None] + delta) >= cols[None, :]
        qk = tl.where(final_mask, qk, float("-inf"))

        m_tile = tl.max(qk, 1)
        m_new = tl.maximum(m_i, m_tile)
        # Guard rows that have seen no valid key yet (avoid -inf - -inf -> nan).
        m_new_safe = tl.where(m_new == float("-inf"), 0.0, m_new)

        p = tl.exp2(qk - m_new_safe[:, None])
        alpha = tl.exp2(m_i - m_new_safe)
        l_i = l_i * alpha + tl.sum(p, 1)

        v_ptrs = (
            V
            + (kv_start + cols)[:, None] * stride_vm
            + cur_kv_head * stride_vh
            + offs_d[None, :]
        )
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_new

    # Epilogue: normalize, cast output to bf16, write base-2 LSE.
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / l_safe[:, None]
    lse_val = m_i + tl.log2(l_i)  # -inf when a row saw no valid key (matches ref)

    o_ptrs = (
        O + (q_start + offs_m)[:, None] * stride_om + cur_head * stride_oh + offs_d[None, :]
    )
    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=mask_m[:, None])

    lse_ptrs = LSE + (q_start + offs_m) * stride_lm + cur_head * stride_lh
    tl.store(lse_ptrs, lse_val, mask=mask_m)


@torch.no_grad()
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    total_kv, num_kv_heads, _ = k.shape
    device = q.device

    batch_size = qo_indptr.shape[0] - 1
    if batch_size <= 0 or total_q == 0:
        output = torch.zeros(
            (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
        )
        lse = torch.full(
            (total_q, num_qo_heads), float("-inf"), dtype=torch.float32, device=device
        )
        return output, lse

    gqa_ratio = num_qo_heads // num_kv_heads

    BLOCK_M = 128
    BLOCK_N = 64
    qk_scale = float(sm_scale) * LOG2E

    # Grid m-block count. Every query row in [0,total_q) is written by exactly one
    # (seq, block) program per head, so torch.empty is safe (see c003 docstring).
    #
    # No single sequence can exceed total_q tokens, so when total_q <= BLOCK_M every
    # sequence fits in one m-block: n_m_blocks = 1 without any device->host sync. This
    # removes the max-reduction + .item() round-trip (a full stream sync) from all
    # launch-bound tiny/small cases (total_q <= 92 dominate the geomean).
    if total_q <= BLOCK_M:
        n_m_blocks = 1
    else:
        q_lens = qo_indptr[1 : batch_size + 1] - qo_indptr[:batch_size]
        max_seqlen = int(q_lens.max().item())
        if max_seqlen == 0:
            output = torch.zeros(
                (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
            )
            lse = torch.full(
                (total_q, num_qo_heads), float("-inf"), dtype=torch.float32, device=device
            )
            return output, lse
        n_m_blocks = triton.cdiv(max_seqlen, BLOCK_M)

    # Every output/lse row is fully written by the kernel (packed layout, one owner per
    # row per head) -> skip the 2 memset kernels the reference's zeros/full would incur.
    output = torch.empty(
        (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.empty(
        (total_q, num_qo_heads), dtype=torch.float32, device=device
    )

    grid = (batch_size, num_qo_heads, n_m_blocks)

    _fwd_kernel[grid](
        q,
        k,
        v,
        output,
        lse,
        qo_indptr,
        kv_indptr,
        qk_scale,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        lse.stride(1),
        GQA_RATIO=gqa_ratio,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=head_dim,
        IS_CAUSAL=True,
        num_warps=8,
        num_stages=3,
    )

    return output, lse
