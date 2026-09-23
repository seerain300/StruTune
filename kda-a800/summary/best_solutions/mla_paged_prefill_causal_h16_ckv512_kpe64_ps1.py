# KDA A800 best solution: mla_paged_prefill_causal_h16_ckv512_kpe64_ps1
# candidate: c002  |  feedback: 68.46x  |  final (authoritative): 55.44x
# campaign formal-kda-20260916 (A800, g0056)  |  evaluations: 2
# source: tasks/formal-kda-20260916--flashinfer--mla_paged_prefill_causal_h16_ckv512_kpe64_ps1/control/candidates/c002/solution.py (sha256-frozen snapshot)

"""c002: baseline fused Triton MLA paged-prefill (causal) kernel.

Parent: c001 (rejected -- compile bug only). The ONLY change from c001 is that
the base-2 LSE factor 1/ln(2) is now inlined as a local literal inside the
kernel instead of being referenced from a module-level Python global
(`_INV_LN2`), which Triton could not access inside a @triton.jit body and which
raised a CompilationError on all five workloads. The algorithm is byte-for-byte
identical to c001; this candidate exercises the baseline algorithm for the first
time to obtain a real correctness + speedup signal.

Target: NVIDIA A800 / sm_80 (Ampere). Triton compute only; PyTorch used solely
for tensor metadata / launch plumbing (host-side tile->sequence mapping).

Design (see docs/plan.md sec 3-4, docs/draft.md):
  * MQA-shaped: 16 query heads are the M (row) axis so a single paged KV gather
    feeds all heads for a query token.
  * One CTA per query token (BLOCK_Q_tok = 1, M = 16).
  * QK: logits = q_nope[16,512] @ Kc^T + q_pe[16,64] @ Kp^T  (bf16 MMA, fp32 acc).
  * Online softmax over KV tiles (running max m, sum l), fp32 throughout.
  * PV: acc[16,512] += softmax(p).to(bf16) @ Kc[N,512].
  * base-2 LSE = (m + log(l)) / log(2)   [1/log(2) inlined as literal].
  * Empty / zero-length rows -> output 0, lse -inf (guarded, no div-by-zero).
No causal tile-skipping yet (that is a later candidate); every KV tile up to
kv_len is scanned.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mla_prefill_kernel(
    q_nope_ptr, q_pe_ptr, ckv_ptr, kpe_ptr, kv_indices_ptr,
    kv_beg_ptr, kv_len_ptr, q_abs_ptr,
    out_ptr, lse_ptr,
    sm_scale,
    stride_qn_t, stride_qn_h, stride_qn_d,
    stride_qp_t, stride_qp_h, stride_qp_d,
    stride_ckv_p, stride_ckv_d,
    stride_kpe_p, stride_kpe_d,
    stride_o_t, stride_o_h, stride_o_d,
    stride_lse_t, stride_lse_h,
    H: tl.constexpr, D_CKV: tl.constexpr, D_KPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)

    kv_beg = tl.load(kv_beg_ptr + pid)
    kv_len = tl.load(kv_len_ptr + pid)
    q_abs = tl.load(q_abs_ptr + pid)

    h_idx = tl.arange(0, H)
    d_ckv = tl.arange(0, D_CKV)
    d_kpe = tl.arange(0, D_KPE)

    # Load this query token's q_nope [H, D_CKV] and q_pe [H, D_KPE] (bf16).
    qn_ptrs = (q_nope_ptr + pid * stride_qn_t
               + h_idx[:, None] * stride_qn_h + d_ckv[None, :] * stride_qn_d)
    qn = tl.load(qn_ptrs)
    qp_ptrs = (q_pe_ptr + pid * stride_qp_t
               + h_idx[:, None] * stride_qp_h + d_kpe[None, :] * stride_qp_d)
    qp = tl.load(qp_ptrs)

    m_i = tl.full([H], -float("inf"), tl.float32)
    l_i = tl.zeros([H], tl.float32)
    acc = tl.zeros([H, D_CKV], tl.float32)

    for start_n in range(0, kv_len, BLOCK_N):
        n = start_n + tl.arange(0, BLOCK_N)
        n_mask = n < kv_len

        tok = tl.load(kv_indices_ptr + kv_beg + n, mask=n_mask, other=0)

        ckv_ptrs = (ckv_ptr + tok[:, None] * stride_ckv_p
                    + d_ckv[None, :] * stride_ckv_d)
        Kc = tl.load(ckv_ptrs, mask=n_mask[:, None], other=0.0)  # [BLOCK_N, 512] bf16
        kpe_ptrs = (kpe_ptr + tok[:, None] * stride_kpe_p
                    + d_kpe[None, :] * stride_kpe_d)
        Kp = tl.load(kpe_ptrs, mask=n_mask[:, None], other=0.0)  # [BLOCK_N, 64] bf16

        # QK (bf16 MMA, fp32 accumulate): [H, BLOCK_N]
        scores = tl.dot(qn, tl.trans(Kc))
        scores += tl.dot(qp, tl.trans(Kp))
        scores = scores * sm_scale

        # causal mask (per-row cutoff with prefix) + tail mask
        valid = (n[None, :] <= q_abs) & n_mask[None, :]
        scores = tl.where(valid, scores, -float("inf"))

        # online softmax update
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), Kc)
        m_i = m_new

    l_safe = l_i > 0
    out = tl.where(l_safe[:, None], acc / l_i[:, None], 0.0)
    # base-2 LSE; 1/ln(2) inlined as a local literal (Triton cannot read a
    # module-level Python global inside the jit body -- this was the c001 bug).
    lse = tl.where(l_safe, (m_i + tl.log(l_i)) * 1.4426950408889634, -float("inf"))

    o_ptrs = (out_ptr + pid * stride_o_t
              + h_idx[:, None] * stride_o_h + d_ckv[None, :] * stride_o_d)
    tl.store(o_ptrs, out.to(tl.bfloat16))
    lse_ptrs = lse_ptr + pid * stride_lse_t + h_idx * stride_lse_h
    tl.store(lse_ptrs, lse)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, H, D_CKV = q_nope.shape
    D_KPE = q_pe.shape[-1]
    device = q_nope.device
    batch = qo_indptr.shape[0] - 1

    output = torch.zeros((total_q, H, D_CKV), dtype=torch.bfloat16, device=device)
    lse = torch.full((total_q, H), -float("inf"), dtype=torch.float32, device=device)
    if total_q == 0 or batch <= 0:
        return output, lse

    if isinstance(sm_scale, torch.Tensor):
        sm_scale = float(sm_scale.item())
    else:
        sm_scale = float(sm_scale)

    # --- host-side per-query-token metadata (launch plumbing, allowed) ---
    qo_i64 = qo_indptr.to(torch.int64)
    kv_i64 = kv_indptr.to(torch.int64)
    q_lens = qo_i64[1:] - qo_i64[:-1]          # [batch]
    kv_lens = kv_i64[1:] - kv_i64[:-1]         # [batch]
    kv_begs = kv_i64[:-1]                       # [batch]
    q_starts = qo_i64[:-1]                      # [batch]

    seq_id = torch.repeat_interleave(
        torch.arange(batch, device=device, dtype=torch.int64), q_lens
    )                                           # [total_q]
    tok_global = torch.arange(total_q, device=device, dtype=torch.int64)
    i_local = tok_global - q_starts[seq_id]     # [total_q]

    kv_beg_per_tok = kv_begs[seq_id].to(torch.int32)
    kv_len_per_tok = kv_lens[seq_id].to(torch.int32)
    q_len_per_tok = q_lens[seq_id]
    q_abs = ((kv_lens[seq_id] - q_len_per_tok) + i_local).to(torch.int32)

    BLOCK_N = 64
    grid = (total_q,)
    _mla_prefill_kernel[grid](
        q_nope, q_pe, ckv_cache, kpe_cache, kv_indices,
        kv_beg_per_tok, kv_len_per_tok, q_abs,
        output, lse,
        sm_scale,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2),
        ckv_cache.stride(0), ckv_cache.stride(2),
        kpe_cache.stride(0), kpe_cache.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        H=H, D_CKV=D_CKV, D_KPE=D_KPE,
        BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return output, lse
