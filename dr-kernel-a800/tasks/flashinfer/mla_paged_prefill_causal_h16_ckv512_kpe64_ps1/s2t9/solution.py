import torch
import math
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    q_nope_ptr,           # *bf16, [Q_total, 16, 512], row-major
    q_pe_ptr,             # *bf16, [Q_total, 16, 64], row-major
    Kc_sel_ptr,           # *bf16, [kv_len, 512], row-major (selected from ckv_cache)
    Kp_sel_ptr,           # *bf16, [kv_len, 64], row-major (selected from kpe_cache)
    output_ptr,           # *bf16, [Q_total, 16, 512], row-major
    lse_ptr,              # *fp32, [Q_total, 16]
    # scalars and constexprs
    q_start,              # int32, start query index for this batch element
    sm_scale,             # fp32, scaling factor
    ln2_inv,              # fp32, 1.0 / ln(2.0)
    # meta-parameters
    Q_TOTAL: tl.constexpr,          # total number of queries (for bounds)
    q_len_const: tl.constexpr,      # number of queries in this batch element
    kv_len_const: tl.constexpr,     # number of selected KV tokens in this batch element
    NUM_HEADS: tl.constexpr,        # 16
    HEAD_DIM_CKV: tl.constexpr,     # 512
    HEAD_DIM_KPE: tl.constexpr,     # 64
):
    # Grid is (len_indptr-1, total_q). Decode program ids into batch b and query i.
    b = tl.program_id(0)  # batch element index (0 to len_indptr - 2)
    i = tl.program_id(1)  # query index within this batch element

    # Bounds check (defensive)
    if b >= Q_TOTAL or i >= q_len_const:
        return

    q_abs = q_start + i  # absolute query index in global q_nope/q_pe

    # Process each head h separately
    for h in tl.static_range(0, NUM_HEADS):
        # Load qn[h, :] and qp[h, :] vectors from q_nope and q_pe for this query
        # q_nope is [Q_total, 16, 512], row-major => offset = q_abs * (16*512) + h * 512
        qn_offset = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        qn_vec = tl.load(q_nope_ptr + qn_offset + tl.arange(0, HEAD_DIM_CKV), mask=tl.arange(0, HEAD_DIM_CKV) < HEAD_DIM_CKV, other=0.0).to(tl.float32)

        qp_offset = q_abs * (NUM_HEADS * HEAD_DIM_KPE) + h * HEAD_DIM_KPE
        qp_vec = tl.load(q_pe_ptr + qp_offset + tl.arange(0, HEAD_DIM_KPE), mask=tl.arange(0, HEAD_DIM_KPE) < HEAD_DIM_KPE, other=0.0).to(tl.float32)

        # Initialize logits for this head
        logits_h = tl.zeros((kv_len_const,), dtype=tl.float32)

        # Compute logits[h, j] for all j in [0, kv_len)
        for j in tl.static_range(0, kv_len_const):
            # Load Kc_sel[j, :] and Kp_sel[j, :]
            Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV), mask=tl.arange(0, HEAD_DIM_CKV) < HEAD_DIM_CKV, other=0.0).to(tl.float32)
            Kp_j = tl.load(Kp_sel_ptr + j * HEAD_DIM_KPE + tl.arange(0, HEAD_DIM_KPE), mask=tl.arange(0, HEAD_DIM_KPE) < HEAD_DIM_KPE, other=0.0).to(tl.float32)
            # Compute dot products
            dot_qn = tl.sum(qn_vec * Kc_j, axis=0)
            dot_qp = tl.sum(qp_vec * Kp_j, axis=0)
            logits_h[j] = dot_qn + dot_qp

        # Scale logits
        logits_h = logits_h * sm_scale

        # Apply causal mask: absolute position is query_abs_pos = prefix_len + i, where prefix_len = kv_len_const - q_len_const
        prefix_len = kv_len_const - q_len_const
        query_abs_pos = prefix_len + i + 1
        j_vec = tl.arange(0, kv_len_const)
        causal_mask = j_vec >= query_abs_pos
        logits_h = tl.where(causal_mask, logits_h, -float("inf"))

        # Compute logsumexp in log2
        m = tl.max(logits_h, axis=0)
        sumexp = tl.sum(tl.exp(logits_h - m), axis=0)
        lse_val = m + (tl.log(sumexp) * ln2_inv)  # scalar fp32
        # Store lse[q_abs, h] as float32
        lse_base = q_abs * NUM_HEADS + h
        tl.store(lse_ptr + lse_base, lse_val)

        # Softmax over j
        exp_logits = tl.exp(logits_h - m)
        sumexp = tl.sum(exp_logits, axis=0)
        softmax = exp_logits / sumexp  # [kv_len_const] vector

        # Output: out[h, :] = sum_j softmax[j] * Kc_sel[j, :]
        out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        for j in tl.static_range(0, kv_len_const):
            Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV), mask=tl.arange(0, HEAD_DIM_CKV) < HEAD_DIM_CKV, other=0.0).to(tl.float32)
            out_vec += softmax[j] * Kc_j

        # Store output vector for this query and head as bfloat16
        out_store = out_vec.to(tl.bfloat16)
        base_out = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        for k in tl.static_range(0, HEAD_DIM_CKV):
            tl.store(output_ptr + base_out + k, out_store[k])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Validate shapes
        assert q_nope.shape[1] == 16 and q_nope.shape[2] == 512, "q_nope must be [Q_total, 16, 512]"
        assert q_pe.shape[1] == 16 and q_pe.shape[2] == 64, "q_pe must be [Q_total, 16, 64]"
        assert ckv_cache.shape[1] == 1 and ckv_cache.shape[2] == 512, "ckv_cache must be [num_pages, 1, 512]"
        assert kpe_cache.shape[1] == 1 and kpe_cache.shape[2] == 64, "kpe_cache must be [num_pages, 1, 64]"
        assert qo_indptr.dim() == 1 and kv_indptr.dim() == 1, "indptrs must be 1D"

        device = q_nope.device
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[2]
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1

        # Output and LSE buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Precompute ln2_inv (host code uses only scalar math)
        ln2_inv = 1.0 / math.log(2.0)

        # For each batch element, gather tok_idx and select Kc_sel/Kp_sel
        for b in range(batch_size):
            # Compute tok_idx for this batch element
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            tok_idx = kv_indices[page_beg:page_end].to(torch.long)  # [kv_len]

            # Select Kc_sel and Kp_sel from caches
            Kc_sel = ckv_cache[tok_idx]  # [kv_len, 512], bfloat16
            Kp_sel = kpe_cache[tok_idx]  # [kv_len, 64], bfloat16

            # q_start for this batch element
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            # Launch Triton kernel: one program per query i
            grid = (batch_size, total_q)
            _forward_single_query_kernel[grid](
                q_nope, q_pe, Kc_sel, Kp_sel, output, lse,
                q_start, float(sm_scale), float(ln2_inv),
                total_q, q_len, (page_end - page_beg), 16, 512, 64,
                num_warps=4, num_stages=2
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
