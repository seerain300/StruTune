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
    # runtime scalar for batch start
    q_start,              # int32
    # constexpr meta-parameters
    sm_scale: tl.constexpr,          # fp32 scaling factor
    ln2_inv: tl.constexpr,           # fp32 = 1.0 / ln(2.0)
    q_len_const: tl.constexpr,       # number of queries in this batch element (unused, but kept for clarity)
    kv_len_const: tl.constexpr,      # number of selected KV tokens
    NUM_HEADS: tl.constexpr,         # 16
    HEAD_DIM_CKV: tl.constexpr,      # 512
    HEAD_DIM_KPE: tl.constexpr,      # 64
):
    # Grid: (len_indptr-1, total_q). Decode program ids.
    b = tl.program_id(0)  # batch element index
    i = tl.program_id(1)  # query index within this batch element

    q_abs = q_start + i  # absolute query index in global q_nope/q_pe

    # Loop over heads; we tile heads in the second grid dimension but here we use a simple loop.
    for h in range(NUM_HEADS):
        # Load qn[h, :] and qp[h, :] for this query and head
        k_vec = tl.arange(0, HEAD_DIM_CKV)  # for Kc
        qn_vec = tl.load(q_nope_ptr + q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV + k_vec, mask=k_vec < HEAD_DIM_CKV, other=0.0)
        k_vec_kp = tl.arange(0, HEAD_DIM_KPE)  # for Kp
        qp_vec = tl.load(q_pe_ptr + q_abs * (NUM_HEADS * HEAD_DIM_KPE) + h * HEAD_DIM_KPE + k_vec_kp, mask=k_vec_kp < HEAD_DIM_KPE, other=0.0)

        # Cast to fp32 for math
        qn_vec = qn_vec.to(tl.float32)
        qp_vec = qp_vec.to(tl.float32)

        # Initialize logits for this head
        logits_h = tl.zeros((kv_len_const,), dtype=tl.float32)

        # Compute logits[h, j] = qn[h]·Kc[j] + qp[h]·Kp[j]
        for j in tl.static_range(0, kv_len_const):
            # Load Kc_sel[j, :] and Kp_sel[j, :]
            Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + k_vec, mask=k_vec < HEAD_DIM_CKV, other=0.0).to(tl.float32)
            Kp_j = tl.load(Kp_sel_ptr + j * HEAD_DIM_KPE + k_vec_kp, mask=k_vec_kp < HEAD_DIM_KPE, other=0.0).to(tl.float32)

            # Dot products
            dot_qn = tl.sum(qn_vec * Kc_j, axis=0)
            dot_qp = tl.sum(qp_vec * Kp_j, axis=0)

            logits_h[j] = dot_qn + dot_qp

        # Scale
        logits_h = logits_h * sm_scale

        # Apply causal mask: j >= prefix_len + i + 1, where prefix_len = kv_len_const - q_len_const
        prefix_len = kv_len_const - q_len_const  # typically 0 if q_len == kv_len
        valid_start = prefix_len + i + 1
        j_vec = tl.arange(0, kv_len_const)
        causal_mask = j_vec >= valid_start
        logits_h = tl.where(causal_mask, logits_h, -float("inf"))

        # logsumexp in log2: lse[h] = max + log(sum exp(logits - max)) * ln2_inv
        m = tl.max(logits_h, axis=0)
        sumexp = tl.sum(tl.exp(logits_h - m), axis=0)
        lse_val = m + (tl.log(sumexp) * ln2_inv)  # scalar fp32

        # Store lse[q_abs, h] as float32
        lse_base = q_abs * NUM_HEADS + h
        tl.store(lse_ptr + lse_base, lse_val)

        # Softmax
        exp_logits = tl.exp(logits_h - m)
        sumexp_soft = tl.sum(exp_logits, axis=0)
        softmax = exp_logits / sumexp_soft  # [kv_len_const]

        # Output: out[h, :] = sum_j softmax[j] * Kc_sel[j, :]
        out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        for j in tl.static_range(0, kv_len_const):
            Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + k_vec, mask=k_vec < HEAD_DIM_CKV, other=0.0).to(tl.float32)
            out_vec += softmax[j] * Kc_j

        # Store output vector for this query and head as bfloat16
        out_store = out_vec.to(tl.bfloat16)
        base_out = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        for k in tl.static_range(0, HEAD_DIM_CKV):
            tl.store(output_ptr + base_out + k, out_store[k])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Shape assertions
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

        # Precompute ln2_inv
        ln2_inv = 1.0 / math.log(2.0)

        # Prepare Kc_sel and Kp_sel per batch element using kv_indptr and kv_indices
        # For each b: tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b+1]]
        for b in range(batch_size):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            if start >= end:
                continue
            tok_idx = kv_indices[start:end].to(torch.int64)  # indices into cache
            Kc_sel = ckv_cache[tok_idx]  # [kv_len, 512], bfloat16
            Kp_sel = kpe_cache[tok_idx]  # [kv_len, 64], bfloat16

            # q_len for this b
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = max(0, q_end - q_start)
            kv_len = end - start

            # Launch Triton kernel once per query
            grid = (batch_size, total_q)
            _forward_single_query_kernel[grid](
                q_nope, q_pe, Kc_sel, Kp_sel, output, lse,
                q_start,
                sm_scale, ln2_inv,
                total_q, q_len, kv_len,
                num_qo_heads, head_dim_ckv, head_dim_kpe,
                num_warps=4, num_stages=2
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
