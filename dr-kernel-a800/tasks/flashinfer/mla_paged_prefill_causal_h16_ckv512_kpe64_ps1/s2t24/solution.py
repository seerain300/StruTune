import torch
import math
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    q_nope_ptr,     # *bf16, [Q_total, 16, 512]
    q_pe_ptr,       # *bf16, [Q_total, 16, 64]
    Kc_sel_ptr,     # *bf16, [kv_len, 512]
    Kp_sel_ptr,     # *bf16, [kv_len, 64]
    output_ptr,     # *bf16, [Q_total, 16, 512]
    lse_ptr,        # *fp32, [Q_total, 16]
    q_abs,          # int32, absolute query index
    sm_scale,       # fp32
    ln2_inv,        # fp32, 1.0 / ln(2.0)
    q_len: tl.constexpr,        # number of queries in this batch element (assumed 1 per program)
    kv_len: tl.constexpr,       # number of selected KV tokens
    NUM_HEADS: tl.constexpr,    # 16
    HEAD_DIM_CKV: tl.constexpr, # 512
    HEAD_DIM_KPE: tl.constexpr  # 64
):
    # Compute per-head logits and produce output for one query i identified by q_abs.
    # We specialize on kv_len and q_len. In this implementation, we handle q_len=1 per kernel program.

    # For robustness, we loop over heads (h in [0, NUM_HEADS))
    for h in range(NUM_HEADS):
        # Load qn[h, :] and qp[h, :]
        qn = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        for k in range(HEAD_DIM_CKV):
            ptr = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV + k
            val = tl.load(q_nope_ptr + ptr).to(tl.float32)
            qn[k] = val

        qp = tl.zeros((HEAD_DIM_KPE,), dtype=tl.float32)
        for k in range(HEAD_DIM_KPE):
            ptr = q_abs * (NUM_HEADS * HEAD_DIM_KPE) + h * HEAD_DIM_KPE + k
            val = tl.load(q_pe_ptr + ptr).to(tl.float32)
            qp[k] = val

        # Compute logits_j = [kv_len], initialize to zeros
        logits_j = tl.zeros((kv_len,), dtype=tl.float32)

        # Accumulate dot-products for each j in [0, kv_len)
        for j in range(kv_len):
            # dot(qn, Kc_sel[j, :])
            dot_qn = 0.0
            for k in range(HEAD_DIM_CKV):
                ptr = j * HEAD_DIM_CKV + k
                Kc_val = tl.load(Kc_sel_ptr + ptr).to(tl.float32)
                dot_qn += qn[k] * Kc_val

            # dot(qp, Kp_sel[j, :])
            dot_qp = 0.0
            for k in range(HEAD_DIM_KPE):
                ptr = j * HEAD_DIM_KPE + k
                Kp_val = tl.load(Kp_sel_ptr + ptr).to(tl.float32)
                dot_qp += qp[k] * Kp_val

            logits_j[j] = dot_qn + dot_qp

        # Scale logits
        logits_j = logits_j * sm_scale

        # Apply causal mask: for absolute position query_abs_pos = prefix_len + i, i=0 here
        # Since q_len is constexpr and we assume one query per program, prefix_len = kv_len - q_len
        prefix_len = kv_len - q_len
        query_abs_pos = prefix_len  # i=0; if multiple queries, grid handles i separately
        for j in range(kv_len):
            if j < query_abs_pos:
                logits_j[j] = -float("inf")

        # logsumexp in base-2
        m = logits_j[0]  # initialize; Triton supports tl.max but we can use a loop to find max
        for j in range(1, kv_len):
            if logits_j[j] > m:
                m = logits_j[j]
        sumexp = 0.0
        for j in range(kv_len):
            sumexp += tl.exp(logits_j[j] - m)
        lse_val = (m + tl.log(sumexp)) * ln2_inv  # per-head lse (base-2)

        # Store lse[q_abs, h] as float32
        tl.store(lse_ptr + q_abs * NUM_HEADS + h, lse_val)

        # Softmax over j
        sumexp_total = 0.0
        for j in range(kv_len):
            sumexp_total += tl.exp(logits_j[j] - m)
        for j in range(kv_len):
            logits_j[j] = tl.exp(logits_j[j] - m) / sumexp_total

        # Output: out[h, :] = sum_j softmax_j[j] * Kc_sel[j, :]
        out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        for j in range(kv_len):
            Kc_j = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
            for k in range(HEAD_DIM_CKV):
                ptr = j * HEAD_DIM_CKV + k
                Kc_val = tl.load(Kc_sel_ptr + ptr).to(tl.float32)
                Kc_j[k] = Kc_val
            out_vec += logits_j[j] * Kc_j  # note: logits_j[j] is softmax[h, j]

        # Store output as bfloat16 at output[q_abs, h, :]
        out_store = out_vec.to(tl.bfloat16)
        base_out = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        for k in range(HEAD_DIM_CKV):
            tl.store(output_ptr + base_out + k, out_store[k])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Validate shapes (original assumptions)
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

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Precompute ln2_inv
        ln2_inv = 1.0 / math.log(2.0)

        # Loop over batch elements; for each element, if q_len > 0, launch kernel once (q_len is assumed 1 per program).
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            q_len_b = q_end - q_start
            kv_len_b = kv_end - kv_start

            # Select tok_idx for this batch element
            tok_idx = kv_indices[kv_start:kv_end].to(torch.int32)

            # Gather selected Kc and Kp (keep original dtype for storage; cast to fp32 in kernel)
            # ckv_cache is [num_pages, 1, 512], reduce 1-dim to [num_pages, 512]
            Kc_sel = ckv_cache[tok_idx].to(torch.bfloat16)  # [kv_len, 512]
            Kp_sel = kpe_cache[tok_idx].to(torch.bfloat16)  # [kv_len, 64]

            # If q_len_b == 0, skip
            if q_len_b == 0:
                continue

            # Launch kernel once per query i; grid is (1, q_len_b)
            # Triton requires static constexpr for loop bounds; we set q_len and kv_len as constexpr meta-parameters.
            # Since our grid second dimension is q_len_b, and Triton programs handle one query per kernel, we can pass q_len_b and kv_len_b as constexpr.
            grid = (1, q_len_b)
            _forward_single_query_kernel[grid](
                q_nope, q_pe,
                Kc_sel, Kp_sel,
                output, lse,
                q_abs=q_start,  # start query index for this batch element
                sm_scale=float(sm_scale),
                ln2_inv=float(ln2_inv),
                q_len=q_len_b, kv_len=kv_len_b,
                NUM_HEADS=16, HEAD_DIM_CKV=512, HEAD_DIM_KPE=64
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
