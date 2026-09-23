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
    ln2_inv,        # fp32
    kv_len: tl.constexpr,      # number of selected KV tokens
    NUM_HEADS: tl.constexpr,   # 16
    HEAD_DIM_CKV: tl.constexpr, # 512
    HEAD_DIM_KPE: tl.constexpr  # 64
):
    # Process one query per program. q_abs identifies the query.
    for h in range(NUM_HEADS):
        # Load qn[h, :] -> [HEAD_DIM_CKV] (bf16 from q_nope), cast to fp32
        qn = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        for k in range(HEAD_DIM_CKV):
            ptr = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV + k
            val = tl.load(q_nope_ptr + ptr).to(tl.float32)
            qn[k] = val

        # Load qp[h, :] -> [HEAD_DIM_KPE] (bf16 from q_pe), cast to fp32
        qp = tl.zeros((HEAD_DIM_KPE,), dtype=tl.float32)
        for k in range(HEAD_DIM_KPE):
            ptr = q_abs * (NUM_HEADS * HEAD_DIM_KPE) + h * HEAD_DIM_KPE + k
            val = tl.load(q_pe_ptr + ptr).to(tl.float32)
            qp[k] = val

        # Compute logits[h, j] for j in [0, kv_len)
        logits_j = tl.zeros((kv_len,), dtype=tl.float32)
        for j in range(kv_len):
            # Kc_sel[j, :] -> [HEAD_DIM_CKV]
            Kc_j = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
            for k in range(HEAD_DIM_CKV):
                ptr = j * HEAD_DIM_CKV + k
                Kc_val = tl.load(Kc_sel_ptr + ptr).to(tl.float32)
                Kc_j[k] = Kc_val

            # Kp_sel[j, :] -> [HEAD_DIM_KPE]
            Kp_j = tl.zeros((HEAD_DIM_KPE,), dtype=tl.float32)
            for k in range(HEAD_DIM_KPE):
                ptr = j * HEAD_DIM_KPE + k
                Kp_val = tl.load(Kp_sel_ptr + ptr).to(tl.float32)
                Kp_j[k] = Kp_val

            # Dot-products
            dot_qn = 0.0
            for k in range(HEAD_DIM_CKV):
                dot_qn += qn[k] * Kc_j[k]
            dot_qp = 0.0
            for k in range(HEAD_DIM_KPE):
                dot_qp += qp[k] * Kp_j[k]

            logits_j[j] = dot_qn + dot_qp

        # Scale by sm_scale
        logits_j = logits_j * sm_scale

        # Apply causal mask: absolute position is query_abs_pos = (kv_len - q_len) + i.
        # Here q_len is 1 (per batch element), and i=0 for this program.
        valid_start = kv_len - 1 + 1  # prefix_len + i+1 = (kv_len - 1) + 1 = kv_len
        for j in range(kv_len):
            if j < valid_start:
                logits_j[j] = -float("inf")

        # logsumexp in log2
        m = tl.max(logits_j, axis=0)
        sumexp = 0.0
        for j in range(kv_len):
            sumexp += tl.exp(logits_j[j] - m)
        lse_val = (m + tl.log(sumexp) * ln2_inv)

        # Softmax over j (stable)
        softmax_j = tl.zeros((kv_len,), dtype=tl.float32)
        for j in range(kv_len):
            softmax_j[j] = tl.exp(logits_j[j] - m) / sumexp

        # Output: out[h, :] = sum_j softmax_j[j] * Kc_sel[j, :]
        out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        for j in range(kv_len):
            Kc_j = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
            for k in range(HEAD_DIM_CKV):
                ptr = j * HEAD_DIM_CKV + k
                Kc_val = tl.load(Kc_sel_ptr + ptr).to(tl.float32)
                Kc_j[k] = Kc_val
            out_vec += softmax_j[j] * Kc_j

        # Store output vector (bfloat16) to output[q_abs, h, :]
        out_store = out_vec.to(tl.bfloat16)
        base_out = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        for k in range(HEAD_DIM_CKV):
            tl.store(output_ptr + base_out + k, out_store[k])

        # Store lse[q_abs, h] as float32
        lse_base = q_abs * NUM_HEADS + h
        tl.store(lse_ptr + lse_base, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure shapes
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

        ln2_inv = 1.0 / math.log(2.0)

        # Iterate over batch elements and launch Triton kernel for each query
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start  # queries in this batch element

            # Gather token indices for this batch element
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            tok_idx = kv_indices[page_beg:page_end]  # [kv_len]

            # Gather selected Kc and Kp per batch element into temporary tensors (keep original dtype, convert inside kernel)
            Kc_sel = ckv_cache[tok_idx]       # [kv_len, 512], bfloat16
            Kp_sel = kpe_cache[tok_idx]       # [kv_len, 64], bfloat16

            # For each query in this batch element
            for i in range(q_len):
                q_abs = q_start + i
                grid = (1, 1)
                _forward_single_query_kernel[grid](
                    q_nope, q_pe, Kc_sel, Kp_sel, output, lse,
                    q_abs, float(sm_scale), float(ln2_inv),
                    kv_len=Kc_sel.shape[0], NUM_HEADS=16, HEAD_DIM_CKV=512, HEAD_DIM_KPE=64
                )

        return output, lse


def run(*args):
    return ModelNew()(*args)
