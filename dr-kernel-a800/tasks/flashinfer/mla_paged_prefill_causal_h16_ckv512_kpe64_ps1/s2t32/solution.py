import torch
import math
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    q_nope_ptr,       # *bf16, shape [Q_total, 16, 512]
    q_pe_ptr,         # *bf16, shape [Q_total, 16, 64]
    Kc_sel_ptr,       # *bf16, shape [kv_len, 512]
    Kp_sel_ptr,       # *bf16, shape [kv_len, 64]
    output_ptr,       # *bf16, shape [Q_total, 16, 512]
    lse_ptr,          # *fp32, shape [Q_total, 16]
    # meta-parameters (specialized per launch)
    q_abs,            # int32, absolute query index within q_nope/q_pe (used for output storage)
    q_len: tl.constexpr,       # number of queries in this batch element (compile-time per program)
    kv_len: tl.constexpr,      # number of selected KV tokens (compile-time per program)
    sm_scale: tl.constexpr,    # fp32 scaling factor
    ln2_inv: tl.constexpr,     # fp32 = 1.0 / ln(2.0)
    NUM_HEADS: tl.constexpr,   # 16
    HEAD_DIM_CKV: tl.constexpr, # 512
    HEAD_DIM_KPE: tl.constexpr, # 64
):
    # This kernel handles one specific query i within a batch element b.
    # q_abs is the global query index: q_abs = qo_indptr[b] + i.

    # Load qn[h, :] and qp[h, :] vectors for all heads h, as fp32
    for h in range(NUM_HEADS):
        qn = tl.load(q_nope_ptr + q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV), mask=tl.arange(0, HEAD_DIM_CKV) < HEAD_DIM_CKV, other=0.0).to(tl.float32)
        qp = tl.load(q_pe_ptr + q_abs * (NUM_HEADS * HEAD_DIM_KPE) + h * HEAD_DIM_KPE + tl.arange(0, HEAD_DIM_KPE), mask=tl.arange(0, HEAD_DIM_KPE) < HEAD_DIM_KPE, other=0.0).to(tl.float32)

        # Initialize logits per head
        logits = tl.full((kv_len,), -float("inf"), dtype=tl.float32)

        # Compute logits for each KV position j
        for j in range(kv_len):
            Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV), mask=tl.arange(0, HEAD_DIM_CKV) < HEAD_DIM_CKV, other=0.0).to(tl.float32)
            Kp_j = tl.load(Kp_sel_ptr + j * HEAD_DIM_KPE + tl.arange(0, HEAD_DIM_KPE), mask=tl.arange(0, HEAD_DIM_KPE) < HEAD_DIM_KPE, other=0.0).to(tl.float32)

            dot_qn = tl.sum(qn * Kc_j, axis=0)  # scalar
            dot_qp = tl.sum(qp * Kp_j, axis=0)  # scalar
            logits[j] = dot_qn + dot_qp

        # Scale
        logits = logits * sm_scale

        # Causal mask: prefix_len = kv_len - q_len
        prefix_len = kv_len - q_len
        valid_start = prefix_len + 1  # i starts at 0 in this program
        j_vec = tl.arange(0, kv_len)
        causal_mask = j_vec >= valid_start
        logits = tl.where(causal_mask, logits, -float("inf"))

        # logsumexp in log2
        m = tl.max(logits, axis=0)
        exp_logits = tl.exp(logits - m)
        sumexp = tl.sum(exp_logits, axis=0)
        lse_val = m + tl.log(sumexp) * ln2_inv  # scalar per head

        # Softmax over j
        exp_logits = tl.exp(logits - m)  # stable
        sumexp = tl.sum(exp_logits, axis=0)
        softmax = exp_logits / sumexp  # [kv_len]

        # Output: out[h, :] = sum_j softmax[j] * Kc_sel[j, :]
        out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        for j in range(kv_len):
            Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV), mask=tl.arange(0, HEAD_DIM_CKV) < HEAD_DIM_CKV, other=0.0).to(tl.float32)
            out_vec += softmax[j] * Kc_j

        # Store output vector for this query and head as bfloat16
        out_store = out_vec.to(tl.bfloat16)
        base_out = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        for k in range(HEAD_DIM_CKV):
            tl.store(output_ptr + base_out + k, out_store[k])

        # Store lse for this query and head as float32
        lse_base = q_abs * NUM_HEADS + h
        tl.store(lse_ptr + lse_base, lse_val)


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

        # Preselect Kc_sel and Kp_sel per batch element: Kc_sel[j, :] = ckv_cache[tok_idx, 0, :]
        for b in range(batch_size):
            # tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b+1]]
            tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]].to(torch.long).to(device)
            # Selected Kc and Kp for this batch element
            Kc_sel = ckv_cache[tok_idx].to(torch.bfloat16).contiguous()  # [kv_len, 512]
            Kp_sel = kpe_cache[tok_idx].to(torch.bfloat16).contiguous()  # [kv_len, 64]

            # Compute q_len and q_start for this batch element
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len_b = max(q_end - q_start, 0)

            # Launch Triton kernel: one program per query i in this batch element
            grid = (1, q_len_b)
            _forward_single_query_kernel[grid](
                q_nope, q_pe,
                Kc_sel, Kp_sel,
                output, lse,
                q_abs=q_start,  # one program per query; for each i we set q_abs to q_start + i by grid second dim loop via host
                q_len=q_len_b,
                kv_len=Kc_sel.shape[0],
                sm_scale=float(sm_scale),
                ln2_inv=float(ln2_inv),
                NUM_HEADS=16,
                HEAD_DIM_CKV=512,
                HEAD_DIM_KPE=64,
                num_warps=4,
                num_stages=2,
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
