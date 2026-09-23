import torch
import math
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    # Pointers to input tensors
    q_nope_ptr,        # *bf16, [Q_total, 16, 512]
    q_pe_ptr,          # *bf16, [Q_total, 16, 64]
    Kc_sel_ptr,        # *bf16, [kv_len, 512]
    Kp_sel_ptr,        # *bf16, [kv_len, 64]
    output_ptr,        # *bf16, [Q_total, 16, 512]
    lse_ptr,           # *fp32, [Q_total, 16]
    # runtime scalars
    q_start,           # int32: start index of queries for this batch element
    # meta-parameters (constexpr for this program)
    q_len: tl.constexpr,        # number of queries in this batch element
    kv_len: tl.constexpr,       # number of selected KV tokens
    sm_scale: tl.constexpr,     # fp32 scaling factor
    ln2_inv: tl.constexpr,      # fp32 = 1.0 / ln(2.0)
    NUM_HEADS: tl.constexpr,    # number of heads (16)
    HEAD_DIM_CKV: tl.constexpr, # head dim for ckv (512)
    HEAD_DIM_KPE: tl.constexpr  # head dim for kpe (64)
):
    # Grid is (len_indptr - 1, q_len), so program id along second dim gives i.
    i = tl.program_id(1)
    b = tl.program_id(0)
    q_abs = q_start + i  # absolute query index in global q_nope

    # Load qn[h, :] and qp[h, :] for all heads h
    # q_nope_ptr layout: [Q_total, NUM_HEADS, HEAD_DIM_CKV], contiguous
    # q_pe_ptr layout: [Q_total, NUM_HEADS, HEAD_DIM_KPE], contiguous
    for h in range(NUM_HEADS):
        base_qn = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        qn = tl.load(q_nope_ptr + base_qn + tl.arange(0, HEAD_DIM_CKV), mask=tl.arange(0, HEAD_DIM_CKV) < HEAD_DIM_CKV, other=0.0).to(tl.float32)
        base_qp = q_abs * (NUM_HEADS * HEAD_DIM_KPE) + h * HEAD_DIM_KPE
        qp = tl.load(q_pe_ptr + base_qp + tl.arange(0, HEAD_DIM_KPE), mask=tl.arange(0, HEAD_DIM_KPE) < HEAD_DIM_KPE, other=0.0).to(tl.float32)

        # Initialize logits for this head
        logits = tl.zeros((kv_len,), dtype=tl.float32)

        # Compute dot products with selected Kc_sel and Kp_sel and accumulate
        for j in range(kv_len):
            # Load Kc_sel[j, :] and Kp_sel[j, :]
            Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV), mask=tl.arange(0, HEAD_DIM_CKV) < HEAD_DIM_CKV, other=0.0).to(tl.float32)
            Kp_j = tl.load(Kp_sel_ptr + j * HEAD_DIM_KPE + tl.arange(0, HEAD_DIM_KPE), mask=tl.arange(0, HEAD_DIM_KPE) < HEAD_DIM_KPE, other=0.0).to(tl.float32)
            dot_qn = tl.sum(qn * Kc_j, axis=0)
            dot_qp = tl.sum(qp * Kp_j, axis=0)
            logits[j] = dot_qn + dot_qp

        # Scale
        logits = logits * sm_scale

        # Causal mask: valid j >= prefix_len + i + 1, prefix_len = kv_len - q_len
        prefix_len = kv_len - q_len
        valid_start = prefix_len + i + 1
        j_vec = tl.arange(0, kv_len)
        causal_mask = j_vec >= valid_start
        logits = tl.where(causal_mask, logits, -float("inf"))

        # logsumexp in log2
        m = tl.max(logits, axis=0)
        sumexp = tl.sum(tl.exp(logits - m), axis=0)
        lse_val = m + tl.log(sumexp) * ln2_inv  # per-head lse in log2
        # Store lse[q_abs, h]
        lse_base = q_abs * NUM_HEADS + h
        tl.store(lse_ptr + lse_base, lse_val)

        # Softmax over j
        logits = logits - m
        exp_logits = tl.exp(logits)
        sumexp = tl.sum(exp_logits, axis=0)
        softmax = exp_logits / sumexp  # [kv_len], per head

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


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Shapes as in original assumptions
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

        # For each batch element, select Kc_sel and Kp_sel from caches based on kv_indices and kv_indptr
        for b in range(batch_size):
            # Determine q_len and kv_len for this batch element
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len_b = max(0, q_end - q_start)

            # Compute tok_idx for this batch element: tokens used as KV
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            kv_len_b = max(0, page_end - page_beg)
            tok_idx = kv_indices[page_beg:page_end]  # [kv_len_b] int32

            # Select Kc_sel and Kp_sel: shape [kv_len_b, head_dim] in bfloat16
            Kc_sel = ckv_cache[tok_idx].to(torch.bfloat16)  # [kv_len_b, 512]
            Kp_sel = kpe_cache[tok_idx].to(torch.bfloat16)  # [kv_len_b, 64]

            # Launch Triton kernel for each query i in this batch element
            # Grid: (len_indptr - 1, q_len_b)
            grid = (1, q_len_b)
            _forward_single_query_kernel[grid](
                q_nope, q_pe, Kc_sel, Kp_sel, output, lse,
                q_start,
                q_len=q_len_b, kv_len=kv_len_b,
                sm_scale=float(sm_scale), ln2_inv=float(ln2_inv),
                NUM_HEADS=16, HEAD_DIM_CKV=512, HEAD_DIM_KPE=64,
                num_warps=4, num_stages=2
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
