import torch
import math
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    # Input pointers (original dtype), we'll cast to fp32 for math
    q_nope_ptr,       # *bf16, [Q_total, 16, 512]
    q_pe_ptr,         # *bf16, [Q_total, 16, 64]
    Kc_sel_ptr,       # *bf16, [kv_len, 512]
    Kp_sel_ptr,       # *bf16, [kv_len, 64]
    output_ptr,       # *bf16, [Q_total, 16, 512]
    lse_ptr,          # *fp32, [Q_total, 16]
    # runtime scalar
    q_start,          # int32
    # constexpr meta-parameters
    q_len: tl.constexpr,     # number of queries in this batch element (meta only, grid controls it)
    kv_len: tl.constexpr,    # number of KV tokens selected for this batch element
    sm_scale: tl.constexpr,  # fp32 scaling factor
    ln2_inv: tl.constexpr,   # fp32 = 1 / ln(2)
    NUM_HEADS: tl.constexpr, # 16
    HEAD_DIM_CKV: tl.constexpr, # 512
    HEAD_DIM_KPE: tl.constexpr, # 64
):
    # Grid: (len_indptr-1, q_len). Decode b and i
    b = tl.program_id(0)
    i = tl.program_id(1)
    q_abs = q_start + i

    # Compute prefix_len for causal mask
    prefix_len = kv_len - q_len
    valid_start = prefix_len + i + 1

    # Iterate over heads and compute everything per head
    for h in range(NUM_HEADS):
        # Load qn[h, :] and qp[h, :]
        k_vec = tl.arange(0, HEAD_DIM_CKV)
        base = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        qn = tl.load(q_nope_ptr + base + k_vec, mask=k_vec < HEAD_DIM_CKV, other=0.0).to(tl.float32)  # [512]

        kpe_vec = tl.arange(0, HEAD_DIM_KPE)
        base_pe = q_abs * (NUM_HEADS * HEAD_DIM_KPE) + h * HEAD_DIM_KPE
        qp = tl.load(q_pe_ptr + base_pe + kpe_vec, mask=kpe_vec < HEAD_DIM_KPE, other=0.0).to(tl.float32)  # [64]

        # Initialize logits and output accumulator
        logits = tl.zeros((kv_len,), dtype=tl.float32)
        out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)

        # Compute logits[j] for all j in [0, kv_len)
        for j in range(kv_len):
            k_vec2 = tl.arange(0, HEAD_DIM_CKV)
            Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + k_vec2, mask=k_vec2 < HEAD_DIM_CKV, other=0.0).to(tl.float32)  # [512]
            kpe_vec2 = tl.arange(0, HEAD_DIM_KPE)
            Kp_j = tl.load(Kp_sel_ptr + j * HEAD_DIM_KPE + kpe_vec2, mask=kpe_vec2 < HEAD_DIM_KPE, other=0.0).to(tl.float32)  # [64]

            # Dot products
            dot_qn = tl.sum(qn * Kc_j, axis=0)  # scalar
            dot_qp = tl.sum(qp * Kp_j, axis=0)  # scalar
            logits[j] = dot_qn + dot_qp

        # Scale
        logits = logits * sm_scale

        # Causal mask: j >= valid_start -> keep logits, else -inf
        j_vec = tl.arange(0, kv_len)
        causal_mask = j_vec >= valid_start
        logits = tl.where(causal_mask, logits, -float("inf"))

        # logsumexp in log2
        m = tl.max(logits, axis=0)  # scalar
        sumexp = tl.sum(tl.exp(logits - m), axis=0)  # scalar
        lse_val = m + tl.log(sumexp) * ln2_inv  # scalar per head

        # Softmax over j
        logits = logits - m
        exp_logits = tl.exp(logits)
        sumexp = tl.sum(exp_logits, axis=0)  # scalar
        softmax = exp_logits / sumexp  # [kv_len] vector

        # Output: out[h, :] = sum_j softmax[j] * Kc_sel[j, :]
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
        # Shape assertions matching original assumptions
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


def run(*args):
    return ModelNew()(*args)
