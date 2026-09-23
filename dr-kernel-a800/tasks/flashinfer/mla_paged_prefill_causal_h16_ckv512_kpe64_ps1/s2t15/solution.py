import torch
import math
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    # Input pointers (original dtype)
    q_nope_ptr,       # *bf16, shape [Q_total, 16, 512], row-major
    q_pe_ptr,         # *bf16, shape [Q_total, 16, 64], row-major
    Kc_sel_ptr,       # *bf16, shape [kv_len, 512], row-major
    Kp_sel_ptr,       # *bf16, shape [kv_len, 64], row-major
    output_ptr,       # *bf16, shape [Q_total, 16, 512], row-major
    lse_ptr,          # *fp32, shape [Q_total, 16]
    # runtime scalar: start query index for this batch element
    q_start,          # int32
    # constexpr meta-parameters
    q_len: tl.constexpr,    # number of queries in this batch element (compile-time for this program)
    kv_len: tl.constexpr,   # number of selected KV tokens (compile-time for this program)
    sm_scale: tl.constexpr,         # fp32 scaling factor
    ln2_inv: tl.constexpr,          # fp32 = 1.0 / ln(2.0)
    NUM_HEADS: tl.constexpr,        # 16
    HEAD_DIM_CKV: tl.constexpr,     # 512
    HEAD_DIM_KPE: tl.constexpr,     # 64
):
    # This program handles one specific (batch element, query) pair.
    # Grid is (len_indptr-1, q_len); q_start and i (second grid dim) are provided.
    i = tl.program_id(1)
    q_abs = q_start + i

    # Prepare head and dim vectors
    h_idx = tl.arange(0, NUM_HEADS)           # [16]
    k_ckv = tl.arange(0, HEAD_DIM_CKV)        # [512]
    k_kpe = tl.arange(0, HEAD_DIM_KPE)        # [64]
    j_vec = tl.arange(0, kv_len)              # [kv_len]

    # Load qn[h, :] and qp[h, :] from q_nope and q_pe (cast to float32 for math)
    qn = tl.zeros((NUM_HEADS, HEAD_DIM_CKV), dtype=tl.float32)
    for h in range(NUM_HEADS):
        qn[h, :] = tl.load(q_nope_ptr + q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV + k_ckv).to(tl.float32)
    qp = tl.zeros((NUM_HEADS, HEAD_DIM_KPE), dtype=tl.float32)
    for h in range(NUM_HEADS):
        qp[h, :] = tl.load(q_pe_ptr + q_abs * (NUM_HEADS * HEAD_DIM_KPE) + h * HEAD_DIM_KPE + k_kpe).to(tl.float32)

    # Load Kc_sel and Kp_sel as float32
    Kc_sel = tl.zeros((kv_len, HEAD_DIM_CKV), dtype=tl.float32)
    Kp_sel = tl.zeros((kv_len, HEAD_DIM_KPE), dtype=tl.float32)
    for j in range(kv_len):
        Kc_sel[j, :] = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + k_ckv, mask=k_ckv < HEAD_DIM_CKV, other=0.0).to(tl.float32)
        Kp_sel[j, :] = tl.load(Kp_sel_ptr + j * HEAD_DIM_KPE + k_kpe, mask=k_kpe < HEAD_DIM_KPE, other=0.0).to(tl.float32)

    # Compute logits per head: logits[h, j] = sum_k qn[h,k]*Kc_sel[j,k] + sum_k qp[h,k]*Kp_sel[j,k]
    logits = tl.zeros((NUM_HEADS, kv_len), dtype=tl.float32)
    for j in range(kv_len):
        # qn[h, :] dot Kc_sel[j, :]
        dot_qn = tl.sum(qn * Kc_sel[j, None, :], axis=1)  # [NUM_HEADS]
        # qp[h, :] dot Kp_sel[j, :]
        dot_qp = tl.sum(qp * Kp_sel[j, None, :], axis=1)  # [NUM_HEADS]
        logits[:, j] = dot_qn + dot_qp

    # Scale
    logits = logits * sm_scale

    # Apply causal mask: j >= prefix_len + i + 1, where prefix_len = kv_len - q_len
    prefix_len = kv_len - q_len
    valid_start = prefix_len + i + 1
    causal_mask = j_vec >= valid_start
    logits = tl.where(causal_mask[None, :], logits, -float("inf"))

    # Logsumexp in log2
    m = tl.max(logits, axis=1)               # [NUM_HEADS]
    exp_logits = tl.exp(logits - m[:, None])  # [NUM_HEADS, kv_len]
    sumexp = tl.sum(exp_logits, axis=1)      # [NUM_HEADS]
    lse_vals = m + tl.log(sumexp) * ln2_inv  # per head, in log2
    # Store lse[q_abs, h] as float32
    for h in range(NUM_HEADS):
        tl.store(lse_ptr + q_abs * NUM_HEADS + h, lse_vals[h])

    # Softmax over j (stable)
    logits = logits - m[:, None]
    exp_logits = tl.exp(logits)              # [NUM_HEADS, kv_len]
    sumexp = tl.sum(exp_logits, axis=1)[:, None]  # [NUM_HEADS, 1]
    softmax = exp_logits / sumexp            # [NUM_HEADS, kv_len]

    # Output: out[h, :] = sum_j softmax[h, j] * Kc_sel[j, :]
    out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
    for h in range(NUM_HEADS):
        for j in range(kv_len):
            Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + k_ckv, mask=k_ckv < HEAD_DIM_CKV, other=0.0).to(tl.float32)
            out_vec += softmax[h, j] * Kc_j
        # Store as bfloat16
        out_store = out_vec.to(tl.bfloat16)
        base_out = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        for k in range(HEAD_DIM_CKV):
            tl.store(output_ptr + base_out + k, out_store[k])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Assertions matching original assumptions
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

        # Iterate over batch elements; select tok_idx and prepare K_sel
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len_b = q_end - q_start

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            kv_len_b = page_end - page_beg

            if q_len_b == 0 or kv_len_b == 0:
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.long).to(device)

            Kc_sel = ckv_cache[tok_idx].to(torch.bfloat16).contiguous()  # [kv_len_b, 512]
            Kp_sel = kpe_cache[tok_idx].to(torch.bfloat16).contiguous()  # [kv_len_b, 64]

            # Launch Triton kernel: one program per query
            grid = (batch_size, q_len_b)
            _forward_single_query_kernel[grid](
                q_nope, q_pe, Kc_sel, Kp_sel, output, lse,
                q_start,  # runtime scalar start index
                q_len_b, kv_len_b,
                sm_scale, ln2_inv,
                NUM_HEADS=16, HEAD_DIM_CKV=512, HEAD_DIM_KPE=64,
                num_warps=4, num_stages=2
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
