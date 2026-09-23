import torch
import math
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    q_nope_ptr,        # *bf16, [Q_total, 16, 512]
    q_pe_ptr,          # *bf16, [Q_total, 16, 64]
    Kc_sel_ptr,        # *bf16, [kv_len, 512]
    Kp_sel_ptr,        # *bf16, [kv_len, 64]
    output_ptr,        # *bf16, [Q_total, 16, 512]
    lse_ptr,           # *fp32, [Q_total, 16]
    q_start,           # int32, start index in q_nope for this batch element
    sm_scale,          # fp32, scaling factor
    ln2_inv,           # fp32, 1.0 / ln(2.0)
    NUM_HEADS: tl.constexpr,      # 16
    HEAD_DIM_CKV: tl.constexpr,   # 512
    HEAD_DIM_KPE: tl.constexpr,   # 64
    q_len: tl.constexpr,          # number of queries in this batch element
    kv_len: tl.constexpr,         # number of selected KV tokens
):
    # Grid is (b, i) where b in [0, len_indptr-2], i in [0, q_len-1]
    i = tl.program_id(1)
    b = tl.program_id(0)
    q_abs = q_start + i

    # Load qn and qp for all heads (rows) as float32
    qn_rows = []  # list of [512] vectors
    qp_rows = []  # list of [64] vectors
    for h in range(NUM_HEADS):
        base_qn = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        qn_row = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        for k in range(HEAD_DIM_CKV):
            val = tl.load(q_nope_ptr + base_qn + k).to(tl.float32)
            qn_row[k] = val
        base_qp = q_abs * (NUM_HEADS * HEAD_DIM_KPE) + h * HEAD_DIM_KPE
        qp_row = tl.zeros((HEAD_DIM_KPE,), dtype=tl.float32)
        for k in range(HEAD_DIM_KPE):
            val = tl.load(q_pe_ptr + base_qp + k).to(tl.float32)
            qp_row[k] = val
        qn_rows.append(qn_row)
        qp_rows.append(qp_row)

    # Compute logits per head: [NUM_HEADS, kv_len]
    logits = tl.zeros((NUM_HEADS, kv_len), dtype=tl.float32)
    for j in range(kv_len):
        # Load Kc_sel[j, :] and Kp_sel[j, :] as float32
        k_vec = tl.arange(0, HEAD_DIM_CKV)
        Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + k_vec, mask=k_vec < HEAD_DIM_CKV, other=0.0).to(tl.float32)  # [512]
        kpe_vec = tl.arange(0, HEAD_DIM_KPE)
        Kp_j = tl.load(Kp_sel_ptr + j * HEAD_DIM_KPE + kpe_vec, mask=kpe_vec < HEAD_DIM_KPE, other=0.0).to(tl.float32)  # [64]

        # Dot products over head dimension
        dot_qn = tl.sum(qn_rows * Kc_j[None, :], axis=1)  # [NUM_HEADS]
        dot_qp = tl.sum(qp_rows * Kp_j[None, :], axis=1)  # [NUM_HEADS]
        logits[:, j] = dot_qn + dot_qp

    # Scale logits
    logits = logits * sm_scale

    # Causal mask: absolute position to start ignoring is prefix_len + i + 1, with prefix_len = kv_len - q_len
    prefix_len = kv_len - q_len
    valid_start = prefix_len + i + 1
    j_vec = tl.arange(0, kv_len)
    causal_mask = j_vec >= valid_start
    logits = tl.where(causal_mask, logits, -float("inf"))

    # Logsumexp in log2
    m = tl.max(logits, axis=1)  # [NUM_HEADS]
    sumexp = tl.sum(tl.exp(logits - m[:, None]), axis=1)  # [NUM_HEADS]
    lse_val = m + tl.log(sumexp) * ln2_inv  # [NUM_HEADS], per head

    # Softmax
    logits = logits - m[:, None]  # stable
    exp_logits = tl.exp(logits)
    sumexp = tl.sum(exp_logits, axis=1)  # [NUM_HEADS]
    softmax = exp_logits / sumexp[:, None]  # [NUM_HEADS, kv_len]

    # Output: out[h, :] = sum_j softmax[h, j] * Kc_sel[j, :]
    for h in range(NUM_HEADS):
        out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        for j in range(kv_len):
            Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV), mask=tl.arange(0, HEAD_DIM_CKV) < HEAD_DIM_CKV, other=0.0).to(tl.float32)
            out_vec += softmax[h, j] * Kc_j
        # Store output vector for this query and head as bfloat16
        out_store = out_vec.to(tl.bfloat16)
        base_out = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        for k in range(HEAD_DIM_CKV):
            tl.store(output_ptr + base_out + k, out_store[k])

        # Store lse for this query and head as float32
        lse_base = q_abs * NUM_HEADS + h
        tl.store(lse_ptr + lse_base, lse_val[h])


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
        ln2_inv = 1.0 / math.log(2.0)

        # Process each batch element: build Kc_sel and Kp_sel based on kv_indptr and kv_indices
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            # tok_idx for this batch element: indices into cache per kv token
            tok_idx = kv_indices[int(kv_indptr[b].item() - 1):int(kv_indptr[b + 1].item()) - 1]  # [kv_len]
            # Select Kc_sel and Kp_sel from cache, cast to bfloat16 for storage
            Kc_sel = ckv_cache[tok_idx].to(torch.bfloat16)  # [kv_len, 512]
            Kp_sel = kpe_cache[tok_idx].to(torch.bfloat16)  # [kv_len, 64]
            kv_len = Kc_sel.shape[0]

            # Launch Triton kernel: one program per query i
            grid = (b, q_len)
            _forward_single_query_kernel[grid](
                q_nope, q_pe, Kc_sel, Kp_sel, output, lse,
                q_start, sm_scale, ln2_inv,
                NUM_HEADS=16, HEAD_DIM_CKV=512, HEAD_DIM_KPE=64,
                q_len=q_len, kv_len=kv_len,
                num_warps=4, num_stages=2,
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
