import torch
import math
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    q_nope_ptr,           # *bf16, [Q_total, 16, 512]
    q_pe_ptr,             # *bf16, [Q_total, 16, 64]
    Kc_sel_ptr,           # *bf16, [kv_len, 512]
    Kp_sel_ptr,           # *bf16, [kv_len, 64]
    output_ptr,           # *bf16, [Q_total, 16, 512]
    lse_ptr,              # *fp32, [Q_total, 16]
    # scalars and constexprs
    q_start,              # int32, start query index for this batch element
    q_len_const,          # int32, number of queries in this batch element (meta)
    kv_len_const,         # int32, number of selected KV tokens in this batch element (meta)
    sm_scale,             # fp32, scaling factor
    ln2_inv,              # fp32, 1.0 / ln(2.0)
    # meta-parameters
    Q_TOTAL: tl.constexpr,          # total number of queries for bounds check
    NUM_HEADS: tl.constexpr,        # 16
    HEAD_DIM_CKV: tl.constexpr,     # 512
    HEAD_DIM_KPE: tl.constexpr,     # 64
):
    # Grid: (len_indptr - 1, q_len). We decode program ids accordingly.
    b = tl.program_id(0)  # batch element index
    i = tl.program_id(1)  # query index within this batch element

    # Absolute query index
    q_abs = q_start + i

    # Iterate over heads; Triton allows runtime loops
    for h in range(NUM_HEADS):
        # Load qn[h, :] and qp[h, :] vectors
        # q_nope_ptr layout: [Q_total, NUM_HEADS, HEAD_DIM_CKV], contiguous
        # Base offset for head h at query q_abs
        base_qn = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        qn_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        for k in range(HEAD_DIM_CKV):
            val = tl.load(q_nope_ptr + base_qn + k).to(tl.float32)
            qn_vec[k] = val

        base_qp = q_abs * (NUM_HEADS * HEAD_DIM_KPE) + h * HEAD_DIM_KPE
        qp_vec = tl.zeros((HEAD_DIM_KPE,), dtype=tl.float32)
        for k in range(HEAD_DIM_KPE):
            val = tl.load(q_pe_ptr + base_qp + k).to(tl.float32)
            qp_vec[k] = val

        # Prepare logits vector for this head
        logits_vec = tl.zeros((kv_len_const,), dtype=tl.float32)

        # Compute logits[h, j] for j in [0, kv_len_const)
        for j in range(kv_len_const):
            # Load Kc_sel[j, :] and Kp_sel[j, :]
            k_vec_ck = tl.arange(0, HEAD_DIM_CKV)
            Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + k_vec_ck, mask=k_vec_ck < HEAD_DIM_CKV, other=0.0).to(tl.float32)  # [512]
            k_vec_kp = tl.arange(0, HEAD_DIM_KPE)
            Kp_j = tl.load(Kp_sel_ptr + j * HEAD_DIM_KPE + k_vec_kp, mask=k_vec_kp < HEAD_DIM_KPE, other=0.0).to(tl.float32)  # [64]

            # Dot products
            dot_qn = tl.sum(qn_vec * Kc_j, axis=0)  # scalar
            dot_qp = tl.sum(qp_vec * Kp_j, axis=0)  # scalar
            logits_vec[j] = dot_qn + dot_qp

        # Scale logits
        logits_vec = logits_vec * sm_scale

        # Apply causal mask: j >= prefix_len + i + 1 where prefix_len = kv_len_const - q_len_const
        prefix_len = kv_len_const - q_len_const
        valid_start = prefix_len + i + 1
        j_vec = tl.arange(0, kv_len_const)
        causal_mask = j_vec >= valid_start
        logits_vec = tl.where(causal_mask, logits_vec, -float("inf"))

        # Compute logsumexp in log2
        m = tl.max(logits_vec, axis=0)
        sumexp = tl.sum(tl.exp(logits_vec - m), axis=0)
        lse_val = m + (tl.log(sumexp) * ln2_inv)
        # Store lse[q_abs, h] as float32
        lse_base = q_abs * NUM_HEADS + h
        tl.store(lse_ptr + lse_base, lse_val)

        # Softmax over j (stable)
        logits_shift = logits_vec - m
        sumexp = tl.sum(tl.exp(logits_shift), axis=0)
        softmax_vec = tl.exp(logits_shift) / sumexp  # [kv_len_const]

        # Output: out[h, :] = sum_j softmax_vec[j] * Kc_sel[j, :]
        out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        for j in range(kv_len_const):
            Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV), mask=tl.arange(0, HEAD_DIM_CKV) < HEAD_DIM_CKV, other=0.0).to(tl.float32)
            out_vec += softmax_vec[j] * Kc_j

        # Store output as bfloat16
        out_base = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        out_store = out_vec.to(tl.bfloat16)
        for k in range(HEAD_DIM_CKV):
            tl.store(output_ptr + out_base + k, out_store[k])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Validate shapes (as in the original assumptions)
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

        # For each batch element, compute tok_idx = kv_indices[page_beg:page_end], then Kc_sel and Kp_sel
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len_const = q_end - q_start

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            kv_len_const = page_end - page_beg

            # tok_idx for this batch element
            tok_idx = kv_indices[page_beg:page_end].to(torch.long).to(device)

            # Select Kc_sel and Kp_sel from cache on device (keep as bfloat16)
            Kc_sel = ckv_cache[tok_idx].to(torch.bfloat16).contiguous()  # [kv_len_const, 512]
            Kp_sel = kpe_cache[tok_idx].to(torch.bfloat16).contiguous()  # [kv_len_const, 64]

            # Launch Triton kernel: one program per query and per batch element
            # Grid: (batch_size, q_len_const)
            grid = (batch_size, q_len_const)
            _forward_single_query_kernel[grid](
                q_nope, q_pe, Kc_sel, Kp_sel, output, lse,
                q_start, q_len_const, kv_len_const, sm_scale, ln2_inv,
                total_q, 16, 512, 64,
                num_warps=4, num_stages=2
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
