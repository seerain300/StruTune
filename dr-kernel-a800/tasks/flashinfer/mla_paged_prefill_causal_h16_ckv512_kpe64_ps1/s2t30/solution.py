import torch
import math
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    # Input tensors (pointers); original dtypes can be bf16/fp16/fp32; we cast to fp32 in-kernel
    q_nope_ptr,        # *bf16, [Q_total, 16, 512], row-major
    q_pe_ptr,          # *bf16, [Q_total, 16, 64], row-major
    Kc_sel_ptr,        # *bf16, [kv_len, 512], row-major
    Kp_sel_ptr,        # *bf16, [kv_len, 64], row-major
    output_ptr,        # *bf16, [Q_total, 16, 512], row-major
    lse_ptr,           # *fp32, [Q_total, 16]
    # runtime scalar
    q_start,           # int32: starting query index for this batch element
    # constexpr meta-parameters
    q_len: tl.constexpr,         # number of queries in this batch element
    kv_len: tl.constexpr,        # number of selected KV tokens for this batch element
    sm_scale: tl.constexpr,      # fp32 scaling factor
    ln2_inv: tl.constexpr,       # fp32 = 1.0 / ln(2.0)
    NUM_HEADS: tl.constexpr,     # 16
    HEAD_DIM_CKV: tl.constexpr,  # 512
    HEAD_DIM_KPE: tl.constexpr,  # 64
):
    # Each program handles one specific (batch b, query i). Grid is (len_indptr-1, q_len).
    # q_start is passed from host for this b.

    # Compute absolute query index for this program
    i = tl.program_id(1)  # second grid dimension is q_len
    q_abs = q_start + i

    # Base offsets for this absolute query index
    base_qn = q_abs * (NUM_HEADS * HEAD_DIM_CKV)
    base_qp = q_abs * (NUM_HEADS * HEAD_DIM_KPE)

    # Compute logits per head and do all math in fp32
    # Initialize lse per head
    lse_vec = tl.zeros((NUM_HEADS,), dtype=tl.float32)

    # For each head, compute logits vector and then softmax/output
    for h in range(NUM_HEADS):
        # Load qn[h, :] and qp[h, :] as fp32
        qn_row = tl.load(q_nope_ptr + base_qn + h * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV), mask=tl.arange(0, HEAD_DIM_CKV) < HEAD_DIM_CKV, other=0.0).to(tl.float32)
        qp_row = tl.load(q_pe_ptr + base_qp + h * HEAD_DIM_KPE + tl.arange(0, HEAD_DIM_KPE), mask=tl.arange(0, HEAD_DIM_KPE) < HEAD_DIM_KPE, other=0.0).to(tl.float32)

        # Initialize logits vector for this head
        logits = tl.zeros((kv_len,), dtype=tl.float32)

        # Compute dot-products and accumulate logits[h, j]
        for j in range(kv_len):
            Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV), mask=tl.arange(0, HEAD_DIM_CKV) < HEAD_DIM_CKV, other=0.0).to(tl.float32)
            Kp_j = tl.load(Kp_sel_ptr + j * HEAD_DIM_KPE + tl.arange(0, HEAD_DIM_KPE), mask=tl.arange(0, HEAD_DIM_KPE) < HEAD_DIM_KPE, other=0.0).to(tl.float32)
            # qn_row and Kc_j both [HEAD_DIM_CKV]
            dot_qn = tl.sum(qn_row * Kc_j, axis=0)
            # same for Kp
            dot_qp = tl.sum(qp_row * Kp_j, axis=0)
            logits[j] = dot_qn + dot_qp

        # Scale
        logits = logits * sm_scale

        # Causal mask: j >= prefix_len + i + 1 where prefix_len = kv_len - q_len
        prefix_len = kv_len - q_len
        valid_start = prefix_len + i + 1
        j_vec = tl.arange(0, kv_len)
        causal_mask = j_vec >= valid_start
        logits = tl.where(causal_mask, logits, -float("inf"))

        # logsumexp in log2
        m = tl.max(logits, axis=0)  # scalar
        sumexp = tl.sum(tl.exp(logits - m), axis=0)  # scalar
        lse_vec[h] = m + tl.log(sumexp) * ln2_inv

        # Softmax
        logits = logits - m
        exp_logits = tl.exp(logits)
        sumexp = tl.sum(exp_logits, axis=0)  # scalar
        softmax = exp_logits / sumexp  # [kv_len] vector

        # Output: out[h, :] = sum_j softmax[j] * Kc_sel[j, :]
        out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        for j in range(kv_len):
            Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV), mask=tl.arange(0, HEAD_DIM_CKV) < HEAD_DIM_CKV, other=0.0).to(tl.float32)
            out_vec += softmax[j] * Kc_j

        # Store output vector as bfloat16
        out_store = out_vec.to(tl.bfloat16)
        base_out = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        for k in range(HEAD_DIM_CKV):
            tl.store(output_ptr + base_out + k, out_store[k])

    # Store lse vector for this (b, i)
    lse_base = q_abs * NUM_HEADS
    for h in range(NUM_HEADS):
        tl.store(lse_ptr + lse_base + h, lse_vec[h])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Assertions to match original assumptions
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

        # Prepare output and LSE buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Precompute ln2_inv
        ln2_inv = float(1.0 / math.log(2.0))

        # For each batch element, compute tok_idx and select Kc_sel, Kp_sel as bfloat16
        # We will build Kc_sel and Kp_sel per batch in host; these are small relative to caches.
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            kv_len = page_end - page_beg

            # Build tok_idx for this batch element
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)

            # Select Kc_sel and Kp_sel
            Kc_sel = ckv_cache[tok_idx, 0, :].to(torch.bfloat16)  # [kv_len, 512]
            Kp_sel = kpe_cache[tok_idx, 0, :].to(torch.bfloat16)  # [kv_len, 64]

            # Launch Triton kernel: one program per query i in this batch element
            grid = (1, q_len)
            _forward_single_query_kernel[grid](
                q_nope, q_pe,
                Kc_sel, Kp_sel,
                output, lse,
                q_start,
                q_len=q_len,
                kv_len=kv_len,
                sm_scale=sm_scale,
                ln2_inv=ln2_inv,
                NUM_HEADS=16,
                HEAD_DIM_CKV=512,
                HEAD_DIM_KPE=64,
                num_warps=4,
                num_stages=2,
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
