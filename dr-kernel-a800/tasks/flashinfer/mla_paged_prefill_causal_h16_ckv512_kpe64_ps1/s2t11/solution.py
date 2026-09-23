import torch
import math
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    # Input pointers
    q_nope_ptr,   # *bf16, shape [Q_total, 16, 512], row-major
    q_pe_ptr,     # *bf16, shape [Q_total, 16, 64], row-major
    Kc_sel_ptr,   # *bf16, shape [kv_len, 512], row-major (per-batch element)
    Kp_sel_ptr,   # *bf16, shape [kv_len, 64], row-major (per-batch element)
    output_ptr,   # *bf16, shape [Q_total, 16, 512], row-major
    lse_ptr,      # *fp32, shape [Q_total, 16]
    # runtime scalar
    q_start,      # int32, start query index for this batch element
    # meta-parameters
    q_len: tl.constexpr,        # number of queries in this batch element
    kv_len: tl.constexpr,       # number of selected KV tokens for this batch element
    sm_scale: tl.constexpr,     # fp32 scaling factor
    ln2_inv: tl.constexpr,      # fp32 = 1.0 / ln(2.0)
    NUM_HEADS: tl.constexpr,    # 16
    HEAD_DIM_CKV: tl.constexpr, # 512
    HEAD_DIM_KPE: tl.constexpr, # 64
):
    # Grid is (len_indptr - 1, q_len). pid0 is batch element index (implicit via q_start),
    # pid1 is query index i.
    i = tl.program_id(1)
    q_abs = q_start + i

    # For each head h, compute logits for all j in [0, kv_len), then lse and output.
    for h in range(NUM_HEADS):
        # Load qn[h, :] and qp[h, :]
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

        # Initialize logits vector for this head
        logits = tl.zeros((kv_len,), dtype=tl.float32)

        # Compute logits[h, j] = sum_k qn[h,k] * Kc_sel[j,k] + sum_k qp[h,k] * Kp_sel[j,k]
        for j in range(kv_len):
            # Load Kc_sel[j, :] as fp32
            Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV), mask=tl.arange(0, HEAD_DIM_CKV) < HEAD_DIM_CKV, other=0.0).to(tl.float32)
            dot_qn = tl.sum(qn_vec * Kc_j, axis=0)

            # Load Kp_sel[j, :]
            Kp_j = tl.load(Kp_sel_ptr + j * HEAD_DIM_KPE + tl.arange(0, HEAD_DIM_KPE), mask=tl.arange(0, HEAD_DIM_KPE) < HEAD_DIM_KPE, other=0.0).to(tl.float32)
            dot_qp = tl.sum(qp_vec * Kp_j, axis=0)

            logits[j] = dot_qn + dot_qp

        # Scale by sm_scale
        logits = logits * sm_scale

        # Apply causal mask: for query i, only j >= prefix_len + i + 1 are valid,
        # where prefix_len = kv_len - q_len (tokens beyond current batch's queries).
        prefix_len = kv_len - q_len
        valid_start = prefix_len + i + 1
        for j in range(kv_len):
            if j < valid_start:
                logits[j] = -float("inf")

        # logsumexp in log2
        m = tl.max(logits, axis=0)
        sumexp = tl.sum(tl.exp(logits - m), axis=0)
        lse_val = m + tl.log(sumexp) * ln2_inv
        lse_base = q_abs * NUM_HEADS + h
        tl.store(lse_ptr + lse_base, lse_val)

        # Softmax
        exp_logits = tl.exp(logits - m)
        sumexp = tl.sum(exp_logits, axis=0)
        softmax = exp_logits / sumexp  # [kv_len]

        # Output: out[h, :] = sum_j softmax[j] * Kc_sel[j, :]
        out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        for j in range(kv_len):
            Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV), mask=tl.arange(0, HEAD_DIM_CKV) < HEAD_DIM_CKV, other=0.0).to(tl.float32)
            out_vec += softmax[j] * Kc_j

        # Store output as bfloat16
        base_out = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        out_store = out_vec.to(tl.bfloat16)
        for k in range(HEAD_DIM_CKV):
            tl.store(output_ptr + base_out + k, out_store[k])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure shapes as in original
        assert q_nope.shape[1] == 16 and q_nope.shape[2] == 512, "q_nope must be [Q_total, 16, 512]"
        assert q_pe.shape[1] == 16 and q_pe.shape[2] == 64, "q_pe must be [Q_total, 16, 64]"
        assert ckv_cache.shape[1] == 1 and ckv_cache.shape[2] == 512, "ckv_cache must be [num_pages, 1, 512]"
        assert kpe_cache.shape[1] == 1 and kpe_cache.shape[2] == 64, "kpe_cache must be [num_pages, 1, 64]"
        assert qo_indptr.dim() == 1 and kv_indptr.dim() == 1, "indptrs must be 1D"
        assert q_nope.device.type == "cuda" and q_pe.device.type == "cuda" and ckv_cache.device.type == "cuda" and kpe_cache.device.type == "cuda", "All tensors must be on CUDA device"

        device = q_nope.device
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[2]
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1

        # Output and LSE buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        ln2_inv = 1.0 / math.log(2.0)

        # For each batch element b, build per-b Kc_sel and Kp_sel
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len_b = q_end - q_start

            # KV token indices for this batch element: tok_idx = kv_indices[ kv_indptr[b]:kv_indptr[b+1] ]
            tok_idx = kv_indices[qo_indptr[b]:qo_indptr[b + 1]].to(device)  # indices into cache
            kv_len_b = int((kv_indptr[b + 1] - kv_indptr[b]).item())

            # Build Kc_sel and Kp_sel for this batch element (bf16)
            Kc_sel_b = ckv_cache[tok_idx.long()].to(torch.bfloat16)  # [kv_len_b, 512]
            Kp_sel_b = kpe_cache[tok_idx.long()].to(torch.bfloat16)  # [kv_len_b, 64]

            # Launch one Triton program per query i in this batch element
            grid = (1, q_len_b)
            _forward_single_query_kernel[grid](
                q_nope, q_pe, Kc_sel_b, Kp_sel_b, output, lse,
                q_start,
                q_len=q_len_b,
                kv_len=kv_len_b,
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
