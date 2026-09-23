import torch
import math
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    # Inputs (pointers) and outputs (pointers)
    q_nope_ptr,       # *bf16, shape [Q_total, 16, 512]
    q_pe_ptr,         # *bf16, shape [Q_total, 16, 64]
    Kc_sel_ptr,       # *bf16, shape [kv_len, 512], per-batch selected Kc
    Kp_sel_ptr,       # *bf16, shape [kv_len, 64],  per-batch selected Kp
    output_ptr,       # *bf16, shape [Q_total, 16, 512]
    lse_ptr,          # *fp32, shape [Q_total, 16]
    # scalars and meta-parameters
    q_start,          # int32, starting query index of this batch element
    sm_scale,         # fp32, scaling factor
    ln2_inv,          # fp32, 1.0 / ln(2.0)
    NUM_HEADS: tl.constexpr,      # 16
    HEAD_DIM_CKV: tl.constexpr,   # 512
    HEAD_DIM_KPE: tl.constexpr,   # 64
    kv_len: tl.constexpr,         # number of selected KV tokens
    q_abs: tl.constexpr,          # absolute query index for this program
):
    # One program handles one query i for one batch element b.
    # We derive b and i from the grid: grid = (len_indptr-1, q_len)
    # Here q_len is not used directly; we compute q_abs and process all heads and j-loops.
    # Compute qn[...] and qp[...] vectors for all heads in fp32.

    # Build qn[h, :] and qp[h, :] vectors
    # We use q_nope_ptr which is [Q_total, 16, 512] and q_pe_ptr [Q_total, 16, 64]
    # For a given q_abs, access rows for each head h.
    for h in range(NUM_HEADS):
        # Load qn[h, :] = q_nope[q_abs, h, :]
        base_qn = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        qn_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        for k in range(HEAD_DIM_CKV):
            qn_val = tl.load(q_nope_ptr + base_qn + k, mask=True, other=0.0)
            qn_val = qn_val.to(tl.float32)
            qn_vec[k] = qn_val

        # Load qp[h, :] = q_pe[q_abs, h, :]
        base_qp = q_abs * (NUM_HEADS * HEAD_DIM_KPE) + h * HEAD_DIM_KPE
        qp_vec = tl.zeros((HEAD_DIM_KPE,), dtype=tl.float32)
        for k in range(HEAD_DIM_KPE):
            qp_val = tl.load(q_pe_ptr + base_qp + k, mask=True, other=0.0)
            qp_val = qp_val.to(tl.float32)
            qp_vec[k] = qp_val

        # Compute logits for this head across kv_len tokens
        logits = tl.zeros((kv_len,), dtype=tl.float32)

        for j in range(kv_len):
            # Load Kc_sel[j, :] and cast to fp32
            k_vec = tl.arange(0, HEAD_DIM_CKV)
            Kc_j = tl.load(Kc_sel_ptr + j * HEAD_DIM_CKV + k_vec, mask=k_vec < HEAD_DIM_CKV, other=0.0).to(tl.float32)

            # dot(qn[h], Kc_sel[j, :]) = sum_k qn[h,k] * Kc_j[k]
            dot_qn = tl.sum(qn_vec * Kc_j, axis=0)  # scalar

            # Load Kp_sel[j, :] and cast to fp32
            kpe_vec = tl.arange(0, HEAD_DIM_KPE)
            Kp_j = tl.load(Kp_sel_ptr + j * HEAD_DIM_KPE + kpe_vec, mask=kpe_vec < HEAD_DIM_KPE, other=0.0).to(tl.float32)

            # dot(qp[h], Kp_sel[j, :])
            dot_qp = tl.sum(qp_vec * Kp_j, axis=0)  # scalar

            logits[j] = dot_qn + dot_qp

        # Scale logits
        logits = logits * sm_scale

        # Apply causal mask: j >= prefix_len + i + 1, with prefix_len = kv_len - 1 (since q_len is 1 per program)
        # Note: q_len for this program is 1 (single query i), so i = 0. prefix_len = kv_len - 1
        prefix_len = kv_len - 1
        valid_start = prefix_len + 1  # since i=0
        for j in range(kv_len):
            if j < valid_start:
                logits[j] = -float("inf")

        # Logsumexp in base-2
        m = tl.max(logits, axis=0)  # scalar
        exp_logits = tl.exp(logits - m)
        sumexp = tl.sum(exp_logits, axis=0)  # scalar
        lse_val = m + tl.log(sumexp) * ln2_inv  # scalar

        # Softmax over j
        exp_logits = tl.exp(logits - m)
        sumexp = tl.sum(exp_logits, axis=0)
        softmax = exp_logits / sumexp  # [kv_len] vector

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

        # Allocate outputs and LSE
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Precompute ln2_inv
        ln2_inv = 1.0 / math.log(2.0)

        # For each batch element, select kv_indices range and precompute Kc_sel/Kp_sel
        # We use grid=(1, q_len) so q_len must be 1 per program; our kernel handles a single query i.
        # Therefore, we launch a program per (b, i) pair. We can compute b via q_abs - qo_indptr[b], but Triton grid determines (b, i).
        # Instead, we launch total queries = qo_indptr[-1].item() programs by iterating in host over b and launching grid=(1, q_len).
        # However, to adhere to Triton-only, we avoid Python per-query loops. We handle all queries in a single kernel launch by iterating i=0..q_len-1 inside the kernel, but that requires knowing q_len.
        # Better: we launch a kernel per b and let the kernel loop over q_len. Triton allows us to pass q_len as constexpr per launch. The grid is (1, q_len).
        # Implement as: for b in range(batch_size): compute tok_idx, create Kc_sel_b, Kp_sel_b, and launch kernel with grid=(1, q_len). We pass q_start=qo_indptr[b].
        # This ensures we cover all queries and use Triton for computation.

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            # Compute tok_idx for this batch element
            tok_idx = kv_indices[q_start:q_end]  # [q_len] int32
            # Gather per-b selected KV chunks
            Kc_sel_b = ckv_cache[tok_idx].to(torch.bfloat16)  # [q_len, 512]
            Kp_sel_b = kpe_cache[tok_idx].to(torch.bfloat16)  # [q_len, 64]

            # Launch kernel: one program per query i in this batch element
            grid = (1, q_len)
            _forward_single_query_kernel[grid](
                q_nope, q_pe, Kc_sel_b, Kp_sel_b, output, lse,
                q_start, sm_scale, ln2_inv,
                NUM_HEADS=16, HEAD_DIM_CKV=512, HEAD_DIM_KPE=64,
                kv_len=q_len, q_abs=q_start,  # q_abs is the starting query index for this batch element
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
