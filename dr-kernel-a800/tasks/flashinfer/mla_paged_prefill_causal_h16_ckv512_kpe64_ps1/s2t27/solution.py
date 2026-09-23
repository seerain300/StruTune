import torch
import math
import triton
import triton.language as tl


@triton.jit
def select_kvs_kernel(
    kv_indices_ptr,         # *int32, shape [num_kv_indices]
    kv_indptr_ptr,          # *int32, shape [len_indptr]
    ckv_cache_ptr,          # *bf16, shape [num_pages, 1, 512]
    kpe_cache_ptr,          # *bf16, shape [num_pages, 1, 64]
    Kc_sel_flat_ptr,        # *bf16, shape [len_indptr - 1, MAX_KV, 512] — we pass only kv_len for b
    Kp_sel_flat_ptr,        # *bf16, shape [len_indptr - 1, MAX_KV, 64] — we pass only kv_len for b
    b,                      # int32, current batch element
    kv_len,                 # int32, number of KV tokens for this b
    HEAD_DIM_CKV: tl.constexpr,  # 512
    HEAD_DIM_KPE: tl.constexpr,  # 64
    MAX_KV: tl.constexpr,        # e.g., 1024, enough to cover any kv_len in provided workloads
):
    # Compute base pointers for this batch element
    base_kvs = b * (MAX_KV * HEAD_DIM_CKV + MAX_KV * HEAD_DIM_KPE)  # not used directly; layout: [KV, DIM], separate buffers
    # Instead, we operate on per-batch slots:
    # For simplicity, we assume Kc_sel_flat_ptr + b and Kp_sel_flat_ptr + b point to arrays of size kv_len*HEAD_DIM.
    # We need to copy kv_indices[b*step : (b+1)*step] -> selected rows of ckv_cache and kpe_cache.

    # First, compute tok_idx for this batch element
    step = tl.program_id(0)  # equals b, but we can also pass via meta. Here we rely on single program per b.
    start = tl.load(kv_indptr_ptr + b)
    end = tl.load(kv_indptr_ptr + b + 1)
    kv_count = end - start  # same as kv_len passed
    # Now copy selected rows into flat buffers at offset b
    Kc_base = Kc_sel_flat_ptr + b * MAX_KV * HEAD_DIM_CKV
    Kp_base = Kp_sel_flat_ptr + b * MAX_KV * HEAD_DIM_KPE

    # Loop over j in [0, kv_count)
    for j in tl.static_range(0, MAX_KV):
        # Safety: only copy if j < kv_count, but MAX_KV should match kv_count; use mask for generality
        if j >= kv_count:
            break
        idx = tl.load(kv_indices_ptr + start + j)  # idx into ckv_cache / kpe_cache
        # Copy ckv: ckv_cache[idx, 0, :] to Kc_sel_flat[b*MAX_KV + j, :]
        # Addressing: ckv_cache_ptr + idx * (1*512) + 0*512 + k_vec
        k_vec = tl.arange(0, HEAD_DIM_CKV)
        vals_ckv = tl.load(ckv_cache_ptr + idx * HEAD_DIM_CKV + k_vec, mask=k_vec < HEAD_DIM_CKV, other=0.0)  # dtype follows pointer (bf16 here)
        # Store to flat buffer: offset = (b*MAX_KV + j) * HEAD_DIM_CKV + k_vec
        store_offset = (b * MAX_KV + j) * HEAD_DIM_CKV + k_vec
        tl.store(Kc_base + j * HEAD_DIM_CKV + k_vec, vals_ckv.to(tl.bfloat16), mask=k_vec < HEAD_DIM_CKV)
        # Copy kpe: kpe_cache[idx, 0, :] to Kp_sel_flat[b*MAX_KV + j, :]
        kpe_vec = tl.arange(0, HEAD_DIM_KPE)
        vals_kpe = tl.load(kpe_cache_ptr + idx * HEAD_DIM_KPE + kpe_vec, mask=kpe_vec < HEAD_DIM_KPE, other=0.0).to(tl.bfloat16)
        store_offset_kpe = (b * MAX_KV + j) * HEAD_DIM_KPE + kpe_vec
        tl.store(Kp_sel_flat_ptr + b * MAX_KV * HEAD_DIM_KPE + j * HEAD_DIM_KPE + kpe_vec, vals_kpe, mask=kpe_vec < HEAD_DIM_KPE)


@triton.jit
def compute_query_kernel(
    q_nope_ptr,             # *bf16, shape [Q_total, 16, 512], row-major
    q_pe_ptr,               # *bf16, shape [Q_total, 16, 64], row-major
    Kc_sel_flat_ptr,        # *bf16, shape [MAX_KV, 512], per b
    Kp_sel_flat_ptr,        # *bf16, shape [MAX_KV, 64], per b
    output_ptr,             # *bf16, shape [Q_total, 16, 512], row-major
    lse_ptr,                # *fp32, shape [Q_total, 16]
    q_start,                # int32: start query index for this batch element
    i,                      # int32: query index within this batch element
    b,                      # int32: batch element index
    kv_len,                 # int32: number of selected KV tokens for this b
    sm_scale,               # fp32 scaling factor
    ln2_inv,                # fp32 = 1.0 / ln(2.0)
    NUM_HEADS: tl.constexpr,  # 16
    HEAD_DIM_CKV: tl.constexpr,  # 512
    HEAD_DIM_KPE: tl.constexpr,  # 64
    MAX_KV: tl.constexpr,        # e.g., 1024, should be >= kv_len
):
    # Compute absolute query index for this program
    q_abs = q_start + i
    h = tl.program_id(2)  # head index

    # Load qn[h] and qp[h] vectors from q_nope and q_pe, cast to fp32
    base_qn = q_abs * HEAD_DIM_CKV + h * HEAD_DIM_CKV
    qn_h = tl.load(q_nope_ptr + base_qn + tl.arange(0, HEAD_DIM_CKV), mask=tl.arange(0, HEAD_DIM_CKV) < HEAD_DIM_CKV, other=0.0).to(tl.float32)  # [512]
    base_qp = q_abs * (NUM_HEADS * HEAD_DIM_KPE) + h * HEAD_DIM_KPE
    qp_h = tl.load(q_pe_ptr + base_qp + tl.arange(0, HEAD_DIM_KPE), mask=tl.arange(0, HEAD_DIM_KPE) < HEAD_DIM_KPE, other=0.0).to(tl.float32)  # [64]

    # Build logits[h, j] for j in [0, kv_len)
    logits = tl.zeros((MAX_KV,), dtype=tl.float32)
    for j in tl.static_range(0, MAX_KV):
        if j >= kv_len:
            break
        Kc_j = tl.load(Kc_sel_flat_ptr + b * MAX_KV * HEAD_DIM_CKV + j * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV), mask=tl.arange(0, HEAD_DIM_CKV) < HEAD_DIM_CKV, other=0.0).to(tl.float32)  # [512]
        Kp_j = tl.load(Kp_sel_flat_ptr + b * MAX_KV * HEAD_DIM_KPE + j * HEAD_DIM_KPE + tl.arange(0, HEAD_DIM_KPE), mask=tl.arange(0, HEAD_DIM_KPE) < HEAD_DIM_KPE, other=0.0).to(tl.float32)  # [64]
        dot_qn = tl.sum(qn_h * Kc_j, axis=0)
        dot_qp = tl.sum(qp_h * Kp_j, axis=0)
        logits[j] = dot_qn + dot_qp

    # Scale logits
    logits = logits * sm_scale

    # Causal mask: for absolute position query_abs_pos = prefix_len + i + 1, where prefix_len = kv_len - q_len
    # We need q_len for this b. Compute q_len from qo_indptr:
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    q_len_b = qo_end - qo_start
    prefix_len = kv_len - q_len_b
    valid_start = prefix_len + i + 1
    for j in tl.static_range(0, MAX_KV):
        if j >= kv_len:
            break
        if j < valid_start:
            logits[j] = -float("inf")

    # Compute logsumexp in log2
    m = tl.max(logits, axis=0)
    sumexp = tl.sum(tl.exp(logits - m), axis=0)
    lse_val = m + tl.log(sumexp) * ln2_inv  # scalar per head
    lse_base = q_abs * NUM_HEADS + h
    tl.store(lse_ptr + lse_base, lse_val)

    # Softmax over j
    exp_logits = tl.exp(logits - lse_val)
    sumexp2 = tl.sum(exp_logits, axis=0)
    softmax = exp_logits / sumexp2  # [kv_len]

    # Output: out[h, :] = sum_j softmax[j] * Kc_sel[j, :]
    out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
    for j in tl.static_range(0, MAX_KV):
        if j >= kv_len:
            break
        Kc_j = tl.load(Kc_sel_flat_ptr + b * MAX_KV * HEAD_DIM_CKV + j * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV), mask=tl.arange(0, HEAD_DIM_CKV) < HEAD_DIM_CKV, other=0.0).to(tl.float32)  # [512]
        out_vec += softmax[j] * Kc_j

    # Store output vector for this query and head as bfloat16
    out_store = out_vec.to(tl.bfloat16)
    base_out = q_abs * (NUM_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
    for k in tl.static_range(0, HEAD_DIM_CKV):
        tl.store(output_ptr + base_out + k, out_store[k])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Assert shapes as in original code
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

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Flatten buffers for selected KVs per batch element
        # We assume MAX_KV >= max(kv_len) across workloads (verified: kv_len ~ 34)
        MAX_KV = 1024
        Kc_sel_flat = torch.empty((batch_size, MAX_KV, head_dim_ckv), dtype=torch.bfloat16, device=device)
        Kp_sel_flat = torch.empty((batch_size, MAX_KV, head_dim_kpe), dtype=torch.bfloat16, device=device)

        # Launch select_kvs_kernel: one program per batch element to fill Kc_sel_flat/Kp_sel_flat
        # We need to pass qo_indptr for computing q_len inside the compute kernel; but select_kvs does not need it.
        # Triton grid: (batch_size,)
        select_kvs_kernel[(batch_size,)](
            kv_indices, kv_indptr, ckv_cache, kpe_cache,
            Kc_sel_flat, Kp_sel_flat,
            0,  # b will be provided as program_id(0)
            MAX_KV,  # kv_len will be filled by for-loop; but we pass a dummy. We will call kernel per b and set b via program_id(0).
            HEAD_DIM_CKV=head_dim_ckv, HEAD_DIM_KPE=head_dim_kpe, MAX_KV=MAX_KV,
            num_warps=4
        )

        # Launch compute_query_kernel for each (b, i, h)
        for b in range(batch_size):
            # q_len_b and kv_len_b
            qo_start = int(qo_indptr[b].item())
            qo_end = int(qo_indptr[b + 1].item())
            q_len_b = qo_end - qo_start
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            kv_len_b = kv_end - kv_start

            # Loop over queries i
            for i in range(q_len_b):
                q_abs = qo_start + i
                # Launch one program per head
                for h in range(num_qo_heads):
                    compute_query_kernel[(1,)](
                        q_nope, q_pe,
                        Kc_sel_flat, Kp_sel_flat,
                        output, lse,
                        qo_start, i, b,
                        kv_len_b,
                        sm_scale, 1.0 / math.log(2.0),
                        NUM_HEADS=num_qo_heads, HEAD_DIM_CKV=head_dim_ckv, HEAD_DIM_KPE=head_dim_kpe, MAX_KV=MAX_KV,
                        qo_indptr=qo_indptr,  # passed for potential use (not used inside)
                        num_warps=4
                    )

        return output, lse


def run(*args):
    return ModelNew()(*args)
