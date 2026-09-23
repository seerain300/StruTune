import math
import torch
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    q_nope_ptr,       # *bf16, [Q, 16, 512]
    q_pe_ptr,         # *bf16, [Q, 16, 64]
    ckv_cache_ptr,    # *bf16, [N, 1, 512]
    kpe_cache_ptr,    # *bf16, [N, 1, 64]
    qo_indptr_ptr,    # *int32, [L]
    kv_indptr_ptr,    # *int32, [L]
    kv_indices_ptr,   # *int32, [M]
    output_ptr,       # *bf16, [Q, 16, 512]
    lse_ptr,          # *fp32, [Q, 16]
    sm_scale,         # scalar float32 (unused in this specific computation, kept for signature)
):
    # program ids
    b = tl.program_id(0)  # batch element index (from qo_indptr)
    i = tl.program_id(1)  # query index within batch

    # absolute query index
    q_abs = tl.load(qo_indptr_ptr + b) + i

    # dimensions (constants for this model)
    num_heads = 16
    head_dim_ckv = 512
    head_dim_kpe = 64

    # loop over heads
    for h in range(0, num_heads):
        # base offsets for q_nope[q_abs, h, :] and q_pe[q_abs, h, :]
        base_qn = q_abs * (num_heads * head_dim_ckv) + h * head_dim_ckv
        base_qp = q_abs * (num_heads * head_dim_kpe) + h * head_dim_kpe

        # load qn[h, :] and qp[h, :] as vectors, compute in float32
        qn = tl.zeros((head_dim_ckv,), dtype=tl.float32)
        for k in range(0, head_dim_ckv):
            val = tl.load(q_nope_ptr + base_qn + k)
            qn[k] = val.to(tl.float32)

        qp = tl.zeros((head_dim_kpe,), dtype=tl.float32)
        for k in range(0, head_dim_kpe):
            val = tl.load(q_pe_ptr + base_qp + k)
            qp[k] = val.to(tl.float32)

        # compute q_len (number of queries in this batch element)
        q_start = tl.load(qo_indptr_ptr + b)
        q_end = tl.load(qo_indptr_ptr + b + 1)
        q_len = q_end - q_start

        # compute kv_len and kv_start
        kv_start = tl.load(kv_indptr_ptr + b)
        kv_end = tl.load(kv_indptr_ptr + b + 1)
        kv_len = kv_end - kv_start

        # prefix_len = number of previously cached tokens for this batch element
        prefix_len = kv_len - q_len

        # compute logits[j] for j in 0..kv_len-1
        logits = tl.zeros((kv_len,), dtype=tl.float32)
        for j in range(0, kv_len):
            idx_j = tl.load(kv_indices_ptr + kv_start + j).to(tl.int64)
            # load Kc_sel[j, :] and Kp_sel[j, :]
            Kc_j = tl.zeros((head_dim_ckv,), dtype=tl.float32)
            for kk in range(0, head_dim_ckv):
                val = tl.load(ckv_cache_ptr + idx_j * head_dim_ckv + kk)
                Kc_j[kk] = val.to(tl.float32)

            Kp_j = tl.zeros((head_dim_kpe,), dtype=tl.float32)
            for kk in range(0, head_dim_kpe):
                val = tl.load(kpe_cache_ptr + idx_j * head_dim_kpe + kk)
                Kp_j[kk] = val.to(tl.float32)

            # dot products
            dot_qn = 0.0
            for k in range(0, head_dim_ckv):
                dot_qn += qn[k] * Kc_j[k]

            dot_qp = 0.0
            for k in range(0, head_dim_kpe):
                dot_qp += qp[k] * Kp_j[k]

            logits[j] = dot_qn + dot_qp

        # apply causal mask: j >= prefix_len + i
        # We need to ignore positions j where j < prefix_len + i
        # Implement masking by setting those logits to -inf before reductions.
        mask_thresh = prefix_len + i
        for j in range(0, kv_len):
            if j < mask_thresh:
                logits[j] = -float("inf")

        # max over logits for stable logsumexp
        max_logit = logits[0]
        for j in range(1, kv_len):
            max_logit = tl.maximum(max_logit, logits[j])

        # sum exp(logits - max_logit)
        sum_exp = 0.0
        for j in range(0, kv_len):
            sum_exp += tl.exp(logits[j] - max_logit)

        # logsumexp and then convert to log2
        logsumexp = max_logit + tl.log(sum_exp)
        lse_log2 = logsumexp / 0.6931471805599453  # 1 / ln(2)

        # store lse[q_abs, h]
        base_lse = q_abs * num_heads + h
        tl.store(lse_ptr + base_lse, lse_log2)

        # softmax over logits
        inv_sum_exp = 1.0 / sum_exp
        softmax = tl.zeros((kv_len,), dtype=tl.float32)
        for j in range(0, kv_len):
            softmax[j] = tl.exp(logits[j] - max_logit) * inv_sum_exp

        # compute output[h, :] = sum_j softmax[j] * Kc_sel[j, :]
        output_vec = tl.zeros((head_dim_ckv,), dtype=tl.float32)
        for j in range(0, kv_len):
            idx_j = tl.load(kv_indices_ptr + kv_start + j)
            Kc_j = tl.zeros((head_dim_ckv,), dtype=tl.float32)
            for kk in range(0, head_dim_ckv):
                val = tl.load(ckv_cache_ptr + idx_j * head_dim_ckv + kk)
                Kc_j[kk] = val.to(tl.float32)
            output_vec += softmax[j] * Kc_j

        # store output[q_abs, h, :]
        base_out = q_abs * (num_heads * head_dim_ckv) + h * head_dim_ckv
        for k in range(0, head_dim_ckv):
            tl.store(output_ptr + base_out + k, output_vec[k].to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton kernel"
        device = q_nope.device

        total_q = int(q_nope.shape[0])
        len_indptr = int(qo_indptr.shape[0])
        Batches = len_indptr - 1

        # Prepare outputs
        outputs = torch.empty((total_q, 16, 512), dtype=torch.bfloat16, device=device)
        lse_out = torch.empty((total_q, 16), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, i)
        grid = (len_indptr - 1, total_q)
        _forward_single_query_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache,
            qo_indptr, kv_indptr, kv_indices,
            outputs, lse_out,
            sm_scale,
        )

        return outputs, lse_out


def run(*args):
    return ModelNew()(*args)
