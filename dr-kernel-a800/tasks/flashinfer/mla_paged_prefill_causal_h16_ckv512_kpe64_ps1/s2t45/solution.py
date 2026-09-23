import math
import torch
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    q_nope_ptr,       # *bf16, [total_q, 16, 512]
    q_pe_ptr,         # *bf16, [total_q, 16, 64]
    ckv_cache_ptr,    # *bf16, [num_pages, 1, 512]
    kpe_cache_ptr,    # *bf16, [num_pages, 1, 64]
    qo_indptr_ptr,    # *int32, [len_indptr]
    kv_indptr_ptr,    # *int32, [len_indptr]
    kv_indices_ptr,   # *int32, [num_kv_indices]
    output_ptr,       # *bf16, [total_q, 16, 512]
    lse_ptr,          # *float32, [total_q, 16]
    total_q,          # int32
    len_indptr,       # int32
    sm_scale,         # float32 scalar (not used in original scaling; kept for API)
):
    # program ids: one program per (b, i)
    b = tl.program_id(0)  # batch element id
    i = tl.program_id(1)  # query index within this batch

    # Compute absolute query position
    qo_b = qo_indptr_ptr + b
    q_start = tl.load(qo_b)  # int32
    q_abs = q_start + i  # absolute query index

    # Dimensions
    num_heads = 16
    head_dim_ckv = 512
    head_dim_kpe = 64

    # Iterate over heads
    for h in range(0, num_heads):
        # Load qn[h, :] and qp[h, :] in fp32 from q_nope and q_pe
        base_qn = q_abs * (num_heads * head_dim_ckv) + h * head_dim_ckv
        qn = tl.zeros((head_dim_ckv,), dtype=tl.float32)
        for k in range(0, head_dim_ckv):
            val = tl.load(q_nope_ptr + base_qn + k)
            qn[k] = val.to(tl.float32)

        base_qp = q_abs * (num_heads * head_dim_kpe) + h * head_dim_kpe
        qp = tl.zeros((head_dim_kpe,), dtype=tl.float32)
        for k in range(0, head_dim_kpe):
            val = tl.load(q_pe_ptr + base_qp + k)
            qp[k] = val.to(tl.float32)

        # Compute prefix_len (tokens before this query in this batch)
        kv_b = kv_indptr_ptr + b
        kv_start = tl.load(kv_b)  # int32
        kv_end = tl.load(kv_indptr_ptr + b + 1)  # int32
        kv_len = kv_end - kv_start  # int32
        q_end = tl.load(qo_indptr_ptr + b + 1)  # int32
        q_len = q_end - q_start  # int32
        prefix_len = kv_len - q_len  # int32

        # Initialize logits
        logits = tl.zeros((kv_len,), dtype=tl.float32)

        # Compute logits per j
        # Note: We need to loop j and load Kc_sel/Kp_sel rows via tok_idx[j]
        for j in range(0, kv_len):
            idx_j = tl.load(kv_indices_ptr + kv_start + j)  # int32

            # Load Kc_sel[j, :] and Kp_sel[j, :]
            Kc_j = tl.zeros((head_dim_ckv,), dtype=tl.float32)
            for kk in range(0, head_dim_ckv):
                val = tl.load(ckv_cache_ptr + idx_j * head_dim_ckv + kk)
                Kc_j[kk] = val.to(tl.float32)
            Kp_j = tl.zeros((head_dim_kpe,), dtype=tl.float32)
            for kk in range(0, head_dim_kpe):
                val = tl.load(kpe_cache_ptr + idx_j * head_dim_kpe + kk)
                Kp_j[kk] = val.to(tl.float32)

            # Compute dot products
            dot_qn = 0.0
            for k in range(0, head_dim_ckv):
                dot_qn += qn[k] * Kc_j[k]
            dot_qp = 0.0
            for k in range(0, head_dim_kpe):
                dot_qp += qp[k] * Kp_j[k]
            logits[j] = dot_qn + dot_qp  # multiply by sm_scale if needed, but original code uses sm_scale in forward op; here we keep original behavior

        # Apply causal mask: j >= prefix_len + i
        j_vec = tl.arange(0, kv_len)
        mask_vec = j_vec >= (prefix_len + i)
        logits = tl.where(mask_vec, logits, -float("inf"))

        # Compute logsumexp (stable)
        max_logit = tl.max(logits)
        sum_exp = 0.0
        for j in range(0, kv_len):
            sum_exp += tl.exp(logits[j] - max_logit)
        lse = tl.log(sum_exp) + max_logit  # original code divides by ln(2) outside, we keep raw logsumexp
        # Store lse to output lse[q_abs, h]
        base_lse = q_abs * num_heads + h
        tl.store(lse_ptr + base_lse, lse)

        # Softmax over logits
        inv_sum_exp = 1.0 / sum_exp
        softmax = tl.zeros((kv_len,), dtype=tl.float32)
        for j in range(0, kv_len):
            softmax[j] = tl.exp(logits[j] - max_logit) * inv_sum_exp

        # Compute output[h, :] = sum_j softmax[j] * Kc_sel[j, :]
        output_vec = tl.zeros((head_dim_ckv,), dtype=tl.float32)
        for j in range(0, kv_len):
            idx_j = tl.load(kv_indices_ptr + kv_start + j)
            Kc_j = tl.zeros((head_dim_ckv,), dtype=tl.float32)
            for kk in range(0, head_dim_ckv):
                val = tl.load(ckv_cache_ptr + idx_j * head_dim_ckv + kk)
                Kc_j[kk] = val.to(tl.float32)
            output_vec += softmax[j] * Kc_j

        # Store output[q_abs, h, :]
        base_out = q_abs * (num_heads * head_dim_ckv) + h * head_dim_ckv
        for k in range(0, head_dim_ckv):
            tl.store(output_ptr + base_out + k, output_vec[k].to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are CUDA for Triton
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
            total_q, len_indptr,
            float(sm_scale),  # pass scalar
            num_warps=4,
        )
        return outputs, lse_out


def run(*args):
    return ModelNew()(*args)
