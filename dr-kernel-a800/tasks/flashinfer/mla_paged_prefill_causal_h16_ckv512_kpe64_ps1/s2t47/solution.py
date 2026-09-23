import math
import torch
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    q_nope_ptr,      # *bf16, [total_q, 16, 512]
    q_pe_ptr,        # *bf16, [total_q, 16, 64]
    ckv_cache_ptr,   # *bf16, [num_pages, 1, 512]
    kpe_cache_ptr,   # *bf16, [num_pages, 1, 64]
    qo_indptr_ptr,   # *int32, [len_indptr]
    kv_indptr_ptr,   # *int32, [len_indptr]
    kv_indices_ptr,  # *int32, [num_kv_indices]
    output_ptr,      # *bf16, [total_q, 16, 512]
    lse_ptr,         # *float32, [total_q, 16]
    sm_scale,        # float32
    total_q,         # int32
    len_indptr,      # int32
    q_len_ptr,       # *int32, [len_indptr - 1]  # q_len per batch
    num_qo_heads: tl.constexpr,     # 16
    head_dim_ckv: tl.constexpr,     # 512
    head_dim_kpe: tl.constexpr,     # 64
):
    b = tl.program_id(0)
    i = tl.program_id(1)

    # Absolute query index
    qo_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    q_abs = qo_start + i

    # Loop over heads
    for h in range(0, num_qo_heads):
        # Load qn[h, :] and qp[h, :] in fp32
        base_qn = q_abs * (num_qo_heads * head_dim_ckv) + h * head_dim_ckv
        idx_k = tl.arange(0, head_dim_ckv)
        qn = tl.load(q_nope_ptr + base_qn + idx_k, mask=idx_k < head_dim_ckv, other=0.0).to(tl.float32)

        base_qp = q_abs * (num_qo_heads * head_dim_kpe) + h * head_dim_kpe
        idx_kp = tl.arange(0, head_dim_kpe)
        qp = tl.load(q_pe_ptr + base_qp + idx_kp, mask=idx_kp < head_dim_kpe, other=0.0).to(tl.float32)

        # kv indices for this batch
        kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
        kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
        kv_len = kv_end - kv_start

        # tok_idx = kv_indices[kv_start:kv_end]
        tok_idx = tl.load(kv_indices_ptr + kv_start + tl.arange(0, kv_len), mask=tl.arange(0, kv_len) < kv_len, other=0).to(tl.int32)

        # q_len for this batch
        q_len_b = tl.load(q_len_ptr + b).to(tl.int32)
        prefix_len = kv_len - q_len_b  # number of previously cached tokens in this batch

        # Compute logits and lse
        max_logit = -float("inf")
        sum_exp = 0.0

        # Loop over j to compute stable logsumexp
        for j in range(0, kv_len):
            idx_j = tok_idx[j]
            Kc_row = tl.load(ckv_cache_ptr + idx_j * head_dim_ckv + tl.arange(0, head_dim_ckv), mask=tl.arange(0, head_dim_ckv) < head_dim_ckv, other=0.0).to(tl.float32)
            Kp_row = tl.load(kpe_cache_ptr + idx_j * head_dim_kpe + tl.arange(0, head_dim_kpe), mask=tl.arange(0, head_dim_kpe) < head_dim_kpe, other=0.0).to(tl.float32)

            dot_qn = tl.sum(qn * Kc_row)
            dot_qp = tl.sum(qp * Kp_row)
            logit = (dot_qn + dot_qp) * sm_scale

            # causal mask: ignore if j < prefix_len + i
            if j >= (prefix_len + i):
                if logit > max_logit:
                    sum_exp = sum_exp * tl.exp(max_logit - logit) + 1.0
                    max_logit = logit
                else:
                    sum_exp += tl.exp(logit - max_logit)

        lse_val = (max_logit + tl.log(sum_exp)) / math.log(2.0)
        base_lse = q_abs * num_qo_heads + h
        tl.store(lse_ptr + base_lse, lse_val)

        # Compute attention output without using Kp (original output only depends on attn over Kc)
        logits_vec = tl.zeros((kv_len,), dtype=tl.float32)
        for j in range(0, kv_len):
            idx_j = tok_idx[j]
            Kc_row = tl.load(ckv_cache_ptr + idx_j * head_dim_ckv + tl.arange(0, head_dim_ckv), mask=tl.arange(0, head_dim_ckv) < head_dim_ckv, other=0.0).to(tl.float32)
            dot_qn = tl.sum(qn * Kc_row)
            dot_qp = tl.sum(qp * tl.load(kpe_cache_ptr + idx_j * head_dim_kpe + tl.arange(0, head_dim_kpe), mask=tl.arange(0, head_dim_kpe) < head_dim_kpe, other=0.0).to(tl.float32))
            logits_vec[j] = (dot_qn + dot_qp) * sm_scale

        # Softmax over valid logits (apply causal mask by zeroing invalid entries for softmax stability)
        # For simplicity, we compute softmax over all entries but set invalid logits to -inf before softmax.
        # Alternatively, recompute with valid set. We'll recompute valid set by zeroing invalid entries.
        valid_mask = (tl.arange(0, kv_len) >= (prefix_len + i))
        # Set invalid logits to -inf
        for j in range(0, kv_len):
            if not valid_mask[j]:
                logits_vec[j] = -float("inf")

        # Stable softmax
        max_log_vec = -float("inf")
        for j in range(0, kv_len):
            max_log_vec = tl.maximum(max_log_vec, logits_vec[j])
        sum_exp_vec = 0.0
        for j in range(0, kv_len):
            sum_exp_vec += tl.exp(logits_vec[j] - max_log_vec)
        inv_sum = 1.0 / sum_exp_vec
        attn = tl.zeros((kv_len,), dtype=tl.float32)
        for j in range(0, kv_len):
            attn[j] = tl.exp(logits_vec[j] - max_log_vec) * inv_sum

        # Compute output[h, :] = sum_j attn[j] * Kc_sel[j, :]
        output_vec = tl.zeros((head_dim_ckv,), dtype=tl.float32)
        for j in range(0, kv_len):
            idx_j = tok_idx[j]
            Kc_row = tl.load(ckv_cache_ptr + idx_j * head_dim_ckv + tl.arange(0, head_dim_ckv), mask=tl.arange(0, head_dim_ckv) < head_dim_ckv, other=0.0).to(tl.float32)
            output_vec += attn[j] * Kc_row

        # Store output[q_abs, h, :]
        base_out = q_abs * (num_qo_heads * head_dim_ckv) + h * head_dim_ckv
        for k in range(0, head_dim_ckv):
            tl.store(output_ptr + base_out + k, output_vec[k].to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton kernel"
        device = q_nope.device

        total_q = int(q_nope.shape[0])
        len_indptr = int(qo_indptr.shape[0])
        Batches = len_indptr - 1

        outputs = torch.empty((total_q, 16, 512), dtype=torch.bfloat16, device=device)
        lse_out = torch.empty((total_q, 16), dtype=torch.float32, device=device)

        # Compute q_len per batch for causal mask
        q_len_list = []
        for b in range(Batches):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len_list.append(q_end - q_start)
        q_len_tensor = torch.tensor(q_len_list, dtype=torch.int32, device=device)

        grid = (len_indptr - 1, total_q)
        _forward_single_query_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache,
            qo_indptr, kv_indptr, kv_indices,
            outputs, lse_out,
            float(sm_scale),
            total_q, len_indptr,
            q_len_tensor,
            num_qo_heads=16,
            head_dim_ckv=512,
            head_dim_kpe=64,
        )

        return outputs, lse_out


def run(*args):
    return ModelNew()(*args)
