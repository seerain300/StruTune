import math
import torch
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    q_nope_ptr,      # *bf16, [total_q, num_qo_heads, head_dim_ckv]
    q_pe_ptr,        # *bf16, [total_q, num_qo_heads, head_dim_kpe]
    ckv_cache_ptr,   # *bf16, [num_pages, 1, head_dim_ckv]
    kpe_cache_ptr,   # *bf16, [num_pages, 1, head_dim_kpe]
    qo_indptr_ptr,   # *int32, [len_indptr]
    kv_indptr_ptr,   # *int32, [len_indptr]
    kv_indices_ptr,  # *int32, [num_kv_indices]
    output_ptr,      # *bf16, [total_q, num_qo_heads, head_dim_ckv]
    lse_ptr,         # *f32,  [total_q, num_qo_heads]
    total_q,         # int
    len_indptr,      # int
    sm_scale,        # f32
    # constexpr parameters
    num_qo_heads: tl.constexpr,       # 16
    head_dim_ckv: tl.constexpr,       # 512
    head_dim_kpe: tl.constexpr,       # 64
):
    # program ids: one per (b, i)
    b = tl.program_id(0)
    i = tl.program_id(1)

    # absolute query index
    q_abs = q_abs = tl.load(qo_indptr_ptr + b).to(tl.int32) + i

    # loop over heads (compile-time unrolled)
    for h in tl.static_range(num_qo_heads):
        # base offsets for q_nope[q_abs, h, :] and q_pe[q_abs, h, :]
        base_qn = q_abs * (num_qo_heads * head_dim_ckv) + h * head_dim_ckv
        base_qp = q_abs * (num_qo_heads * head_dim_kpe) + h * head_dim_kpe

        # load qn[h, :] and qp[h, :] as float32 vectors
        offs_qn = tl.arange(0, head_dim_ckv)
        qn = tl.load(q_nope_ptr + base_qn + offs_qn, mask=offs_qn < head_dim_ckv, other=0.0).to(tl.float32)

        offs_qp = tl.arange(0, head_dim_kpe)
        qp = tl.load(q_pe_ptr + base_qp + offs_qp, mask=offs_qp < head_dim_kpe, other=0.0).to(tl.float32)

        # compute kv length and tok_idx for this batch
        kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
        kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
        kv_len = kv_end - kv_start

        # prefix_len = number of tokens already processed in this batch
        # q_len = number of queries in this batch
        q_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
        q_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
        q_len = q_end - q_start

        prefix_len = kv_len - q_len

        # initialize accumulators
        logits = tl.zeros((kv_len,), dtype=tl.float32)
        for j in range(0, kv_len):
            idx_j = tl.load(kv_indices_ptr + kv_start + j).to(tl.int32)
            # load Kc_sel_row[j, :] and Kp_sel_row[j, :]
            Kc_j = tl.load(ckv_cache_ptr + idx_j * head_dim_ckv + tl.arange(0, head_dim_ckv),
                           mask=tl.arange(0, head_dim_ckv) < head_dim_ckv,
                           other=0.0).to(tl.float32)
            Kp_j = tl.load(kpe_cache_ptr + idx_j * head_dim_kpe + tl.arange(0, head_dim_kpe),
                           mask=tl.arange(0, head_dim_kpe) < head_dim_kpe,
                           other=0.0).to(tl.float32)
            # dot products
            dot_qn = tl.sum(qn * Kc_j, axis=0)
            dot_qp = tl.sum(qp * Kp_j, axis=0)
            logits[j] = dot_qn + dot_qp

        # apply causal mask: j >= prefix_len + i
        abs_pos = prefix_len + i
        causal = j >= abs_pos
        for j in range(0, kv_len):
            logits[j] = tl.where(causal, -float('inf'), logits[j])

        # stable logsumexp in fp32, then convert to log2
        max_logit = tl.max(logits, axis=0)
        sum_exp = 0.0
        for j in range(0, kv_len):
            sum_exp += tl.exp(logits[j] - max_logit)
        lse_val = (max_logit + tl.log(sum_exp)) * (1.0 / 0.6931471805599453)  # ln(2)

        # store lse[q_abs, h]
        tl.store(lse_ptr + q_abs * num_qo_heads + h, lse_val)

        # softmax over logits
        inv_sum_exp = 1.0 / sum_exp
        softmax = tl.zeros((kv_len,), dtype=tl.float32)
        for j in range(0, kv_len):
            softmax[j] = tl.exp(logits[j] - max_logit) * inv_sum_exp

        # compute output[h, :] = sum_j softmax[j] * Kc_sel[j, :]
        output_vec = tl.zeros((head_dim_ckv,), dtype=tl.float32)
        for j in range(0, kv_len):
            idx_j = tl.load(kv_indices_ptr + kv_start + j).to(tl.int32)
            Kc_j = tl.load(ckv_cache_ptr + idx_j * head_dim_ckv + tl.arange(0, head_dim_ckv),
                           mask=tl.arange(0, head_dim_ckv) < head_dim_ckv,
                           other=0.0).to(tl.float32)
            output_vec += softmax[j] * Kc_j

        # store output[q_abs, h, :] as bfloat16
        base_out = q_abs * (num_qo_heads * head_dim_ckv) + h * head_dim_ckv
        for k in range(0, head_dim_ckv):
            tl.store(output_ptr + base_out + k, output_vec[k].to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton kernel"

        total_q = int(q_nope.shape[0])
        len_indptr = int(qo_indptr.shape[0])
        num_qo_heads = 16
        head_dim_ckv = 512
        head_dim_kpe = 64

        # Prepare outputs
        outputs = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=q_nope.device)
        lse_out = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q_nope.device)

        # Launch Triton kernel: one program per (b, i)
        grid = (len_indptr - 1, total_q)
        _forward_single_query_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices,
            outputs, lse_out, total_q, len_indptr, float(sm_scale),
            num_qo_heads=num_qo_heads, head_dim_ckv=head_dim_ckv, head_dim_kpe=head_dim_kpe,
            num_warps=4,  # tune as needed
            num_stages=2  # tune as needed
        )

        return outputs, lse_out


def run(*args):
    return ModelNew()(*args)
