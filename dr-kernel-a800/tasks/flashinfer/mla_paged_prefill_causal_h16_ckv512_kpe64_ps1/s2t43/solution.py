import math
import torch
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    q_nope_ptr,         # *bf16, [q_total, num_heads, head_dim_ckv]
    q_pe_ptr,           # *bf16, [q_total, num_heads, head_dim_kpe]
    ckv_cache_ptr,      # *bf16, [num_pages, 1, head_dim_ckv]
    kpe_cache_ptr,      # *bf16, [num_pages, 1, head_dim_kpe]
    qo_indptr_ptr,      # *int32, [len_indptr]
    kv_indptr_ptr,      # *int32, [len_indptr]
    kv_indices_ptr,     # *int32, [num_kv_indices]
    output_ptr,         # *bf16, [q_total, num_heads, head_dim_ckv]
    lse_ptr,            # *f32,  [q_total, num_heads]
    q_total: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim_ckv: tl.constexpr,
    head_dim_kpe: tl.constexpr,
    sm_scale,            # float scalar (host passes Python float)
):
    # Program ids: b in [0, len_indptr-1), i in [0, q_total)
    b = tl.program_id(0)
    i = tl.program_id(1)

    # Compute absolute query index
    qo_indptr_b = tl.load(qo_indptr_ptr + b)  # int32
    qo_indptr_b_plus = tl.load(qo_indptr_ptr + b + 1)
    q_abs = qo_indptr_b + i

    # Load qn and qp for all heads h
    for h in range(num_heads):
        qn_offset = q_abs * (num_heads * head_dim_ckv) + h * head_dim_ckv
        qn = tl.zeros((head_dim_ckv,), dtype=tl.float32)
        for k in range(0, head_dim_ckv):
            val = tl.load(q_nope_ptr + qn_offset + k)
            qn[k] = val.to(tl.float32)

        qp_offset = q_abs * (num_heads * head_dim_kpe) + h * head_dim_kpe
        qp = tl.zeros((head_dim_kpe,), dtype=tl.float32)
        for k in range(0, head_dim_kpe):
            val = tl.load(q_pe_ptr + qp_offset + k)
            qp[k] = val.to(tl.float32)

        # Compute kv_len and select tok_idx for this batch
        kv_indptr_b = tl.load(kv_indptr_ptr + b)
        kv_indptr_b_plus = tl.load(kv_indptr_ptr + b + 1)
        kv_len = kv_indptr_b_plus - kv_indptr_b

        prefix_len = kv_len - (qo_indptr_b_plus - qo_indptr_b)

        # Prepare logits vector for this head
        logits = tl.zeros((kv_len,), dtype=tl.float32)

        # Loop over j and compute logits[j] = (qn · Kc_sel[j]) + (qp · Kp_sel[j])
        # Select Kc_sel[j, :] and Kp_sel[j, :] from caches using kv_indices[b]
        for j in range(0, kv_len):
            tok_idx_j = tl.load(kv_indices_ptr + j + kv_indptr_b)
            Kc_sel_ptr = ckv_cache_ptr + tok_idx_j * head_dim_ckv  # cache has [num_pages, 1, D] -> stride D per row
            Kp_sel_ptr = kpe_cache_ptr + tok_idx_j * head_dim_kpe

            # Dot with qn
            dot_qn = 0.0
            for kk in range(0, head_dim_ckv):
                val = tl.load(Kc_sel_ptr + kk)
                dot_qn += qn[kk] * val.to(tl.float32)
            # Dot with qp
            dot_qp = 0.0
            for kk in range(0, head_dim_kpe):
                val = tl.load(Kp_sel_ptr + kk)
                dot_qp += qp[kk] * val.to(tl.float32)

            logits[j] = dot_qn + dot_qp

        # Apply causal mask: only j >= prefix_len + i are valid
        valid = j >= (prefix_len + i)
        logits = tl.where(valid, logits, -1.0e20)

        # Stable logsumexp over logits: m = max(logits), sum = sum(exp(logits - m))
        m = tl.max(logits, axis=0)
        logits_shift = logits - m
        exp_logits = tl.exp(logits_shift)
        sum_exp = tl.sum(exp_logits, axis=0)
        lse_val = (m + tl.log(sum_exp)) / math.log(2.0)  # logsumexp in log2
        tl.store(lse_ptr + q_abs * num_heads + h, lse_val)

        # Softmax over logits (stable), then attention output
        softmax = exp_logits / sum_exp  # already exp(logits - m)
        output_vec = tl.zeros((head_dim_ckv,), dtype=tl.float32)
        for j in range(0, kv_len):
            tok_idx_j = tl.load(kv_indices_ptr + j + kv_indptr_b)
            Kc_sel_ptr = ckv_cache_ptr + tok_idx_j * head_dim_ckv
            for k in range(0, head_dim_ckv):
                val = tl.load(Kc_sel_ptr + k)
                output_vec[k] += softmax[j] * val.to(tl.float32)

        # Store output[q_abs, h, :]
        base_out = q_abs * (num_heads * head_dim_ckv) + h * head_dim_ckv
        for k in range(0, head_dim_ckv):
            tl.store(output_ptr + base_out + k, output_vec[k].to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure shapes and device
        assert q_nope.ndim == 3, "q_nope must be [q_total, num_heads, head_dim_ckv]"
        assert q_pe.ndim == 3, "q_pe must be [q_total, num_heads, head_dim_kpe]"
        assert ckv_cache.ndim == 3 and ckv_cache.shape[1] == 1, "ckv_cache must be [num_pages, 1, head_dim_ckv]"
        assert kpe_cache.ndim == 3 and kpe_cache.shape[1] == 1, "kpe_cache must be [num_pages, 1, head_dim_kpe]"
        assert qo_indptr.ndim == 1 and qo_indptr.shape[0] > 1, "qo_indptr must be [len_indptr]"
        assert kv_indptr.ndim == 1 and kv_indptr.shape[0] > 1, "kv_indptr must be [len_indptr]"
        assert kv_indices.ndim == 1, "kv_indices must be [num_kv_indices]"

        q_total = q_nope.shape[0]
        num_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Assert constants as in original
        assert num_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"

        device = q_nope.device
        len_indptr = qo_indptr.shape[0]

        # Allocate outputs
        outputs = torch.empty((q_total, num_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse_out = torch.empty((q_total, num_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, i)
        grid = (len_indptr - 1, q_total)
        _forward_single_query_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices,
            outputs, lse_out,
            q_total=q_total,
            num_heads=num_heads,
            head_dim_ckv=head_dim_ckv,
            head_dim_kpe=head_dim_kpe,
            sm_scale=float(sm_scale),  # pass scalar
            num_warps=4,
        )

        return outputs, lse_out


def run(*args):
    return ModelNew()(*args)
