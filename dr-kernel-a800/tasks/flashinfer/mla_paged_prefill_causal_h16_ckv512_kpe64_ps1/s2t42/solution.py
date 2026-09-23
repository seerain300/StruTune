import math
import torch
import triton
import triton.language as tl


@triton.jit
def _forward_kernel(
    q_nope_ptr,        # *bf16, [total_q, 16, 512]
    q_pe_ptr,          # *bf16, [total_q, 16, 64]
    ckv_cache_ptr,     # *bf16, [num_pages, 1, 512]
    kpe_cache_ptr,     # *bf16, [num_pages, 1, 64]
    qo_indptr_ptr,     # *int32, [len_indptr]
    kv_indptr_ptr,     # *int32, [len_indptr]
    kv_indices_ptr,    # *int32, [num_kv_indices]
    output_ptr,        # *bf16, [total_q, 16, 512]
    lse_ptr,           # *float32, [total_q, 16]
    total_q: tl.int32,             # int
    num_heads: tl.int32,           # int, e.g. 16
    head_dim_ckv: tl.int32,        # int, e.g. 512
    head_dim_kpe: tl.int32,        # int, e.g. 64
    len_indptr: tl.int32,          # int
    sm_scale,                      # scalar float32
    i  # query index (int32)
):
    # program ids: pid0 = batch element index (b), pid1 = query index within that batch (i)
    b = tl.program_id(0)
    q_abs = qo_indptr_ptr[b] + i  # absolute query index

    # Compute kv_len for this batch element
    kv_start = kv_indptr_ptr[b]
    kv_end = kv_indptr_ptr[b + 1]
    kv_len = kv_end - kv_start

    # Compute q_len for this batch element
    q_start = qo_indptr_ptr[b]
    q_end = qo_indptr_ptr[b + 1]
    q_len = q_end - q_start

    # Compute prefix_len (number of previously cached tokens)
    prefix_len = kv_len - q_len

    # Load qn[h, :] and qp[h, :] for all heads h
    num_heads_const = 16  # matches original code requirement
    for h in range(0, num_heads_const):
        # qn[h, :] from q_nope[q_abs, h, :]
        qn_base = q_abs * (num_heads_const * head_dim_ckv) + h * head_dim_ckv
        qn_vec = tl.zeros((head_dim_ckv,), dtype=tl.float32)
        for k in range(0, head_dim_ckv):
            val = tl.load(q_nope_ptr + qn_base + k)
            qn_vec[k] = val.to(tl.float32)

        # qp[h, :] from q_pe[q_abs, h, :]
        qp_base = q_abs * (num_heads_const * head_dim_kpe) + h * head_dim_kpe
        qp_vec = tl.zeros((head_dim_kpe,), dtype=tl.float32)
        for k in range(0, head_dim_kpe):
            val = tl.load(q_pe_ptr + qp_base + k)
            qp_vec[k] = val.to(tl.float32)

        # Initialize outputs
        output_vec = tl.zeros((head_dim_ckv,), dtype=tl.float32)
        logits = tl.zeros((kv_len,), dtype=tl.float32)
        max_log = tl.full((), -float('inf'), tl.float32)
        sum_exp = tl.zeros((), dtype=tl.float32)

        # Compute logits for each j
        for j in range(0, kv_len):
            # Select Kc_sel[j, :] = ckv_cache[tok_idx[j], 0, :]
            tok_idx_j = tl.load(kv_indices_ptr + j + kv_start).to(tl.int64)
            Kc_row_base = tok_idx_j * head_dim_ckv  # since ckv_cache shape [num_pages, 1, 512], row index is tok_idx_j
            Kc_row = tl.zeros((head_dim_ckv,), dtype=tl.float32)
            for kk in range(0, head_dim_ckv):
                val = tl.load(ckv_cache_ptr + Kc_row_base + kk)
                Kc_row[kk] = val.to(tl.float32)

            # Select Kp_sel[j, :] = kpe_cache[tok_idx[j], 0, :]
            Kp_row_base = tok_idx_j * head_dim_kpe
            Kp_row = tl.zeros((head_dim_kpe,), dtype=tl.float32)
            for kk in range(0, head_dim_kpe):
                val = tl.load(kpe_cache_ptr + Kp_row_base + kk)
                Kp_row[kk] = val.to(tl.float32)

            # Compute logits[j] = (qn · Kc_sel[j]) + (qp · Kp_sel[j])
            dot_qn = 0.0
            for kk in range(0, head_dim_ckv):
                dot_qn += qn_vec[kk] * Kc_row[kk]
            dot_qp = 0.0
            for kk in range(0, head_dim_kpe):
                dot_qp += qp_vec[kk] * Kp_row[kk]
            logits[j] = dot_qn + dot_qp

        # Apply causal mask: j >= prefix_len + i
        # Stable logsumexp in log2
        # First pass: find max for stability
        # We'll iterate and update max_log
        for j in range(0, kv_len):
            if (j >= (prefix_len + i)):
                # Update max_log
                if logits[j] > max_log:
                    max_log = logits[j]

        # Second pass: compute sum_exp = sum(exp(logits - max_log))
        for j in range(0, kv_len):
            if (j >= (prefix_len + i)):
                sum_exp += tl.exp(logits[j] - max_log)

        # Compute lse (logsumexp) in log2
        lse_val = (max_log + tl.log(sum_exp)) / math.log(2.0)

        # Store lse[q_abs, h]
        lse_addr = q_abs * (num_heads_const * 1) + h  # lse is [Q, num_heads]
        tl.store(lse_ptr + lse_addr, lse_val)

        # Third pass: compute softmax and attention output
        for j in range(0, kv_len):
            if (j >= (prefix_len + i)):
                exp_val = tl.exp(logits[j] - max_log)
                prob = exp_val / (sum_exp + 0.0)
                # out[h, :] += prob * Kc_sel[j, :]
                Kc_row_base = tok_idx_j * head_dim_ckv
                for k in range(0, head_dim_ckv):
                    Kc_elem = tl.load(ckv_cache_ptr + Kc_row_base + k).to(tl.float32)
                    output_vec[k] += prob * Kc_elem

        # Store output[q_abs, h, :]
        out_base = q_abs * (num_heads_const * head_dim_ckv) + h * head_dim_ckv
        for k in range(0, head_dim_ckv):
            tl.store(output_ptr + out_base + k, output_vec[k].to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on the same device
        device = q_nope.device
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        ckv_cache = ckv_cache.contiguous()
        kpe_cache = kpe_cache.contiguous()
        qo_indptr = qo_indptr.to(device=device, dtype=torch.int32)
        kv_indptr = kv_indptr.to(device=device, dtype=torch.int32)
        kv_indices = kv_indices.to(device=device, dtype=torch.int32)

        total_q = q_nope.shape[0]
        assert q_nope.shape[1] == 16, "num_qo_heads must be 16"
        head_dim_ckv = q_nope.shape[-1]
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert q_pe.shape[-1] == 64, "head_dim_kpe must be 64"
        len_indptr = qo_indptr.shape[0]
        assert len_indptr > 0

        # Output buffers
        outputs = torch.empty((total_q, 16, 512), dtype=torch.bfloat16, device=device)
        lse_out = torch.empty((total_q, 16), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, i)
        grid = (len_indptr - 1, total_q)
        _forward_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices,
            outputs, lse_out,
            total_q, 16, head_dim_ckv, 64, len_indptr,
            sm_scale,  # scalar float
            lambda META: (len_indptr - 1, total_q),  # (b, i)
        )

        return outputs, lse_out


def run(*args):
    return ModelNew()(*args)
