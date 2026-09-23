import torch
import math
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    q_nope_ptr,            # *bf16, [Q_total, 16, 512]
    q_pe_ptr,              # *bf16, [Q_total, 16, 64]
    Kc_sel_ptr,            # *bf16, [L_kv, 512]
    Kp_sel_ptr,            # *bf16, [L_kv, 64]
    output_ptr,            # *bf16, [Q_total, 16, 512]
    lse_ptr,               # *float32, [Q_total, 16]
    sm_scale,              # float32 scalar
    Q_total,               # int32
    num_heads,             # int32, should be 16
    head_dim_ckv,          # int32, 512
    head_dim_kpe,          # int32, 64
    qo_indptr_ptr,         # *int32, [len_indptr]
    kv_indptr_ptr,         # *int32, [len_indptr]
    kv_indices_ptr,        # *int32, [num_kv_indices]
    Batches,               # int32, len_indptr - 1
    L_kv,                  # int32, kv_len for this batch element
    Q_len,                 # int32, q_len for this batch element
    q_start,               # int32, starting query index of this batch element
    i,                     # int32, query index within this batch element
):
    q_abs = q_start + i

    # Loop over heads
    for h in range(0, num_heads):
        # Load qn[h, :] = q_nope[q_abs, h, :]
        base = q_abs * (head_dim_ckv + head_dim_kpe) + h * (head_dim_ckv + head_dim_kpe)
        qn = tl.zeros((head_dim_ckv,), dtype=tl.float32)
        for k in range(0, head_dim_ckv):
            val = tl.load(q_nope_ptr + base + k)
            qn[k] = val.to(tl.float32)

        # Load qp[h, :] = q_pe[q_abs, h, :]
        qp = tl.zeros((head_dim_kpe,), dtype=tl.float32)
        for k in range(0, head_dim_kpe):
            val = tl.load(q_pe_ptr + base + head_dim_ckv + k)
            qp[k] = val.to(tl.float32)

        # Compute logits[j] = (qn · Kc_sel[j]) + (qp · Kp_sel[j])
        logits = tl.zeros((L_kv,), dtype=tl.float32)
        for j in range(0, L_kv):
            # Load Kc_sel[j, :] and Kp_sel[j, :]
            Kc_row = tl.zeros((head_dim_ckv,), dtype=tl.float32)
            for kk in range(0, head_dim_ckv):
                val = tl.load(Kc_sel_ptr + j * head_dim_ckv + kk)
                Kc_row[kk] = val.to(tl.float32)

            Kp_row = tl.zeros((head_dim_kpe,), dtype=tl.float32)
            for kk in range(0, head_dim_kpe):
                val = tl.load(Kp_sel_ptr + j * head_dim_kpe + kk)
                Kp_row[kk] = val.to(tl.float32)

            dot_qn = 0.0
            for kk in range(0, head_dim_ckv):
                dot_qn += qn[kk] * Kc_row[kk]

            dot_qp = 0.0
            for kk in range(0, head_dim_kpe):
                dot_qp += qp[kk] * Kp_row[kk]

            logits[j] = dot_qn + dot_qp

        # Scale logits
        logits = logits * sm_scale

        # Causal mask: j >= prefix_len + i + 1, where prefix_len = L_kv - Q_len
        prefix_len = L_kv - Q_len
        # Set invalid positions to -inf
        for j in range(0, L_kv):
            if (prefix_len + i + 1) > j:
                logits[j] = -1e20

        # Stable logsumexp: max over logits
        max_log = -1e20
        for j in range(0, L_kv):
            if logits[j] > max_log:
                max_log = logits[j]

        # sum_exp = sum(exp(logits - max_log))
        sum_exp = 0.0
        for j in range(0, L_kv):
            sum_exp += tl.exp(logits[j] - max_log)

        # lse in log2
        lse_val = tl.log(sum_exp) + max_log
        lse_val = lse_val / tl.log(2.0)  # convert natural log to log2

        # Store lse[q_abs, h]
        tl.store(lse_ptr + q_abs * num_heads + h, lse_val)

        # Softmax over j
        softmax = tl.zeros((L_kv,), dtype=tl.float32)
        for j in range(0, L_kv):
            if (prefix_len + i + 1) <= j:
                softmax[j] = tl.exp(logits[j] - max_log) / sum_exp
            else:
                softmax[j] = 0.0

        # Output[h, :] = sum_j softmax[j] * Kc_sel[j, :]
        output_vec = tl.zeros((head_dim_ckv,), dtype=tl.float32)
        for j in range(0, L_kv):
            if (prefix_len + i + 1) <= j:
                Kc_row = tl.zeros((head_dim_ckv,), dtype=tl.float32)
                for kk in range(0, head_dim_ckv):
                    val = tl.load(Kc_sel_ptr + j * head_dim_ckv + kk)
                    Kc_row[kk] = val.to(tl.float32)
                output_vec += softmax[j] * Kc_row

        # Store output[q_abs, h, :]
        base_out = q_abs * (num_heads * head_dim_ckv) + h * head_dim_ckv
        for k in range(0, head_dim_ckv):
            tl.store(output_ptr + base_out + k, output_vec[k].to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure shapes and types
        assert q_nope.shape[1] == 16, "num_qo_heads must be 16"
        assert q_nope.shape[-1] == 512, "head_dim_ckv must be 512"
        assert q_pe.shape[-1] == 64, "head_dim_kpe must be 64"
        len_indptr = qo_indptr.shape[0]
        assert qo_indptr.shape == kv_indptr.shape == (len_indptr,)

        total_q = q_nope.shape[0]
        device = q_nope.device

        # Prepare outputs
        outputs = torch.empty((total_q, 16, 512), dtype=torch.bfloat16, device=device)
        lse_out = torch.empty((total_q, 16), dtype=torch.float32, device=device)

        # For each batch element, select the Kc_sel and Kp_sel rows based on kv_indices
        Batches = len_indptr - 1
        for b in range(Batches):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            kv_len = kv_end - kv_start

            # tok_idx = kv_indices[kv_start:kv_end]
            tok_idx = kv_indices[kv_start:kv_end].to(device=device, dtype=torch.int64)  # keep on device

            # Select Kc_sel and Kp_sel: [kv_len, 512] and [kv_len, 64] in bfloat16
            # Use .index_select for device-side selection
            Kc_sel = ckv_cache.index_select(0, tok_idx.long()).to(torch.bfloat16)  # [kv_len, 512]
            Kp_sel = kpe_cache.index_select(0, tok_idx.long()).to(torch.bfloat16)  # [kv_len, 64]

        # Launch Triton kernel: one program per (b, i)
        grid = (Batches, q_len)
        _forward_single_query_kernel[grid](
            q_nope, q_pe,
            Kc_sel, Kp_sel,
            outputs, lse_out,
            sm_scale,
            total_q, 16, 512, 64,
            qo_indptr, kv_indptr, kv_indices,
            Batches,  # L_kv per b
            q_len,    # Q_len per b
            q_start,  # per b
            0,        # i handled inside kernel via grid second dim
            num_warps=1,  # simple kernel; can tune later
            num_stages=1,
        )

        return outputs, lse_out


def run(*args):
    return ModelNew()(*args)
