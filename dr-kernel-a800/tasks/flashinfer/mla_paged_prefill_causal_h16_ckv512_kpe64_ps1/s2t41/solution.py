import math
import torch
import triton
import triton.language as tl


@triton.jit
def _forward_single_query_kernel(
    q_nope_ptr,         # *bf16, [total_q, 16, 512]
    q_pe_ptr,           # *bf16, [total_q, 16, 64]
    ckv_cache_ptr,      # *bf16, [num_pages, 1, 512]
    kpe_cache_ptr,      # *bf16, [num_pages, 1, 64]
    qo_indptr_ptr,      # *int32, [len_indptr]
    kv_indptr_ptr,      # *int32, [len_indptr]
    kv_indices_ptr,     # *int32, [num_kv_indices]
    output_ptr,         # *bf16, [total_q, 16, 512]
    lse_ptr,            # *fp32, [total_q, 16]
    sm_scale: tl.constexpr,       # scale for logits
    inv_ln2: tl.constexpr,        # 1 / ln(2)
    num_heads: tl.constexpr,      # 16
    head_dim_ckv: tl.constexpr,   # 512
    head_dim_kpe: tl.constexpr,   # 64
):
    # 2D grid: (len_indptr-1, total_q). program_id(0) = b, program_id(1) = i
    b = tl.program_id(0)
    i = tl.program_id(1)

    # Read q_start and q_end
    q_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    q_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    q_len = q_end - q_start

    # Absolute query index
    q_abs = q_start + i

    # Loop over heads
    for h in range(num_heads):
        # Load qn[h, :] and qp[h, :]
        base_qn = q_abs * (num_heads * head_dim_ckv) + h * head_dim_ckv
        qn = tl.zeros((head_dim_ckv,), dtype=tl.float32)
        j = 0
        while j < head_dim_ckv:
            val = tl.load(q_nope_ptr + base_qn + j).to(tl.float32)
            qn[j] = val
            j += 1

        base_qp = q_abs * (num_heads * head_dim_kpe) + h * head_dim_kpe
        qp = tl.zeros((head_dim_kpe,), dtype=tl.float32)
        j = 0
        while j < head_dim_kpe:
            val = tl.load(q_pe_ptr + base_qp + j).to(tl.float32)
            qp[j] = val
            j += 1

        # Compute kv_len and tok_idx for this batch element
        kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
        kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
        kv_len = kv_end - kv_start

        tok_idx = tl.zeros((kv_len,), dtype=tl.int32)
        # Fill tok_idx[j] = kv_indices[kv_start + j]
        j = 0
        while j < kv_len:
            idx = kv_start + j
            tok = tl.load(kv_indices_ptr + idx).to(tl.int32)
            tok_idx[j] = tok
            j += 1

        # Prepare Kc_sel and Kp_sel buffers in float32
        Kc_sel = tl.zeros((kv_len, head_dim_ckv), dtype=tl.float32)
        Kp_sel = tl.zeros((kv_len, head_dim_kpe), dtype=tl.float32)

        # Fill Kc_sel and Kp_sel rows selected by tok_idx
        j = 0
        while j < kv_len:
            base_ckv = tok_idx[j] * head_dim_ckv
            kk = 0
            while kk < head_dim_ckv:
                val_ckv = tl.load(ckv_cache_ptr + base_ckv + kk).to(tl.float32)
                Kc_sel[j, kk] = val_ckv
                kk += 1

            base_kpe = tok_idx[j] * head_dim_kpe
            kk = 0
            while kk < head_dim_kpe:
                val_kpe = tl.load(kpe_cache_ptr + base_kpe + kk).to(tl.float32)
                Kp_sel[j, kk] = val_kpe
                kk += 1

            j += 1

        # Compute logits for each j
        logits = tl.zeros((kv_len,), dtype=tl.float32)
        j = 0
        while j < kv_len:
            dot_qn = 0.0
            kk = 0
            while kk < head_dim_ckv:
                dot_qn += qn[kk] * Kc_sel[j, kk]
                kk += 1
            dot_qp = 0.0
            kk = 0
            while kk < head_dim_kpe:
                dot_qp += qp[kk] * Kp_sel[j, kk]
                kk += 1
            logits[j] = dot_qn + dot_qp
            j += 1

        # Apply scaling and causal mask: only j in [prefix_len + i + 1, kv_len)
        prefix_len = kv_len - q_len
        start_j = prefix_len + i + 1

        # For lse computation: use -inf for invalid j so they don't affect max/sum
        j = 0
        while j < kv_len:
            if start_j <= kv_len and start_j > j:
                # j is invalid for this query: set to -inf
                logits[j] = -float('inf')
            j += 1

        # Scale logits
        j = 0
        while j < kv_len:
            logits[j] = logits[j] * sm_scale
            j += 1

        # Compute logsumexp in log2: lse = log(sum(exp(logits - m))) + m; then * inv_ln2
        m = -float('inf')
        j = 0
        while j < kv_len:
            if logits[j] > m:
                m = logits[j]
            j += 1

        sum_exp = 0.0
        j = 0
        while j < kv_len:
            sum_exp += tl.exp(logits[j] - m)
            j += 1

        lse_val = m + tl.log(sum_exp) * inv_ln2

        # Compute softmax over valid j (invalid j are -inf and thus have exp=0)
        denom = 0.0
        j = 0
        while j < kv_len:
            denom += tl.exp(logits[j])
            j += 1

        softmax = tl.zeros((kv_len,), dtype=tl.float32)
        j = 0
        while j < kv_len:
            softmax[j] = tl.exp(logits[j]) / denom
            j += 1

        # Compute attention output: out[h, :] = sum_j softmax[j] * Kc_sel[j, :]
        output_vec = tl.zeros((head_dim_ckv,), dtype=tl.float32)
        j = 0
        while j < kv_len:
            # valid_j = (start_j <= kv_len) and (start_j > j) is false here because we already masked logits.
            # However, we must still guard the contribution: since softmax[j] = exp(-inf) for invalid j, it's 0.
            # So we can simply accumulate without extra guard, but we can enforce by recomputing active_j per j.
            active_j = (start_j <= kv_len) and (start_j > j)
            # Kc_sel[j, :]
            Kc_row = tl.zeros((head_dim_ckv,), dtype=tl.float32)
            base_kc = tok_idx[j] * head_dim_ckv
            kk = 0
            while kk < head_dim_ckv:
                val = tl.load(ckv_cache_ptr + base_kc + kk).to(tl.float32)
                Kc_row[kk] = val
                kk += 1
            # Only add if active_j (softmax[j] is 0 when j invalid)
            output_vec += softmax[j] * Kc_row
            j += 1

        # Store output[q_abs, h, :] as bfloat16
        base_out = q_abs * (num_heads * head_dim_ckv) + h * head_dim_ckv
        k = 0
        while k < head_dim_ckv:
            tl.store(output_ptr + base_out + k, output_vec[k].to(tl.bfloat16))
            k += 1

        # Store lse[q_abs, h] as float32
        tl.store(lse_ptr + q_abs * num_heads + h, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure dtypes/device
        device = q_nope.device
        q_nope = q_nope.to(device=device, dtype=torch.bfloat16)
        q_pe = q_pe.to(device=device, dtype=torch.bfloat16)
        ckv_cache = ckv_cache.to(device=device, dtype=torch.bfloat16)
        kpe_cache = kpe_cache.to(device=device, dtype=torch.bfloat16)
        qo_indptr = qo_indptr.to(device=device, dtype=torch.int32)
        kv_indptr = kv_indptr.to(device=device, dtype=torch.int32)
        kv_indices = kv_indices.to(device=device, dtype=torch.int32)

        total_q = q_nope.shape[0]
        num_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]
        assert num_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"

        # Outputs
        outputs = torch.empty((total_q, num_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse_out = torch.empty((total_q, num_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: grid = (len_indptr - 1, total_q)
        len_indptr = qo_indptr.shape[0]
        grid = (len_indptr - 1, total_q)
        inv_ln2 = 1.0 / math.log(2.0)

        _forward_single_query_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices,
            outputs, lse_out,
            sm_scale=sm_scale, inv_ln2=inv_ln2,
            num_heads=num_heads, head_dim_ckv=head_dim_ckv, head_dim_kpe=head_dim_kpe,
            num_warps=4,  # modest default; can tune
            num_stages=2
        )

        return outputs, lse_out


def run(*args):
    return ModelNew()(*args)
