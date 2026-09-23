import torch
import math
import triton
import triton.language as tl


@triton.jit
def _fused_attention_kernel(
    q_ptr, k_ptr, v_ptr, qo_indptr_ptr, kv_indptr_ptr,
    output_ptr, lse_ptr, sm_scale
):
    # Grid over (batch, q_token, qo_head)
    b = tl.program_id(0)
    q_token = tl.program_id(1)
    qo_head = tl.program_id(2)

    # Load batch segments
    qo_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    qo_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)

    if qo_start >= qo_end or kv_start >= kv_end:
        return

    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start
    gqa_ratio = 4  # fixed because num_qo_heads=32, num_kv_heads=8

    # q vector for this q_token and qo_head
    q_index = qo_start + q_token
    base_q = q_index * (32 * 128) + qo_head * 128
    q_vec = tl.load(q_ptr + base_q)  # [128], float32

    # Prepare accumulators
    sum_exp = 0.0  # for LSE
    out_vec = tl.zeros([128], dtype=tl.float32)

    # Loop over KV tokens and repeats for expanded K/V
    for j in range(0, num_kv_tokens):
        for r in range(0, gqa_ratio):
            # kv head mapping for GQA: use qo_head % 8
            kv_head = qo_head % 8
            kv_index = kv_start + j
            base_k = kv_index * (8 * 128) + kv_head * 128
            k_vec = tl.load(k_ptr + base_k)  # [128], float32

            # Dot product over head_dim = 128
            dot = 0.0
            for d in range(0, 128):
                dot += q_vec[d] * k_vec[d]

            # Scale logits
            val = dot * sm_scale  # scalar

            # Causal mask: for q_token, valid KV positions are [0, min(q_token + 1 + (num_kv_tokens - num_q_tokens), num_kv_tokens))
            # Compute delta
            delta = num_kv_tokens - num_q_tokens
            # mask_ok if j < (q_token + 1 + delta)
            mask_ok = j < (q_token + 1 + delta)
            # Apply mask via tl.where
            val = tl.where(mask_ok, val, -float("inf"))

            # Accumulate sum_exp for LSE
            sum_exp += tl.exp(val)

            # Load corresponding v vector
            base_v = kv_index * (8 * 128) + kv_head * 128
            v_vec = tl.load(v_ptr + base_v)  # [128], float32

            # Compute softmax weight: exp(val - lse) where lse will be computed later
            # We'll recompute softmax in the second pass using saved logits; but here we only need to write per-(j,r) contribution.
            # Instead, we compute weight = exp(val - lse) in the second loop.
            # For now, continue to accumulate out_vec using a temporary weight=1.0 (we'll overwrite in the second pass).
            # To correctly compute output, we need lse; we'll do it in the second loop.

    # Second pass: compute LSE and then output using softmax over all valid (j, r)
    # Compute lse in base-2: lse = log(sum_exp) / log(2.0)
    lse_val = tl.log(sum_exp) / math.log(2.0)

    # Now compute output: for each (j, r), weight = exp(val - lse), then out += weight * v_vec
    for j in range(0, num_kv_tokens):
        for r in range(0, gqa_ratio):
            kv_head = qo_head % 8
            kv_index = kv_start + j
            base_k = kv_index * (8 * 128) + kv_head * 128
            k_vec = tl.load(k_ptr + base_k)  # [128]
            dot = 0.0
            for d in range(0, 128):
                dot += q_vec[d] * k_vec[d]
            val = dot * sm_scale
            mask_ok = j < (q_token + 1 + (num_kv_tokens - num_q_tokens))
            val = tl.where(mask_ok, val, -float("inf"))
            weight = tl.exp(val - lse_val)  # softmax weight
            base_v = kv_index * (8 * 128) + kv_head * 128
            v_vec = tl.load(v_ptr + base_v)
            for d in range(0, 128):
                out_vec[d] += weight * v_vec[d]

    # Store output vector for this (q_index, qo_head, :)
    base_out = q_index * (32 * 128) + qo_head * 128
    for d in range(0, 128):
        tl.store(output_ptr + base_out + d, out_vec[d])

    # Store lse for this (b, q_token, qo_head)
    lse_index = b * (num_q_tokens * 32) + q_token * 32 + qo_head
    tl.store(lse_ptr + lse_index, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors and contiguity
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        qo_indptr = qo_indptr.contiguous()
        kv_indptr = kv_indptr.contiguous()

        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        assert head_dim == 128, "head_dim must be 128"
        device = q.device

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        len_indptr = qo_indptr.shape[0]
        grid = (len_indptr - 1, q.shape[0], num_qo_heads)  # (b, q_token, qo_head)

        _fused_attention_kernel[grid](
            q, k, v, qo_indptr, kv_indptr,
            output, lse, sm_scale
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
