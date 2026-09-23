import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_logits_and_lse_kernel(
    q_ptr, k_ptr, v_ptr, qo_indptr_ptr, kv_indptr_ptr,
    output_logits_ptr, lse_ptr,
    total_q, num_qo_heads, head_dim, num_kv_heads, gqa_ratio, sm_scale
):
    # Grid: (b, q_token, qo_head)
    b = tl.program_id(0)
    q_token = tl.program_id(1)
    qo_head = tl.program_id(2)

    # Load batch bounds
    qo_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    qo_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)

    if qo_start >= qo_end or kv_start >= kv_end:
        return

    # num_q_tokens and num_kv_tokens for this batch
    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start

    # delta per original code: num_kv_tokens - num_q_tokens
    delta = num_kv_tokens - num_q_tokens

    # q vector: q[b, q_token, qo_head, :]
    q_vec = q_ptr + (qo_start + q_token) * (num_qo_heads * head_dim) + qo_head * head_dim

    # Output logits buffer layout: [len_indptr, total_q, num_qo_heads, out_len]
    out_len = num_kv_tokens * gqa_ratio
    base = (b * (total_q * num_qo_heads) + q_token * num_qo_heads + qo_head) * out_len

    # Accumulator for logsumexp
    sum_exp = 0.0  # scalar float32

    # Loop over original KV heads j and expanded positions r
    for j in range(0, num_kv_heads):
        for r in range(0, gqa_ratio):
            # Causal mask: kv_pos < q_token + 1 + delta, where kv_pos = j * gqa_ratio + r
            kv_pos = j * gqa_ratio + r
            if kv_pos < (q_token + 1 + delta):
                # k expanded address: k[kv_start + j, r, :]
                k_idx = kv_start + j
                k_vec = k_ptr + k_idx * (num_kv_heads * head_dim) + j * head_dim + r
                # Dot product over head_dim
                dot = 0.0
                for d in range(0, head_dim):
                    q_val = tl.load(q_vec + d)
                    k_val = tl.load(k_vec + d)
                    dot += q_val * k_val
                val = dot * sm_scale
            else:
                val = -float("inf")

            # Store logits at structured index: (b, q_token, qo_head, kv_pos)
            tl.store(output_logits_ptr + base + kv_pos, val)

            # Accumulate for LSE
            sum_exp += val if val > -1e20 else 0.0

    # Compute LSE in base-2: log(sum(exp(logits))) / log(2)
    lse_val = math.log(sum_exp) / math.log(2.0) if sum_exp > 0 else -float("inf")
    lse_index = (b * total_q + q_token) * num_qo_heads + qo_head
    tl.store(lse_ptr + lse_index, lse_val)


@triton.jit
def _compute_output_kernel(
    q_ptr, k_ptr, v_ptr, qo_indptr_ptr, kv_indptr_ptr,
    output_logits_ptr, lse_ptr, output_ptr,
    total_q, num_qo_heads, head_dim, num_kv_heads, gqa_ratio, sm_scale
):
    # Grid: (b, q_token, qo_head)
    b = tl.program_id(0)
    q_token = tl.program_id(1)
    qo_head = tl.program_id(2)

    # Load batch bounds
    qo_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    qo_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)

    if qo_start >= qo_end or kv_start >= kv_end:
        return

    # num_q_tokens and num_kv_tokens for this batch
    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start
    delta = num_kv_tokens - num_q_tokens

    # Load q vector
    q_vec = q_ptr + (qo_start + q_token) * (num_qo_heads * head_dim) + qo_head * head_dim

    # Load LSE for this (q_token, qo_head)
    lse_index = (b * total_q + q_token) * num_qo_heads + qo_head
    lse_val = tl.load(lse_ptr + lse_index)

    # Output buffer layout: [len_indptr, total_q, num_qo_heads, head_dim]
    out_base = (b * (total_q * num_qo_heads) + q_token * num_qo_heads + qo_head) * head_dim

    # Accumulator for output vector
    out_vec = tl.zeros([head_dim], dtype=tl.float32)

    out_len = num_kv_tokens * gqa_ratio
    base = (b * (total_q * num_qo_heads) + q_token * num_qo_heads + qo_head) * out_len

    # Compute attention weights and accumulate output
    # Loop over all expanded positions kv_pos; apply causal mask and softmax normalization
    for kv_pos in range(0, out_len):
        if (kv_pos % gqa_ratio == 0) and (kv_start + (kv_pos // gqa_ratio) < (q_token + 1 + delta)):
            val = tl.load(output_logits_ptr + base + kv_pos)
            attn = tl.exp(val - lse_val)
            # Corresponding j and r
            j = kv_pos // gqa_ratio
            r = kv_pos % gqa_ratio
            # v expanded vector address: v[kv_start + j, r, :]
            v_idx = kv_start + j
            v_exp_vec = v_ptr + v_idx * (num_kv_heads * head_dim) + j * head_dim + r
            dot_v = 0.0
            for d in range(0, head_dim):
                v_val = tl.load(v_exp_vec + d)
                q_val = tl.load(q_vec + d)
                dot_v += q_val * v_val
            out_vec += attn * dot_v

    # Store output vector
    for d in range(0, head_dim):
        tl.store(output_ptr + out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA and contiguous
        device = q.device
        assert q.device == k.device == v.device, "q, k, v must be on the same device"
        assert device.type == "cuda", "This Triton implementation requires CUDA device"

        q_f32 = q.contiguous().to(torch.float32)
        k_f32 = k.contiguous().to(torch.float32)
        v_f32 = v.contiguous().to(torch.float32)
        qo_indptr = qo_indptr.contiguous().to(torch.int32)
        kv_indptr = kv_indptr.contiguous().to(torch.int32)

        # Shapes (fixed as per original)
        total_q = int(q_f32.shape[0])
        num_qo_heads = int(q_f32.shape[1])
        head_dim = int(q_f32.shape[2])
        total_kv = int(k_f32.shape[0])
        num_kv_heads = int(k_f32.shape[1])
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128, "Fixed head dims required"
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        len_indptr = int(qo_indptr.shape[0])
        num_batches = len_indptr

        # Allocate buffers
        # output_logits: [len_indptr, total_q, num_qo_heads, out_len]
        out_len = total_kv * gqa_ratio
        output_logits = torch.empty(
            (num_batches, total_q, num_qo_heads, out_len),
            dtype=torch.float32, device=device
        )
        lse = torch.empty(
            (num_batches, total_q, num_qo_heads),
            dtype=torch.float32, device=device
        )
        output = torch.empty(
            (total_q, num_qo_heads, head_dim),
            dtype=torch.float32, device=device
        )

        # Launch kernel 1: compute logits and lse
        _compute_logits_and_lse_kernel[(num_batches, total_q, num_qo_heads)](
            q_f32, k_f32, v_f32, qo_indptr, kv_indptr,
            output_logits, lse,
            total_q, num_qo_heads, head_dim, num_kv_heads, gqa_ratio, sm_scale
        )

        # Launch kernel 2: compute output
        _compute_output_kernel[(num_batches, total_q, num_qo_heads)](
            q_f32, k_f32, v_f32, qo_indptr, kv_indptr,
            output_logits, lse, output,
            total_q, num_qo_heads, head_dim, num_kv_heads, gqa_ratio, sm_scale
        )

        # Return output and lse (float32), shaped [total_q, num_qo_heads] for lse
        lse_final = lse.squeeze(1)  # reduce batch dim (for len_indptr=2)
        return output, lse_final


def run(*args):
    return ModelNew()(*args)
