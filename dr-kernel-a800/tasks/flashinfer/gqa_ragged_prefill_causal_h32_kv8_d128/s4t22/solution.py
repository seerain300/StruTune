import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_logits_and_lse_kernel(
    q_ptr, k_ptr, v_ptr, qo_indptr_ptr, kv_indptr_ptr,
    logits_flat_ptr, lse_ptr,
    total_q, num_qo_heads, head_dim, num_kv_heads, gqa_ratio
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

    # num tokens for this batch
    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start
    delta = num_kv_tokens - num_q_tokens

    # q vector: [Q, 32, 128] -> q[b, q_token, qo_head, :]
    q_vec = q_ptr + (qo_start + q_token) * (num_qo_heads * head_dim) + qo_head * head_dim

    # We will write logits_flat per (b, q_token, qo_head) in order of j, r
    out_len = num_kv_tokens * gqa_ratio
    base = (b * (total_q * num_qo_heads) + q_token * num_qo_heads + qo_head) * out_len

    sum_exp = 0.0  # scalar for logsumexp

    # Loop over original KV heads and expanded positions
    for j in range(0, num_kv_heads):
        for r in range(0, gqa_ratio):
            kv_pos = j * gqa_ratio + r
            # causal mask: allow if kv_pos < q_token + 1 + delta
            if kv_pos < (q_token + 1 + delta):
                # k address: k[kv_start + j, j, r]
                k_idx = kv_start + j
                k_vec = k_ptr + k_idx * (num_kv_heads * head_dim) + j * head_dim + r
                dot = 0.0
                for d in range(0, head_dim):
                    q_val = tl.load(q_vec + d)
                    k_val = tl.load(k_vec + d)
                    dot += q_val * k_val
                val = dot  # already scaled by sm_scale outside if needed
            else:
                val = -float("inf")

            # Store logits at linear index
            tl.store(logits_flat_ptr + base + kv_pos, val)
            # Accumulate for LSE (ignoring -inf)
            sum_exp += val if val > -1e20 else 0.0

    # Compute LSE in base-2: log(sum_exp) / log(2)
    lse_val = math.log(sum_exp) / math.log(2.0) if sum_exp > 0 else -float("inf")
    lse_index = (b * total_q + q_token) * num_qo_heads + qo_head
    tl.store(lse_ptr + lse_index, lse_val)


@triton.jit
def _compute_output_kernel(
    q_ptr, k_ptr, v_ptr, qo_indptr_ptr, kv_indptr_ptr,
    logits_flat_ptr, lse_ptr, output_ptr,
    total_q, num_qo_heads, head_dim, num_kv_heads, gqa_ratio
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

    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start
    delta = num_kv_tokens - num_q_tokens

    # q vector
    q_vec = q_ptr + (qo_start + q_token) * (num_qo_heads * head_dim) + qo_head * head_dim

    # Load lse
    lse_index = (b * total_q + q_token) * num_qo_heads + qo_head
    lse_val = tl.load(lse_ptr + lse_index)

    out_base = (b * (total_q * num_qo_heads) + q_token * num_qo_heads + qo_head) * head_dim
    out_vec = tl.zeros([head_dim], dtype=tl.float32)

    out_len = num_kv_tokens * gqa_ratio
    base = (b * (total_q * num_qo_heads) + q_token * num_qo_heads + qo_head) * out_len

    # First pass: compute sum of numerator for softmax
    sum_num = 0.0
    for kv_pos in range(0, out_len):
        val = tl.load(logits_flat_ptr + base + kv_pos)
        sum_num += tl.exp(val - lse_val)

    # Second pass: compute output by weighted sum with v_expanded
    for kv_pos in range(0, out_len):
        val = tl.load(logits_flat_ptr + base + kv_pos)
        attn = tl.exp(val - lse_val)
        j = kv_pos // gqa_ratio
        r = kv_pos % gqa_ratio
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
        # Triton requires CUDA tensors
        device = q.device
        assert q.device == k.device == v.device, "q, k, v must be on the same device"
        assert device.type == "cuda", "This Triton implementation requires a CUDA device"

        # Convert to float32 and ensure contiguity
        q_f32 = q.contiguous().to(torch.float32)
        k_f32 = k.contiguous().to(torch.float32)
        v_f32 = v.contiguous().to(torch.float32)
        qo_indptr = qo_indptr.contiguous().to(torch.int32)
        kv_indptr = kv_indptr.contiguous().to(torch.int32)

        # Shapes
        total_q = int(q_f32.shape[0])
        num_qo_heads = int(q_f32.shape[1])
        head_dim = int(q_f32.shape[2])
        total_kv = int(k_f32.shape[0])
        num_kv_heads = int(k_f32.shape[1])

        # Fixed assertions as in original
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128, "Fixed dims required"
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        len_indptr = int(qo_indptr.shape[0])

        # Allocate buffers
        logits_flat = torch.empty(
            (len_indptr * total_q * num_qo_heads * (num_kv_heads * gqa_ratio)),
            dtype=torch.float32, device=device
        )
        lse = torch.empty(
            (len_indptr * total_q * num_qo_heads),
            dtype=torch.float32, device=device
        )
        output = torch.empty(
            (total_q, num_qo_heads, head_dim),
            dtype=torch.float32, device=device
        )

        # Launch Triton kernels
        _compute_logits_and_lse_kernel[(len_indptr, total_q, num_qo_heads)](
            q_f32, k_f32, v_f32, qo_indptr, kv_indptr,
            logits_flat, lse,
            total_q, num_qo_heads, head_dim, num_kv_heads, gqa_ratio
        )

        _compute_output_kernel[(len_indptr, total_q, num_qo_heads)](
            q_f32, k_f32, v_f32, qo_indptr, kv_indptr,
            logits_flat, lse, output,
            total_q, num_qo_heads, head_dim, num_kv_heads, gqa_ratio
        )

        # Reshape lse to [total_q, num_qo_heads]
        lse = lse.view(total_q, num_qo_heads)

        return output, lse


def run(*args):
    return ModelNew()(*args)
