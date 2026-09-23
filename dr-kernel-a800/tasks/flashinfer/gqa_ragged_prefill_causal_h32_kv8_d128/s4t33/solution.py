import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_lse_kernel(q_ptr, k_ptr, v_ptr,  # inputs
                       qo_indptr, kv_indptr,  # int32 per-batch offsets
                       sm_scale,              # float32 scale
                       output_lse_ptr,       # float32 [len_indptr, total_q, 32] to write LSE
                       total_q, total_kv, len_indptr,
                       num_qo_heads, num_kv_heads,
                       head_dim,
                       gqa_ratio: tl.constexpr,
                       BLOCK_Q: tl.constexpr,  # number of q tokens to process per program (we set 1)
                       BLOCK_K: tl.constexpr):  # number of k tokens chunk to compute dot (we set 1)
    # program ids
    b = tl.program_id(0)          # batch element
    q_token = tl.program_id(1)    # query token index
    qo_head = tl.program_id(2)    # query head index

    # compute ranges
    qo_start = tl.load(qo_indptr + b).to(tl.int32)
    qo_end = tl.load(qo_indptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr + b + 1).to(tl.int32)

    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start
    out_len = num_kv_tokens * gqa_ratio

    # if empty batch segment, skip
    if (num_q_tokens <= 0) or (num_kv_tokens <= 0):
        return

    # pointers for this q vector: q[b, q_token, qo_head, :]
    q_off = (qo_start * num_qo_heads * head_dim) + (qo_head * head_dim) + (q_token * head_dim)
    q_vec = tl.load(q_ptr + q_off)

    # initialize sum_exp for logsumexp
    sum_exp = 0.0  # float32 scalar

    # iterate over expanded KV heads: j over original kv heads, r over 4 repeats
    # We compute masked dot products for each expanded position kv_pos in [0, out_len)
    for j in range(0, num_kv_heads):
        for r in range(0, gqa_ratio):
            kv_pos = j * gqa_ratio + r
            if kv_pos >= out_len:
                break
            # corresponding original kv index
            kq_idx = kv_start + (kv_pos // gqa_ratio)  # since gqa_ratio == 4, kq_idx is the row in k/batch
            # load k and v vectors for this expanded head position
            k_off = (kv_start * num_kv_heads * head_dim) + (j * head_dim) + (kq_idx * head_dim)
            v_off = (kv_start * num_kv_heads * head_dim) + (j * head_dim) + (kq_idx * head_dim)
            k_vec = tl.load(k_ptr + k_off)
            v_vec = tl.load(v_ptr + v_off)

            # compute dot product
            dot = 0.0
            for d in range(0, head_dim):
                dot += q_vec[d] * k_vec[d]

            # scaling
            dot = dot * sm_scale

            # causal mask: a query at q_token can attend to kv positions < q_token + 1 + delta
            # where delta = num_kv_tokens - num_q_tokens (can be negative; original asserts ensure non-empty segments)
            delta = num_kv_tokens - num_q_tokens
            bound = q_token + 1 + delta
            if kv_pos >= bound:
                dot = -float("inf")
            else:
                # compute exp for LSE
                sum_exp += tl.exp(dot)

    # store LSE per (b, q_token, qo_head) into output_lse_ptr[b, q_token, qo_head]
    lse_val = tl.log(sum_exp) / math.log(2.0)
    out_lse_off = b * (total_q * num_qo_heads) + q_token * num_qo_heads + qo_head
    tl.store(output_lse_ptr + out_lse_off, lse_val)


@triton.jit
def compute_output_kernel(q_ptr, k_ptr, v_ptr,      # inputs
                          qo_indptr, kv_indptr,    # int32 per-batch offsets
                          sm_scale,               # float32 scale
                          output_ptr,            # float32 [total_q, 32, 128] to write output
                          total_q, total_kv, len_indptr,
                          num_qo_heads, num_kv_heads,
                          head_dim,
                          gqa_ratio: tl.constexpr,
                          BLOCK_Q: tl.constexpr,   # 1
                          BLOCK_K: tl.constexpr):  # 1
    # program ids
    b = tl.program_id(0)          # batch element
    q_token = tl.program_id(1)    # query token index
    qo_head = tl.program_id(2)    # query head index

    # compute ranges
    qo_start = tl.load(qo_indptr + b).to(tl.int32)
    qo_end = tl.load(qo_indptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr + b + 1).to(tl.int32)

    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start
    out_len = num_kv_tokens * gqa_ratio

    # if empty batch segment, skip
    if (num_q_tokens <= 0) or (num_kv_tokens <= 0):
        return

    # pointers for this q vector: q[b, q_token, qo_head, :]
    q_off = (qo_start * num_qo_heads * head_dim) + (qo_head * head_dim) + (q_token * head_dim)
    q_vec = tl.load(q_ptr + q_off)

    # compute attention weights: softmax over out_len positions
    # We compute them on-the-fly inside the kernel without storing logits
    denom = 0.0
    for j in range(0, num_kv_heads):
        for r in range(0, gqa_ratio):
            kv_pos = j * gqa_ratio + r
            if kv_pos >= out_len:
                break
            kq_idx = kv_start + (kv_pos // gqa_ratio)
            k_off = (kv_start * num_kv_heads * head_dim) + (j * head_dim) + (kq_idx * head_dim)
            k_vec = tl.load(k_ptr + k_off)
            dot = 0.0
            for d in range(0, head_dim):
                dot += q_vec[d] * k_vec[d]
            dot = dot * sm_scale
            # causal mask for stability
            delta = num_kv_tokens - num_q_tokens
            bound = q_token + 1 + delta
            if kv_pos >= bound:
                dot = -float("inf")
            # accumulate denom
            denom += tl.exp(dot)

    # compute output vector output[b, q_token, qo_head, :]
    out_vec = tl.zeros((head_dim,), dtype=tl.float32)
    for j in range(0, num_kv_heads):
        for r in range(0, gqa_ratio):
            kv_pos = j * gqa_ratio + r
            if kv_pos >= out_len:
                break
            kq_idx = kv_start + (kv_pos // gqa_ratio)
            v_off = (kv_start * num_kv_heads * head_dim) + (j * head_dim) + (kq_idx * head_dim)
            v_vec = tl.load(v_ptr + v_off)
            k_off = (kv_start * num_kv_heads * head_dim) + (j * head_dim) + (kq_idx * head_dim)
            k_vec = tl.load(k_ptr + k_off)
            dot = 0.0
            for d in range(0, head_dim):
                dot += q_vec[d] * k_vec[d]
            dot = dot * sm_scale
            delta = num_kv_tokens - num_q_tokens
            bound = q_token + 1 + delta
            if kv_pos >= bound:
                dot = -float("inf")
            weight = tl.exp(dot) / denom
            out_vec += weight * v_vec

    # store output[b, q_token, qo_head, :]
    out_off = (q_token * (num_qo_heads * head_dim)) + (qo_head * head_dim)
    for d in range(0, head_dim):
        tl.store(output_ptr + (b * (total_q * num_qo_heads * head_dim)) + out_off + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        Triton-optimized forward that returns (output, lse) with:
          - output: float32 tensor [total_q, 32, 128]
          - lse: float32 tensor [len_indptr, total_q, 32]
        """
        # Ensure inputs are contiguous and float32 for stable math
        device = q.device
        q = q.contiguous().to(torch.float32)
        k = k.contiguous().to(torch.float32)
        v = v.contiguous().to(torch.float32)

        total_q = q.shape[0]
        total_kv = k.shape[0]
        num_qo_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        head_dim = q.shape[2]
        len_indptr = qo_indptr.shape[0]

        # Output and LSE buffers
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        # lse buffer for each (b, q_token, qo_head)
        lse = torch.full((len_indptr, total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Grid: (len_indptr, total_q, num_qo_heads) one program per (b, q_token, head)
        grid = (len_indptr, total_q, num_qo_heads)

        # Launch kernel 1: compute LSE
        compute_lse_kernel[grid](
            q, k, v,
            qo_indptr, kv_indptr,
            sm_scale,
            lse,
            total_q, total_kv, len_indptr,
            num_qo_heads, num_kv_heads,
            head_dim,
            gqa_ratio=4,
            BLOCK_Q=1, BLOCK_K=1,
        )

        # Launch kernel 2: compute output
        compute_output_kernel[grid](
            q, k, v,
            qo_indptr, kv_indptr,
            sm_scale,
            output,
            total_q, total_kv, len_indptr,
            num_qo_heads, num_kv_heads,
            head_dim,
            gqa_ratio=4,
            BLOCK_Q=1, BLOCK_K=1,
        )

        # Return output and lse (per-batch, per-query, per-head)
        return output, lse


def run(*args):
    return ModelNew()(*args)
