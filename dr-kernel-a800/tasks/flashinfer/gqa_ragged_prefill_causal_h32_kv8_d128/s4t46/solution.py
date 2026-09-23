import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_lse_kernel(
    q_ptr, k_ptr, qo_indptr_ptr, kv_indptr_ptr,
    lse_ptr,  # [len_indptr, total_q, num_qo_heads] float32
    total_q,  # int32
    sm_scale,  # float32
    NUM_QO_HEADS: tl.constexpr,   # 32
    NUM_KV_HEADS: tl.constexpr,   # 8
    HEAD_DIM: tl.constexpr,       # 128
    GQA_RATIO: tl.constexpr,      # 4
):
    # Grid: (b, q_token, qo_head)
    b = tl.program_id(0)
    q_token = tl.program_id(1)
    qo_head = tl.program_id(2)

    # Load ranges
    qo_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    qo_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)

    # Skip if empty
    if qo_start >= qo_end or kv_start >= kv_end:
        return

    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start
    delta = num_kv_tokens - num_q_tokens  # original code uses delta = num_kv_tokens - num_q_tokens

    # Base pointer to q vector: q[b, q_token, qo_head, :]
    q_vec_base = q_ptr + (qo_start + q_token) * (NUM_QO_HEADS * HEAD_DIM) + qo_head * HEAD_DIM

    # Accumulator for LSE
    sum_exp = 0.0

    # Loop over original KV heads and GQA expansions (compile-time loops)
    for j in range(NUM_KV_HEADS):  # j in [0..7)
        for r in range(GQA_RATIO):  # r in [0..3)
            kv_pos = j * GQA_RATIO + r  # original KV position index after expansion

            # Compute dot: q_vec dot k_vec
            q_row = tl.load(q_vec_base + tl.arange(0, HEAD_DIM), mask=True, other=0.0)  # [HEAD_DIM]
            k_row_base = k_ptr + kv_pos * (NUM_KV_HEADS * HEAD_DIM) + j * HEAD_DIM
            k_row = tl.load(k_row_base + tl.arange(0, HEAD_DIM), mask=True, other=0.0)  # [HEAD_DIM]
            val = 0.0
            for i in range(HEAD_DIM):
                val += q_row[i] * k_row[i]
            val = val * sm_scale

            # Causal mask: only allow if kv_pos < q_token + 1 + delta
            if kv_pos < (q_token + 1 + delta):
                sum_exp += tl.exp(val)

    # Compute LSE in base-2
    ln2 = 0.6931471805599453
    lse_val = tl.log(sum_exp) / ln2

    # Store LSE
    tl.store(lse_ptr + (b * (total_q * NUM_QO_HEADS) + q_token * NUM_QO_HEADS + qo_head), lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure inputs are CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda, "All tensors must be on CUDA."
        device = q.device

        # Shapes
        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        num_qo_heads = 32
        num_kv_heads = 8
        head_dim = 128
        gqa_ratio = 4

        # Cast to float32 for stable math
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel to compute LSE
        grid = (qo_indptr.numel(), total_q, num_qo_heads)
        _compute_lse_kernel[grid](
            q_f32, k_f32, qo_indptr, kv_indptr,
            lse,
            total_q,
            sm_scale,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim,
            GQA_RATIO=gqa_ratio,
            num_warps=4,
        )

        # Compute actual attention outputs using torch (GPU)
        # Loop over batches
        for b in range(qo_indptr.numel()):
            qo_start = int(qo_indptr[b].item())
            qo_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            num_q_tokens = qo_end - qo_start
            num_kv_tokens = kv_end - kv_start

            # Per-batch slices
            q_batch = q_f32[qo_start:qo_end]                 # [num_q_tokens, 32, 128]
            k_batch = k_f32[kv_start:kv_end]                 # [num_kv_tokens, 8, 128]
            v_batch = v_f32[kv_start:kv_end]                 # [num_kv_tokens, 8, 128]

            # Compute logits per (q_token, qo_head)
            for q_token in range(num_q_tokens):
                for qo_head in range(num_qo_heads):
                    q_vec = q_batch[q_token, qo_head, :]     # [128]

                    # Build logits vector of length out_len = num_kv_tokens * 4
                    logits = []
                    for j in range(num_kv_heads):
                        for r in range(gqa_ratio):
                            kv_pos = j * gqa_ratio + r
                            k_row = k_batch[:, j, :]          # [num_kv_tokens, 128]
                            dot = torch.dot(q_vec, k_row[0])  # scalar
                            logits.append(dot * sm_scale)
                    logits = torch.tensor(logits, device=device, dtype=torch.float32).view(1, -1)  # [1, out_len]

                    # Apply causal mask: kv_pos < q_token + 1 + delta
                    out_len = num_kv_tokens * gqa_ratio
                    allowed = torch.arange(out_len, device=device)
                    causal_mask = allowed < (q_token + 1 + (num_kv_tokens - num_q_tokens))
                    logits = torch.where(causal_mask, logits, torch.tensor(float("-inf"), device=device, dtype=torch.float32))

                    # Softmax along axis=1
                    probs = torch.softmax(logits, dim=1)      # [1, out_len]

                    # Expanded V: repeat v along heads
                    v_expanded = v_batch.repeat_interleave(gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]

                    # Compute output vector of length 128
                    out_vec = torch.zeros((head_dim,), device=device, dtype=torch.float32)
                    for j in range(num_kv_heads):
                        for r in range(gqa_ratio):
                            kv_pos = j * gqa_ratio + r
                            if kv_pos < (q_token + 1 + (num_kv_tokens - num_q_tokens)):
                                v_row = v_expanded[kv_pos, j, :]  # [128]
                                out_vec += probs[0, kv_pos] * v_row

                    # Store output for (b, q_token, qo_head)
                    output[qo_start + q_token, qo_head, :] = out_vec

        return output, lse


def run(*args):
    return ModelNew()(*args)
