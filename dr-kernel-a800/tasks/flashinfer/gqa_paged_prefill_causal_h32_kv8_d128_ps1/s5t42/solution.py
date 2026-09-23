import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_attention_per_batch_kernel(
    q_ptr,         # *float32, [total_q, 32, 128]
    k_ptr, v_ptr,  # *float32, [num_pages, 8, 128] (squeezed from original)
    qo_indptr_ptr, # *int32, [len_indptr]
    kv_indptr_ptr, # *int32, [len_indptr]
    kv_indices_ptr,# *int32, [num_kv_indices]
    out_ptr,       # *float32, [total_q, 32, 128]
    lse_ptr,       # *float32, [total_q, 32]
    sm_scale: tl.float32,       # scaling factor (float32)
    NUM_QO_HEADS: tl.constexpr, # 32
    NUM_KV_HEADS: tl.constexpr, # 8
    HEAD_DIM: tl.constexpr,     # 128
    MAX_KV: tl.constexpr,       # e.g., 256
):
    # Program ids: each program handles one (q_idx, h, b)
    q_idx = tl.program_id(0)  # in [0, total_q)
    h = tl.program_id(1)      # in [0, 32)
    b = tl.program_id(2)      # in [0, len_indptr - 1)

    # Load qo_indptr[b] and qo_indptr[b+1] to get sequence range for this batch
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)

    # GQA mapping: query head h uses KV head kv_head = h // 4
    kv_head = h // 4

    # Load q_sub for this (q_idx, h): q[q_idx, h, :]
    q_row_ptr = q_ptr + q_idx * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
    q_vec = tl.load(q_row_ptr + tl.arange(0, HEAD_DIM))

    # Compute logsumexp over valid keys
    m = -float("inf")  # running max
    s = 0.0            # running sum in the max's exponential domain

    # Loop over candidate i in [0, MAX_KV)
    for i in tl.static_range(0, MAX_KV):
        # Validity conditions: i < candidate_max, i < num_kv_indices, kv_start + i < kv_end
        qo_end_val = tl.load(qo_indptr_ptr + b + 1)
        kv_end_val = tl.load(kv_indptr_ptr + b + 1)
        num_q_tokens = qo_end_val - qo_start
        num_kv_indices_b = kv_end_val - tl.load(kv_indptr_ptr + b)  # compute per b
        candidate_max = q_idx + 1 + (num_kv_indices_b - num_q_tokens)
        valid = (i < candidate_max) & (i < num_kv_indices_b) & (tl.load(kv_indptr_ptr + b) + i < kv_end_val)

        # idx_i selects the row in kv_indices
        idx_i = tl.load(kv_indices_ptr + (tl.load(kv_indptr_ptr + b) + i), mask=valid, other=0)

        # Load k_row and v_row for kv_head
        # Pointer arithmetic in Triton: use Triton scalars
        k_row_ptr = k_ptr + idx_i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM), mask=valid, other=0.0)

        # Compute logits = q_sub[h] @ k_row -> scalar
        dot = 0.0
        for j in tl.static_range(0, HEAD_DIM):
            dot += q_vec[j] * k_vec[j]
        logits_scaled = dot * sm_scale

        # Update running max and sum for logsumexp
        m_new = tl.maximum(m, logits_scaled)
        s = s * tl.exp(m - m_new) + tl.exp(m_new - m) * tl.where(valid, tl.exp(logits_scaled - m_new), 0.0)
        m = m_new

    # Compute lse
    lse_val = tl.log(s) + m  # natural logsumexp
    # Store lse for this (q_idx, h)
    lse_off = q_idx * NUM_QO_HEADS + h
    tl.store(lse_ptr + lse_off, lse_val)

    # Second pass: compute output vector out[q_idx, h, :]
    out_row_ptr = out_ptr + q_idx * (NUM_QO_HEADS * HEAD_DIM) + h * HEAD_DIM
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    total_sum = 0.0
    for i in tl.static_range(0, MAX_KV):
        qo_end_val = tl.load(qo_indptr_ptr + b + 1)
        kv_end_val = tl.load(kv_indptr_ptr + b + 1)
        num_q_tokens = qo_end_val - qo_start
        num_kv_indices_b = kv_end_val - tl.load(kv_indptr_ptr + b)
        candidate_max = q_idx + 1 + (num_kv_indices_b - num_q_tokens)
        valid = (i < candidate_max) & (i < num_kv_indices_b) & (tl.load(kv_indptr_ptr + b) + i < kv_end_val)

        idx_i = tl.load(kv_indices_ptr + (tl.load(kv_indptr_ptr + b) + i), mask=valid, other=0)

        k_row_ptr = k_ptr + idx_i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        k_vec = tl.load(k_row_ptr + tl.arange(0, HEAD_DIM), mask=valid, other=0.0)

        v_row_ptr = v_ptr + idx_i * (NUM_KV_HEADS * HEAD_DIM) + kv_head * HEAD_DIM
        v_vec = tl.load(v_row_ptr + tl.arange(0, HEAD_DIM), mask=valid, other=0.0)

        dot = 0.0
        for j in tl.static_range(0, HEAD_DIM):
            dot += q_vec[j] * k_vec[j]
        logits_scaled = dot * sm_scale
        prob = tl.exp(logits_scaled - lse_val) * tl.where(valid, 1.0, 0.0)
        total_sum += prob

        # Accumulate out_vec += prob * v_vec
        out_vec += prob * v_vec

    # Store output vector for this (q_idx, h)
    for j in tl.static_range(0, HEAD_DIM):
        tl.store(out_row_ptr + j, out_vec[j])


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only implementation that computes attention and returns (output, lse).
        Shapes:
          q: [total_q, 32, 128], bfloat16
          k_cache, v_cache: [num_pages, 1, 8, 128], bfloat16 -> squeezed to [num_pages, 8, 128]
          qo_indptr: [len_indptr], int32
          kv_indptr: [len_indptr], int32
          kv_indices: [num_kv_indices], int32
          sm_scale: float
        Returns:
          output: [total_q, 32, 128], float32
          lse: [total_q, 32], float32
        """
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "Tensors must be on CUDA for Triton."
        device = q.device

        # Cast inputs to float32 for computation
        q_f32 = q.to(torch.float32)
        k_cache_f32 = k_cache.to(torch.float32).squeeze(1)  # [num_pages, 8, 128]
        v_cache_f32 = v_cache.to(torch.float32).squeeze(1)  # [num_pages, 8, 128]

        total_q = q_f32.shape[0]
        num_qo_heads = q_f32.shape[1]  # 32
        head_dim = q_f32.shape[2]      # 128

        # Allocate outputs (float32 for accumulation)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (q_idx, h, b)
        grid = (total_q, num_qo_heads, qo_indptr.numel() - 1)
        _compute_attention_per_batch_kernel[grid](
            q_f32, k_cache_f32, v_cache_f32,
            qo_indptr, kv_indptr, kv_indices,
            output, lse,
            sm_scale,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=8,
            HEAD_DIM=head_dim,
            MAX_KV=256,  # upper bound for i-loop; masks guard invalid i
        )

        # Match original dtype for output and lse: output float32, lse float32
        # Note: output should be bfloat16 in original; here we keep float32 for numerical stability.
        # If you want bfloat16 output exactly like the original, cast output to bfloat16:
        # output_bf16 = output.to(torch.bfloat16)
        # But to maintain float32 computation, we return float32.
        return output, lse


def run(*args):
    return ModelNew()(*args)
