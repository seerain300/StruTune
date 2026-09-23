import math
import torch

import triton
import triton.language as tl


@triton.jit
def _compute_lse_and_out_per_triplet(
    q_ptr,            # *float32, [total_q, num_qo_heads, head_dim]
    k_ptr, v_ptr,     # *float32, [num_pages, num_kv_heads, head_dim, 1] (4D)
    qo_indptr_ptr,    # *int32, [len_indptr]
    kv_indptr_ptr,    # *int32, [len_indptr]
    kv_indices_ptr,   # *int32, [num_kv_indices]
    out_ptr,          # *float32, [total_q, num_qo_heads, head_dim]
    lse_ptr,          # *float32, [total_q, num_qo_heads]
    sm_scale,         # float32 scalar
    GQA_RATIO: tl.constexpr,   # e.g., 4
    MAX_KV: tl.constexpr,      # e.g., 256
    HEAD_DIM: tl.constexpr,    # e.g., 128
):
    # Grid is (len_indptr, total_q, num_qo_heads)
    b = tl.program_id(0)
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    # Load segment starts
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    # Guard q_idx < num_q_tokens_in_b
    if q_idx >= (qo_end - qo_start):
        return

    global_q_idx = qo_start + q_idx

    # Causal + length masking
    num_q_tokens_in_b = qo_end - qo_start
    num_kv_indices_in_b = kv_end - kv_start
    candidate_max = q_idx + 1 + (num_kv_indices_in_b - num_q_tokens_in_b)

    kv_head = h // GQA_RATIO

    # Load q_sub vector: q[global_q_idx, h, :]
    q_off = global_q_idx * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
    q_sub = tl.load(q_ptr + q_off + tl.arange(0, HEAD_DIM))

    # Accumulate logits for logsumexp
    logits = tl.zeros((), dtype=tl.float32)
    max_logits = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)

    for i in tl.static_range(0, MAX_KV):
        m_i = (i < candidate_max) & (i < num_kv_indices_in_b) & ((kv_start + i) < kv_end)
        # If m_i is false, continue without loads
        if m_i:
            idx = tl.load(kv_indices_ptr + (kv_start + i))  # int32 scalar
            # k_ptr and v_ptr are 4D: [num_pages, num_kv_heads, HEAD_DIM, 1]
            # idx selects the num_pages dimension; kv_head selects num_kv_heads; HEAD_DIM selects width; trailing dim=1 ignored
            k_off = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM  # last dim is size 1, so just + (kv_head * HEAD_DIM)
            k_sub = tl.load(k_ptr + k_off + tl.arange(0, HEAD_DIM))
            v_off = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            v_sub = tl.load(v_ptr + v_off + tl.arange(0, HEAD_DIM))

            dot = tl.sum(q_sub * k_sub, axis=0)  # scalar
            logits += dot

            scaled = logits * sm_scale
            max_logits = tl.maximum(max_logits, scaled)
            sum_exp = sum_exp * tl.exp(max_logits - scaled) + 1.0
            max_logits = max_logits

    # lse = logsumexp(logits_scaled) / log(2)
    log2 = 0.6931471805599453  # math.log(2)
    lse_val = max_logits + tl.log(sum_exp)  # logsumexp over scaled logits
    lse_val = lse_val / log2

    # Store lse for this (b, q_idx, h)
    lse_off = global_q_idx * (num_qo_heads) + h
    tl.store(lse_ptr + lse_off, lse_val)

    # Compute output: out = softmax(logits_scaled) @ v_sub across i
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for i in tl.static_range(0, MAX_KV):
        m_i = (i < candidate_max) & (i < num_kv_indices_in_b) & ((kv_start + i) < kv_end)
        if m_i:
            idx = tl.load(kv_indices_ptr + (kv_start + i))
            k_off = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            k_sub = tl.load(k_ptr + k_off + tl.arange(0, HEAD_DIM))
            v_off = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            v_sub = tl.load(v_ptr + v_off + tl.arange(0, HEAD_DIM))

            dot = tl.sum(q_sub * k_sub, axis=0)  # scalar
            scaled = dot * sm_scale
            prob = tl.exp(scaled - max_logits) / sum_exp  # softmax probability
            # out_vec += prob * v_sub
            for j in tl.static_range(0, HEAD_DIM):
                out_vec[j] += prob * v_sub[j]

    # Store output for this (b, q_idx, h)
    out_off = global_q_idx * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
    for j in tl.static_range(0, HEAD_DIM):
        tl.store(out_ptr + out_off + j, out_vec[j])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed constants from original code
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.GQA_RATIO = self.num_qo_heads // self.num_kv_heads  # 4
        self.MAX_KV = 256  # safe upper bound for typical candidate_max

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors and contiguity
        device = q.device
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All inputs must be CUDA tensors for Triton kernels"

        q_f32 = q.to(torch.float32).contiguous()              # [total_q, 32, 128]
        # Keep 4D shapes for k and v so Triton can unpack pointers reliably: [num_pages, 8, 128, 1]
        k4 = k_cache.to(torch.float32).contiguous().view(k_cache.shape[0], self.num_kv_heads, self.head_dim, 1)
        v4 = v_cache.to(torch.float32).contiguous().view(v_cache.shape[0], self.num_kv_heads, self.head_dim, 1)

        len_indptr = qo_indptr.shape[0]
        total_q = q_f32.shape[0]
        num_qo_heads = q_f32.shape[1]
        head_dim = q_f32.shape[2]
        assert num_qo_heads == 32 and head_dim == 128, "q must have shape [*, 32, 128]"
        assert k4.shape[1] == 8 and k4.shape[3] == 1 and k4.shape[2] == 128, "k must have shape [*, 8, 128, 1] after view"
        assert v4.shape[1] == 8 and v4.shape[3] == 1 and v4.shape[2] == 128, "v must have shape [*, 8, 128, 1] after view"

        # Allocate outputs (float32 for compute)
        out_f32 = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse_f32 = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, q_idx, h)
        grid = (len_indptr, total_q, num_qo_heads)
        _compute_lse_and_out_per_triplet[grid](
            q_f32, k4, v4, qo_indptr, kv_indptr, kv_indices,
            out_f32, lse_f32, float(sm_scale),
            GQA_RATIO=self.GQA_RATIO, MAX_KV=self.MAX_KV, HEAD_DIM=self.head_dim,
            num_warps=4, num_stages=2
        )

        # Cast output to bfloat16 for final result
        out_bf16 = out_f32.to(torch.bfloat16)
        return out_bf16, lse_f32


def run(*args):
    return ModelNew()(*args)
