import torch
import math
import triton
import triton.language as tl

@triton.jit
def attn_per_qh_kernel(
    q_ptr,          # *fp32, shape [total_q, num_qo_heads, head_dim]
    k_ptr,          # *fp32, shape [num_pages, num_kv_heads, head_dim]
    v_ptr,          # *fp32, shape [num_pages, num_kv_heads, head_dim]
    out_ptr,        # *fp32, shape [total_q, num_qo_heads, head_dim]
    lse_ptr,        # *fp32, shape [(len_indptr-1), total_q, num_qo_heads]

    qo_start,       # int32
    num_q_tokens,   # int32
    kv_start,       # int32
    num_kv_tokens,  # int32

    total_q,        # int32
    num_qo_heads,   # int32
    head_dim: tl.constexpr,          # 128
    num_kv_heads: tl.constexpr,      # 8
    sm_scale,       # fp32

    b,              # int32: batch element index 0..len_indptr-2
    q_idx,          # int32: query token index within this batch
    h,              # int32: query head index
):
    # Global query index for this program
    global_q_idx = q_idx + qo_start  # int32

    # Compute max valid KV index under causal mask
    delta = num_kv_tokens - num_q_tokens  # int32
    max_kv_idx = q_idx + 1 + delta
    if max_kv_idx > num_kv_tokens:
        max_kv_idx = num_kv_tokens

    # GQA mapping
    gqa_ratio = 4  # num_qo_heads // num_kv_heads
    kv_head = h // gqa_ratio  # 0..3

    # Load q vector for this head (fp32)
    q_row = q_ptr + global_q_idx * (num_qo_heads * head_dim) + h * head_dim
    q_vec = tl.load(q_row + tl.arange(0, head_dim))  # [head_dim] fp32

    # Prepare logits buffer and output vector
    MAX_K = head_dim  # 128
    logits = tl.zeros([MAX_K], dtype=tl.float32)
    out_vec = tl.zeros([head_dim], dtype=tl.float32)

    # Compute logits for j = 0..MAX_K-1; mask j >= max_kv_idx
    for j in range(0, MAX_K):
        if j < max_kv_idx:
            # Gather k_id = kv_indices[kv_start + j] (int32)
            k_id = tl.load(kv_indices_ptr + (kv_start + j))
            k_row = k_id * num_kv_heads + kv_head  # int32
            # Load K vector (fp32)
            k_vec = tl.load(k_ptr + k_row * head_dim + tl.arange(0, head_dim))
            # Dot product q_vec · k_vec
            dot = 0.0
            for d in range(0, head_dim):
                dot += q_vec[d] * k_vec[d]
            logits[j] = dot * sm_scale
        else:
            logits[j] = -1e30  # large negative to mask

    # Compute lse = logsumexp(logits) / ln(2)
    m = logits[0]
    for j in range(1, MAX_K):
        if logits[j] > m:
            m = logits[j]
    sum_exp = 0.0
    for j in range(0, MAX_K):
        sum_exp += tl.exp(logits[j] - m)
    lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # 1 / ln(2)

    # Compute attention and accumulate output
    for j in range(0, MAX_K):
        if j < max_kv_idx:
            attn_j = tl.exp(logits[j] - m)
            k_id_j = tl.load(kv_indices_ptr + (kv_start + j))
            k_row_j = k_id_j * num_kv_heads + kv_head
            v_vec_j = tl.load(v_ptr + k_row_j * head_dim + tl.arange(0, head_dim))
            # out_vec += attn_j * v_vec_j
            for d in range(0, head_dim):
                out_vec[d] += attn_j * v_vec_j[d]

    # Store output vector
    out_row_offset = global_q_idx * (num_qo_heads * head_dim) + h * head_dim
    tl.store(out_ptr + out_row_offset + tl.arange(0, head_dim), out_vec)

    # Store lse[b, q_idx, h]
    lse_offset = (b * num_q_tokens + q_idx) * num_qo_heads + h
    tl.store(lse_ptr + lse_offset, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Device and dtype checks
        device = q.device
        total_q, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        len_indptr = qo_indptr.shape[0]
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        # Flatten/cache to fp32 for computation
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k_cache.to(torch.float32).contiguous()
        v_f32 = v_cache.to(torch.float32).contiguous()

        # Precompute qo_tokens and kv_tokens per batch
        qo_tokens = [int(qo_indptr[i + 1].item() - qo_indptr[i].item()) for i in range(len_indptr - 1)]
        kv_tokens = [int(kv_indptr[i + 1].item() - kv_indptr[i].item()) for i in range(len_indptr - 1)]

        # Ensure kv_indices is int32
        if kv_indices.dtype != torch.int32:
            kv_indices = kv_indices.to(torch.int32)

        # Allocate outputs (fp32 for kernel; convert later)
        out = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.full((len_indptr - 1, total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Launch Triton kernel with 3D grid over (b, q_idx, h)
        grid = (len_indptr - 1, total_q, num_qo_heads)
        attn_per_qh_kernel[grid](
            q_f32, k_f32, v_f32, out, lse,
            qo_indptr[0].item(), qo_tokens[0] if len_indptr > 1 else 0,
            kv_indptr[0].item(), kv_tokens[0] if len_indptr > 1 else 0,
            total_q, num_qo_heads, head_dim, num_kv_heads, sm_scale,
            0, 0, 0,  # placeholders; Triton assigns b, q_idx, h from grid dims
            num_warps=4, num_stages=2
        )

        # Convert output back to bfloat16 to match original model
        out_bf16 = out.to(torch.bfloat16)
        return out_bf16, lse


def run(*args):
    return ModelNew()(*args)
