import torch
import math
import triton
import triton.language as tl


@triton.jit
def compute_logits_and_lse_kernel(
    q_ptr, k_ptr, qo_indptr_ptr, kv_indptr_ptr, lse_ptr,
    total_q, num_qo_heads, head_dim, num_kv_heads, gqa_ratio, sm_scale
):
    # Grid: (batch, q_token, qo_head)
    b = tl.program_id(0)
    q_token = tl.program_id(1)
    qo_head = tl.program_id(2)

    # Load bounds
    qo_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    qo_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)

    if qo_start >= qo_end or kv_start >= kv_end:
        return

    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start
    delta = num_kv_tokens - num_q_tokens

    # Load Q vector for this (q_token, qo_head)
    q_index = qo_start + q_token
    base_q = q_index * (num_qo_heads * head_dim) + qo_head * head_dim
    q_vec = tl.load(q_ptr + base_q)  # [head_dim], float32

    # Prepare indices for kv tokens
    j_idx = tl.arange(0, num_kv_tokens)  # [num_kv_tokens]
    kv_index = kv_start + j_idx  # [num_kv_tokens]
    kv_head = qo_head % num_kv_heads  # 8
    base_k = kv_index * (num_kv_heads * head_dim) + kv_head * head_dim  # [num_kv_tokens]

    # Load K rows: [num_kv_tokens, head_dim]
    k_rows = tl.load(k_ptr + base_k, mask=j_idx < num_kv_tokens, other=0.0)  # [num_kv_tokens, head_dim]

    # Compute dot products per kv token: [num_kv_tokens]
    dot_row = tl.sum(q_vec[:, None] * k_rows, axis=0)  # [num_kv_tokens]
    logits_row = dot_row * sm_scale  # [num_kv_tokens]

    # Causal mask: can attend up to q_token + 1 + delta
    mask_ok = j_idx < (q_token + 1 + delta)  # [num_kv_tokens]
    logits_row = tl.where(mask_ok, logits_row, -float("inf"))

    # LSE in base-2: log(sum(exp(logits))) / log(2)
    sum_exp = tl.sum(tl.exp(logits_row))  # scalar
    lse_val = tl.log(sum_exp) / math.log(2.0)

    # Store lse at (b, q_token, qo_head)
    out_lse_idx = b * (num_q_tokens * num_qo_heads) + q_token * num_qo_heads + qo_head
    tl.store(lse_ptr + out_lse_idx, lse_val)


@triton.jit
def compute_output_kernel(
    q_ptr, k_ptr, v_ptr, qo_indptr_ptr, kv_indptr_ptr, output_ptr, lse_ptr, sm_scale,
    total_q, num_qo_heads, head_dim, num_kv_heads, gqa_ratio
):
    # Grid: (batch, q_token, qo_head)
    b = tl.program_id(0)
    q_token = tl.program_id(1)
    qo_head = tl.program_id(2)

    # Load bounds
    qo_start = tl.load(qo_indptr_ptr + b).to(tl.int32)
    qo_end = tl.load(qo_indptr_ptr + b + 1).to(tl.int32)
    kv_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)

    if qo_start >= qo_end or kv_start >= kv_end:
        return

    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start
    delta = num_kv_tokens - num_q_tokens

    # Load Q vector for this (q_token, qo_head)
    q_index = qo_start + q_token
    base_q = q_index * (num_qo_heads * head_dim) + qo_head * head_dim
    q_vec = tl.load(q_ptr + base_q)  # [head_dim], float32

    # Load lse for this (q_token, qo_head)
    out_lse_idx = b * (num_q_tokens * num_qo_heads) + q_token * num_qo_heads + qo_head
    lse_val = tl.load(lse_ptr + out_lse_idx)  # scalar float32
    scale = 1.0 / (math.log(2.0) * lse_val)

    # Prepare indices for kv tokens
    j_idx = tl.arange(0, num_kv_tokens)  # [num_kv_tokens]
    kv_index = kv_start + j_idx  # [num_kv_tokens]
    kv_head = qo_head % num_kv_heads  # 8
    base_k = kv_index * (num_kv_heads * head_dim) + kv_head * head_dim  # [num_kv_tokens]

    # Load K rows: [num_kv_tokens, head_dim]
    k_rows = tl.load(k_ptr + base_k, mask=j_idx < num_kv_tokens, other=0.0)  # [num_kv_tokens, head_dim]

    # Compute logits row and attention weights
    dot_row = tl.sum(q_vec[:, None] * k_rows, axis=0)  # [num_kv_tokens]
    logits_row = dot_row * sm_scale  # [num_kv_tokens]
    mask_ok = j_idx < (q_token + 1 + delta)  # [num_kv_tokens]
    logits_row = tl.where(mask_ok, logits_row, -float("inf"))
    attn_row = tl.exp(logits_row * scale)  # [num_kv_tokens]

    # Load V per kv token and head: [num_kv_tokens, head_dim]
    v_base = kv_index * (num_kv_heads * head_dim) + kv_head * head_dim  # [num_kv_tokens]
    v_rows = tl.load(v_ptr + v_base, mask=j_idx < num_kv_tokens, other=0.0)  # [num_kv_tokens, head_dim]

    # Output vector for this (q_token, qo_head): out = attn_row @ v_rows
    out_vec = tl.zeros([head_dim], dtype=tl.float32)
    for d in range(0, head_dim):
        out_vec[d] = tl.sum(attn_row[:, None] * v_rows[:, d], axis=0)

    # Store output at (q_token, qo_head)
    out_base = q_index * (num_qo_heads * head_dim) + qo_head * head_dim
    tl.store(output_ptr + out_base, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure bfloat16 inputs as in original
        assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
        device = q.device

        total_q = int(qo_indptr[-1].item())
        num_qo_heads = 32
        head_dim = 128
        num_kv_heads = 8
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        # Cast inputs to float32 for compute
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernels
        grid = (qo_indptr.numel() - 1, total_q, num_qo_heads)
        compute_logits_and_lse_kernel[grid](
            q_f32, k_f32, qo_indptr, kv_indptr, lse,
            total_q, num_qo_heads, head_dim, num_kv_heads, gqa_ratio, float(sm_scale),
            num_warps=4, num_stages=2
        )

        compute_output_kernel[grid](
            q_f32, k_f32, v_f32, qo_indptr, kv_indptr, output, lse, float(sm_scale),
            total_q, num_qo_heads, head_dim, num_kv_heads, gqa_ratio,
            num_warps=4, num_stages=2
        )

        # Return output and lse (float32, matching original run behavior)
        return output, lse


def run(*args):
    return ModelNew()(*args)
