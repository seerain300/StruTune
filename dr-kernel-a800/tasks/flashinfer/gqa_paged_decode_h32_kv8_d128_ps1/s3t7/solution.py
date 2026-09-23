import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_lse_gqa_kernel(
    q_ptr,          # *bfloat16, [B, H, D]
    k_ptr,          # *float32,  [N, D] (k_cache.squeeze(1).to(float32))
    v_ptr,          # *float32,  [N, D] (v_cache.squeeze(1).to(float32))
    kv_indptr_ptr,  # *int32,    [B+1]
    kv_indices_ptr, # *int32,    [num_kv_indices]
    out_ptr,        # *bfloat16, [B, H, D]
    lse_ptr,        # *float32,  [B, H]
    sm_scale,       # float32 scalar
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # num query heads
    D: tl.constexpr,          # head dim (e.g., 128)
    N: tl.constexpr,          # num kv heads (e.g., 8)
    gqa_ratio: tl.constexpr,  # H // N (e.g., 4)
):
    # Grid: (B, H)
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Read the token range for this batch
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    num_tokens = end - start

    # GQA mapping: each query head h uses KV head kvh = h // (H // N)
    kvh = h // gqa_ratio

    # Initialize LSE accumulators
    max_logit = -float("inf")
    sum_exp = 0.0

    # First pass: compute max and sum_exp over tokens
    t = 0
    while t < num_tokens:
        tok_idx = tl.load(kv_indices_ptr + start + t).to(tl.int32)

        # Load q[b, h, :] as float32
        q_base = q_ptr + b * (H * D) + h * D
        q_vec = tl.load(q_base, other=0.0).to(tl.float32)  # [D]

        # Load k[kvh, tok_idx, :] and v[kvh, tok_idx, :] as float32
        # k_ptr and v_ptr have shape [N, D]; row kvh is contiguous length D
        k_row_ptr = k_ptr + kvh * D
        v_row_ptr = v_ptr + kvh * D

        k_vec = tl.load(k_row_ptr + tok_idx * D, other=0.0).to(tl.float32)  # [D]
        v_vec = tl.load(v_row_ptr + tok_idx * D, other=0.0).to(tl.float32)  # [D]

        # Compute dot product q·k for this token
        dot_qk = 0.0
        for d in range(0, D):
            dot_qk += q_vec[d] * k_vec[d]
        logits_scaled = dot_qk * sm_scale  # scalar

        # Update max and sum_exp for logsumexp
        new_max = tl.maximum(max_logit, logits_scaled)
        # sum_exp_new = sum_exp * exp(max_logit - new_max) + exp(logits_scaled - new_max)
        sum_exp = sum_exp * tl.exp(max_logit - new_max) + tl.exp(logits_scaled - new_max)
        max_logit = new_max

        t += 1

    # Compute base-2 LSE
    ln2 = 0.6931471805599453
    lse_val = max_logit + tl.log(sum_exp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute output[b, h, :] = sum_t softmax(logits_scaled)[t] * v[t]
    t = 0
    acc = tl.zeros((D,), dtype=tl.float32)
    while t < num_tokens:
        tok_idx = tl.load(kv_indices_ptr + start + t).to(tl.int32)

        q_base = q_ptr + b * (H * D) + h * D
        q_vec = tl.load(q_base, other=0.0).to(tl.float32)  # [D]

        k_row_ptr = k_ptr + kvh * D
        v_row_ptr = v_ptr + kvh * D

        k_vec = tl.load(k_row_ptr + tok_idx * D, other=0.0).to(tl.float32)  # [D]
        v_vec = tl.load(v_row_ptr + tok_idx * D, other=0.0).to(tl.float32)  # [D]

        dot_qk = 0.0
        for d in range(0, D):
            dot_qk += q_vec[d] * k_vec[d]
        logits_scaled = dot_qk * sm_scale  # scalar

        # Compute attention for this token
        attn = tl.exp(logits_scaled - max_logit) / sum_exp  # scalar

        # Accumulate output: acc += attn * v_vec
        acc += attn * v_vec

        t += 1

    # Store output as bfloat16
    out_vec = acc.to(tl.bfloat16)
    out_ptr_row = out_ptr + b * (H * D) + h * D
    tl.store(out_ptr_row + tl.arange(0, D), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All tensors must be CUDA tensors."

        # Shapes
        B, H, D = q.shape

        # Squeeze dim-1 from k_cache/v_cache: original uses [P, 1, N, D]
        # Keep dtype float32 for stable math in Triton
        k_squeezed = k_cache.squeeze(1).to(torch.float32)  # [N, D], float32
        v_squeezed = v_cache.squeeze(1).to(torch.float32)  # [N, D], float32

        device = q.device

        # Output and LSE
        output = torch.empty((B, H, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device).fill_(-float("inf"))

        # GQA ratio: H // N = 4
        gqa_ratio = H // 8  # num_kv_heads == 8 (asserted in original)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)

        attention_lse_gqa_kernel[grid](
            q, k_squeezed, v_squeezed, kv_indptr, kv_indices, output, lse, sm_scale,
            B, H, D, 8, gqa_ratio,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
