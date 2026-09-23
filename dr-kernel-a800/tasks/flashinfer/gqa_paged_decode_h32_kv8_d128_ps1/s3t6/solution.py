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
    kv_indices_ptr, # *int32,    [num_tokens_total]
    output_ptr,     # *bfloat16, [B, H, D]
    lse_ptr,        # *float32,  [B, H], initialized to -inf
    sm_scale,       # float32 scalar
    B,              # int
    H,              # int
    D,              # int
    N,              # int (num kv heads, e.g., 8)
    gqa_ratio,      # int (H // N, e.g., 4)
):
    # Grid: one program per (b, h)
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Read token range for this batch
    start = tl.load(kv_indptr_ptr + b)
    end = tl.load(kv_indptr_ptr + b + 1)
    num_tokens = end - start

    # GQA mapping from query head to KV head
    kvh = h // gqa_ratio

    # Initialize LSE accumulators
    max_logit = -float("inf")
    sum_exp = 0.0
    lse_off = b * H + h

    # Accumulator for output vector [D] in float32
    acc = tl.zeros((D,), dtype=tl.float32)

    # Loop over tokens t
    t = 0
    while t < num_tokens:
        # Compute tok_idx for this token
        tok_idx = tl.load(kv_indices_ptr + start + t)

        # Load q vector for (b, h) and cast to float32
        q_off = b * (H * D) + h * D
        q_vec = tl.load(q_ptr + q_off).to(tl.float32)  # [D]

        # Load k_vec and v_vec for kvh and tok_idx from k_ptr and v_ptr (shape [N, D])
        k_off = kvh * D + tok_idx * D
        v_off = kvh * D + tok_idx * D
        k_vec = tl.load(k_ptr + k_off + tl.arange(0, D))  # [D]
        v_vec = tl.load(v_ptr + v_off + tl.arange(0, D))  # [D]

        # Compute logits_scaled = dot(q_vec, k_vec) * sm_scale
        dot_qk = tl.sum(q_vec * k_vec, axis=0)  # scalar
        logits_scaled = dot_qk * sm_scale

        # Update LSE: max_logit and sum_exp across tokens (base-2 logsumexp)
        if t == 0:
            max_logit = logits_scaled
        else:
            max_logit = tl.maximum(max_logit, logits_scaled)
        sum_exp = sum_exp + tl.exp(logits_scaled - max_logit)

        # Compute attention weight for this token
        attn = tl.exp(logits_scaled - max_logit) / sum_exp

        # Accumulate output vector
        acc += attn * v_vec

        t += 1

    # Store LSE in base-2: lse = max + log(sum_exp) / ln(2)
    lse_val = max_logit + tl.log(sum_exp) * (1.4426950408889634)  # 1 / ln(2)
    tl.store(lse_ptr + lse_off, lse_val)

    # Store acc cast to bfloat16 to output[b, h, :]
    out_off = b * (H * D) + h * D
    tl.store(output_ptr + out_off + tl.arange(0, D), acc.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All tensors must be CUDA tensors."

        # Shapes
        B, H, D = q.shape

        # Squeeze dim 1 for k_cache and v_cache to match original: [P, 1, N, D] -> [N, D]
        # Keep dtype float32 for math
        k_squeezed = k_cache.squeeze(1).to(torch.float32)
        v_squeezed = v_cache.squeeze(1).to(torch.float32)

        device = q.device

        # Output and LSE
        output = torch.empty((B, H, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device).fill_(-float("inf"))

        # GQA ratio
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
