import math
import torch
import triton
import triton.language as tl


@triton.jit
def gqa_attention_kernel_bh(
    q_ptr,          # *bfloat16, [B, H, D], contiguous
    k_ptr,          # *float32,  [N, D], contiguous (k_cache.squeeze(1).to(float32))
    v_ptr,          # *float32,  [N, D], contiguous (v_cache.squeeze(1).to(float32))
    kv_indptr_ptr,  # *int32,    [B+1]
    kv_indices_ptr, # *int32,    [num_kv_indices]
    out_ptr,        # *bfloat16, [B, H, D]
    lse_ptr,        # *float32,  [B, H], initialized to -inf
    sm_scale,       # float32 scalar
    B: tl.constexpr,          # batch size (meta)
    H: tl.constexpr,          # num query heads (meta)
    D: tl.constexpr,          # head dim (meta, e.g., 128)
    N: tl.constexpr,          # num kv heads (meta, e.g., 8)
    gqa_ratio: tl.constexpr,  # H // N (meta, e.g., 4)
    BLOCK_T: tl.constexpr,    # token chunk size (meta, e.g., 128)
):
    # Each program handles one (b, h) pair
    b = tl.program_id(0)
    h = tl.program_id(1)

    # GQA mapping: each query head h uses KV head kvh = h // gqa_ratio
    kvh = h // gqa_ratio

    # Read the token range for this batch
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    num_tokens = end - start  # number of tokens for this batch element

    # Prepare output accumulator in float32
    acc = tl.zeros((D,), dtype=tl.float32)

    # First pass: compute max over tokens for numerical stability and sum of exp
    max_logit = -float("inf")
    sum_exp = 0.0

    for t_off in range(0, BLOCK_T):
        t = start + t_off
        valid = t < num_tokens

        # Load token index
        tok_idx = tl.load(kv_indices_ptr + t, mask=valid, other=0).to(tl.int32)

        # Load q vector for this (b, h): q layout [B, H, D] contiguous; offset = b*(H*D) + h*D
        q_off = b * (H * D) + h * D
        q_vec = tl.load(q_ptr + q_off, mask=valid, other=0.0).to(tl.float32)  # [D]

        # Load k for this token and kvh; k_ptr[kvh, tok_idx, :] -> k_ptr[kvh*D + tok_idx*D + offs]
        k_off = kvh * D + tok_idx * D
        k_chunk = tl.load(k_ptr + k_off + tl.arange(0, D), mask=valid, other=0.0)  # [D]
        dot_qk = tl.sum(q_vec * k_chunk, axis=0)  # scalar

        # Scale
        logits_scaled = dot_qk * sm_scale

        # Update max and sum_exp (ignore invalid tokens by setting logits_scaled = -inf)
        logits_scaled = tl.where(valid, logits_scaled, -float("inf"))

        # Maintain running max and sum_exp across chunks
        max_logit = tl.maximum(max_logit, logits_scaled)
        sum_exp = tl.where(valid, sum_exp + tl.exp(logits_scaled - max_logit), sum_exp)

    # Compute lse[b, h] = logsumexp(logits_scaled, axis=0) / log(2.0)
    lse_off = b * H + h
    lse_val = max_logit + tl.log(sum_exp) / tl.log(2.0)
    tl.store(lse_ptr + lse_off, lse_val)

    # Second pass: compute attn for each token chunk and accumulate output
    for t_off in range(0, BLOCK_T):
        t = start + t_off
        valid = t < num_tokens

        tok_idx = tl.load(kv_indices_ptr + t, mask=valid, other=0).to(tl.int32)

        q_off = b * (H * D) + h * D
        q_vec = tl.load(q_ptr + q_off, mask=valid, other=0.0).to(tl.float32)

        k_off = kvh * D + tok_idx * D
        k_chunk = tl.load(k_ptr + k_off + tl.arange(0, D), mask=valid, other=0.0)  # [D]

        dot_qk = tl.sum(q_vec * k_chunk, axis=0)  # scalar
        logits_scaled = dot_qk * sm_scale

        attn = tl.where(valid, tl.exp(logits_scaled - max_logit) / sum_exp, 0.0)

        v_off = kvh * D + tok_idx * D
        v_chunk = tl.load(v_ptr + v_off + tl.arange(0, D), mask=valid, other=0.0)  # [D]
        acc += attn * v_chunk

    # Store output[b, h, :] in bfloat16
    out_off = b * (H * D) + h * D
    tl.store(out_ptr + out_off, acc.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure on CUDA
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All tensors must be CUDA tensors."

        # Shapes
        B, H, D = q.shape
        # k_cache, v_cache: [P, 1, N, D] -> squeeze dim 1 to [N, D]
        k_squeezed = k_cache.squeeze(1).to(torch.float32)
        v_squeezed = v_cache.squeeze(1).to(torch.float32)
        device = q.device

        # Output and LSE
        output = torch.empty((B, H, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device).fill_(-float("inf"))

        # GQA ratio
        gqa_ratio = H // 8  # num_kv_heads == 8 in original; H//N = 4

        # Launch Triton kernel: grid over (B, H)
        grid = (B, H)
        gqa_attention_kernel_bh[grid](
            q, k_squeezed, v_squeezed, kv_indptr, kv_indices, output, lse, sm_scale,
            B=B, H=H, D=D, N=8, gqa_ratio=gqa_ratio, BLOCK_T=128,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
