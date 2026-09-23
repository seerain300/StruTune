import math
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_and_attention_single_bh(
    q_ptr,                 # *fp32, pointer to q[b, h] vector, length = HEAD_DIM
    K_ptr,                 # *fp32, pointer to K tokens for this batch, shape [NUM_TOKENS, KV_HEADS, HEAD_DIM], contiguous
    V_ptr,                 # *fp32, pointer to V tokens for this batch, shape [NUM_TOKENS, KV_HEADS, HEAD_DIM], contiguous
    OUT_ptr,               # *fp32, output vector for this head [HEAD_DIM]
    LSE_ptr,               # *fp32, single scalar lse for this (b, h)
    NUM_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KV_HEADS: tl.constexpr,
    SM_SCALE: tl.float32,      # 1.0 / sqrt(HEAD_DIM)
    LOG2_INVERSE: tl.float32,  # 1.0 / ln(2.0) ≈ 1.4426950408889634
):
    # Grid is 2D: (B, num_qo_heads). We get b from the grid as well.
    b = tl.program_id(1)
    h = tl.program_id(0)

    # First pass: compute lse = logsumexp(logits_scaled) / ln(2)
    running_max = -float("inf")
    running_sum = 0.0

    for t in range(0, NUM_TOKENS):
        # Load q vector for this head: [HEAD_DIM]
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))
        # Load k vector for token t for this kv_head: [HEAD_DIM]
        # We need to compute kv_head mapping: gqa_ratio = num_qo_heads // num_kv_heads
        # Here we assume num_qo_heads=32, num_kv_heads=8, so gqa_ratio=4.
        kv_head = h // 4
        # K_ptr layout: [NUM_TOKENS, KV_HEADS, HEAD_DIM], contiguous
        # Pointer to k_vec for this t and kv_head: base = t * KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
        base = t * KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
        k_vec = tl.load(K_ptr + base + tl.arange(0, HEAD_DIM))
        # Compute logits_scaled for this token
        logits_t = tl.sum(q_vec * k_vec, axis=0)  # scalar
        scaled = logits_t * SM_SCALE
        # Numerically stable update
        running_max = tl.maximum(running_max, scaled)
        running_sum = running_sum * tl.exp(running_max - running_max) + tl.exp(scaled - running_max)

    # Compute lse and scale by 1/ln(2)
    lse_val = tl.log(running_sum) + running_max
    lse_val = lse_val * LOG2_INVERSE
    # Store lse as a scalar at LSE_ptr[b * num_qo_heads + h]
    tl.store(LSE_ptr + b * 32 + h, lse_val)

    # Second pass: compute output vector for this head
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for t in range(0, NUM_TOKENS):
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))
        kv_head = h // 4
        base = t * KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
        k_vec = tl.load(K_ptr + base + tl.arange(0, HEAD_DIM))
        v_vec = tl.load(V_ptr + base + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        logits_t = tl.sum(q_vec * k_vec, axis=0)
        scaled = logits_t * SM_SCALE
        attn = tl.exp(scaled - lse_val)  # softmax over tokens
        out_vec += attn * v_vec

    # Store output vector
    tl.store(OUT_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "All tensors must be on CUDA for Triton"
        B, num_qo_heads, head_dim = q.shape
        num_pages, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        # Allocate outputs (compute in fp32, cast to bfloat16 at end)
        output = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Compute per-batch num_tokens from kv_indptr
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens_b = end - start
            if num_tokens_b == 0:
                # Nothing to do; set zeros
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather token indices for this batch
            token_indices = []
            for t in range(num_tokens_b):
                idx = start + t  # 0..num_tokens_b-1
                token_idx = int(kv_indices[idx].item())  # must be in [0, num_pages)
                token_indices.append(token_idx)

            # Gather K_t and V_t directly: [num_tokens_b, num_kv_heads, head_dim]
            K_t = k_cache[token_indices]  # [num_tokens_b, 8, 128]
            V_t = v_cache[token_indices]  # [num_tokens_b, 8, 128]

            # Cast to fp32 and make contiguous
            K_t = K_t.to(torch.float32).contiguous()
            V_t = V_t.to(torch.float32).contiguous()

            # Launch Triton kernel once per (b, h)
            grid = (num_qo_heads, B)  # (heads, batch) to compute each (b, h)
            softmax_and_attention_single_bh[grid](
                # q[b, h] for all h: create pointers by stride across batch
                q[b],                     # [32, 128] per b
                K_t,                      # [num_tokens_b, 8, 128]
                V_t,                      # [num_tokens_b, 8, 128]
                output[b],                # [32, 128] -> pointer per h
                lse[b],                   # [32] -> pointer per h
                NUM_TOKENS=num_tokens_b,
                HEAD_DIM=head_dim,
                KV_HEADS=num_kv_heads,
                SM_SCALE=sm_scale,
                LOG2_INVERSE=1.4426950408889634,
            )

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
