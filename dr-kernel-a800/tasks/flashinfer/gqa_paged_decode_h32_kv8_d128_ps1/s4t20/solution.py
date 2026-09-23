import math
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_and_attention_single_bh(
    q_ptr,                 # *fp32, pointer to q[b, h] vector, length = HEAD_DIM
    K_ptr,                 # *fp32, pointer to K tokens for this batch, shape [NUM_TOKENS, HEAD_DIM], contiguous
    V_ptr,                 # *fp32, pointer to V tokens for this batch, shape [NUM_TOKENS, HEAD_DIM], contiguous
    OUT_ptr,               # *fp32, output vector for this head [HEAD_DIM]
    LSE_ptr,               # *fp32, single scalar lse for this (b, h), 1-element tensor
    NUM_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.float32,       # 1.0 / sqrt(HEAD_DIM) (here 1/sqrt(128))
    LOG2_INVERSE: tl.float32,   # 1.0 / ln(2.0) ≈ 1.4426950408889634
):
    # One program instance per (b, h) where h = program_id(0)
    h = tl.program_id(0)

    # First pass: compute lse = logsumexp(logits_scaled) / ln(2)
    running_max = -float("inf")
    running_sum = 0.0

    for t in range(0, NUM_TOKENS):
        # Load q vector for this head: [HEAD_DIM]
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))
        # Load k vector for token t: [HEAD_DIM]
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))
        logits_t = tl.sum(q_vec * k_vec, axis=0)  # scalar
        scaled = logits_t * SM_SCALE
        # numerically stable update
        running_max = tl.maximum(running_max, scaled)
        running_sum += tl.exp(scaled - running_max)

    # Compute lse = logsumexp(scaled) / ln(2)
    lse = tl.log(running_sum) + running_max
    lse = lse * LOG2_INVERSE  # divide by ln(2)
    # Store scalar lse at LSE_ptr[0]
    tl.store(LSE_ptr + 0, lse)

    # Second pass: compute output[b, h] = sum_j exp(scaled_j - lse) * V_j
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for t in range(0, NUM_TOKENS):
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))
        v_vec = tl.load(V_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))
        logits_t = tl.sum(q_vec * k_vec, axis=0)  # scalar
        scaled = logits_t * SM_SCALE
        attn = tl.exp(scaled - lse)  # softmax over tokens
        out_vec += attn * v_vec

    # Store output vector
    tl.store(OUT_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "All tensors must be on CUDA device for Triton"
        assert q.dtype == torch.bfloat16 and k_cache.dtype == torch.bfloat16 and v_cache.dtype == torch.bfloat16
        B, num_qo_heads, head_dim = q.shape
        N, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and head_dim == 128 and num_kv_heads == 8

        # Allocate outputs
        output = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)  # fp32 compute
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Compute per-batch num_tokens from kv_indptr
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens_b = end - start

            # If no tokens, output and lse are zeros
            if num_tokens_b == 0:
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather K_t and V_t for this batch: per-token indices are kv_indices[start:start+num_tokens_b]
            # Build list of token indices for this batch
            token_indices = []
            for t in range(num_tokens_b):
                idx = start + t  # 0..num_tokens_b-1
                token_idx = int(kv_indices[idx].item())
                token_indices.append(token_idx)

            # Now gather K_t and V_t: [num_tokens_b, 1, 8, 128]
            K_t = k_cache[token_indices]  # [num_tokens_b, 1, 8, 128]
            V_t = v_cache[token_indices]  # [num_tokens_b, 1, 8, 128]

            # Cast to fp32 for compute
            K_t = K_t.to(torch.float32).contiguous()
            V_t = V_t


def run(*args):
    return ModelNew()(*args)
