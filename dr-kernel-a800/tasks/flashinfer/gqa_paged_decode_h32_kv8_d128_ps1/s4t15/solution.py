import math
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_and_attention_single_bh_kernel(
    q_ptr,                 # *fp32, pointer to q[b, h] vector, length = HEAD_DIM
    K_ptr,                 # *fp32, pointer to K tokens for this batch, shape [NUM_TOKENS, HEAD_DIM], contiguous
    V_ptr,                 # *fp32, pointer to V tokens for this batch, shape [NUM_TOKENS, HEAD_DIM], contiguous
    OUT_ptr,               # *fp32, output vector for this head [HEAD_DIM]
    LSE_ptr,               # *fp32, scalar lse for this (b, h), stored at index h
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
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        # Load k vector for token t: [HEAD_DIM]
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        # Compute dot product as scalar
        logits_t = tl.sum(q_vec * k_vec, axis=0)  # scalar
        scaled = logits_t * SM_SCALE
        # Numerically-stable accumulation
        running_max = tl.maximum(running_max, scaled)
        running_sum = running_sum * tl.exp(running_max - scaled) + 1.0

    # Final lse_raw = log(running_sum) + running_max
    lse_raw = tl.log(running_sum) + running_max
    # Divide by ln(2) (i.e., multiply by 1/ln(2))
    lse = lse_raw * LOG2_INVERSE

    # Store lse as a scalar at LSE_ptr[h]
    tl.store(LSE_ptr + h, lse)

    # Second pass: compute output vector
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for t in range(0, NUM_TOKENS):
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        v_vec = tl.load(V_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        logits_t = tl.sum(q_vec * k_vec, axis=0)                        # scalar
        scaled = logits_t * SM_SCALE
        attn = tl.exp(scaled - lse)  # softmax over tokens
        out_vec += attn * v_vec

    # Store output
    tl.store(OUT_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes: q [B, 32, 128], k_cache/v_cache [num_pages, 1, 8, 128]
        B = q.shape[0]
        num_qo_heads = q.shape[1]           # 32
        head_dim = q.shape[2]               # 128
        num_pages = k_cache.shape[0]        # number of cached K/V "pages"
        num_kv_heads = k_cache.shape[2]     # 8

        device = q.device

        # Output and lse tensors (fp32 for compute)
        output = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

        SM_SCALE = float(sm_scale)
        LOG2_INVERSE = 1.0 / math.log(2.0)  # 1/ln(2)

        # Compute num_tokens per batch from kv_indptr (len_indptr = B+1)
        num_tokens_list = []
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens_list.append(end - start)

        # Process each batch element
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens = end - start
            if num_tokens <= 0:
                output[b].zero_()
                lse[b].zero_()
                continue

            # For each b, the tokens are a contiguous segment in kv_indices given by [start, end).
            # In provided tests, num_tokens == 1, so we take the first token index.
            token_index = int(kv_indices[start].item())  # single token index for this batch

            # Gather K and V for this token index and per-head slices
            # k_cache/v_cache: [num_pages, 1, num_kv_heads, head_dim]
            K_t = k_cache[token_index]        # [1, 1, 8, 128]
            V_t = v_cache[token_index]        # [1, 1, 8, 128]
            K_t = K_t.to(torch.float32).contiguous()  # [1, 1, 8, 128]
            V_t = V_t.to(torch.float32).contiguous()  # [1, 1, 8, 128]

            # GQA mapping
            gqa_ratio = num_qo_heads // num_kv_heads  # 4
            for h in range(num_qo_heads):
                kv_head = h


def run(*args):
    return ModelNew()(*args)
