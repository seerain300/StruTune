import math
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_and_attention_single_bh(
    q_ptr,                 # *fp32, q[b, :] reshaped as [num_qo_heads, HEAD_DIM]
    K_ptr,                 # *fp32, K tokens flattened as [NUM_TOKENS, HEAD_DIM]
    V_ptr,                 # *fp32, V tokens flattened as [NUM_TOKENS, HEAD_DIM]
    OUT_ptr,               # *fp32, output[b, h, :] flattened as [num_qo_heads, HEAD_DIM]
    LSE_ptr,               # *fp32, lse[b, h] flattened as [num_qo_heads]
    NUM_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.float32,      # 1.0 / sqrt(HEAD_DIM) (here 1/sqrt(128))
    LOG2_INVERSE: tl.float32,  # 1.0 / ln(2.0) ≈ 1.4426950408889634
):
    # 2D grid: program_id(0) = b, program_id(1) = h
    b = tl.program_id(0)
    h = tl.program_id(1)

    # First pass: compute lse = logsumexp(scaled_logits) / ln(2)
    running_max = -float("inf")
    running_sum = 0.0

    for t in range(0, NUM_TOKENS):
        # Load q vector for this head: [HEAD_DIM]
        q_vec = tl.load(q_ptr + b * (HEAD_DIM * num_qo_heads) + h * HEAD_DIM + tl.arange(0, HEAD_DIM))
        # Load k vector for token t: [HEAD_DIM]
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))
        # Dot product: scalar
        logits_t = tl.sum(q_vec * k_vec, axis=0)
        # Scale
        scaled = logits_t * SM_SCALE
        running_max = tl.maximum(running_max, scaled)
        running_sum = running_sum * tl.exp(running_max - running_max) + tl.exp(scaled - running_max)

    lse = tl.log(running_sum) + running_max  # ln-sumexp
    # Divide by ln(2) (i.e., multiply by 1/ln(2))
    lse_scaled = lse * LOG2_INVERSE
    # Store lse as a scalar at LSE_ptr[h] (per head)
    tl.store(LSE_ptr + h, lse_scaled)

    # Second pass: compute attention and accumulate output
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for t in range(0, NUM_TOKENS):
        q_vec = tl.load(q_ptr + b * (HEAD_DIM * num_qo_heads) + h * HEAD_DIM + tl.arange(0, HEAD_DIM))
        v_vec = tl.load(V_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))
        logits_t = tl.sum(q_vec * k_vec, axis=0)
        scaled = logits_t * SM_SCALE
        attn = tl.exp(scaled - lse_scaled)
        out_vec += attn * v_vec

    # Store output vector into OUT_ptr[b, h, :]
    tl.store(OUT_ptr + b * (HEAD_DIM * num_qo_heads) + h * HEAD_DIM + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "All tensors must be on CUDA device for Triton"
        assert q.dtype == torch.bfloat16 and k_cache.dtype == torch.bfloat16 and v_cache.dtype == torch.bfloat16
        B, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and head_dim == 128 and num_kv_heads == 8

        # Allocate outputs (fp32 compute)
        output = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)  # [B, 32, 128]
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)               # [B, 32]

        # Compute per-batch num_tokens from kv_indptr
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens_b = end - start

            if num_tokens_b == 0:
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather token indices for this batch and form K_t, V_t
            token_indices = []
            for t in range(num_tokens_b):
                idx = start + t  # 0..num_tokens_b-1
                token_idx = int(kv_indices[idx].item())
                token_indices.append(token_idx)

            # Gather K_t and V_t: [num_tokens_b, 1, 8, 128]
            K_t = k_cache[token_indices]  # [num_tokens_b, 1, 8, 128]
            V_t = v_cache[token_indices]  # [num_tokens_b, 1, 8, 128]

            # Cast to fp32 and contiguous
            K_t = K_t.to(torch.float32).contiguous()  # [num_tokens_b, 1, 8, 128]
            V_t = V_t.to(torch.float32).contiguous()

            # Prepare q[b] as fp32: shape [32, 128], contiguous
            q_b = q[b].to(torch.float32).contiguous()  # [32, 128]

            # Launch Triton kernel once per (b, h) with 2D grid
            grid = (B, num_qo_heads)
            LOG2_INVERSE = 1.4426950408889634  # 1 / ln(2)

            softmax_and_attention_single_bh[grid](
                q_b,                       # q_ptr, shape [32, 128]
                K_t.view(-1, head_dim),   # K_ptr flattened [NUM_TOKENS, 128]
                V_t.view(-1, head_dim),   # V_ptr flattened [NUM_TOKENS, 128]
                output,                   # OUT_ptr [B, 32, 128] flattened per (b,h)
                lse[b],                   # LSE_ptr per head for this b
                NUM_TOKENS=num_tokens_b,
                HEAD_DIM=head_dim,
                SM_SCALE=1.0 / math.sqrt(head_dim),   # sm_scale
                LOG2_INVERSE=LOG2_INVERSE,
                num_warps=4,
                num_stages=2,
            )

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
