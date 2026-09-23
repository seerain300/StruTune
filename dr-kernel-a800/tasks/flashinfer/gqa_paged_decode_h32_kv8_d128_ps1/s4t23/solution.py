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
    LSE_ptr,               # *fp32, lse scalar for this (b, h)
    NUM_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.float32,      # 1.0 / sqrt(HEAD_DIM)
    LOG2_INVERSE: tl.float32,  # 1.0 / ln(2.0) ≈ 1.4426950408889634
):
    # 2D grid: axis 0 is batch, axis 1 is head
    b = tl.program_id(0)
    h = tl.program_id(1)

    # First pass: compute lse = logsumexp(scaled_logits) / ln(2)
    running_max = -float("inf")
    running_sum = 0.0

    for t in range(0, NUM_TOKENS):
        # Load q vector for this head: [HEAD_DIM]
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))
        # Load k vector for token t: [HEAD_DIM]
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))
        # Compute dot product scalar
        logits_t = tl.sum(q_vec * k_vec, axis=0)
        # Scale
        scaled = logits_t * SM_SCALE
        # Numerically stable logsumexp
        running_max = tl.maximum(running_max, scaled)
        running_sum = running_sum * tl.exp(running_max - running_max) + tl.exp(scaled - running_max)

    # lse = logsumexp(scaled) / ln(2) = log(running_sum) + running_max, then divide by ln(2)
    lse_val = tl.log(running_sum) + running_max
    lse_val = lse_val * LOG2_INVERSE
    tl.store(LSE_ptr + b * 32 + h, lse_val)

    # Second pass: compute output vector
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for t in range(0, NUM_TOKENS):
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        v_vec = tl.load(V_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))  # [HEAD_DIM]
        logits_t = tl.sum(q_vec * k_vec, axis=0)                        # scalar
        scaled = logits_t * SM_SCALE
        attn = tl.exp(scaled - lse_val)  # softmax over tokens
        out_vec += attn * v_vec

    # Store output
    tl.store(OUT_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors and dtypes
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "All tensors must be on CUDA device"
        assert q.dtype == torch.bfloat16 and k_cache.dtype == torch.bfloat16 and v_cache.dtype == torch.bfloat16
        B, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and head_dim == 128 and num_kv_heads == 8

        # Output buffers (fp32 for compute)
        output = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Compute per-batch num_tokens from kv_indptr
        # For each batch b, tokens are in [kv_indptr[b]: kv_indptr[b+1])
        # Ensure indices are within bounds
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens_b = end - start

            # If no tokens, zeros
            if num_tokens_b <= 0:
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather token indices for this batch
            # Note: kv_indices is a 1D int32 tensor. We assume it's valid and within [0, num_pages).
            token_indices = []
            for t in range(num_tokens_b):
                idx = start + t  # 0..num_tokens_b-1
                token_idx = int(kv_indices[idx].item())
                token_indices.append(token_idx)

            # Build K_t and V_t: [num_tokens_b, 1, 8, 128]
            K_t = k_cache[token_indices]  # [num_tokens_b, 1, 8, 128]
            V_t = v_cache[token_indices]  # [num_tokens_b, 1, 8, 128]

            # Cast to fp32 for compute and ensure contiguous
            K_t = K_t.to(torch.float32).contiguous()
            V_t = V_t.to(torch.float32).contiguous()

            # Precompute q vector per head
            q_vecs = q[b].to(torch.float32).contiguous()  # [32, 128]

            # For each head h, launch Triton kernel
            gqa_ratio = num_qo_heads // num_kv_heads  # 4
            for h in range(num_qo_heads):
                kv_head = h // gqa_ratio  # 0..7

                # Prepare pointers for this head: K_t[:, 0, kv_head, :], V_t[:, 0, kv_head, :]
                # Shapes: [num_tokens_b, 128]
                K_t_h = K_t[:, 0, kv_head, :]  # [num_tokens_b, 128]
                V_t_h = V_t[:, 0, kv_head, :]  # [num_tokens_b, 128]

                # q vector for this head: q_vecs[h]
                q_vec = q_vecs[h]  # [128]

                # For Triton, we need contiguous 1D pointers of length NUM_TOKENS*HEAD_DIM
                # But simpler: launch kernel with q_ptr pointing to q_vec, K_ptr/V_ptr pointing to per-token rows.
                # We'll pass NUM_TOKENS and HEAD_DIM as meta-parameters; pointers must be contiguous [NUM_TOKENS, HEAD_DIM].
                # Convert to contiguous [num_tokens_b, 128] -> [num_tokens_b*128] via flatten, then create 1D pointers.
                K_t_h_flat = K_t_h.contiguous().view(-1)              # [num_tokens_b * HEAD_DIM]
                V_t_h_flat = V_t_h.contiguous().view(-1)              # [num_tokens_b * HEAD_DIM]
                NUM_TOKENS = num_tokens_b
                HEAD_DIM = 128

                # Launch Triton kernel with 2D grid: (B, num_qo_heads)
                softmax_and_attention_single_bh[(B, num_qo_heads)](
                    q_vec,  # *fp32, 1D vector [HEAD_DIM]
                    K_t_h_flat,  # *fp32, 1D [NUM_TOKENS*HEAD_DIM]
                    V_t_h_flat,  # *fp32, 1D [NUM_TOKENS*HEAD_DIM]
                    output[b * num_qo_heads + h],  # OUT_ptr is [HEAD_DIM] but we pass per-head vector directly
                    lse[b * num_qo_heads + h],     # store scalar lse at lse[b, h]
                    NUM_TOKENS=NUM_TOKENS,
                    HEAD_DIM=HEAD_DIM,
                    SM_SCALE=float(sm_scale),      # 1.0 / sqrt(128) or provided
                    LOG2_INVERSE=1.4426950408889634  # 1 / ln(2.0)
                )

    def _write_output(self, output_flat_ptr, h, out_vec):
        # Helper to store out_vec into output[b,h,:]
        # Not used here; we directly store per-head output in the kernel.
        pass


def run(*args):
    return ModelNew()(*args)
