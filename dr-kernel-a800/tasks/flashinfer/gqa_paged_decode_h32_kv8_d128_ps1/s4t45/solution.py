import torch
import math
import triton
import triton.language as tl

# Triton kernel: computes attention output for one (batch, head) pair and its lse
# Inputs:
#   q_ptr: [B, num_qo_heads, head_dim], fp32, contiguous
#   k_ptr: [num_tokens, num_kv_heads, head_dim], fp32, contiguous
#   v_ptr: [num_tokens, num_kv_heads, head_dim], fp32, contiguous
# Outputs:
#   out_ptr: [num_qo_heads, head_dim], fp32
#   lse_ptr: [num_qo_heads], fp32
@triton.jit
def softmax_and_attention_single_bh(
    q_ptr, k_ptr, v_ptr,
    out_ptr, lse_ptr,
    num_tokens: tl.int32,
    sm_scale: tl.float32,
    head_dim: tl.int32,
    num_kv_heads: tl.int32,
    GQA_RATIO: tl.int32,       # num_qo_heads // num_kv_heads, e.g., 4
):
    # program id for head
    h = tl.program_id(1)
    kv_head = h // GQA_RATIO  # map Q head to KV head via GQA

    # First pass: numerically-stable logsumexp over scaled logits
    running_max = -float("inf")
    running_sum = 0.0

    for t in range(0, num_tokens):
        # load q vector for this head
        q_off = h * head_dim
        q_vec = tl.load(q_ptr + q_off, mask=True, other=0.0)  # [head_dim]

        # load k vector for token t and kv_head
        k_off = t * (num_kv_heads * head_dim) + kv_head * head_dim
        k_vec = tl.load(k_ptr + k_off, mask=True, other=0.0)  # [head_dim]

        # logits = dot(q_vec, k_vec)
        logits_t = 0.0
        for i in range(0, head_dim):
            logits_t += q_vec[i] * k_vec[i]

        scaled = logits_t * sm_scale
        running_max = tl.maximum(running_max, scaled)
        # update sum: new_token contributes exp(scaled - running_max)
        running_sum = running_sum * tl.exp(-running_max + scaled) + 1.0

    # lse = log(running_sum) + running_max, then divide by ln(2)
    lse_val = tl.log(running_sum) + running_max
    LOG2_INVERSE = 1.4426950408889634  # 1 / ln(2)
    lse_val = lse_val * LOG2_INVERSE
    tl.store(lse_ptr + h, lse_val)

    # Second pass: compute output vector
    out_vec = tl.zeros([head_dim], dtype=tl.float32)
    for t in range(0, num_tokens):
        q_off = h * head_dim
        q_vec = tl.load(q_ptr + q_off, mask=True, other=0.0)  # [head_dim]

        k_off = t * (num_kv_heads * head_dim) + kv_head * head_dim
        k_vec = tl.load(k_ptr + k_off, mask=True, other=0.0)  # [head_dim]

        v_off = t * (num_kv_heads * head_dim) + kv_head * head_dim
        v_vec = tl.load(v_ptr + v_off, mask=True, other=0.0)  # [head_dim]

        logits_t = 0.0
        for i in range(0, head_dim):
            logits_t += q_vec[i] * k_vec[i]

        scaled = logits_t * sm_scale
        attn = tl.exp(scaled - lse_val)  # softmax over scaled logits
        # accumulate output
        for i in range(0, head_dim):
            out_vec[i] += attn * v_vec[i]

    # store output vector for this head
    out_off = h * head_dim
    tl.store(out_ptr + out_off, out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        q: [B, num_qo_heads, head_dim], dtype bfloat16 (cast to fp32 in-kernel)
        k_cache: [num_pages, num_kv_heads, head_dim], dtype bfloat16 (cast to fp32)
        v_cache: [num_pages, num_kv_heads, head_dim], dtype bfloat16 (cast to fp32)
        kv_indptr: [B+1], int32
        kv_indices: [num_kv_indices], int32
        sm_scale: float32 scalar, 1/sqrt(128) in the original
        Returns:
        - output: [B, num_qo_heads, head_dim], bfloat16
        - lse: [B, num_qo_heads], float32
        """
        B, num_qo_heads, head_dim = q.shape
        _, num_kv_heads, _ = k_cache.shape

        # Compute per-batch number of tokens: num_tokens[b] = kv_indptr[b+1] - kv_indptr[b]
        assert kv_indptr.shape[0] == B + 1, "kv_indptr length must be B+1"
        num_tokens_list = []
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens_list.append(end - start)
        num_tokens_list = torch.tensor(num_tokens_list, dtype=torch.int32, device=q.device)

        # Output buffers in fp32 for computation
        out = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # For each batch b
        for b in range(B):
            num_tokens_b = int(num_tokens_list[b].item())
            if num_tokens_b == 0:
                out[b].zero_()
                lse[b].zero_()
                continue

            # Build token indices for this batch element
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            token_indices = kv_indices[start:start + num_tokens_b].to(torch.int32).contiguous()

            # Gather K_t and V_t: [num_tokens_b, num_kv_heads, head_dim]
            # k_cache and v_cache shapes: [num_pages, num_kv_heads, head_dim]
            K_t = k_cache[token_indices, :, :].contiguous().to(torch.float32)
            V_t = v_cache[token_indices, :, :].contiguous().to(torch.float32)

            # Q for this batch: [num_qo_heads, head_dim], fp32
            Q_b = q[b].contiguous().to(torch.float32)

            # Launch Triton kernel: one program per (b, h). grid[1] = num_qo_heads
            grid = (B, num_qo_heads)
            softmax_and_attention_single_bh[grid](
                Q_b, K_t, V_t,
                out[b], lse[b],
                num_tokens_b,
                sm_scale,
                head_dim,
                num_kv_heads,
                num_qo_heads // num_kv_heads,
                num_warps=4,
            )

        # Cast output to bfloat16 to match original
        output = out.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
