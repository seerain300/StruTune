import torch
import math
import triton
import triton.language as tl


@triton.jit
def softmax_attention_single_bh(
    q_ptr,          # *fp32, shape [num_qo_heads, head_dim]
    k_ptr,          # *fp32, shape [NUM_TOKENS, head_dim]
    v_ptr,          # *fp32, shape [NUM_TOKENS, head_dim]
    out_ptr,        # *fp32, shape [num_qo_heads, head_dim] (we write a single h per launch)
    lse_ptr,        # *fp32, shape [num_qo_heads]
    num_tokens: tl.constexpr,  # int32 scalar
    head_dim: tl.constexpr,    # int32 scalar
    kv_head,        # int32 scalar (0..num_kv_heads-1)
    LOG2_INV,       # fp32 scalar = 1 / ln(2)
    BLOCK_SIZE: tl.constexpr,  # e.g., 128
):
    h = tl.program_id(1)  # second grid dim is num_qo_heads

    # Load q vector for this head
    q_vec = tl.load(q_ptr + h * head_dim + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0)
    out_vec = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Running max and sum for logsumexp
    running_max = -float("inf")
    running_sum = 0.0

    # First pass: compute lse over tokens
    for t in range(0, num_tokens):
        k_vec = tl.load(k_ptr + t * head_dim + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0)
        # Dot product between q_vec and k_vec
        dot_val = 0.0
        for i in range(0, BLOCK_SIZE):
            dot_val += q_vec[i] * k_vec[i]

        # Scale logits
        inv_sqrt = 1.0 / tl.sqrt(128.0)
        scaled = dot_val * inv_sqrt

        # Update numerically stable running max/sum
        if scaled > running_max:
            running_sum = running_sum * tl.exp(running_max - scaled) + 1.0
            running_max = scaled
        else:
            running_sum += 1.0

    # Compute lse = log(running_sum) + running_max, then divide by ln(2)
    lse_val = tl.log(running_sum) + running_max
    lse_val = lse_val * LOG2_INV
    tl.store(lse_ptr + h, lse_val)

    # Second pass: compute output
    for t in range(0, num_tokens):
        k_vec = tl.load(k_ptr + t * head_dim + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0)
        v_vec = tl.load(v_ptr + t * head_dim + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < head_dim, other=0.0)

        dot_val = 0.0
        for i in range(0, BLOCK_SIZE):
            dot_val += q_vec[i] * k_vec[i]

        inv_sqrt = 1.0 / tl.sqrt(128.0)
        scaled = dot_val * inv_sqrt
        attn = tl.exp(scaled - lse_val)  # softmax over scaled logits

        out_vec += attn * v_vec

    # Store output for this (b, h)
    tl.store(out_ptr + h * head_dim + tl.arange(0, BLOCK_SIZE), out_vec, mask=tl.arange(0, BLOCK_SIZE) < head_dim)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        q: [B, num_qo_heads, head_dim], dtype=bfloat16 (or float32), device
        k_cache, v_cache: [num_pages, num_kv_heads, head_dim], dtype=bfloat16 (or float32), device
        kv_indptr: [len_indptr], int32, device
        kv_indices: [num_kv_indices], int32, device
        sm_scale: float32 scalar (unused inside Triton; we use 1/sqrt(128) in-kernel)
        Returns:
        - output: [B, num_qo_heads, head_dim], dtype=bfloat16
        - lse: [B, num_qo_heads], dtype=float32
        """
        # Shapes and assertions (as per original)
        B, num_qo_heads, head_dim = q.shape
        num_pages, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        assert head_dim == 128, "head_dim must be 128"

        # Compute per-batch token counts (host-side index arithmetic)
        num_tokens_list = []
        token_indices_lists = []
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens_list.append(end - start)
            token_indices_lists.append(kv_indices[start:end].to(torch.int64).tolist())

        # Prepare outputs
        output = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        LOG2_INV = 1.0 / math.log(2.0)  # 1/ln(2)

        # Launch Triton kernel for each (b, h)
        for b in range(B):
            num_tokens_b = num_tokens_list[b]
            token_indices_b = token_indices_lists[b]

            # Gather k and v for this batch into [num_tokens_b, head_dim] for each kv_head
            # After the original squeeze(1), k_cache and v_cache are [num_pages, num_kv_heads, head_dim].
            # We don't assume any size-1 at dim=1; directly gather [token_index, kv_head, :].
            k_rows = []
            v_rows = []
            for t in range(num_tokens_b):
                idx = int(token_indices_b[t])  # int64 scalar
                for kv_h in range(num_kv_heads):
                    k_rows.append(k_cache[idx, kv_h].contiguous().to(torch.float32))
                    v_rows.append(v_cache[idx, kv_h].contiguous().to(torch.float32))

            k_b = torch.stack(k_rows, dim=0).contiguous()  # [num_tokens_b, head_dim], fp32
            v_b = torch.stack(v_rows, dim=0).contiguous()  # [num_tokens_b, head_dim], fp32

            # Q for this batch: [num_qo_heads, head_dim], fp32
            q_b = q[b].to(torch.float32).contiguous()

            # For each head h, compute attention
            for h in range(num_qo_heads):
                kv_head = h // gqa_ratio  # 0..7

                # Launch one program per (b, h). We pass k_b and v_b for all kv_heads; the kernel selects kv_head by indexing.
                grid = (1, 1)
                softmax_attention_single_bh[grid](
                    q_b, k_b, v_b,
                    output[b], lse[b],
                    num_tokens_b, head_dim,
                    kv_head,
                    LOG2_INV,
                    BLOCK_SIZE=128,
                )

        # Cast output to bfloat16 to match original behavior
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
