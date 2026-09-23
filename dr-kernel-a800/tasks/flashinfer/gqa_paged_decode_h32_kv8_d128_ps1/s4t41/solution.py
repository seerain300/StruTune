import math
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_and_attention_single_bh(
    OUT_ptr,   # *float32, shape [head_dim]
    LSE_ptr,   # *float32, scalar
    Q_ptr,     # *float32, shape [head_dim]
    K_ptr,     # *float32, shape [num_tokens_total, head_dim]
    V_ptr,     # *float32, shape [num_tokens_total, head_dim]
    num_tokens_b,            # int runtime: number of tokens for this batch
    sm_scale,                # float32
    LOG2_INVERSE,            # float32 (1 / ln(2))
    head_dim: tl.constexpr,         # compile-time constant
    KV_HEAD: tl.constexpr,          # compile-time constant: h // 4
):
    # First pass: compute lse = logsumexp(scaled_logits) / ln(2)
    running_max = -float("inf")
    running_sum = 0.0

    t = 0
    while t < num_tokens_b:
        # Load q vector for head h (1D of length head_dim)
        i = tl.arange(0, head_dim)
        q_vec = tl.load(Q_ptr + i)

        # Load k vector for token t: row index t selects one token; per-head slice via KV_HEAD
        # K_ptr layout: [num_tokens_total, head_dim] contiguous
        k_row = tl.load(K_ptr + t * head_dim + i)

        # Compute logits and scaled logits
        logits = tl.sum(q_vec * k_row, axis=0)  # scalar
        scaled = logits * sm_scale

        # Update running max and sum (stable logsumexp)
        running_max_new = tl.maximum(running_max, scaled)
        running_sum = running_sum * tl.exp(running_max - running_max_new) + tl.exp(scaled - running_max_new)
        running_max = running_max_new

        t += 1

    # lse = log(sum(exp(scaled - max))) + max, divide by ln(2)
    lse_value = tl.log(running_sum) + running_max
    lse_value = lse_value * LOG2_INVERSE
    tl.store(LSE_ptr, lse_value)

    # Second pass: compute output = sum_j softmax(scaled)_j * V_j for this batch
    out_vec = tl.zeros((head_dim,), dtype=tl.float32)
    t = 0
    while t < num_tokens_b:
        i = tl.arange(0, head_dim)
        q_vec = tl.load(Q_ptr + i)
        k_row = tl.load(K_ptr + t * head_dim + i)
        v_row = tl.load(V_ptr + t * head_dim + i)

        logits = tl.sum(q_vec * k_row, axis=0)
        scaled = logits * sm_scale

        attn = tl.exp(scaled - lse_value)  # scalar attention for this token

        out_vec += attn * v_row

        t += 1

    # Store output vector
    j = tl.arange(0, head_dim)
    tl.store(OUT_ptr + j, out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes
        batch_size, num_qo_heads, head_dim = q.shape
        num_pages, num_kv_heads, _ = k_cache.shape  # k_cache: [num_pages, num_kv_heads, head_dim]
        len_indptr = kv_indptr.shape[0]
        num_kv_indices = kv_indices.shape[0]

        # Assertions as in original
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128
        assert len_indptr == batch_size + 1
        assert num_kv_indices == kv_indptr[-1].item()

        # Output buffers
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q.device)

        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        # Compute per-batch token counts and build flattened token indices
        # We need total tokens across all batches to create K_t/V_t of shape [num_tokens_total, head_dim].
        total_tokens = 0
        for b in range(batch_size):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            total_tokens += (end - start)

        # Build token indices array: flat [total_tokens], and per-batch starts
        flat_indices = torch.empty((total_tokens,), dtype=torch.long, device=q.device)
        batch_starts = torch.empty((batch_size,), dtype=torch.long, device=q.device)
        current = 0
        for b in range(batch_size):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            batch_tokens = end - start
            batch_starts[b] = current
            flat_indices[current:current + batch_tokens] = kv_indices[start:end]
            current += batch_tokens

        # For each batch b, we will pass K_t and V_t corresponding to flat_indices[batch_starts[b]:batch_starts[b]+num_tokens_b]
        # But since the kernel expects a single K_ptr/V_ptr of length total_tokens, we can slice per b inside the kernel by num_tokens_b.
        # Prepare K_t and V_t per (b, h): flatten per-batch tokens for this head and pass them to the kernel by selecting via num_tokens_b.

        # Iterate over batch and heads; launch Triton kernel per (b, h)
        for b in range(batch_size):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens_b = end - start

            # If no tokens, lse = -inf and output zero
            if num_tokens_b <= 0:
                lse[b, :] = -float("inf")
                output[b].zero_()
                continue

            # For each head h
            for h in range(num_qo_heads):
                kv_head = h // gqa_ratio

                # Gather K and V for this batch and kv_head: [num_tokens_b, head_dim]
                # We need k_cache[token_idx, kv_head, :] for each token_idx in this batch
                # Using flat_indices and batch_starts, token_idx is flat_indices[batch_starts[b] + t]
                k_rows = torch.empty((num_tokens_b, head_dim), dtype=torch.bfloat16, device=q.device)
                v_rows = torch.empty((num_tokens_b, head_dim), dtype=torch.bfloat16, device=q.device)

                # Fill k_rows and v_rows
                for t in range(num_tokens_b):
                    idx = int(flat_indices[batch_starts[b] + t].item())
                    k_rows[t] = k_cache[idx, kv_head, :]
                    v_rows[t] = v_cache[idx, kv_head, :]

                # Cast to float32 for Triton
                K_t = k_rows.to(torch.float32).contiguous()  # [num_tokens_b, head_dim]
                V_t = v_rows.to(torch.float32).contiguous()  # [num_tokens_b, head_dim]

                # Q vector for this head
                q_vec = q[b, h, :].to(torch.float32).contiguous()  # [head_dim]

                # Output vector and lse scalar
                out_vec = torch.empty((head_dim,), dtype=torch.float32, device=q.device)
                lse_scalar = torch.empty((), dtype=torch.float32, device=q.device)

                # Launch Triton kernel for (b, h)
                grid = (1, 1)
                softmax_and_attention_single_bh[grid](
                    out_vec,                # OUT_ptr
                    lse_scalar,             # LSE_ptr
                    q_vec,                  # Q_ptr
                    K_t.view(-1),           # K_ptr flattened to [num_tokens_b * head_dim]
                    V_t.view(-1),           # V_ptr flattened
                    num_tokens_b,           # runtime: number of tokens for this batch
                    float(sm_scale),        # sm_scale
                    1.4426950408889634,     # LOG2_INVERSE
                    head_dim,               # constexpr
                    h // gqa_ratio,         # KV_HEAD constexpr
                )

                # Store results
                output[b, h, :] = out_vec.to(torch.bfloat16)
                lse[b, h] = lse_scalar.item()  # store as float32 scalar

        return output, lse


def run(*args):
    return ModelNew()(*args)
