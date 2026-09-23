import torch
import triton
import triton.language as tl


@triton.jit
def attn_gqa_token_kernel(
    q_ptr,          # *float32, shape: [num_q_tokens, num_qo_heads, head_dim]
    k_ptr,          # *float32, shape: [num_kv_tokens, num_kv_heads, head_dim]
    v_ptr,          # *float32, shape: [num_kv_tokens, num_kv_heads, head_dim]
    out_ptr,        # *bfloat16, shape: [num_q_tokens, num_qo_heads, head_dim]
    lse_ptr,        # *float32, shape: [num_q_tokens, num_qo_heads]
    # meta-parameters (compile-time constants for this launch)
    num_qo_heads: tl.constexpr,   # 32
    num_kv_heads: tl.constexpr,   # 8
    head_dim: tl.constexpr,       # 128
    gqa_ratio: tl.constexpr,      # 4 (num_qo_heads // num_kv_heads)
    sm_scale,                     # float32 scalar
    # runtime parameters
    q_start,                      # int32: start token index for this batch
    num_q_tokens,                 # int32: number of tokens in this batch
    kv_start,                     # int32: start kv index for this batch
    num_kv_tokens,                # int32: number of kv tokens in this batch
    kv_indices_ptr,               # *int32: [num_kv_indices]
    q_batch_stride,               # int32: stride to move between tokens in q_ptr (num_qo_heads * head_dim)
    kv_head_stride,               # int32: stride between kv_heads in k_ptr/v_ptr (head_dim)
):
    # One program per token
    t = tl.program_id(0)  # token index within this batch
    if t >= num_q_tokens:
        return

    # For each query head h, compute attention against selected kv tokens
    for h in range(num_qo_heads):
        kv_head = h // gqa_ratio  # 0..7

        # Initialize accumulators for LSE and output
        max_logits = tl.full((), -float("inf"), tl.float32)
        sum_exp = tl.zeros((), dtype=tl.float32)
        out_vec = tl.zeros([head_dim], dtype=tl.float32)

        # Loop over kv tokens
        for kv_idx in range(num_kv_tokens):
            kv_index = tl.load(kv_indices_ptr + kv_idx)  # int32 index into k/v cache
            # Compute offsets for k and v
            # k_ptr is [num_kv_tokens, num_kv_heads, head_dim] with elements contiguous along head_dim
            k_offset = kv_index * (num_kv_heads * head_dim) + kv_head * head_dim
            # v_ptr similarly
            v_offset = kv_index * (num_kv_heads * head_dim) + kv_head * head_dim

            # Load q vector for this head and this token
            q_off = (t + q_start) * q_batch_stride + h * head_dim
            q_vec = tl.load(q_ptr + q_off + tl.arange(0, head_dim), mask=tl.arange(0, head_dim) < head_dim, other=0.0)

            # Load k vector
            k_vec = tl.load(k_ptr + k_offset + tl.arange(0, head_dim), mask=tl.arange(0, head_dim) < head_dim, other=0.0)

            # Dot product
            dot = tl.sum(q_vec * k_vec, axis=0)  # scalar
            logits_scaled = dot * sm_scale

            # Update logsumexp components
            # Compute max over all tokens
            # Note: we need to incorporate logits_scaled into running max/sum in a way Triton supports.
            # We'll do it in two passes: first compute max, second compute sum of exp(logits_scaled - max)
            # However, Triton supports dynamic loop; we can use tl.where and accumulate.
            # Approach: maintain max and sum_exp per scalar, not per vector.
            # We can compute per-token max using tl.maximum and accumulate sum_exp.
            # Since this is scalar, we update max and sum_exp accordingly.
            # We'll initialize max_logits above; for first iteration, max_logits = logits_scaled.
            # For subsequent iterations, update using new max and rescale sum_exp.
            # Handle first iteration explicitly:
            if kv_idx == 0:
                max_logits = logits_scaled
            else:
                new_max = tl.maximum(max_logits, logits_scaled)
                # sum_exp rescales when new_max > max_logits
                sum_exp = sum_exp * tl.exp(max_logits - new_max) + tl.exp(logits_scaled - new_max)
                max_logits = new_max

        # Now compute softmax using the final max and sum_exp
        # soft = exp((logits_scaled - max_logits) * sm_scale) / sum_exp
        # But we don't have logits_scaled beyond the first; we need to recompute for each kv_idx and accumulate.
        # To correctly compute softmax, we must go through a second loop to compute per-token softmax contributions.
        # Let's recompute in a second pass over kv tokens, compute softmax, and accumulate output.
        for kv_idx in range(num_kv_tokens):
            kv_index = tl.load(kv_indices_ptr + kv_idx)
            k_offset = kv_index * (num_kv_heads * head_dim) + kv_head * head_dim
            v_offset = kv_index * (num_kv_heads * head_dim) + kv_head * head_dim

            q_off = (t + q_start) * q_batch_stride + h * head_dim
            q_vec = tl.load(q_ptr + q_off + tl.arange(0, head_dim), mask=tl.arange(0, head_dim) < head_dim, other=0.0)
            k_vec = tl.load(k_ptr + k_offset + tl.arange(0, head_dim), mask=tl.arange(0, head_dim) < head_dim, other=0.0)

            dot = tl.sum(q_vec * k_vec, axis=0)
            logits_scaled = dot * sm_scale
            soft = tl.exp(logits_scaled - max_logits) / sum_exp  # scalar

            v_vec = tl.load(v_ptr + v_offset + tl.arange(0, head_dim), mask=tl.arange(0, head_dim) < head_dim, other=0.0)
            out_vec += soft * v_vec

        # Store output and lse for this (token, head)
        out_off = (t) * (num_qo_heads * head_dim) + h * head_dim  # since out is contiguous [num_q_tokens, num_qo_heads, head_dim]
        tl.store(out_ptr + out_off, out_vec.to(tl.bfloat16))

        lse_off = (t) * num_qo_heads + h
        tl.store(lse_ptr + lse_off, max_logits)  # LSE is logsumexp(logits_scaled)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Original asserts: num_qo_heads == 32, num_kv_heads == 8, head_dim == 128
        device = q.device

        # Cast to float32 for math
        q_f32 = q.to(torch.float32).contiguous()
        # Flatten/reshape k/v cache: [num_pages, 1, 8, 128] -> [num_pages, 8, 128]
        k_cache_flat = k_cache.squeeze(1).to(torch.float32).contiguous()
        v_cache_flat = v_cache.squeeze(1).to(torch.float32).contiguous()

        total_q = q_f32.shape[0]
        num_qo_heads = 32
        num_kv_heads = 8
        head_dim = 128
        gqa_ratio = num_qo_heads // num_kv_heads
        sm_scale = float(sm_scale)  # ensure float

        # Prepare output and LSE
        out = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Iterate over batches defined by qo_indptr and kv_indptr
        len_indptr = qo_indptr.shape[0]
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            # Skip empty batches
            if (q_end - q_start) <= 0 or (kv_end - kv_start) <= 0:
                continue

            # Slice q for this batch
            q_batch = q_f32[q_start:q_end]  # [num_q_tokens, 32, 128]
            num_q_tokens = q_batch.shape[0]

            # Gather k and v based on kv_indices for this batch
            num_kv_tokens = kv_end - kv_start
            # Create indices tensor for this batch
            kv_indices_batch = kv_indices[kv_start:kv_end]  # [num_kv_tokens]
            # Index into k_cache_flat and v_cache_flat
            # k_cache_flat: [num_pages, 8, 128], v_cache_flat similarly
            k_batch = k_cache_flat.index_select(0, kv_indices_batch)  # [num_kv_tokens, 8, 128]
            v_batch = v_cache_flat.index_select(0, kv_indices_batch)  # [num_kv_tokens, 8, 128]

            # Launch Triton kernel: one program per token in this batch
            grid = (num_q_tokens,)
            # Strides: q_stride between tokens is num_qo_heads * head_dim; within head is head_dim
            q_batch_stride = num_qo_heads * head_dim
            kv_head_stride = head_dim  # per head, elements are contiguous

            attn_gqa_token_kernel[grid](
                q_batch, k_batch, v_batch,
                out, lse,
                num_qo_heads=32, num_kv_heads=8, head_dim=128, gqa_ratio=4, sm_scale=sm_scale,
                q_start=q_start, num_q_tokens=num_q_tokens,
                kv_start=kv_start, num_kv_tokens=num_kv_tokens,
                kv_indices_ptr=kv_indices_batch,
                q_batch_stride=q_batch_stride, kv_head_stride=kv_head_stride,
                num_warps=4, num_stages=2
            )

        return out, lse


def run(*args):
    return ModelNew()(*args)
