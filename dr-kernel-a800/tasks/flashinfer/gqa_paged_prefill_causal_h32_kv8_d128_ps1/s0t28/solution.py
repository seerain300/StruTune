import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per batch. Computes output[total_q, 32, 128] (fp32) and lse[total_q, 32] (fp32)
# for that batch. Host allocates output and lse as zeros and adds each batch's contribution to produce
# the final result. This keeps all computation in Triton and avoids Python-side loops over batches.
if TRITON_AVAILABLE:
    @triton.jit
    def _batch_compute_kernel(
        q_ptr, k_ptr, v_ptr,
        qo_indptr_ptr, kv_indptr_ptr, kv_indices_ptr,
        output_ptr, lse_ptr,
        sm_scale: tl.float32,
        total_q: tl.int32, num_qo_heads: tl.int32, head_dim: tl.int32, gqa_ratio: tl.int32,
        b: tl.int32,  # batch index in [0, len_indptr - 2]
        len_indptr: tl.int32,
    ):
        # Load batch boundaries
        q_start = tl.load(qo_indptr_ptr + b)
        q_end = tl.load(qo_indptr_ptr + b + 1)
        kv_start = tl.load(kv_indptr_ptr + b)
        kv_end = tl.load(kv_indptr_ptr + b)

        num_q_tokens = q_end - q_start
        num_kv_tokens = kv_end - kv_start

        # Early return if nothing to do
        if (num_q_tokens <= 0) or (num_kv_tokens <= 0):
            return

        # Build idx_vec for this batch: kv_indices[kv_start:kv_end]
        idx_vec = []
        i = 0
        while i < num_kv_tokens:
            idx_val = tl.load(kv_indices_ptr + kv_start + i)
            idx_vec.append(idx_val)
            i += 1

        # Process each query token
        q_idx = 0
        while q_idx < num_q_tokens:
            global_q_idx = q_start + q_idx

            # Causal masking: max_kv_idx = min(q_idx + 1 + (num_kv_tokens - num_q_tokens), num_kv_tokens)
            delta = num_kv_tokens - num_q_tokens
            max_kv_idx = num_kv_tokens
            if (q_idx + 1 + delta) < num_kv_tokens:
                max_kv_idx = q_idx + 1 + delta

            # For each head h
            h = 0
            while h < num_qo_heads:
                kv_head = h // gqa_ratio  # GQA mapping: 32 heads -> 8 kv_heads, ratio=4

                # Load q_vec[h] for this token: [128]
                q_off = global_q_idx * (num_qo_heads * head_dim) + h * head_dim
                q_vec = tl.load(q_ptr + q_off + tl.arange(0, head_dim))
                q_vec = q_vec.to(tl.float32)  # ensure fp32

                # Initialize lse accumulators for this head
                running_max = tl.full((), -float("inf"), dtype=tl.float32)
                lse_sum = tl.full((), 0.0, dtype=tl.float32)

                # Loop over KV indices i up to max_kv_idx
                i = 0
                while i < max_kv_idx:
                    idx = idx_vec[i]
                    # Compute offsets for K and V rows
                    # k_ptr, v_ptr layout: [num_pages, num_kv_heads, head_dim] flattened
                    k_off = idx * (num_kv_heads * head_dim) + kv_head * head_dim
                    v_off = idx * (num_kv_heads * head_dim) + kv_head * head_dim

                    # Load k_row and v_row vectors
                    k_row = tl.load(k_ptr + k_off + tl.arange(0, head_dim))
                    v_row = tl.load(v_ptr + v_off + tl.arange(0, head_dim))
                    k_row = k_row.to(tl.float32)
                    v_row = v_row.to(tl.float32)

                    # Compute dot product q_vec @ k_row (scalar)
                    dot_val = tl.zeros((), dtype=tl.float32)
                    # Unroll across head_dim in chunks of 16 for efficiency
                    for d in range(0, head_dim, 16):
                        q_chunk = q_vec[d : d + 16]
                        k_chunk = k_row[d : d + 16]
                        dot_val += tl.sum(q_chunk * k_chunk, axis=0)

                    # Scale logits
                    logits_i = dot_val * sm_scale

                    # Streaming logsumexp update
                    new_max = tl.maximum(running_max, logits_i)
                    sum_term = lse_sum * tl.exp(running_max - logits_i) + tl.exp(new_max - logits_i)
                    running_max = new_max
                    lse_sum = sum_term

                    i += 1

                # Compute lse_value = (max + log(sum)) / ln(2)
                lse_value = (running_max + tl.log(lse_sum)) * (1.0 / math.log(2.0))

                # Compute output vector: out_vec[h] = sum_i exp(logits_i - max) * v_row[i]
                out_vec = tl.zeros((head_dim,), dtype=tl.float32)
                i = 0
                while i < max_kv_idx:
                    idx = idx_vec[i]
                    k_off = idx * (num_kv_heads * head_dim) + kv_head * head_dim
                    v_off = idx * (num_kv_heads * head_dim) + kv_head * head_dim
                    k_row = tl.load(k_ptr + k_off + tl.arange(0, head_dim))
                    v_row = tl.load(v_ptr + v_off + tl.arange(0, head_dim))
                    k_row = k_row.to(tl.float32)
                    v_row = v_row.to(tl.float32)
                    dot_val = tl.zeros((), dtype=tl.float32)
                    for d in range(0, head_dim, 16):
                        q_chunk = q_vec[d : d + 16]
                        k_chunk = k_row[d : d + 16]
                        dot_val += tl.sum(q_chunk * k_chunk, axis=0)
                    logits_i = dot_val * sm_scale
                    attn_i = tl.exp(logits_i - running_max)
                    out_vec += attn_i * v_row

                    i += 1

                # Store output: output[global_q_idx, h, :] in fp32
                out_addr = output_ptr + (global_q_idx * num_qo_heads + h) * head_dim
                tl.store(out_addr + tl.arange(0, head_dim), out_vec)

                # Update lse[global_q_idx, h] += lse_value / ln(2)
                lse_addr = lse_ptr + (global_q_idx * num_qo_heads + h)
                curr_lse = tl.load(lse_addr)
                new_lse = curr_lse + lse_value
                tl.store(lse_addr, new_lse)

                h += 1

            q_idx += 1

        b += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA and contiguous
        if q.device.type != "cuda":
            q = q.cuda(non_blocking=True)
        if k_cache.device.type != "cuda":
            k_cache = k_cache.cuda(non_blocking=True)
        if v_cache.device.type != "cuda":
            v_cache = v_cache.cuda(non_blocking=True)
        if qo_indptr.device.type != "cuda":
            qo_indptr = qo_indptr.cuda(non_blocking=True)
        if kv_indptr.device.type != "cuda":
            kv_indptr = kv_indptr.cuda(non_blocking=True)
        if kv_indices.device.type != "cuda":
            kv_indices = kv_indices.cuda(non_blocking=True)

        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()

        # Flatten k_cache and v_cache by squeezing time dim (always 1 in provided inputs)
        k_cache_flat = k_cache.squeeze(1)  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1)  # [num_pages, 8, 128]

        total_q, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        # Shape assertions as in original
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128, "Shape assertions must hold."
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        len_indptr = qo_indptr.shape[0]
        B = len_indptr - 1  # number of batches

        # Allocate output and lse buffers (fp32 for computation; will cast output to bfloat16 on return)
        output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per batch
        grid = (B,)
        _batch_compute_kernel[grid](
            q, k_cache_flat, v_cache_flat,
            qo_indptr, kv_indptr, kv_indices,
            output, lse,
            sm_scale,
            total_q, num_qo_heads, head_dim, gqa_ratio,
            B, len_indptr,
            num_warps=4, num_stages=2
        )

        # Cast output to bfloat16 to match original behavior
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
