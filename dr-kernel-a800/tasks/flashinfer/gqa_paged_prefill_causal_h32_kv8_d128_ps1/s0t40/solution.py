import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: process one batch per program (program_id = b), compute per-query/per-head outputs and lse contributions.
if TRITON_AVAILABLE:
    @triton.jit
    def _batch_compute_kernel(
        q_ptr, k_ptr, v_ptr,
        qo_indptr_ptr, kv_indptr_ptr, kv_indices_ptr,
        output_ptr, lse_ptr,
        sm_scale: tl.float32,
        total_q: tl.int32, num_qo_heads: tl.int32, head_dim: tl.int32,
        gqa_ratio: tl.int32,
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

        # Nothing to do: return early
        if (num_q_tokens <= 0) or (num_kv_tokens <= 0) or (b >= len_indptr - 1):
            return

        # Build local vector of KV indices for this batch: idx_vec = kv_indices[kv_start:kv_end]
        idx_vec = []
        i = 0
        while i < num_kv_tokens:
            idx_val = tl.load(kv_indices_ptr + kv_start + i)
            idx_vec.append(idx_val)
            i += 1

        # Process each query token in the batch
        q_idx = 0
        while q_idx < num_q_tokens:
            global_q_idx = q_start + q_idx

            # Causal masking: delta = num_kv_tokens - num_q_tokens
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
                    # Compute offsets for K and V rows (flattened [num_pages, num_kv_heads, head_dim])
                    k_off = idx * (num_kv_heads * head_dim) + kv_head * head_dim
                    v_off = idx * (num_kv_heads * head_dim) + kv_head * head_dim

                    # Load k_row and v_row vectors
                    k_row = tl.load(k_ptr + k_off + tl.arange(0, head_dim))
                    v_row = tl.load(v_ptr + v_off + tl.arange(0, head_dim))
                    k_row = k_row.to(tl.float32)
                    v_row = v_row.to(tl.float32)

                    # Compute dot product q_vec @ k_row (scalar)
                    dot_val = tl.zeros((), dtype=tl.float32)
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
                    attn = tl.exp(logits_i - running_max)  # softmax over i (causal)
                    out_vec += attn * v_row

                    i += 1

                # Store output: output[global_q_idx, h, :] in float32 (host will cast to bfloat16)
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
        return


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure Triton availability
        if not TRITON_AVAILABLE:
            # Fallback: original PyTorch computation (kept for correctness if Triton is unavailable)
            total_q, num_qo_heads, head_dim = q.shape
            num_pages, _, num_kv_heads, _ = k_cache.shape
            assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128, "Shape assertions must hold."
            gqa_ratio = num_qo_heads // num_kv_heads

            q_f32 = q.to(torch.float32)
            k_cache_flat = k_cache.squeeze(1).to(torch.float32)
            v_cache_flat = v_cache.squeeze(1).to(torch.float32)

            output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16)
            lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32)

            len_indptr = qo_indptr.shape[0]
            for b in range(len_indptr - 1):
                q_start = int(qo_indptr[b].item())
                q_end = int(qo_indptr[b + 1].item())
                kv_start = int(kv_indptr[b].item())
                kv_end = int(kv_indptr[b + 1].item())

                if q_start >= q_end or kv_start >= kv_end:
                    continue

                num_q_tokens = q_end - q_start
                num_kv_tokens = kv_end - kv_start

                # kv_indices slice
                idx_vec = kv_indices[kv_start:kv_end].tolist()

                # Iterate tokens and heads
                for q_idx in range(num_q_tokens):
                    global_q_idx = q_start + q_idx
                    delta = num_kv_tokens - num_q_tokens
                    max_kv_idx = num_kv_tokens if (q_idx + 1 + delta) >= num_kv_tokens else (q_idx + 1 + delta)

                    for h in range(num_qo_heads):
                        kv_head = h // gqa_ratio
                        q_pos = q_f32[global_q_idx, h]  # [head_dim]
                        k_batch = k_cache_flat[idx_vec[:max_kv_idx], kv_head]  # [max_kv_idx, head_dim]
                        v_batch = v_cache_flat[idx_vec[:max_kv_idx], kv_head]  # [max_kv_idx, head_dim]

                        logits = torch.matmul(q_pos, k_batch.transpose(0, 1))  # [max_kv_idx]
                        logits_scaled = logits * sm_scale
                        lse_val = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
                        lse[global_q_idx, h] = lse_val

                        attn = torch.softmax(logits_scaled, dim=-1)  # [max_kv_idx]
                        out_head = torch.matmul(attn, v_batch)  # [head_dim]
                        output[global_q_idx, h] = out_head.to(torch.bfloat16)

            return output, lse

        # Triton path: ensure tensors on CUDA and contiguous
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "All tensors must be on CUDA for Triton"
        assert qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Indices must be on CUDA for Triton"
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        qo_indptr = qo_indptr.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Flatten k_cache and v_cache by squeezing time dim (1)
        k_cache_flat = k_cache.squeeze(1)  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1)  # [num_pages, 8, 128]

        total_q, num_qo_heads, head_dim = q.shape
        num_pages, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128, "Shape assertions must hold."
        gqa_ratio = num_qo_heads // num_kv_heads  # 4
        len_indptr = qo_indptr.shape[0]
        B = len_indptr - 1  # number of batches

        # Allocate output and lse buffers (fp32 for computation)
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=q.device)

        # Launch one Triton program per batch
        grid = (B,)
        _batch_compute_kernel[grid](
            q, k_cache_flat, v_cache_flat,
            qo_indptr, kv_indptr, kv_indices,
            output, lse,
            sm_scale,
            total_q, num_qo_heads, head_dim,
            gqa_ratio,
            B, len_indptr,
            num_warps=4, num_stages=2
        )

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
