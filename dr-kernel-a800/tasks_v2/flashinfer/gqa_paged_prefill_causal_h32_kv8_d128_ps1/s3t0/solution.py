import math
import torch
import triton
import triton.language as tl


# Kernel: compute logits_scaled = q @ k.T for a single head
# q: [head_dim], k: [num_kv, head_dim], out: [num_kv] (scaled)
@triton.jit
def _compute_logits_scaled_kernel(
    q_ptr, k_ptr, out_ptr,
    head_dim: tl.constexpr,
    num_kv: tl.constexpr,
    sm_scale: tl.float32,
):
    # This kernel assumes num_kv <= head_dim; we handle num_kv via masking
    # We compute out[i] = sum_j q[j] * k[i, j] * sm_scale
    # For simplicity, we do a direct vectorized multiply and sum for i in a small range.
    # Triton supports tl.dot for 1D vectors; here we implement a loop across head_dim.
    for i in range(0, num_kv):
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(0, head_dim):
            qj = tl.load(q_ptr + j)
            ki_j = tl.load(k_ptr + i * head_dim + j)
            acc += qj * ki_j
        tl.store(out_ptr + i, acc * sm_scale)


# Kernel: compute logsumexp of a vector, store lse / log(2.0) to out_ptr
# inp: [H], out: [H] float32
@triton.jit
def _logsumexp_kernel(inp_ptr, out_ptr, n_elements: tl.constexpr):
    # For each element h in [0, n_elements):
    # m = max(inp); sum_exp = sum(exp(inp - m)); lse = log(sum_exp) + m; out[h] = lse / log2
    log2 = 0.6931471805599453
    # We will write out one element at a time; Triton kernel does elementwise computation.
    # Note: Triton requires a compile-time loop; we use for i in range(n_elements).
    for i in range(0, n_elements):
        val = tl.load(inp_ptr + i)
        # Compute m
        m = val
        for j in range(0, n_elements):
            vj = tl.load(inp_ptr + j)
            m = tl.maximum(m, vj)
        # Compute sum_exp
        sum_exp = tl.zeros((), dtype=tl.float32)
        for j in range(0, n_elements):
            vj = tl.load(inp_ptr + j)
            sum_exp += tl.exp(vj - m)
        lse = tl.log(sum_exp) + m
        # Store lse / log2
        tl.store(out_ptr + i, lse / log2)


# Kernel: compute attention for a single head given logits_scaled and produce out
# inp: logits_scaled [num_kv], k: [num_kv, head_dim], v: [head_dim, num_kv], out: [head_dim]
@triton.jit
def _softmax_matmul_kernel(
    inp_ptr, k_ptr, v_ptr, out_ptr,
    head_dim: tl.constexpr, num_kv: tl.constexpr, sm_scale: tl.float32
):
    # Compute softmax of inp_ptr -> attn
    log2 = 0.6931471805599453
    m = tl.load(inp_ptr + 0)  # initialize m with first element
    for i in range(0, num_kv):
        val = tl.load(inp_ptr + i)
        m = tl.maximum(m, val)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for i in range(0, num_kv):
        val = tl.load(inp_ptr + i)
        sum_exp += tl.exp(val - m)
    for i in range(0, num_kv):
        val = tl.load(inp_ptr + i)
        attn_i = tl.exp(val - m) / sum_exp
        # Compute out[h] += attn_i * v[h, i]
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(0, head_dim):
            v_j_i = tl.load(v_ptr + j * num_kv + i)
            acc += attn_i * v_j_i
        tl.store(out_ptr + j, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure on CUDA and contiguous
        device = q.device
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be on CUDA for Triton."

        # Flatten k_cache and v_cache by squeezing the singleton "page_size" dim (always 1)
        k_cache_flat = k_cache.squeeze(1).contiguous()  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1).contiguous()  # [num_pages, 8, 128]

        total_q, num_qo_heads, head_dim = q.shape
        assert num_qo_heads == 32, "num_qo_heads must be 32"
        assert head_dim == 128, "head_dim must be 128"

        num_pages, _, num_kv_heads, _ = k_cache.shape
        assert num_kv_heads == 8, "num_kv_heads must be 8"

        len_indptr = qo_indptr.shape[0]
        num_kv_indices = kv_indices.shape[0]

        # Output tensors
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # q in float32 for compute
        q_f32 = q.to(torch.float32).contiguous()  # [total_q, 32, 128]
        # For each batch b (0..len_indptr-2)
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())

            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                # No queries or KV for this batch element
                continue

            # Gather cached K/V groups for this batch
            page_ids = kv_indices[kv_start:kv_end].to(torch.long).contiguous()  # [num_kv_tokens]
            num_kv_tokens = page_ids.shape[0]

            # Fetch K/V for these groups
            k_batch = k_cache_flat[page_ids]  # [num_kv_tokens, 8, 128]
            v_batch = v_cache_flat[page_ids]  # [num_kv_tokens, 8, 128]

            # Queries for this batch
            q_batch = q_f32[q_start:q_end]  # [num_q_tokens, 32, 128]
            num_q_tokens = q_batch.shape[0]

            # Delta for causal-like masking
            # Original code logic: q_idx + 1 + (num_kv_tokens - num_q_tokens), clamp to num_kv_tokens
            # We'll compute max_kv_idx per token q_idx
            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx

                # Apply causal-like limit
                max_kv_idx = min(q_idx + 1 + (num_kv_tokens - num_q_tokens), num_kv_tokens)
                if max_kv_idx <= 0:
                    continue

                q_pos = q_batch[q_idx]  # [32, 128]
                gqa_ratio = num_qo_heads // num_kv_heads  # = 4

                # Loop over heads, compute per-head attention
                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio  # 0..3

                    # Select vectors for this head
                    q_head = q_pos[h]  # [128]
                    k_head = k_batch[:max_kv_idx, kv_head]  # [max_kv_idx, 128]
                    v_head = v_batch[:max_kv_idx, kv_head]  # [max_kv_idx, 128]

                    # Compute logits_scaled = q @ k.T
                    num_kv = max_kv_idx
                    logits = torch.empty(num_kv, dtype=torch.float32, device=device)
                    # Launch Triton kernel to compute logits
                    _compute_logits_scaled_kernel[(1,)](
                        q_head, k_head, logits,
                        head_dim=128,
                        num_kv=num_kv,
                        sm_scale=sm_scale
                    )
                    # Compute lse = logsumexp(logits_scaled) / log(2)
                    lse_elem = torch.empty(1, dtype=torch.float32, device=device)
                    _logsumexp_kernel[(1,)](
                        logits, lse_elem, n_elements=num_kv
                    )
                    lse[global_q_idx, h] = lse_elem[0]

                    # Compute attention and output
                    out_head = torch.empty(128, dtype=torch.float32, device=device)
                    _softmax_matmul_kernel[(1,)](
                        logits, k_head, v_head, out_head,
                        head_dim=128,
                        num_kv=num_kv,
                        sm_scale=sm_scale
                    )
                    output[global_q_idx, h] = out_head.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
