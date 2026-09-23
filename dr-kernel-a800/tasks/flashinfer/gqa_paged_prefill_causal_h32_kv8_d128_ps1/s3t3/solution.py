import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute logits = q @ k.T
# q_vec: [HEAD_DIM] float32
# k_mat: [NUM_KV, HEAD_DIM] float32, row-major
# logits_out: [NUM_KV] float32
@triton.jit
def _dot_logits_kernel(q_vec_ptr, k_mat_ptr, logits_out_ptr,
                        HEAD_DIM: tl.constexpr, NUM_KV: tl.constexpr):
    # For each i in [0, NUM_KV), compute logits[i] = sum_j q[j] * k[i, j]
    for i in range(NUM_KV):
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(HEAD_DIM):
            qj = tl.load(q_vec_ptr + j)
            ki_j = tl.load(k_mat_ptr + i * HEAD_DIM + j)
            acc += qj * ki_j
        tl.store(logits_out_ptr + i, acc)


# Triton kernel: compute scalar logsumexp of inp vector of length VEC_SIZE and write to out_ptr[0]
# inp_ptr: [VEC_SIZE] float32
# out_ptr: [1] float32
# VEC_SIZE: tl.constexpr
@triton.jit
def _logsumexp_scalar_kernel(inp_ptr, out_ptr, VEC_SIZE: tl.constexpr):
    m = tl.load(inp_ptr + 0)
    for j in range(1, VEC_SIZE):
        vj = tl.load(inp_ptr + j)
        m = tl.maximum(m, vj)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for j in range(0, VEC_SIZE):
        vj = tl.load(inp_ptr + j)
        sum_exp += tl.exp(vj - m)
    lse = tl.log(sum_exp) + m
    # divide by ln(2)
    tl.store(out_ptr, lse / 0.6931471805599651)


# Triton kernel: compute softmax over inp (length NUM_KV) and write to attn_out_ptr[0..NUM_KV-1]
@triton.jit
def _softmax_vector_kernel(inp_ptr, attn_out_ptr, NUM_KV: tl.constexpr):
    m = tl.load(inp_ptr + 0)
    for j in range(1, NUM_KV):
        vj = tl.load(inp_ptr + j)
        m = tl.maximum(m, vj)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for j in range(0, NUM_KV):
        vj = tl.load(inp_ptr + j)
        sum_exp += tl.exp(vj - m)
    for j in range(0, NUM_KV):
        vj = tl.load(inp_ptr + j)
        attn = tl.exp(vj - m) / sum_exp
        tl.store(attn_out_ptr + j, attn)


# Triton kernel: compute out = v @ attn, where
# attn_ptr: [NUM_KV] float32
# v_mat_ptr: [HEAD_DIM, NUM_KV] float32, row-major
# out_vec_ptr: [HEAD_DIM] float32
@triton.jit
def _matmul_vec_vec_kernel(attn_ptr, v_mat_ptr, out_vec_ptr,
                           HEAD_DIM: tl.constexpr, NUM_KV: tl.constexpr):
    out = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for j in range(NUM_KV):
        aj = tl.load(attn_ptr + j)
        # v_mat layout: [HEAD_DIM, NUM_KV], row-major
        # element (r, j) = v_mat_ptr + r * NUM_KV + j
        for r in range(HEAD_DIM):
            vj_r = tl.load(v_mat_ptr + r * NUM_KV + j)
            out[r] += aj * vj_r
    tl.store(out_vec_ptr + tl.arange(0, HEAD_DIM), out)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors and contiguity
        device = q.device
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be on CUDA for Triton."
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        qo_indptr = qo_indptr.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Flatten k_cache/v_cache by squeezing the singleton "page_size" dimension
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

        # Convert inputs to float32 for compute
        q_f32 = q.to(torch.float32).contiguous()
        k_cache_f32 = k_cache_flat.to(torch.float32).contiguous()
        v_cache_f32 = v_cache_flat.to(torch.float32).contiguous()

        # Loop over batch segments
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())

            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Cached K/V groups for this batch
            page_ids = kv_indices[kv_start:kv_end].to(torch.long).contiguous()  # [num_kv_tokens]
            num_kv_tokens = page_ids.shape[0]

            # Fetch K/V as float32
            k_batch = k_cache_f32[page_ids]  # [num_kv_tokens, 8, 128], float32
            v_batch = v_cache_f32[page_ids]  # [num_kv_tokens, 8, 128], float32

            # Queries for this segment (float32)
            q_batch = q_f32[q_start:q_end]  # [num_q_tokens, 32, 128], float32
            num_q_tokens = q_batch.shape[0]

            # Causal-like max_kv_idx per query position
            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx

                max_kv_idx = min(q_idx + 1 + (num_kv_tokens - num_q_tokens), num_kv_tokens)
                if max_kv_idx <= 0:
                    continue

                # Query head vector
                q_pos = q_batch[q_idx]  # [32, 128], float32
                gqa_ratio = num_qo_heads // num_kv_heads  # = 4
                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio  # 0..3

                    # Select vectors for this head
                    q_head = q_pos[h].contiguous()     # [128], float32
                    k_head = k_batch[:max_kv_idx, kv_head].contiguous()  # [max_kv_idx, 128], float32
                    v_head = v_batch[:max_kv_idx, kv_head].contiguous()  # [max_kv_idx, 128], float32

                    # 1) Compute logits = q @ k.T (float32), scaled by sm_scale
                    logits = torch.empty(max_kv_idx, dtype=torch.float32, device=device)
                    _dot_logits_kernel[(1,)](
                        q_head, k_head, logits,
                        HEAD_DIM=128,
                        NUM_KV=max_kv_idx,
                    )
                    logits_scaled = logits * sm_scale

                    # 2) Compute scalar LSE = logsumexp(logits_scaled) / ln(2)
                    lse_elem = torch.empty(1, dtype=torch.float32, device=device)
                    _logsumexp_scalar_kernel[(1,)](
                        logits_scaled, lse_elem,
                        VEC_SIZE=max_kv_idx,
                    )
                    lse[global_q_idx, h] = lse_elem[0]

                    # 3) Compute attention weights (softmax over logits_scaled)
                    attn = torch.empty(max_kv_idx, dtype=torch.float32, device=device)
                    _softmax_vector_kernel[(1,)](
                        logits_scaled, attn,
                        NUM_KV=max_kv_idx,
                    )

                    # 4) Compute output vector: out = v @ attn (float32)
                    out_head_f32 = torch.empty(128, dtype=torch.float32, device=device)
                    _matmul_vec_vec_kernel[(1,)](
                        attn, v_head, out_head_f32,
                        HEAD_DIM=128,
                        NUM_KV=max_kv_idx,
                    )

                    # Store output (cast to bfloat16 to match original)
                    output[global_q_idx, h] = out_head_f32.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
