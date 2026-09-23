import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute logits = q_vec @ k_rows.T
# q_vec_ptr: [HEAD_DIM] float32
# k_rows_ptr: [L, HEAD_DIM] float32, row-major (L is num_kv)
# logits_out_ptr: [L] float32
@triton.jit
def _dot_logits_kernel(q_vec_ptr, k_rows_ptr, logits_out_ptr,
                        HEAD_DIM: tl.constexpr, L: tl.constexpr):
    for i in range(L):
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(HEAD_DIM):
            qj = tl.load(q_vec_ptr + j)
            ki_j = tl.load(k_rows_ptr + i * HEAD_DIM + j)
            acc += qj * ki_j
        tl.store(logits_out_ptr + i, acc)


# Triton kernel: softmax over a 1D vector of length VEC_SIZE (we use VEC_SIZE=L)
# inp_ptr: [L] float32
# out_ptr: [L] float32
@triton.jit
def _softmax_1d_kernel(inp_ptr, out_ptr, L: tl.constexpr):
    # compute max
    m = tl.load(inp_ptr + 0)
    for j in range(1, L):
        vj = tl.load(inp_ptr + j)
        m = tl.maximum(m, vj)
    # compute sum_exp
    sum_exp = tl.zeros((), dtype=tl.float32)
    for j in range(L):
        sum_exp += tl.exp(tl.load(inp_ptr + j) - m)
    # write normalized
    for j in range(L):
        vj = tl.load(inp_ptr + j)
        tl.store(out_ptr + j, tl.exp(vj - m) / sum_exp)


# Triton kernel: logsumexp over a 1D vector of length VEC_SIZE (we use VEC_SIZE=L)
# inp_ptr: [L] float32
# out_ptr: [1] float32, stores log(sum(exp(inp))) = log(sum_exp) + m
@triton.jit
def _logsumexp_1d_kernel(inp_ptr, out_ptr, L: tl.constexpr):
    # compute max
    m = tl.load(inp_ptr + 0)
    for j in range(1, L):
        vj = tl.load(inp_ptr + j)
        m = tl.maximum(m, vj)
    # compute sum_exp
    sum_exp = tl.zeros((), dtype=tl.float32)
    for j in range(L):
        sum_exp += tl.exp(tl.load(inp_ptr + j) - m)
    lse = tl.log(sum_exp) + m
    tl.store(out_ptr, lse)


# Triton kernel: matvec: out_vec = sum_k v_rows[k, :] * attn[k]
# v_rows_ptr: [HEAD_DIM, L] float32, row-major
# attn_ptr: [L] float32
# out_vec_ptr: [HEAD_DIM] float32
@triton.jit
def _matvec_kernel(v_rows_ptr, attn_ptr, out_vec_ptr,
                   HEAD_DIM: tl.constexpr, L: tl.constexpr):
    for j in range(HEAD_DIM):
        acc = tl.zeros((), dtype=tl.float32)
        for k in range(L):
            vk = tl.load(v_rows_ptr + k * HEAD_DIM + j)
            a = tl.load(attn_ptr + k)
            acc += vk * a
        tl.store(out_vec_ptr + j, acc)


# Triton kernel: cast a 1D float32 vector to bfloat16 (writes to out_bf16_ptr)
# inp_ptr: [N] float32
# out_bf16_ptr: [N] bfloat16
@triton.jit
def _cast_bf16_1d_kernel(inp_ptr, out_bf16_ptr, N: tl.constexpr):
    for i in range(N):
        v = tl.load(inp_ptr + i)
        tl.store(out_bf16_ptr + i, v.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # We assume q is [total_q, 32, 128], k_cache/v_cache are [num_pages, 8, 128]
        total_q, num_qo_heads, head_dim = q.shape
        num_pages, num_kv_groups, kv_dim, _ = k_cache.shape
        assert num_qo_heads == 32 and head_dim == 128 and num_kv_groups == 8 and kv_dim == 128
        device = q.device

        # Flatten k/v caches over num_pages and num_kv_groups
        k_cache_flat = k_cache.reshape(num_pages * num_kv_groups, 128).to(torch.float32)
        v_cache_flat = v_cache.reshape(num_pages * num_kv_groups, 128).to(torch.float32)

        # Prepare output and lse
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        gqa_ratio = num_qo_heads // 8  # 4

        # Convert indices to int32 contiguous for kernel
        qo_indptr = qo_indptr.to(torch.int32)
        kv_indptr = kv_indptr.to(torch.int32)
        kv_indices = kv_indices.to(torch.int32)

        # Move q to float32 for compute
        q_f32 = q.to(torch.float32)

        # Iterate segments
        len_indptr = qo_indptr.shape[0]
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            num_q_tokens = q_end - q_start

            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            num_segments_b = kv_end - kv_start

            if num_q_tokens <= 0 or num_segments_b <= 0:
                continue

            # Get kv_ids for this segment
            kv_ids = kv_indices[kv_start:kv_end].contiguous()  # [num_segments_b]

            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx

                # Causal-like bound (similar to original)
                delta = num_segments_b - num_q_tokens
                max_kv_idx = min(q_idx + 1 + delta, num_segments_b)
                if max_kv_idx <= 0:
                    continue

                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio  # map to 8 KV heads

                    # Load q_vec [128] as float32
                    q_vec = q_f32[global_q_idx, h]  # [128]

                    # Load K/V rows [max_kv_idx, 128]
                    # Map kv_ids to flattened indices: idx = kv_ids[i] * num_kv_groups + kv_head
                    k_rows = k_cache_flat[kv_ids[:max_kv_idx] * num_kv_groups + kv_head]  # [L, 128]
                    v_rows = v_cache_flat[kv_ids[:max_kv_idx] * num_kv_groups + kv_head]  # [L, 128]

                    # Compute logits = q_vec @ k_rows.T using Triton
                    logits = torch.empty((max_kv_idx,), dtype=torch.float32, device=device)
                    _dot_logits_kernel[(1,)](q_vec, k_rows, logits, HEAD_DIM=128, L=max_kv_idx)

                    # Scale logits
                    logits_scaled = logits * sm_scale

                    # Triton logsumexp over L elements
                    lse_buf = torch.empty((1,), dtype=torch.float32, device=device)
                    _logsumexp_1d_kernel[(1,)](logits_scaled, lse_buf, L=max_kv_idx)
                    lse_val = lse_buf[0] / math.log(2.0)  # divide by ln(2)
                    lse[global_q_idx, h] = lse_val

                    # Triton softmax over L elements
                    attn = torch.empty((max_kv_idx,), dtype=torch.float32, device=device)
                    _softmax_1d_kernel[(1,)](logits_scaled, attn, L=max_kv_idx)

                    # Compute output = attn @ v_rows using Triton (matvec)
                    out_vec = torch.empty((head_dim,), dtype=torch.float32, device=device)
                    _matvec_kernel[(1,)](v_rows, attn, out_vec, HEAD_DIM=128, L=max_kv_idx)

                    # Cast output to bfloat16 using Triton kernel
                    out_bf = torch.empty((head_dim,), dtype=torch.bfloat16, device=device)
                    _cast_bf16_1d_kernel[(head_dim,)](out_vec, out_bf, N=head_dim)
                    # Store into output tensor at [global_q_idx, h, :]
                    output[global_q_idx, h] = out_bf

        return output, lse


def run(*args):
    return ModelNew()(*args)
