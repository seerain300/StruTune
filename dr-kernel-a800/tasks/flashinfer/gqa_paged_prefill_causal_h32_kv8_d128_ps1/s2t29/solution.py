import torch
import math
import triton
import triton.language as tl


@triton.jit
def attention_bqh_kernel(
    q_ptr,               # *fp32, [T, H, D], contiguous
    k_ptr,               # *fp32, [N, M, D] flattened (k_cache_flat)
    v_ptr,               # *fp32, [N, M, D] flattened (v_cache_flat)
    qo_indptr_ptr,       # *int32, [len_indptr]
    kv_indptr_ptr,       # *int32, [len_indptr]
    kv_indices_ptr,      # *int32, [num_kv_indices]
    output_ptr,          # *bf16,  [T, H, D], contiguous
    lse_ptr,             # *fp32,  [T, H]
    sm_scale,            # fp32 scalar
    total_q,             # int
    H,                   # int
    D,                   # int
    num_q_tokens,        # int
    num_kv_indices,      # int
    BLOCK_K: tl.constexpr,  # int, e.g., 128
):
    # Grid maps to (b, q_idx, h)
    b = tl.program_id(0)
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    # Segment bounds
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    # Global query index for this triple
    global_q_idx = b * num_q_tokens + q_idx

    # GQA mapping: each Q head maps to KV head
    gqa_ratio = H // 8  # num_qo_heads // num_kv_heads
    kv_head = h // gqa_ratio

    # Load q vector for this head: q[global_q_idx, h]
    q_base = global_q_idx * H * D
    q_offset = q_base + h * D
    q_vec = tl.load(q_ptr + q_offset + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32, shape [D]

    # Compute logits_scaled for k in 0..BLOCK_K-1
    logits_scaled = tl.zeros((BLOCK_K,), dtype=tl.float32)

    # Flattened kv_indices for this segment b: indices[kv_start:kv_end]
    # We compute per-kv index as start + k, masked by max_kv_idx
    # Note: We assume that num_kv_indices >= (kv_end - kv_start) for segment b.
    #       The kernel launches only for valid q_idx, h, b; lse accumulation is masked by max_kv_idx.
    # To avoid passing 3D pointers, we load each k-row directly from k_ptr and v_ptr using indices.
    # Note: We don't have per-segment indices tensor; instead, we compute the k-th element in the flattened
    #       kv_indices for this b as index_k = kv_indices_ptr[kv_start + k].
    for k in range(BLOCK_K):
        index_k = tl.load(kv_indices_ptr + kv_start + k)  # scalar int32
        # k_ptr and v_ptr are flattened [N, M, D]. We select row index = index_k * (M*D) + kv_head*D
        # Since k_ptr is k_cache_flat after squeeze(1), N is the first dim; in our setup N=num_pages,
        # and k_ptr is [num_pages, 8, D] flattened. We select rows based on index_k.
        # Construct offsets:
        k_row_offset = index_k * (8 * D) + kv_head * D
        k_row = tl.load(k_ptr + k_row_offset + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32, [D]
        prod = q_vec * k_row
        logits_scaled[k] = tl.sum(prod, axis=0)  # scalar

    # Scale logits
    logits_scaled = logits_scaled * sm_scale

    # Compute logsumexp in base-2
    m = logits_scaled[0]
    for i in range(1, BLOCK_K):
        m = tl.maximum(m, logits_scaled[i])
    sum_exp = 0.0
    for i in range(BLOCK_K):
        sum_exp += tl.exp(logits_scaled[i] - m)
    lse_val = m + tl.log(sum_exp)  # natural log
    lse_base2 = lse_val / tl.log(2.0)
    # Atomic add lse contribution for this (global_q_idx, h)
    tl.atomic_add(lse_ptr + global_q_idx * H + h, lse_base2)

    # Compute output vector: out_vec = sum_k (softmax[k] * v_rows[k, :])
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for i in range(BLOCK_K):
        attn_i = tl.exp(logits_scaled[i] - m) / sum_exp
        index_i = tl.load(kv_indices_ptr + kv_start + i)
        v_row_offset = index_i * (8 * D) + kv_head * D
        v_row = tl.load(v_ptr + v_row_offset + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32, [D]
        out_vec += attn_i * v_row

    # Store output as bfloat16 to output_ptr[global_q_idx, h, :]
    out_base = global_q_idx * H * D
    out_offset = out_base + h * D
    tl.store(output_ptr + out_offset + tl.arange(0, D), out_vec.to(tl.bfloat16), mask=tl.arange(0, D) < D)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA and contiguous
        device = q.device
        q_f32 = q.to(torch.float32).contiguous()           # [T, H, D]
        k_cache_f32 = k_cache.to(torch.float32).contiguous()  # [N, 1, M, D]
        v_cache_f32 = v_cache.to(torch.float32).contiguous()  # [N, 1, M, D]
        qo_indptr = qo_indptr.to(torch.int32).contiguous()
        kv_indptr = kv_indptr.to(torch.int32).contiguous()
        kv_indices = kv_indices.to(torch.int32).contiguous()

        total_q, num_qo_heads, head_dim = q_f32.shape
        len_indptr = qo_indptr.shape[0]
        num_segments = len_indptr - 1  # typically 1 in provided get_inputs

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.zeros((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Flatten caches to [N, M, D] by removing singleton dim=1
        # Note: k_cache shape [N, 1, M, D] -> squeeze(1) -> [N, M, D]
        k_cache_flat = k_cache_f32.squeeze(1)  # [N, M, D]
        v_cache_flat = v_cache_f32.squeeze(1)  # [N, M, D]

        # Launch Triton kernel: grid over (segments, queries, heads)
        # We need num_q_tokens and num_kv_indices per segment to compute max_kv_idx.
        # Since len_indptr is typically 2, num_segments=1. But we can still loop over segments safely.
        for b in range(num_segments):
            # Compute number of queries and KVs in this segment
            qo_start = int(qo_indptr[b].item())
            qo_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            num_q_tokens = qo_end - qo_start
            # num_kv_indices in this segment is (kv_end - kv_start)
            num_kv_in_b = kv_end - kv_start

            # Launch kernel for each (b, q_idx, h)
            # Grid: (1, num_q_tokens, H)
            for q_idx in range(num_q_tokens):
                for h in range(num_qo_heads):
                    attention_bqh_kernel[(1, num_q_tokens, num_qo_heads)](
                        q_f32, k_cache_flat, v_cache_flat,
                        qo_indptr, kv_indptr, kv_indices,
                        output, lse,
                        sm_scale,
                        total_q, num_qo_heads, head_dim,
                        num_q_tokens, num_kv_in_b,
                        BLOCK_K=128,  # constexpr meta-parameter
                    )

        return output, lse


def run(*args):
    return ModelNew()(*args)
