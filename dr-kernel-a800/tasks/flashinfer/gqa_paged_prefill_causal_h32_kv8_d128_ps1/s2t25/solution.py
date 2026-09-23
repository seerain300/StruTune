import torch
import math
import triton
import triton.language as tl


@triton.jit
def attention_single_q_idx_h_kernel(
    q_ptr,           # *fp32, [T, H, D], flattened
    qo_indptr_ptr,   # *int32, [len_indptr]
    kv_indptr_ptr,   # *int32, [len_indptr]
    kv_indices_ptr,  # *int32, [num_kv_indices]
    output_ptr,      # *bf16,  [T, H, D], flattened
    lse_ptr,         # *fp32,  [T, H]
    sm_scale,        # fp32 scalar
    T,               # int32 total_q
    H,               # int32 num_qo_heads
    D,               # int32 head_dim
    k_ptr,           # *fp32,  [K, D] contiguous for this (b,h)
    v_ptr,           # *fp32,  [K, D] contiguous for this (b,h)
    BLOCK_K: tl.constexpr,  # maximum number of KV rows considered
):
    # Grid maps program_id(0)=b, program_id(1)=q_idx, program_id(2)=h
    b = tl.program_id(0)
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start
    if (num_q_tokens <= 0) or (num_kv_tokens <= 0):
        return

    global_q_idx = qo_start + q_idx

    # GQA: map query head to KV head
    gqa_ratio = H // 8  # 32 // 8 = 4
    kv_head = h // gqa_ratio

    # Causal-like masking: max_kv_idx = min(q_idx + 1 + (num_kv_tokens - num_q_tokens), num_kv_tokens)
    delta = num_kv_tokens - num_q_tokens
    max_kv_idx = tl.minimum(q_idx + 1 + delta, num_kv_tokens)

    # Load q vector for this head: q[global_q_idx, h]
    q_base = global_q_idx * H * D
    q_offset = q_base + h * D
    q_vec = tl.load(q_ptr + q_offset + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32

    # Compute logits for k in [0..BLOCK_K-1], masked by max_kv_idx
    logits_scaled = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k in range(BLOCK_K):
        valid = k < max_kv_idx
        # k_ptr is [K, D] contiguous. We load the k-th row for this (b,h)
        k_row = tl.load(k_ptr + k * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32
        prod = q_vec * k_row
        logits_scaled[k] = tl.sum(prod, axis=0)

    # Scale logits
    logits_scaled = logits_scaled * sm_scale

    # logsumexp in natural log, then convert to base-2
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

    # Compute softmax of logits_scaled (masked k >= max_kv_idx contribute 0)
    for i in range(BLOCK_K):
        # set invalid logits to -inf so exp goes to 0
        logits_scaled[i] = -float('inf') if (i >= max_kv_idx) else logits_scaled[i]
    denom = 0.0
    for i in range(BLOCK_K):
        denom += tl.exp(logits_scaled[i] - m)
    softmax_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for i in range(BLOCK_K):
        softmax_vals[i] = tl.exp(logits_scaled[i] - m) / denom

    # Compute output vector: out_vec += softmax[k] * v_rows[k, :]
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for k in range(BLOCK_K):
        valid = k < max_kv_idx
        v_row = tl.load(v_ptr + k * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32
        out_vec += softmax_vals[k] * v_row

    # Store output as bfloat16 to output_ptr[global_q_idx, h, :]
    out_base = global_q_idx * H * D
    out_offset = out_base + h * D
    tl.store(output_ptr + out_offset + tl.arange(0, D), out_vec.to(tl.bfloat16), mask=tl.arange(0, D) < D)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors and contiguity
        device = q.device
        q_f32 = q.to(torch.float32).contiguous()  # [T, H, D]
        # k_cache and v_cache are [N,1,8,128]; squeeze dim=1 => [N,8,128]
        k_cache_f32 = k_cache.to(torch.float32).contiguous()
        v_cache_f32 = v_cache.to(torch.float32).contiguous()
        qo_indptr = qo_indptr.to(torch.int32).contiguous()
        kv_indptr = kv_indptr.to(torch.int32).contiguous()
        kv_indices = kv_indices.to(torch.int32).contiguous()

        total_q, num_qo_heads, head_dim = q_f32.shape
        len_indptr = qo_indptr.shape[0]
        num_segments = len_indptr - 1

        # Flatten caches: [N, 8, D]
        k_cache_flat = k_cache_f32.squeeze(1)  # [N, 8, D]
        v_cache_flat = v_cache_f32.squeeze(1)  # [N, 8, D]

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.zeros((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Precompute k_segments[h] and v_segments[h] per segment b for all heads h
        k_segments_list = [[] for _ in range(num_qo_heads)]
        v_segments_list = [[] for _ in range(num_qo_heads)]

        # For each segment b
        for b in range(num_segments):
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            num_kv_tokens = kv_end - kv_start

            qo_start = int(qo_indptr[b].item())
            qo_end = int(qo_indptr[b + 1].item())

            # Build k_segments[h] and v_segments[h] for this segment
            for h in range(num_qo_heads):
                gqa_ratio = num_qo_heads // 8  # 4
                kv_head = h // gqa_ratio
                indices = kv_indices[kv_start:kv_end].to(torch.int32)  # length num_kv_tokens
                # k_cache_flat shape: [N, 8, D]
                N, M, D2 = k_cache_flat.shape
                assert D2 == head_dim, "head_dim mismatch"
                row_offsets = (indices * (8 * D2) + kv_head * D2).to(torch.int32)
                k_rows = k_cache_flat[row_offsets]  # [num_kv_tokens, D]
                v_rows = v_cache_flat[row_offsets]  # [num_kv_tokens, D]
                k_segments_list[h].append(k_rows)
                v_segments_list[h].append(v_rows)

        # Launch the Triton kernel for each (b, q_idx, h)
        grid = (num_segments, total_q, num_qo_heads)

        for b in range(num_segments):
            qo_start = int(qo_indptr[b].item())
            qo_end = int(qo_indptr[b + 1].item())
            num_q_tokens = qo_end - qo_start
            for q_idx in range(num_q_tokens):
                global_q_idx = qo_start + q_idx
                for h in range(num_qo_heads):
                    kv_start = int(kv_indptr[b].item())
                    kv_end = int(kv_indptr[b + 1].item())
                    num_kv_tokens = kv_end - kv_start
                    delta = num_kv_tokens - num_q_tokens
                    max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)

                    # Gather k_segments[h] and v_segments[h] for this (b)
                    k_list = k_segments_list[h]
                    v_list = v_segments_list[h]
                    # k_list[b] corresponds to this segment; build k_sub and v_sub up to max_kv_idx
                    k_seg = k_list[b].to(torch.float32).contiguous()  # [num_kv_tokens, D]
                    v_seg = v_list[b].to(torch.float32).contiguous()  # [num_kv_tokens, D]
                    k_sub = k_seg[:max_kv_idx].contiguous()          # [max_kv_idx, D]
                    v_sub = v_seg[:max_kv_idx].contiguous()          # [max_kv_idx, D]

                    attention_single_q_idx_h_kernel[grid](
                        q_f32, qo_indptr, kv_indptr, kv_indices,
                        output, lse, sm_scale,
                        total_q, num_qo_heads, head_dim,
                        k_sub, v_sub,
                        BLOCK_K=max_kv_idx,  # pass as constexpr
                    )

        return output, lse


def run(*args):
    return ModelNew()(*args)
