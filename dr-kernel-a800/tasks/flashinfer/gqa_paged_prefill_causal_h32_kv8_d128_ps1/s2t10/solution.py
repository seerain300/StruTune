import torch
import math
import triton
import triton.language as tl


@triton.jit
def attention_single_q_idx_h_kernel(
    q_ptr,          # *fp32, [T, H, D], flattened
    qo_indptr_ptr,  # *int32, [len_indptr]
    kv_indptr_ptr,  # *int32, [len_indptr]
    kv_indices_ptr, # *int32, [num_kv_indices]
    output_ptr,     # *bf16, [T, H, D], flattened
    lse_ptr,        # *fp32, [T, H]
    sm_scale,       # fp32
    T,              # int32, total_q
    H,              # int32, num_qo_heads
    D,              # int32, head_dim
    k_ptr,          # *fp32, [BLOCK_K, D] contiguous (padded rows)
    v_ptr,          # *fp32, [BLOCK_K, D] contiguous (padded rows)
    BLOCK_K: tl.constexpr,
):
    # Grid: (b in 0..len_indptr-2, q_idx in 0..num_q_tokens-1, h in 0..H-1)
    b = tl.program_id(0)
    q_idx = tl.program_id(1)
    h = tl.program_id(2)

    # Compute segment starts/ends
    qo_start = tl.load(qo_indptr_ptr + b)
    qo_end = tl.load(qo_indptr_ptr + b + 1)
    kv_start = tl.load(kv_indptr_ptr + b)
    kv_end = tl.load(kv_indptr_ptr + b + 1)

    num_q_tokens = qo_end - qo_start
    num_kv_tokens = kv_end - kv_start
    if (num_q_tokens <= 0) or (num_kv_tokens <= 0):
        return

    global_q_idx = qo_start + q_idx
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
        k_row = tl.load(k_ptr + k * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32
        prod = q_vec * k_row
        logits_scaled[k] = tl.sum(prod, axis=0)

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

    # Compute softmax of logits_scaled (masked k >= max_kv_idx contribute 0)
    for i in range(BLOCK_K):
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
        q_f32 = q.to(torch.float32).contiguous()
        # k_cache and v_cache are [N,1,8,128]; squeeze dim=1 => [N,8,128]
        k_cache_f32 = k_cache.to(torch.float32).contiguous()
        v_cache_f32 = v_cache.to(torch.float32).contiguous()
        qo_indptr = qo_indptr.to(torch.int32).contiguous()
        kv_indptr = kv_indptr.to(torch.int32).contiguous()
        kv_indices = kv_indices.to(torch.int32).contiguous()

        total_q, num_qo_heads, head_dim = q_f32.shape
        len_indptr = qo_indptr.shape[0]

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.zeros((total_q, num_qo_heads), dtype=torch.float32, device=device)

        num_segments = len_indptr - 1

        # GQA mapping: 8 KV heads shared across 32 Q heads
        gqa_ratio = num_qo_heads // 8

        # Launch Triton kernel per (b, q_idx, h): compute k_ptr and v_ptr on host, then call kernel
        for b in range(num_segments):
            qo_start = int(qo_indptr[b].item())
            qo_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            num_q_tokens = qo_end - qo_start
            num_kv_tokens = kv_end - kv_start
            if num_q_tokens <= 0 or num_kv_tokens <= 0:
                continue

            for q_idx in range(qo_start, qo_end):
                for h in range(num_qo_heads):
                    delta = num_kv_tokens - num_q_tokens
                    max_kv_idx = min(q_idx + 1 + delta, num_kv_tokens)

                    # Prepare k_ptr and v_ptr: shape [BLOCK_K, D] with first max_kv_idx rows loaded, rest zeros.
                    BLOCK_K = 128  # match head_dim=128; mask handles smaller K
                    k_ptr = torch.zeros((BLOCK_K, head_dim), dtype=torch.float32, device=device)
                    v_ptr = torch.zeros((BLOCK_K, head_dim), dtype=torch.float32, device=device)

                    if max_kv_idx > 0:
                        selected_indices = kv_indices[kv_start:kv_start + max_kv_idx]  # [K]
                        kv_head = h // gqa_ratio  # GQA mapping
                        # Gather rows from k_cache_flat [N, 8, D] by computing row offsets:
                        # row_offset = index * (8 * D) + kv_head * D
                        row_offsets = selected_indices * (8 * head_dim) + kv_head * head_dim  # [K]
                        # k_rows = k_cache_f32.squeeze(1).reshape(-1, head_dim)[row_offsets]  # [K, D]
                        # But torch tensor lookup by tensor indices requires advanced indexing. We instead
                        # gather via torch's indexing by constructing a list of rows. Triton requires static
                        # tensors; we gather using torch ops on host and copy into k_ptr/v_ptr.
                        # For clarity: we directly gather using squeeze and indices by building rows.
                        # We can do this via:
                        k_rows = k_cache_f32.squeeze(1)[:, :, :]  # [N, 8, D] -> we need rows for selected indices
                        # We need rows: k_rows[selected_indices, kv_head, :] -> shape [K, D]
                        # Torch allows advanced indexing: k_cache_f32.squeeze(1)[selected_indices[:, None], kv_head, :]
                        # To pass rows, we first extract:
                        # k_rows_list = [k_cache_f32.squeeze(1)[selected_indices[i], kv_head, :] for i in range(K)]
                        # But building a list inside forward is not ideal. Instead, we perform torch.index_select:
                        # PyTorch allows index_select along dim=0 for [N,8,D] if we create a view; however, direct
                        # index_select on squeezed [N,8,D] is awkward. So we gather using torch.gather along dim=0
                        # by constructing expanded indices. To keep it simple and correct, we do per-row indexing
                        # via a small loop. Given K is small in typical workloads, this is fine.
                        for i in range(max_kv_idx):
                            idx = selected_indices[i].item()  # Python int
                            k_row_i = k_cache_f32.squeeze(1)[idx, kv_head, :]  # [D]
                            v_row_i = v_cache_f32.squeeze(1)[idx, kv_head, :]  # [D]
                            k_ptr[i, :] = k_row_i
                            v_ptr[i, :] = v_row_i

                    # Call Triton kernel for this (b, q_idx, h)
                    attention_single_q_idx_h_kernel[(1,)](
                        q_f32, qo_indptr, kv_indptr, kv_indices, output, lse, sm_scale,
                        total_q, num_qo_heads, head_dim,
                        k_ptr, v_ptr, BLOCK_K=BLOCK_K
                    )

        return output, lse


def run(*args):
    return ModelNew()(*args)
