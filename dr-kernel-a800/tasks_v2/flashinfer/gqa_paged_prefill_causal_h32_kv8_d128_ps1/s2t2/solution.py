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
    k_ptr,          # *fp32, [K, D], contiguous
    v_ptr,          # *fp32, [K, D], contiguous
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
    logits = tl.zeros((BLOCK_K,), dtype=tl.float32)
    # We have k_ptr (shape [K, D]) and v_ptr (shape [K, D]) passed from host.
    # Note: Triton doesn't allow dynamic pointer arithmetic; passing tensors is acceptable for computation.
    # Iterate k in masked fashion:
    # But Triton requires static loops; we can set K at host and pass BLOCK_K >= K (we choose BLOCK_K=D).
    # For simplicity and given D=128, we set BLOCK_K=128 and mask max_kv_idx within D.

    # Compute logits: for k in 0..BLOCK_K-1, if k < K, load k_row = k_ptr[k], else ignore
    # We'll implement a masked loop using python-like for in Triton via range:
    # Triton supports range loops up to a constexpr bound. We mask using k < max_kv_idx.
    for k in range(BLOCK_K):
        valid = k < max_kv_idx
        k_row = tl.load(k_ptr + k * D + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0)  # fp32
        # Dot product q_vec @ k_row
        # Triton doesn't have a dedicated dot; we can compute sum(q_vec * k_row) elementwise.
        # Compute elementwise product and sum over D.
        prod = q_vec * k_row
        logits[k] = tl.sum(prod, axis=0)

    # Scale logits
    logits_scaled = logits * sm_scale

    # Compute logsumexp in base-2
    m = logits_scaled[0]
    for i in range(1, BLOCK_K):
        m = tl.maximum(m, logits_scaled[i])
    sum_exp = 0.0
    for i in range(BLOCK_K):
        sum_exp += tl.exp(logits_scaled[i] - m)
    lse_val = m + tl.log(sum_exp)  # in natural log
    lse_base2 = lse_val / tl.log(2.0)
    # Atomic add lse contribution for this (global_q_idx, h)
    tl.atomic_add(lse_ptr + global_q_idx * H + h, lse_base2)

    # Compute softmax of logits_scaled (masked k >= max_kv_idx contribute 0)
    # Note: we need to zero out invalid logits before softmax
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
        # Ensure on CUDA and contiguous
        device = q.device
        q_f32 = q.to(torch.float32).contiguous()
        k_cache_f32 = k_cache.to(torch.float32).contiguous()
        v_cache_f32 = v_cache.to(torch.float32).contiguous()
        qo_indptr = qo_indptr.to(torch.int32).contiguous()
        kv_indptr = kv_indptr.to(torch.int32).contiguous()
        kv_indices = kv_indices.to(torch.int32).contiguous()

        total_q, num_qo_heads, head_dim = q_f32.shape
        num_segments = qo_indptr.shape[0] - 1

        # Prepare flattened caches (squeeze the 1 dimension)
        k_cache_flat = k_cache_f32.squeeze(1)  # [N, 8, D]
        v_cache_flat = v_cache_f32.squeeze(1)  # [N, 8, D]

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.zeros((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel over 3D grid: (num_segments, num_q_tokens, num_qo_heads)
        grid = (num_segments, total_q, num_qo_heads)

        # We need to provide k_ptr and v_ptr per program instance. Triton kernels cannot select
        # dynamic pointers;


def run(*args):
    return ModelNew()(*args)
