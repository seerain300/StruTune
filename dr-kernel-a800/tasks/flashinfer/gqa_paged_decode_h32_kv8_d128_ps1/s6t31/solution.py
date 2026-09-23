import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (b, h). Computes output vector and lse for that pair.
@triton.jit
def _forward_kernel_bh(
    q_ptr,               # *ptr to q, shape [B, H, D], float32
    k_ptr,               # *ptr to k_cache_flat, shape [N_total, num_kv_heads, D], float32
    v_ptr,               # *ptr to v_cache_flat, shape [N_total, num_kv_heads, D], float32
    kv_indices_ptr,      # *ptr to int32 indices of tokens, shape [num_kv_indices]
    kv_indptr_ptr,       # *ptr to int32 indptr, shape [B+1]
    lse_ptr,             # *ptr to lse, shape [B, H], float32
    out_ptr,             # *ptr to output, shape [B, H, D], float32
    B,                   # int (runtime)
    H,                   # int (runtime)
    D: tl.constexpr,     # head dimension, compile-time constant (128)
    num_kv_heads: tl.constexpr,  # 8
    sm_scale,            # scalar float32
    N_TOTAL: tl.constexpr,        # loop bound (e.g., 128), mask beyond actual_num_tokens
):
    # Program ids for batch and head
    b = tl.program_id(0)  # int32
    h = tl.program_id(1)  # int32

    # Compute q vector for this (b, h): q[b, h, :]
    q_base = q_ptr + b * H * D + h * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        q_vec[d] = tl.load(q_base + d).to(tl.float32)

    # Gather kv indices for this batch from kv_indptr
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start  # int32

    # GQA mapping: kv_head = h // (H // num_kv_heads) = h // 4 when H=32, num_kv_heads=8
    kv_head = h // 4

    # First pass: compute logsumexp in base-2 across tokens
    m = -float('inf')
    sumexp = 0.0
    ln2 = 0.6931471805599453

    for nn in range(0, N_TOTAL):
        mask = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=mask, other=0).to(tl.int32)

        # k_row = k_cache[idx, kv_head, :]
        k_row_ptr = k_ptr + idx * num_kv_heads * D + kv_head * D
        k_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_val = tl.load(k_row_ptr + d).to(tl.float32)
            k_row[d] = k_val

        # dot = q_vec · k_row
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_row[d]

        logit = dot * sm_scale
        new_m = tl.maximum(m, logit)
        sumexp = sumexp * tl.exp(m - new_m) + tl.exp(logit - new_m)
        m = new_m

    # lse for this (b, h) in base-2
    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute softmax per token and accumulate output
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for nn in range(0, N_TOTAL):
        mask = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=mask, other=0).to(tl.int32)

        # k_row and v_row for this token
        k_row_ptr = k_ptr + idx * num_kv_heads * D + kv_head * D
        v_row_ptr = v_ptr + idx * num_kv_heads * D + kv_head * D

        k_row = tl.zeros((D,), dtype=tl.float32)
        v_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_val = tl.load(k_row_ptr + d).to(tl.float32)
            v_val = tl.load(v_row_ptr + d).to(tl.float32)
            k_row[d] = k_val
            v_row[d] = v_val

        # Recompute dot and logits_scaled
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_row[d]
        logit = dot * sm_scale

        # softmax per token in base-2: exp(logit - m) / (sumexp * ln2)
        softmax = tl.exp(logit - m) / (sumexp * ln2)
        for d in range(0, D):
            out_vec[d] += softmax * v_row[d]

    # Store output vector
    for d in range(0, D):
        tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only implementation: forward must not use any PyTorch ops.
        assert TRITON_AVAILABLE, "Triton is not available."
        assert q.dim() == 3 and k_cache.dim() == 4 and v_cache.dim() == 4
        B, H, D = q.shape
        # Enforce original constraints
        assert H == 32, "num_qo_heads must be 32"
        assert D == 128, "head_dim must be 128"

        # Ensure tensors are on the same device and contiguous
        device = q.device
        q_f32 = q.contiguous().to(torch.float32)                        # [B, H, D]
        k_cache_f32 = k_cache.contiguous().to(torch.float32)           # [N_pages, 1, num_kv_heads, D]
        v_cache_f32 = v_cache.contiguous().to(torch.float32)           # [N_pages, 1, num_kv_heads, D]
        N_total = k_cache_f32.shape[0]
        k_flat = k_cache_f32.view(N_total, -1, D).contiguous()         # [N_total, num_kv_heads, D]
        v_flat = v_cache_f32.view(N_total, -1, D).contiguous()         # [N_total, num_kv_heads, D]

        # kv_indptr and kv_indices must be int32 on device
        kv_indptr_i32 = kv_indptr.to(torch.int32).contiguous()         # [B+1]
        kv_indices_i32 = kv_indices.to(torch.int32).contiguous()       # [num_kv_indices]

        # Allocate outputs
        out = torch.empty((B, H, D), dtype=torch.float32, device=device)  # final output in


def run(*args):
    return ModelNew()(*args)
