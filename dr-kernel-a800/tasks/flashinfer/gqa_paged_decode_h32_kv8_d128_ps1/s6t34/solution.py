import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _forward_bh_kernel(
    q_ptr,               # *float32, [B, H, D]
    k_ptr,               # *float32, [N_total, num_kv_heads, D]
    v_ptr,               # *float32, [N_total, num_kv_heads, D]
    kv_indices_ptr,      # *int32, [num_tokens]
    kv_indptr_ptr,       # *int32, [B+1]
    lse_ptr,             # *float32, [B, H]
    out_ptr,             # *float32, [B, H, D]
    B: tl.constexpr,     # batch size (meta for indexing)
    H: tl.constexpr,     # num query heads (meta for indexing)
    D: tl.constexpr,     # head dimension (128)
    NUM_KV_HEADS: tl.constexpr,  # num_kv_heads (8)
    SM_SCALE: tl.constexpr,      # scalar sm_scale (float32)
    N_TOTAL: tl.constexpr,       # max tokens per batch element (compile-time bound, e.g., 128)
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

    # GQA mapping
    kv_head = h // 4  # since H=32 and num_kv_heads=8

    # First pass: compute streaming logsumexp in base-2 across tokens
    m = -float("inf")
    sumexp = 0.0
    ln2 = 0.6931471805599453
    for nn in range(0, N_TOTAL):
        mask = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=mask, other=0).to(tl.int32)

        # k_row = k[idx, kv_head, :]
        k_row_ptr = k_ptr + idx * NUM_KV_HEADS * D + kv_head * D
        k_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_row[d] = tl.load(k_row_ptr + d).to(tl.float32)

        # dot = q_vec · k_row
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_row[d]

        logit = dot * SM_SCALE
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

        # v_row = v[idx, kv_head, :]
        v_row_ptr = v_ptr + idx * NUM_KV_HEADS * D + kv_head * D
        v_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            v_row[d] = tl.load(v_row_ptr + d).to(tl.float32)

        # Recompute dot and logits_scaled
        k_row_ptr = k_ptr + idx * NUM_KV_HEADS * D + kv_head * D
        k_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_row[d] = tl.load(k_row_ptr + d).to(tl.float32)
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_row[d]
        logit = dot * SM_SCALE

        # softmax in base-2
        softmax = tl.exp(logit - m) / (sumexp * ln2)
        for d in range(0, D):
            out_vec[d] += softmax * v_row[d]

    # Store output vector
    for d in range(0, D):
        tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only implementation: forward must not use any PyTorch ops.
        assert TRITON_AVAILABLE, "Triton is not available."
        assert q.dim() == 3 and k_cache.dim() == 4 and v_cache.dim() == 4
        B, H, D = q.shape
        assert H == 32, "num_qo_heads must be 32"
        assert D == 128, "head_dim must be 128"

        # Ensure tensors are on the same device and contiguous, cast to float32 for compute
        device = q.device
        q_f32 = q.contiguous().to(torch.float32)                        # [B, H, D]
        k_cache_f32 = k_cache.contiguous().to(torch.float32)           # [N_pages, 1, num_kv_heads, D]
        v_cache_f32 = v_cache.contiguous().to(torch.float32)           # [N_pages, 1, num_kv_heads, D]

        # Flatten to [N_total, num_kv_heads, D]
        N_total = k_cache_f32.shape[0]
        k_flat = k_cache_f32.view(N_total, 8, D).contiguous()          # [N_total, 8, D]
        v_flat = v_cache_f32.view(N_total, 8, D).contiguous()          # [N_total, 8, D]

        # kv_indptr and kv_indices must be int32 on device
        kv_indptr_i32 = kv_indptr.to(torch.int32).contiguous()         # [B+1]
        kv_indices_i32 = kv_indices.to(torch.int32).contiguous()       # [num_tokens]

        # Allocate outputs (float32 for kernel)
        out_f32 = torch.empty((B, H, D), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        _forward_bh_kernel[grid](
            q_f32, k_flat, v_flat, kv_indices_i32, kv_indptr_i32, lse, out_f32,
            B, H, D, 8, float(sm_scale), 128,  # N_TOTAL=128 for safety
        )

        # Return output in bfloat16 as in original, and lse in float32
        output = out_f32.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
