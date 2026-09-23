import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (b, h). Performs full computation in Triton.
@triton.jit
def _forward_kernel_bh(
    q_ptr,                  # *float32, [B, H, D]
    k_ptr,                  # *float32, [N_total, num_kv_heads, D]
    v_ptr,                  # *float32, [N_total, num_kv_heads, D]
    kv_indices_ptr,         # *int32, [num_tokens]
    kv_indptr_ptr,          # *int32, [B+1]
    lse_ptr,                # *float32, [B, H]
    out_ptr,                # *float32, [B, H, D]
    B: tl.int32,            # batch size (runtime)
    H: tl.int32,            # num query heads (runtime)
    D: tl.constexpr,        # head dimension, e.g., 128
    num_kv_heads: tl.int32, # e.g., 8
    sm_scale: tl.float32,   # scaling factor (e.g., 1/sqrt(D))
    N_TOTAL: tl.constexpr,  # loop bound (e.g., 128)
):
    b = tl.program_id(0)  # 0..B-1
    h = tl.program_id(1)  # 0..H-1

    # Load q[b, h, :] as float32
    q_base = q_ptr + b * H * D + h * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        q_vec[d] = tl.load(q_base + d)

    # Token range for this batch
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start

    # GQA mapping: kv_head = h // (H // num_kv_heads) = h // 4
    kv_head = h // (H // num_kv_heads)

    # First pass: streaming logsumexp over tokens
    m = -float("inf")
    sumexp = 0.0
    ln2 = 0.6931471805599453  # log(2)
    for nn in range(0, N_TOTAL):
        valid = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=valid, other=0).to(tl.int32)

        # k_ptr and v_ptr are [N_total, num_kv_heads, D]; each head has contiguous D
        k_base = k_ptr + idx * (num_kv_heads * D) + kv_head * D
        v_base = v_ptr + idx * (num_kv_heads * D) + kv_head * D

        # Load k_row and compute dot with q_vec
        k_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_row[d] = tl.load(k_base + d)

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

    # Second pass: compute softmax and accumulate output
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for nn in range(0, N_TOTAL):
        valid = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=valid, other=0).to(tl.int32)

        k_base = k_ptr + idx * (num_kv_heads * D) + kv_head * D
        v_base = v_ptr + idx * (num_kv_heads * D) + kv_head * D

        v_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            v_row[d] = tl.load(v_base + d)

        # Recompute dot and logits_scaled
        k_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_row[d] = tl.load(k_base + d)
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_row[d]
        logit = dot * sm_scale

        softmax = tl.exp(logit - m) / (sumexp * ln2)
        out_vec += v_row * softmax

    tl.store(out_base, out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only forward: no PyTorch ops here
        assert TRITON_AVAILABLE, "Triton is not available."
        B, H, D = q.shape
        # Cast to float32 for kernel math
        q = q.to(torch.float32)
        k_cache = k_cache.to(torch.float32)
        v_cache = v_cache.to(torch.float32)

        # Output buffers
        output = torch.empty((B, H, D), dtype=torch.float32)
        lse = torch.empty((B, H), dtype=torch.float32)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        _forward_kernel_bh[grid](
            q, k_cache, v_cache, kv_indices, kv_indptr,
            lse, output,
            B, H, D, 8, float(sm_scale), 128
        )
        # Return bfloat16 output to match original signature
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
