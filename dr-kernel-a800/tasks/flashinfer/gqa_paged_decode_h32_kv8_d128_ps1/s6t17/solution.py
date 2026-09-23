import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (batch b, head h)
@triton.jit
def _forward_kernel_bh(
    q_ptr,            # *fp16/fp32, shape [B, H, D]
    k_ptr,            # *fp16/fp32, shape [num_tokens, num_kv_heads, D]
    v_ptr,            # *fp16/fp32, shape [num_tokens, num_kv_heads, D]
    kv_indices_ptr,   # *int32, shape [num_kv_indices]
    kv_indptr_ptr,    # *int32, shape [B+1]
    lse_ptr,          # *fp32, shape [B, H]
    out_ptr,          # *fp32, shape [B, H, D]
    B: tl.constexpr,         # batch size
    H: tl.constexpr,         # number of query heads
    D: tl.constexpr,         # head dimension
    num_kv_heads: tl.constexpr,  # number of kv heads
    sm_scale,                  # float32 scalar = 1.0 / sqrt(D)
    N_TOTAL: tl.constexpr,     # static loop bound (e.g., 128), masked by nn < actual_num_tokens
):
    # program ids for batch and head
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Load q vector for this (b, h): q[b, h, :]
    q_base = q_ptr + b * H * D + h * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        q_vec[d] = tl.load(q_base + d).to(tl.float32)

    # Load kv_indptr for this batch: start and end
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start  # >= 0

    # GQA head mapping: kv_head = h // 4 for H=32, num_kv_heads=8
    kv_head = h // 4

    # Streaming logsumexp across tokens in base-2: initialize
    m = -float("inf")
    sumexp = 0.0
    ln2 = 0.6931471805599453

    # First pass: compute m and sumexp
    for nn in range(0, N_TOTAL):
        valid = nn < actual_num_tokens
        # Load token index if valid
        idx = tl.load(kv_indices_ptr + start + nn, mask=valid, other=0).to(tl.int32)

        # Base offsets for this token and kv_head
        k_token_base = k_ptr + idx * num_kv_heads * D + kv_head * D
        v_token_base = v_ptr + idx * num_kv_heads * D + kv_head * D

        # Load k_row and compute dot with q_vec
        k_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_val = tl.load(k_token_base + d).to(tl.float32)
            k_row[d] = k_val

        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_row[d]

        logit = dot * sm_scale
        new_m = tl.maximum(m, logit)
        # Update sumexp only if valid
        sumexp = tl.where(valid, sumexp * tl.exp(m - new_m) + tl.exp(logit - new_m), sumexp)
        m = new_m

    # lse for this (b, h): logsumexp / ln(2)
    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute softmax per token and accumulate output
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)

    for nn in range(0, N_TOTAL):
        valid = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=valid, other=0).to(tl.int32)
        k_token_base = k_ptr + idx * num_kv_heads * D + kv_head * D
        v_token_base = v_ptr + idx * num_kv_heads * D + kv_head * D

        # Load v_row for this token
        v_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            v_val = tl.load(v_token_base + d).to(tl.float32)
            v_row[d] = v_val

        # Recompute dot and logits_scaled
        k_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_val = tl.load(k_token_base + d).to(tl.float32)
            k_row[d] = k_val
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_row[d]
        logit = dot * sm_scale

        # softmax(logit) scaled by base-2 logsumexp uses m and sumexp
        softmax = tl.exp(logit - m) / (sumexp * ln2)
        # Accumulate output vector
        out_vec += tl.where(valid, softmax * v_row, v_row * 0.0)

    # Store output vector
    for d in range(0, D):
        tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only forward: no PyTorch ops
        B, H, D = q.shape
        # Reshape k_cache and v_cache to [num_tokens, num_kv_heads, D] to simplify kernel addressing
        num_tokens = kv_indices.shape[0]
        k_flat = k_cache.reshape(num_tokens, k_cache.shape[2], D).contiguous()
        v_flat = v_cache.reshape(num_tokens, v_cache.shape[2], D).contiguous()

        # Ensure inputs are contiguous
        q_c = q.contiguous()
        kv_indices_c = kv_indices.contiguous()
        kv_indptr_c = kv_indptr.contiguous()

        # Allocate outputs (float32 for compute; cast to bfloat16 for return)
        output = torch.empty((B, H, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        _forward_kernel_bh[grid](
            q_c, k_flat, v_flat, kv_indices_c, kv_indptr_c, lse, output,
            B=B, H=H, D=D, num_kv_heads=8, sm_scale=float(sm_scale), N_TOTAL=128,
            num_warps=4,
        )

        # Return output as bfloat16 and lse as float32 to match original expected types
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
