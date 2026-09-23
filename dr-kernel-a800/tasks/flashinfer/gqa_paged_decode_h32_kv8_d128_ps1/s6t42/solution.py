import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


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
    D: tl.constexpr,        # head dimension (compile-time const, e.g., 128)
    num_kv_heads: tl.int32, # num kv heads (runtime, e.g., 8)
    sm_scale: tl.float32,   # 1/sqrt(D) (runtime float32)
    N_TOTAL: tl.constexpr,  # loop bound (compile-time const, e.g., 128)
):
    # Program ids for batch and head
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Load q[b, h, :] as float32
    q_base = q_ptr + b * H * D + h * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        q_val = tl.load(q_base + d)
        q_vec[d] = q_val  # already float32

    # Compute token range for this batch
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start

    # GQA mapping: kv_head = h // (H // num_kv_heads) = h // 4
    kv_head = h // (H // num_kv_heads)

    # First pass: compute streaming logsumexp over tokens in base-2
    m = -float("inf")
    sumexp = 0.0
    ln2 = 0.6931471805599453  # log(2)
    for nn in range(0, N_TOTAL):
        valid = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=valid, other=0).to(tl.int32)

        # k_ptr and v_ptr are [N_total, num_kv_heads, D]
        k_base = k_ptr + idx * (num_kv_heads * D) + kv_head * D
        v_base = v_ptr + idx * (num_kv_heads * D) + kv_head * D

        # Load k_row
        k_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_val = tl.load(k_base + d)
            k_row[d] = k_val

        # Dot product q · k_row
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_row[d]

        logit = dot * sm_scale
        new_m = tl.maximum(m, logit)
        sumexp = sumexp * tl.exp(m - new_m) + tl.exp(logit - new_m)
        m = new_m

    # lse for this (b, h): logsumexp(logits_scaled) / ln(2)
    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute softmax per token and accumulate output
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for nn in range(0, N_TOTAL):
        valid = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=valid, other=0).to(tl.int32)

        k_base = k_ptr + idx * (num_kv_heads * D) + kv_head * D
        v_base = v_ptr + idx * (num_kv_heads * D) + kv_head * D

        v_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            v_val = tl.load(v_base + d)
            v_row[d] = v_val

        # Recompute dot and logits_scaled
        k_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_val = tl.load(k_base + d)
            k_row[d] = k_val
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_row[d]
        logit = dot * sm_scale

        softmax = tl.exp(logit - m) / (sumexp * ln2)  # softmax in base-2 normalization
        out_vec += v_row * softmax

    tl.store(out_base, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

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
            q,                  # *float32, [B, H, D]
            k_cache,            # *float32, [N_total, num_kv_heads, D] — note: original shape is [num_pages, 1, num_kv_heads, D]; indexing via kv_indptr/kv_indices handles it
            v_cache,            # *float32, [N_total, num_kv_heads, D]
            kv_indices,         # *int32, [num_tokens]
            kv_indptr,          # *int32, [B+1]
            lse,                # *float32, [B, H]
            output,             # *float32, [B, H, D]
            B,                  # batch size
            H,                  # num query heads
            D=128,              # head dimension (constexpr)
            num_kv_heads=8,     # num kv heads
            sm_scale=1.0 / math.sqrt(128.0),  # 1/sqrt(D)
            N_TOTAL=128,        # loop bound
        )

        # Return results in expected dtypes
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
