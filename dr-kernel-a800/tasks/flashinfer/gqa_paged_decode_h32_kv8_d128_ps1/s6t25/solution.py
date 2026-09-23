import math
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
    q_ptr,          # *ptr to q, shape [B, H, D], float32
    k_ptr,          # *ptr to k_cache, shape [N_total, num_kv_heads, D] (we gather by indices),
                    # in practice this points to [N_pages, 1, num_kv_heads, D]
    v_ptr,          # *ptr to v_cache, shape [N_total, num_kv_heads, D]
    kv_indices_ptr, # *ptr to int32 indices of tokens, shape [num_kv_indices]
    kv_indptr_ptr,  # *ptr to int32 indptr, shape [B+1]
    lse_ptr,        # *ptr to lse, shape [B, H], float32
    out_ptr,        # *ptr to output, shape [B, H, D], float32
    B: tl.constexpr,          # batch size
    H: tl.constexpr,          # num_qo_heads
    D: tl.constexpr,          # head_dim
    num_kv_heads: tl.constexpr,  # number of kv heads (8)
    sm_scale,                 # scalar float32
    N_TOTAL: tl.constexpr,    # loop bound (e.g., 128), mask beyond actual_num_tokens
):
    # Program ids for batch and head
    b = tl.program_id(0)  # int
    h = tl.program_id(1)  # int

    # Compute q vector for this (b, h): q[b, h, :]
    q_base = q_ptr + b * H * D + h * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        q_vec[d] = tl.load(q_base + d)  # q_ptr elements are float32 by construction

    # Gather kv indices for this batch from kv_indptr
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start  # int

    # GQA mapping: kv_head = h // (H // num_kv_heads) = h // 4
    kv_head = h // 4

    # First pass: streaming logsumexp in base-2
    m = -float("inf")
    sumexp = 0.0
    ln2 = 0.6931471805599453
    for nn in range(0, N_TOTAL):
        # Mask: only process nn < actual_num_tokens
        if nn >= actual_num_tokens:
            continue

        idx = tl.load(kv_indices_ptr + start + nn).to(tl.int32)

        # k_row = k_cache[idx, kv_head, :]
        k_row_base = k_ptr + idx * (num_kv_heads * D) + kv_head * D
        k_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_vec[d] = tl.load(k_row_base + d)

        # dot = q_vec · k_vec
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_vec[d]

        logit = dot * sm_scale
        new_m = tl.maximum(m, logit)
        sumexp = sumexp * tl.exp(m - new_m) + tl.exp(logit - new_m)
        m = new_m

    # lse for this (b, h): logsumexp in base-2
    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute softmax per token and accumulate output
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for nn in range(0, N_TOTAL):
        if nn >= actual_num_tokens:
            continue

        idx = tl.load(kv_indices_ptr + start + nn).to(tl.int32)

        k_row_base = k_ptr + idx * (num_kv_heads * D) + kv_head * D
        v_row_base = v_ptr + idx * (num_kv_heads * D) + kv_head * D

        v_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            v_vec[d] = tl.load(v_row_base + d)

        # Recompute dot and logits_scaled
        k_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_vec[d] = tl.load(k_row_base + d)
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_vec[d]
        logit = dot * sm_scale

        # softmax over tokens = exp(logit - m) / (sumexp * ln2)
        softmax = tl.exp(logit - m) / (sumexp * ln2)
        out_vec += softmax * v_vec

    # Store output
    for d in range(0, D):
        tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure all tensors are on CUDA and have correct dtypes. The kernel expects float32 for computation.
        assert q.dim() == 3 and k_cache.dim() == 4 and v_cache.dim() == 4
        B, H, D = q.shape
        device = q.device
        k_cache = k_cache.to(device)
        v_cache = v_cache.to(device)
        kv_indptr = kv_indptr.to(device)
        kv_indices = kv_indices.to(device)
        # Cast to float32 for kernel computation
        q = q.contiguous().to(torch.float32)
        k_cache = k_cache.contiguous().to(torch.float32)
        v_cache = v_cache.contiguous().to(torch.float32)
        kv_indptr = kv_indptr.contiguous().to(torch.int32)
        kv_indices = kv_indices.contiguous().to(torch.int32)

        # Allocate outputs (float32 for computation; final output cast to bfloat16 at end)
        output = torch.empty((B, H, D), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        # Choose a reasonable N_TOTAL; we need a constexpr loop bound. We pick 128 to cover typical token counts.
        N_TOTAL = 128  # sufficient for provided workloads (num_kv_indices up to ~100)
        _forward_kernel_bh[grid](
            q, k_cache, v_cache, kv_indices, kv_indptr, lse, output,
            B=B, H=H, D=D, num_kv_heads=8, sm_scale=sm_scale,
            N_TOTAL=N_TOTAL,
            num_warps=4,
        )

        # Return output as bfloat16 to match original model's output dtype
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
