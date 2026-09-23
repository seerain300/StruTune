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
    k_ptr,          # *ptr to k_cache, shape [N_total, 1, num_kv_heads, D], but we only use indices
    v_ptr,          # *ptr to v_cache, shape [N_total, 1, num_kv_heads, D], similarly
    kv_indices_ptr, # *ptr to int32 indices, shape [num_kv_indices]
    kv_indptr_ptr,  # *ptr to int32 indptr, shape [B+1]
    lse_ptr,        # *ptr to lse, shape [B, H], float32
    out_ptr,        # *ptr to output, shape [B, H, D], bfloat16
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    sm_scale,       # scalar float32
    N_TOTAL: tl.constexpr,   # loop bound (e.g., 128), mask beyond actual_num_tokens
    num_kv_heads: tl.constexpr,  # 8
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
    actual_num_tokens = end - start  # int32 scalar

    # First pass: streaming logsumexp to compute m and sumexp across tokens
    m = -float("inf")
    sumexp = 0.0  # scalar float32
    ln2 = 0.6931471805599453  # log(2)
    for nn in range(0, N_TOTAL):
        if nn < actual_num_tokens:
            idx = tl.load(kv_indices_ptr + start + nn).to(tl.int32)

            # k_cache layout: [N_total, 1, num_kv_heads, D]
            # stride along N_total = 1 * num_kv_heads * D = num_kv_heads * D
            k_row_base = k_ptr + idx * (num_kv_heads * D) + h // (H // num_kv_heads) * D
            # Load k_row vector and compute dot with q_vec
            k_row = tl.zeros((D,), dtype=tl.float32)
            for d in range(0, D):
                k_row[d] = tl.load(k_row_base + d).to(tl.float32)
            dot = 0.0
            for d in range(0, D):
                dot += q_vec[d] * k_row[d]
            logit = dot * sm_scale

            new_m = tl.maximum(m, logit)
            sumexp = sumexp * tl.exp(m - new_m) + tl.exp(logit - new_m)
            m = new_m

    # lse for this (b, h)
    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute softmax per token and accumulate output
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)  # compute in float32, cast later
    for nn in range(0, N_TOTAL):
        if nn < actual_num_tokens:
            idx = tl.load(kv_indices_ptr + start + nn).to(tl.int32)

            # v_cache layout: [N_total, 1, num_kv_heads, D]
            v_row_base = v_ptr + idx * (num_kv_heads * D) + h // (H // num_kv_heads) * D
            v_row = tl.zeros((D,), dtype=tl.float32)
            for d in range(0, D):
                v_row[d] = tl.load(v_row_base + d).to(tl.float32)

            # Recompute dot and logits_scaled
            k_row_base = k_ptr + idx * (num_kv_heads * D) + h // (H // num_kv_heads) * D
            k_row = tl.zeros((D,), dtype=tl.float32)
            for d in range(0, D):
                k_row[d] = tl.load(k_row_base + d).to(tl.float32)
            dot = 0.0
            for d in range(0, D):
                dot += q_vec[d] * k_row[d]
            logit = dot * sm_scale

            softmax = tl.exp(logit - m) / (sumexp * ln2)  # base-2 softmax
            out_vec += softmax * v_row

    # Store output as bfloat16
    out_bf = out_vec.to(tl.bfloat16)
    for d in range(0, D):
        tl.store(out_base + d, out_bf[d])


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Only allocate outputs and launch Triton kernel; no PyTorch ops in forward.
        B, H, D = q.shape
        assert H == 32 and D == 128, "This Triton kernel expects H=32, D=128"
        num_kv_heads = 8
        # Ensure device and dtype compatibility
        # q must be float32 for the kernel
        if q.dtype != torch.float32:
            q = q.to(torch.float32)
        # k_cache and v_cache will be read as float32 in the kernel
        # Output: [B, H, D] bfloat16
        out = torch.empty((B, H, D), dtype=torch.bfloat16, device=q.device)
        # lse: [B, H] float32
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device)

        # Prepare pointers
        q_ptr = q
        k_ptr = k_cache
        v_ptr = v_cache
        kv_indices_ptr = kv_indices
        kv_indptr_ptr = kv_indptr

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        _forward_kernel_bh[grid](
            q_ptr, k_ptr, v_ptr, kv_indices_ptr, kv_indptr_ptr, lse, out,
            B=B, H=H, D=D, sm_scale=sm_scale, N_TOTAL=128, num_kv_heads=num_kv_heads,
            num_warps=4,  # reasonable default; can be tuned
        )

        return out, lse


def run(*args):
    return ModelNew()(*args)
