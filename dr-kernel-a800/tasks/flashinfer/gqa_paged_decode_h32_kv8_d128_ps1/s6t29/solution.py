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
    q_ptr,          # *fp16, shape [B, H, D]
    k_ptr,          # *fp16, shape [N_total, 1, num_kv_heads, D] but gathered per token index
    v_ptr,          # *fp16, shape [N_total, 1, num_kv_heads, D], similarly gathered
    kv_indices_ptr, # *int32, shape [num_kv_indices]
    kv_indptr_ptr,  # *int32, shape [B+1]
    lse_ptr,        # *float32, shape [B, H]
    out_ptr,        # *float32, shape [B, H, D] (we'll return bfloat16 from forward)
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    num_kv_heads: tl.constexpr,
    sm_scale,       # scalar float32
    N_TOTAL: tl.constexpr,  # compile-time loop bound, e.g., 128 (>= D)
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

    # GQA mapping: kv_head = h // (H // num_kv_heads) = h // 4
    kv_head = h // 4

    # First pass: streaming logsumexp in base-2 across tokens
    m = -float("inf")
    sumexp = 0.0
    ln2 = 0.6931471805599453
    for nn in range(0, N_TOTAL):
        idx = start + nn  # scalar int32
        # Mask: only process if idx < end
        if idx >= end:
            continue

        # k_row = k_cache[idx, kv_head, :]
        k_row_base = k_ptr + idx * (num_kv_heads * D) + kv_head * D
        k_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_row[d] = tl.load(k_row_base + d).to(tl.float32)

        # dot = q_vec · k_row
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_row[d]

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
        idx = start + nn
        if idx >= end:
            continue

        # k_row and dot again (recompute is fine for short D)
        k_row_base = k_ptr + idx * (num_kv_heads * D) + kv_head * D
        k_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_row[d] = tl.load(k_row_base + d).to(tl.float32)

        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_row[d]
        logit = dot * sm_scale

        softmax = tl.exp(logit - m) / (sumexp * ln2)
        # v_row = v_cache[idx, kv_head, :]
        v_row_base = v_ptr + idx * (num_kv_heads * D) + kv_head * D
        v_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            v_row[d] = tl.load(v_row_base + d).to(tl.float32)

        out_vec += softmax * v_row

    tl.store(out_base, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # We assume inputs are on CUDA and contiguous. If not, make them so.
        device = q.device
        B = q.shape[0]
        H = q.shape[1]
        D = q.shape[2]

        # Convert to float16/bfloat16 as in original, but kernel loads as fp16 and converts to fp32
        # We'll pass original tensors directly; Triton will convert via .to(tl.float32) in kernel.

        # Outputs: output as float32 in kernel, will cast to bfloat16 at return; lse as float32
        output = torch.empty((B, H, D), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        _forward_kernel_bh[grid](
            q, k_cache, v_cache, kv_indices, kv_indptr,
            lse, output,
            B=B, H=H, D=D, num_kv_heads=8,
            sm_scale=float(sm_scale),
            N_TOTAL=128,
            num_warps=4, num_stages=2
        )

        # Return output as bfloat16 and lse as float32, matching original behavior
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
