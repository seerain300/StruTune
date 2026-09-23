import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _forward_kernel_bh(
    q_ptr,             # *ptr to q, shape [B, H, D], float32
    k_ptr,             # *ptr to k_cache, shape [N_pages, num_kv_heads, D], float32
    v_ptr,             # *ptr to v_cache, shape [N_pages, num_kv_heads, D], float32
    kv_indices_ptr,    # *ptr to int32 indices, shape [num_kv_indices]
    kv_indptr_ptr,     # *ptr to int32 indptr, shape [B+1]
    lse_ptr,           # *ptr to lse, shape [B, H], float32
    out_ptr,           # *ptr to output, shape [B, H, D], float32
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    num_kv_heads: tl.constexpr,
    sm_scale,          # scalar float32
    N_TOTAL: tl.constexpr,  # loop bound (e.g., 128), mask beyond actual_num_tokens
):
    # Program ids for batch and head
    b = tl.program_id(0)  # int
    h = tl.program_id(1)  # int

    # Load q[b, h, :] as a vector of length D (float32)
    q_base = q_ptr + b * H * D + h * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        q_vec[d] = tl.load(q_base + d)

    # Compute token range for this batch from kv_indptr
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start  # int

    # GQA mapping: kv_head = h // (H // num_kv_heads) = h // 4
    kv_head = h // 4

    # First pass: compute logsumexp over tokens in base-2
    m = -float("inf")
    sumexp = 0.0
    ln2 = 0.6931471805599453
    for nn in range(0, N_TOTAL):
        if nn >= actual_num_tokens:
            continue
        idx = tl.load(kv_indices_ptr + start + nn).to(tl.int32)

        # Load k_row for this idx and kv_head: k[idx, kv_head, :]
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

    # lse for this (b, h) in base-2
    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute softmax per token and accumulate output
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for nn in range(0, N_TOTAL):
        if nn >= actual_num_tokens:
            continue
        idx = tl.load(kv_indices_ptr + start + nn).to(tl.int32)

        # Load v_row for this idx and kv_head: v[idx, kv_head, :]
        v_row_base = v_ptr + idx * (num_kv_heads * D) + kv_head * D
        v_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            v_row[d] = tl.load(v_row_base + d)

        # Recompute dot and logit
        k_row_base = k_ptr + idx * (num_kv_heads * D) + kv_head * D
        k_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_vec[d] = tl.load(k_row_base + d)
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_vec[d]
        logit = dot * sm_scale

        # softmax over base-2 logsumexp
        softmax = tl.exp(logit - m) / (sumexp * ln2)
        out_vec += v_row * softmax

    # Store output vector for this (b, h)
    for d in range(0, D):
        tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only forward: no PyTorch ops allowed here
        assert TRITON_AVAILABLE, "Triton is not available"
        B, H, D = q.shape
        num_kv_heads = k_cache.shape[2]  # expected 8
        # We'll enforce the fixed dims expected by the original logic
        assert H == 32 and D == 128 and num_kv_heads == 8, "Fixed dims must match"

        device = q.device
        # Ensure inputs are float32 for kernel math and on CUDA
        q_f32 = q if q.dtype == torch.float32 else q.to(torch.float32)
        k_f32 = k_cache if k_cache.dtype == torch.float32 else k_cache.to(torch.float32)
        v_f32 = v_cache if v_cache.dtype == torch.float32 else v_cache.to(torch.float32)
        kv_indptr_i32 = kv_indptr if kv_indptr.dtype == torch.int32 else kv_indptr.to(torch.int32)
        kv_indices_i32 = kv_indices if kv_indices.dtype == torch.int32 else kv_indices.to(torch.int32)

        # Allocate outputs (float32 for kernel; will convert to bfloat16)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)
        output = torch.empty((B, H, D), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        _forward_kernel_bh[grid](
            q_f32, k_f32, v_f32, kv_indices_i32, kv_indptr_i32,
            lse, output,
            B=B, H=H, D=D, num_kv_heads=num_kv_heads,
            sm_scale=float(sm_scale),  # scalar float32
            N_TOTAL=128,                # loop bound; mask out nn >= actual_num_tokens
            num_warps=4, num_stages=2
        )

        # Return results with expected dtypes: output bfloat16, lse float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
