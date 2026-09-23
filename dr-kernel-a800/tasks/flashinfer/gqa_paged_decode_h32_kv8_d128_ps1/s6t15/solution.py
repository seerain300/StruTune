import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _forward_kernel_bh(
    q_ptr,          # *ptr, float32, [B, H, D]
    k_ptr,          # *ptr, float32, [num_kv_indices, num_kv_heads, D] (gathered by idx)
    v_ptr,          # *ptr, float32, [num_kv_indices, num_kv_heads, D] (gathered by idx)
    kv_indices_ptr, # *ptr, int32, [num_kv_indices]
    kv_indptr_ptr,  # *ptr, int32, [B+1]
    lse_ptr,        # *ptr, float32, [B, H]
    out_ptr,        # *ptr, float32, [B, H, D]
    B: tl.constexpr,             # batch size
    H: tl.constexpr,             # num query heads (32)
    D: tl.constexpr,             # head dim (128)
    num_kv_heads: tl.constexpr,  # 8
    sm_scale,                    # float32 scalar = 1.0 / sqrt(D)
    N_TOTAL: tl.constexpr,       # loop bound (e.g., 128), mask nn < actual_num_tokens
):
    # program ids
    b = tl.program_id(0)  # batch index
    h = tl.program_id(1)  # head index

    # Load q vector for (b, h)
    q_base = q_ptr + b * H * D + h * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        q_vec[d] = tl.load(q_base + d)

    # Compute start/end for this batch in kv_indptr
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start  # runtime int

    # GQA mapping: kv_head = h // 4
    kv_head = h // 4

    # Streaming logsumexp in base-2 across tokens for this (b, h)
    m = -float("inf")
    sumexp = 0.0
    ln2 = 0.6931471805599453  # log(2)

    for nn in range(0, N_TOTAL):
        # mask beyond actual_num_tokens
        in_range = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=in_range, other=0).to(tl.int32)

        # Load k_vec for this token and kv_head
        k_base = k_ptr + idx * num_kv_heads * D + kv_head * D
        k_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_vec[d] = tl.load(k_base + d)

        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_vec[d]

        logit = dot * sm_scale
        # Streaming update
        new_m = tl.maximum(m, logit)
        sumexp = sumexp * tl.exp(m - new_m) + tl.exp(logit - new_m)
        m = new_m

    # lse = m + log(sumexp) / ln(2)
    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute softmax per token and accumulate output
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for nn in range(0, N_TOTAL):
        in_range = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=in_range, other=0).to(tl.int32)

        k_base = k_ptr + idx * num_kv_heads * D + kv_head * D
        v_base = v_ptr + idx * num_kv_heads * D + kv_head * D

        k_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_vec[d] = tl.load(k_base + d)

        v_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            v_vec[d] = tl.load(v_base + d)

        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_vec[d]
        logit = dot * sm_scale

        # softmax per token in base-2 normalization
        softmax = tl.exp(logit - m) / (sumexp * ln2)
        # accumulate output
        for d in range(0, D):
            out_vec[d] += softmax * v_vec[d]

    # store output
    for d in range(0, D):
        tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure Triton is available; if not, raise to enforce Triton-only path.
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton not available, but ModelNew.forward must use Triton kernels.")

        B, H, D = q.shape  # q: [B, H, D]
        num_kv_heads = 8

        # Convert inputs to float32 for computation; ensure contiguity
        q_f32 = q.contiguous().to(torch.float32)         # [B, H, D]
        k_f32 = k_cache.contiguous().to(torch.float32)   # [N, 1, num_kv_heads, D]
        v_f32 = v_cache.contiguous().to(torch.float32)   # [N, 1, num_kv_heads, D]

        # Output and lse buffers as float32
        out = torch.empty((B, H, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        N_TOTAL = 128  # covers typical num_tokens up to 128; masks beyond actual_num_tokens

        _forward_kernel_bh[grid](
            q_f32, k_f32, v_f32, kv_indices, kv_indptr, lse, out,
            B=B, H=H, D=D, num_kv_heads=num_kv_heads, sm_scale=float(sm_scale),
            N_TOTAL=N_TOTAL,
            num_warps=4,
        )

        # Return outputs as bfloat16 (original expects bfloat16) and lse as float32
        return out.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
