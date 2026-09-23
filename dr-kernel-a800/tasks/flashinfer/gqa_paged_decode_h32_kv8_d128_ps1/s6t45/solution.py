import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _forward_bh_kernel(
    q_ptr,          # *ptr to q, shape [B, H, D]
    k_ptr,          # *ptr to k_cache, shape [N_total, 1, num_kv_heads, D]
    v_ptr,          # *ptr to v_cache, shape [N_total, 1, num_kv_heads, D]
    kv_indices_ptr, # *ptr to int32, shape [num_kv_indices]
    kv_indptr_ptr,  # *ptr to int32, shape [B+1]
    lse_ptr,        # *ptr to lse, shape [B, H], float32
    out_ptr,        # *ptr to output, shape [B, H, D], float32
    B, H, D,        # runtime ints
    num_kv_heads: tl.constexpr,  # 8
    sm_scale,       # scalar float32
    N_TOTAL: tl.constexpr,       # loop bound (e.g., 128), masked by nn < actual_num_tokens
):
    # Program ids: one per (b, h)
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Load q vector for (b, h)
    q_base = q_ptr + b * H * D + h * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        q_vec[d] = tl.load(q_base + d).to(tl.float32)

    # Compute token range for this batch
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start  # int32

    # GQA mapping
    kv_head = h // 4

    # First pass: compute logsumexp in base-2
    m = -float("inf")  # scalar
    sumexp = 0.0       # scalar
    ln2 = 0.6931471805599453  # log(2.0)

    for nn in range(0, N_TOTAL):
        cond = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=cond, other=0).to(tl.int32)

        # k_row and v_row pointers: k[idx, kv_head, :], v[idx, kv_head, :]
        # Layout is [N_total, 1, num_kv_heads, D]. For fixed kv_head, stride across N_total is num_kv_heads*D, then D per element.
        k_row_ptr = k_ptr + idx * (num_kv_heads * D) + kv_head * D
        v_row_ptr = v_ptr + idx * (num_kv_heads * D) + kv_head * D

        # Load k vector
        k_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_vec[d] = tl.load(k_row_ptr + d).to(tl.float32)

        # Dot product q·k
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_vec[d]

        logit = dot * sm_scale
        new_m = tl.maximum(m, logit)
        sumexp = sumexp * tl.exp(m - new_m) + tl.exp(logit - new_m)
        m = new_m

    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: accumulate output vector
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for nn in range(0, N_TOTAL):
        cond = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn, mask=cond, other=0).to(tl.int32)

        k_row_ptr = k_ptr + idx * (num_kv_heads * D) + kv_head * D
        v_row_ptr = v_ptr + idx * (num_kv_heads * D) + kv_head * D

        # Load v vector
        v_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            v_vec[d] = tl.load(v_row_ptr + d).to(tl.float32)

        # Recompute dot and logit
        k_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_vec[d] = tl.load(k_row_ptr + d).to(tl.float32)
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_vec[d]
        logit = dot * sm_scale

        # softmax = exp(logit) / sumexp (since sumexp is sum of exp(scaled logits))
        softmax = tl.exp(logit) / sumexp
        out_vec += softmax * v_vec

    # Store final output
    for d in range(0, D):
        tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only forward: no PyTorch ops
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All tensors must be on CUDA for Triton kernel execution."

        B, H, D = q.shape
        num_kv_heads = 8

        # Output and lse buffers (compute in float32 for stability)
        out = torch.empty((B, H, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        _forward_bh_kernel[grid](
            q, k_cache, v_cache, kv_indices, kv_indptr, lse, out,
            B, H, D,
            num_kv_heads=num_kv_heads,
            sm_scale=float(sm_scale),
            N_TOTAL=128,  # compile-time bound; masked by nn < actual_num_tokens
        )

        # Return output in bfloat16 and lse in float32, in list format to match run(...)
        out_bf16 = out.to(torch.bfloat16)
        return [out_bf16, lse]


def run(*args):
    return ModelNew()(*args)
