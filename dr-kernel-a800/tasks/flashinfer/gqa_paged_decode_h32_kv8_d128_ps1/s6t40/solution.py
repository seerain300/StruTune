import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (b, h)
# Compute for this (b, h):
# - q_vec = q[h, :, b]  where q is permuted to [H, D, B] in forward
# - For tokens n in [kv_indptr[b], kv_indptr[b+1)):
#     - k_row = k_cache[indices[start+n], kv_head, :]
#     - v_row = v_cache[indices[start+n], kv_head, :]
#     - dot = q_vec · k_row
#     - logit = dot * sm_scale
#     - Accumulate logsumexp in base-2 over tokens
#     - attn = softmax(logit)
#     - out[b, h, :] += attn * v_row
@triton.jit
def _forward_kernel_bh(
    qh_ptr,                 # *fp16, [H, D, B], pre-permuted: q.permute(1,2,0).contiguous()
    k_ptr,                  # *fp16, [N_total, 1, num_kv_heads, D]
    v_ptr,                  # *fp16, [N_total, 1, num_kv_heads, D]
    kv_indices_ptr,         # *int32, [num_tokens]
    kv_indptr_ptr,          # *int32, [B+1]
    lse_ptr,                # *float32, [B, H]
    out_ptr,                # *float32, [B, H, D] (we compute in fp32)
    H: tl.int32,            # e.g., 32
    D: tl.constexpr,        # e.g., 128
    B: tl.int32,            # batch size
    num_kv_heads: tl.int32, # e.g., 8
    sm_scale: tl.float32,   # 1/sqrt(D) in fp32
    N_TOTAL: tl.constexpr,  # loop bound, e.g., 128
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Load q[b, h, :] from qh_ptr where qh_ptr is [H, D, B]
    q_base = qh_ptr + h * D * B + b * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        q_val = tl.load(q_base + d)
        q_vec[d] = q_val.to(tl.float32)

    # Compute start/end for this batch b
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start  # int

    # GQA mapping: kv_head = h // (H // num_kv_heads) = h // 4
    kv_head = h // (H // num_kv_heads)

    # First pass: streaming logsumexp in base-2
    m = -float("inf")  # scalar
    sumexp = 0.0       # scalar
    ln2 = 0.6931471805599453
    for nn in range(0, N_TOTAL):
        valid = nn < actual_num_tokens
        # idx of the token (only valid tokens contribute)
        idx = tl.load(kv_indices_ptr + start + nn).to(tl.int32)

        # k_row: k_cache[idx, kv_head, :]
        # k_ptr layout: [N_total, 1, num_kv_heads, D] => element offset = idx * (1 * num_kv_heads * D) + kv_head * D
        k_row_ptr = k_ptr + idx * num_kv_heads * D + kv_head * D
        k_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_val = tl.load(k_row_ptr + d)
            k_row[d] = k_val.to(tl.float32)

        # dot = q_vec · k_row
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_row[d]

        logit = dot * sm_scale
        new_m = tl.maximum(m, logit)
        # sumexp update via streaming: sumexp_new = sumexp*exp(m-new_m) + exp(logit-new_m)
        sumexp = sumexp * tl.exp(m - new_m) + tl.exp(logit - new_m)
        m = new_m

    # lse[b, h] = logsumexp(logit) / ln(2) = m + log(sumexp) / ln2
    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute softmax and accumulate output
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for nn in range(0, N_TOTAL):
        valid = nn < actual_num_tokens
        idx = tl.load(kv_indices_ptr + start + nn).to(tl.int32)
        k_row_ptr = k_ptr + idx * num_kv_heads * D + kv_head * D
        v_row_ptr = v_ptr + idx * num_kv_heads * D + kv_head * D

        v_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            v_val = tl.load(v_row_ptr + d)
            v_row[d] = v_val.to(tl.float32)

        # Recompute dot
        k_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_val = tl.load(k_row_ptr + d)
            k_row[d] = k_val.to(tl.float32)
        dot = 0.0
        for d in range(0, D):
            dot += q_vec[d] * k_row[d]
        logit = dot * sm_scale

        # softmax over tokens (base-2 normalization in lse already handled; we just use sumexp)
        softmax = tl.exp(logit - m) / (sumexp * ln2)
        out_vec += v_row * softmax

    # Store result
    for d in range(0, D):
        tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # We expect the same signature as the original: run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale)
        # We'll reconstruct q, k_cache, v_cache, etc. from args. In most cases, args are the same tensors.
        q = args[0]
        k_cache = args[1]
        v_cache = args[2]
        kv_indptr = args[3]
        kv_indices = args[4]
        sm_scale = args[5]

        # Extract shapes
        B, H, D = q.shape  # q: [B, H, D]
        num_kv_heads = 8  # fixed as per original code
        device = q.device

        # We will not use any PyTorch math in the kernel. We need q[h, :, b] for the kernel. Easiest: permute to [H, D, B].
        # Note: This is a pure data permutation (no math), similar to what the original code implicitly handles via layout.
        q_perm = q.permute(1, 2, 0).contiguous()  # [H, D, B], device=q.device

        # Allocate outputs
        lse = torch.empty((B, H), dtype=torch.float32, device=device)  # we'll compute in fp32
        output_fp32 = torch.empty((B, H, D), dtype=torch.float32, device=device)

        # Launch Triton kernel: grid over (b, h)
        grid = (B, H)
        _forward_kernel_bh[grid](
            q_perm, k_cache, v_cache, kv_indices, kv_indptr, lse, output_fp32,
            H=H, D=D, B=B, num_kv_heads=num_kv_heads, sm_scale=float(sm_scale), N_TOTAL=128,
        )

        # Return output as bfloat16 (matching original), and lse as float32.
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
