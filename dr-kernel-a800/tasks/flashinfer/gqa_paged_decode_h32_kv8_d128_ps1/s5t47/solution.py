import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def fused_gqa_forward_kernel(
    q_ptr,            # *bf16, [B, H, D]
    k_ptr,            # *bf16, [Np, 1, K, D] (we index via idx_ptr; no tl.load on indptr)
    v_ptr,            # *bf16, [Np, 1, K, D]
    idx_ptr,          # *int32, [N] flattened token indices
    out_ptr,          # *bf16, [B, H, D]
    lse_ptr,          # *float32, [B, H]
    sm_scale,         # float32 scalar
    H: tl.constexpr,  # num query heads (32)
    D: tl.constexpr,  # head_dim (128)
    K: tl.constexpr,  # num kv heads (8)
    gqa_ratio: tl.constexpr,  # 4
    start,            # int32, start of token range for this batch
    end,              # int32, end of token range for this batch
    MAX_TOKENS: tl.constexpr    # upper bound iterations (e.g., 1024)
):
    # One program per (batch, head)
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Compute corresponding KV head for GQA
    kv_head = h // gqa_ratio  # in [0, K-1]

    # Prepare output vector as fp32
    out_vec = tl.zeros([D], dtype=tl.float32)

    # Load q vector for this head (as fp32)
    q_offset = b * H * D + h * D
    q_vec = tl.load(q_ptr + q_offset).to(tl.float32)  # [D]

    # Pass 1: compute sum_exp = sum(exp((q·k) * sm_scale)) across tokens
    sum_exp = 0.0
    for t in range(MAX_TOKENS):
        if t >= (end - start):
            break
        # idx for this token (scalar int32)
        idx_t = tl.load(idx_ptr + start + t).to(tl.int32)
        # k: [K, D] row for kv_head
        k_row_offset = idx_t * K * D + kv_head * D
        v_row_offset = idx_t * K * D + kv_head * D
        k_vec = tl.load(k_ptr + k_row_offset).to(tl.float32)  # [D]
        v_vec = tl.load(v_ptr + v_row_offset).to(tl.float32)  # [D]

        # Dot product over D
        dot = 0.0
        for d in range(D):
            dot += q_vec[d] * k_vec[d]
        logits_scaled = dot * sm_scale
        sum_exp += tl.exp(logits_scaled)

    # Compute LSE (base-2): lse = log(sum_exp) / ln(2)
    ln2 = 0.6931471805599453
    lse_base2 = tl.log(sum_exp) / ln2

    # Pass 2: compute output vector out_vec += attn * v[token] for each token
    for t in range(MAX_TOKENS):
        if t >= (end - start):
            break
        idx_t = tl.load(idx_ptr + start + t).to(tl.int32)
        k_row_offset = idx_t * K * D + kv_head * D
        v_row_offset = idx_t * K * D + kv_head * D
        k_vec = tl.load(k_ptr + k_row_offset).to(tl.float32)  # [D]
        v_vec = tl.load(v_ptr + v_row_offset).to(tl.float32)  # [D]

        dot = 0.0
        for d in range(D):
            dot += q_vec[d] * k_vec[d]
        logits_scaled = dot * sm_scale
        attn = tl.exp((logits_scaled - lse_base2) * sm_scale)

        for d in range(D):
            out_vec[d] += attn * v_vec[d]

    # Store outputs
    out_offset = b * H * D + h * D
    tl.store(out_ptr + out_offset, out_vec.to(tl.bfloat16))

    lse_offset = b * H
    tl.store(lse_ptr + lse_offset, lse_base2)  # stored as base-2 LSE


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure device and contiguity
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Extract shapes
        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        device = q.device
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Compute start/end for each batch on host (int32)
        # Note: get inputs use kv_indptr of shape [B+1], so we can index directly
        start = kv_indptr[:batch_size].to(torch.int32)
        end = kv_indptr[1:batch_size + 1].to(torch.int32)
        num_tokens = (end - start).to(torch.int32)

        # Flatten kv_indices to int32
        idx = kv_indices.to(torch.int32)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size * num_qo_heads,)
        fused_gqa_forward_kernel[grid](
            q, k_cache, v_cache, idx, output, lse,
            sm_scale,
            H=32, D=128, K=8, gqa_ratio=4,
            start=start, end=end,
            MAX_TOKENS=1024,
            num_warps=4, num_stages=1
        )
        return output, lse


def run(*args):
    return ModelNew()(*args)
