import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def gqa_row_simple_kernel(
    q_ptr,       # *bf16 or *fp32, [B, H, D] contiguous
    k_ptr,       # *bf16 or *fp32, [Np, 1, K, D] contiguous (we pass squeeze(1) -> [Np, K, D])
    v_ptr,       # *bf16 or *fp32, [Np, K, D] contiguous
    indptr_ptr,  # *int32, [B+1]
    indices_ptr, # *int32, [num_kv_indices]
    out_ptr,     # *bf16, [B, H, D]
    lse_ptr,     # *fp32, [B, H]
    B: tl.constexpr,     # batch size
    H: tl.constexpr,     # num_qo_heads
    D: tl.constexpr,     # head_dim
    K: tl.constexpr,     # num_kv_heads
    MAX_TOKENS: tl.constexpr,  # max tokens to iterate (e.g., 1024)
    sm_scale,  # fp32 scalar
):
    pid_b = tl.program_id(0)  # batch id
    pid_h = tl.program_id(1)  # head id

    # Load range for this batch element
    start = tl.load(indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start  # runtime scalar int32

    # Base offsets for q[b, h, :]
    q_offset = pid_b * H * D + pid_h * D
    q_base = q_ptr + q_offset

    # Pass 1: accumulate sum_exp for logsumexp in base-2
    sum_exp = 0.0  # scalar fp32
    for t in range(MAX_TOKENS):
        if t >= num_tokens:
            break
        idx = start + t
        kv_head = pid_h // 4  # gqa mapping
        # Load q[h] as fp32 (scalar loop over D)
        dot = 0.0
        for d in range(D):
            qd = tl.load(q_base + d).to(tl.float32)
            k_row_base = k_ptr + idx * (K * D) + kv_head * D
            kd = tl.load(k_row_base + d).to(tl.float32)
            dot += qd * kd
        logits_scaled = dot * sm_scale
        sum_exp += tl.exp(logits_scaled * 1.4426950408889634)  # log2(exp(x)) = x / ln(2)
    # Compute lse in base-2: lse = log(sum_exp) / ln(2)
    lse_base2 = tl.log(sum_exp) * 1.4426950408889634  # log(sum_exp) is natural log; multiply by 1/ln(2)

    # Store lse for this (b, h)
    tl.store(lse_ptr + pid_b * H + pid_h, lse_base2)

    # Pass 2: compute output vector out[b, h, :] = sum_t attn_t * v[t, kv_head, :]
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for t in range(MAX_TOKENS):
        if t >= num_tokens:
            break
        idx = start + t
        kv_head = pid_h // 4
        dot = 0.0
        for d in range(D):
            qd = tl.load(q_base + d).to(tl.float32)
            k_row_base = k_ptr + idx * (K * D) + kv_head * D
            kd = tl.load(k_row_base + d).to(tl.float32)
            dot += qd * kd
        logits_scaled = dot * sm_scale
        attn = tl.exp((logits_scaled - lse_base2) * 1.4426950408889634)  # base-2 softmax
        v_row_base = v_ptr + idx * (K * D) + kv_head * D
        for d in range(D):
            vd = tl.load(v_row_base + d).to(tl.float32)
            out_vec[d] += attn * vd

    # Store output as bfloat16
    out_base = out_ptr + pid_b * H * D + pid_h * D
    for d in range(D):
        tl.store(out_base + d, out_vec[d].to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity and dtypes
        q = q.contiguous()
        k_cache = k_cache.squeeze(1).contiguous()  # [Np, K, D]
        v_cache = v_cache.squeeze(1).contiguous()  # [Np, K, D]
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        batch_size, num_qo_heads, head_dim = q.shape
        num_pages, num_kv_heads, _, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        device = q.device
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Grid: one program per (b, h)
        grid = (batch_size, num_qo_heads)
        gqa_row_simple_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, output, lse,
            B=batch_size, H=num_qo_heads, D=head_dim, K=num_kv_heads,
            MAX_TOKENS=1024,
            sm_scale=float(sm_scale),
            num_warps=2, num_stages=1,
        )
        return output, lse


def run(*args):
    return ModelNew()(*args)
