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
    q_ptr,          # *bf16, [B, H, D]
    k_ptr,          # *bf16, [Np, 1, K, D]
    v_ptr,          # *bf16, [Np, 1, K, D]
    indptr_ptr,     # *int32, [B+1]
    indices_ptr,    # *int32, [num_kv_indices]
    out_ptr,        # *bf16, [B, H, D]
    lse_ptr,        # *f32,  [B, H]
    sm_scale: tl.constexpr,   # fp32 scalar
    D: tl.constexpr,           # head_dim (e.g., 128)
    H: tl.constexpr,           # num_qo_heads (e.g., 32)
    K: tl.constexpr,           # num_kv_heads (e.g., 8)
    GQA_RATIO: tl.constexpr,   # H // K (e.g., 4)
    MAX_TOKENS: tl.constexpr,  # max tokens processed in scalar loop
):
    # One program per (batch b, query head h)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Load q vector for this (b, h): q[b, h, :]
    q_offset = pid_b * H * D + pid_h * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in tl.range(0, D):
        q_elem = tl.load(q_ptr + q_offset + d).to(tl.float32)
        q_vec[d] = q_elem

    # Load kv range for this batch element
    start = tl.load(indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start

    # Pass 1: compute sum_exp = sum(exp((q·k) * sm_scale)) across tokens (scalar iterations guarded)
    sum_exp = 0.0  # fp32
    for t in tl.range(0, MAX_TOKENS):
        if t >= num_tokens:
            break
        token_idx = tl.load(indices_ptr + start + t).to(tl.int32)
        kv_head = pid_h // GQA_RATIO

        # k[token, kv_head, :] and v[token, kv_head, :]
        k_offset = token_idx * K * D + kv_head * D
        v_offset = token_idx * K * D + kv_head * D

        k_vec = tl.zeros((D,), dtype=tl.float32)
        for dd in tl.range(0, D):
            k_elem = tl.load(k_ptr + k_offset + dd).to(tl.float32)
            k_vec[dd] = k_elem

        dot = 0.0
        for dd in tl.range(0, D):
            dot += q_vec[dd] * k_vec[dd]

        sum_exp += tl.exp(dot * sm_scale)

    # lse in base-2: logsumexp is log(sum_exp); convert to base-2
    ln2 = 0.6931471805599453
    lse_base2 = tl.log(sum_exp) / ln2
    tl.store(lse_ptr + pid_b * H + pid_h, lse_base2)

    # Pass 2: accumulate output vector
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for t in tl.range(0, MAX_TOKENS):
        if t >= num_tokens:
            break
        token_idx = tl.load(indices_ptr + start + t).to(tl.int32)
        kv_head = pid_h // GQA_RATIO

        k_offset = token_idx * K * D + kv_head * D
        v_offset = token_idx * K * D + kv_head * D

        k_vec = tl.zeros((D,), dtype=tl.float32)
        for dd in tl.range(0, D):
            k_elem = tl.load(k_ptr + k_offset + dd).to(tl.float32)
            k_vec[dd] = k_elem

        v_vec = tl.zeros((D,), dtype=tl.float32)
        for dd in tl.range(0, D):
            v_elem = tl.load(v_ptr + v_offset + dd).to(tl.float32)
            v_vec[dd] = v_elem

        dot = 0.0
        for dd in tl.range(0, D):
            dot += q_vec[dd] * k_vec[dd]

        attn = tl.exp((dot - lse_base2) * sm_scale)
        for dd in tl.range(0, D):
            out_vec[dd] += attn * v_vec[dd]

    # Store output as bfloat16
    out_offset = pid_b * H * D + pid_h * D
    for dd in tl.range(0, D):
        tl.store(out_ptr + out_offset + dd, tl.cast(out_vec[dd], tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton path: ensure contiguity and proper dtypes
        q = q.contiguous()
        k_cache = k_cache.contiguous()   # [Np, 1, K, D], D=128
        v_cache = v_cache.contiguous()   # [Np, 1, K, D], D=128
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        device = q.device
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        grid = (batch_size, num_qo_heads)
        # Launch Triton kernel; specialize constants
        gqa_row_simple_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, output, lse,
            sm_scale,
            D=128, H=32, K=8, GQA_RATIO=4, MAX_TOKENS=1024,
            num_warps=2, num_stages=1,
        )
        return output, lse


def run(*args):
    return ModelNew()(*args)
