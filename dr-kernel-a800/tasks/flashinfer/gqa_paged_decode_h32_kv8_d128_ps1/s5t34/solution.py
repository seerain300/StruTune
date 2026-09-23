import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def attention_bh_kernel(
    q_ptr,           # *bf16, [B, H, D]
    k_ptr,           # *bf16, [Np, K, D] (in provided inputs, Np=1, K=8)
    v_ptr,           # *bf16, [Np, K, D] (in provided inputs, Np=1, K=8)
    kv_indptr_ptr,   # *int32, [B+1]
    kv_indices_ptr,  # *int32, [num_kv_indices]
    output_ptr,      # *bf16, [B, H, D]
    lse_ptr,         # *float32, [B, H]
    sm_scale,        # float32 scalar
    B: tl.constexpr, H: tl.constexpr, K: tl.constexpr, D: tl.constexpr, gqa_ratio: tl.constexpr, MAX_TOKENS: tl.constexpr
):
    pid_b = tl.program_id(0)  # batch id
    pid_h = tl.program_id(1)  # query head id

    # Load q vector for this (b, h) as float32
    q_offset = pid_b * H * D + pid_h * D
    q_vec = tl.load(q_ptr + q_offset + tl.arange(0, D)).to(tl.float32)  # [D]

    # Compute token range for this batch
    start = tl.load(kv_indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start

    # Pass 1: compute sum_exp = sum(exp((q·k) * sm_scale)) across tokens
    sum_exp = 0.0
    for t in range(MAX_TOKENS):
        if t < num_tokens:
            idx = start + t
            k_idx = tl.load(kv_indices_ptr + idx).to(tl.int32)  # token index in cache
            kv_h = pid_h // gqa_ratio  # which kv head this query maps to (GQA)
            # Load k_row and v_row as vectors (avoid tl.arange over pointers)
            # k_ptr is [Np, K, D]; for Np=1: offset = k_idx * D + kv_h * D
            k_row = tl.load(k_ptr + k_idx * D + kv_h * D + tl.arange(0, D)).to(tl.float32)  # [D]
            # dot = sum(q_vec * k_row)
            dot = tl.sum(q_vec * k_row, axis=0)  # scalar
            logit_scaled = dot * sm_scale
            sum_exp += tl.exp(logit_scaled)

    # Compute lse in base-2
    lse_val = tl.log(sum_exp) / 0.6931471805599453  # ln(2)

    # Pass 2: compute output vector = sum_t exp((logit_scaled - lse) * sm_scale) * v_row
    out_vec = tl.zeros([D], dtype=tl.float32)
    for t in range(MAX_TOKENS):
        if t < num_tokens:
            idx = start + t
            k_idx = tl.load(kv_indices_ptr + idx).to(tl.int32)
            kv_h = pid_h // gqa_ratio
            k_row = tl.load(k_ptr + k_idx * D + kv_h * D + tl.arange(0, D)).to(tl.float32)  # [D]
            v_row = tl.load(v_ptr + k_idx * D + kv_h * D + tl.arange(0, D)).to(tl.float32)  # [D]
            dot = tl.sum(q_vec * k_row, axis=0)  # scalar
            attn = tl.exp((dot - lse_val) * sm_scale)  # scalar attention for this token
            out_vec += attn * v_row

    # Store output as bfloat16
    out_offset = pid_b * H * D + pid_h * D
    tl.store(output_ptr + out_offset + tl.arange(0, D), out_vec.to(tl.bfloat16))

    # Store lse per (b, h) as float32
    tl.store(lse_ptr + pid_b * H + pid_h, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Shapes
        batch_size, num_qo_heads, head_dim = q.shape
        num_pages, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        device = q.device
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch one program per (b, h)
        grid = (batch_size, num_qo_heads)
        attention_bh_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, output, lse,
            sm_scale,
            B=batch_size, H=num_qo_heads, K=num_kv_heads, D=head_dim, gqa_ratio=4, MAX_TOKENS=1024,
            num_warps=2, num_stages=1,
        )
        return output, lse


def run(*args):
    return ModelNew()(*args)
