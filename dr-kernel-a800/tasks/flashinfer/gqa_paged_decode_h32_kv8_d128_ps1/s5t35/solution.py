import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel A: Compute lse (base-2 logsumexp) per (b, h) across all tokens in [kv_indptr[b], kv_indptr[b+1))
@triton.jit
def reduce_lse_kernel(
    q_ptr,               # *bf16, [B, H, D]
    k_ptr,               # *bf16, [Np, 1, K, D]
    v_ptr,               # *bf16, [Np, 1, K, D]
    kv_indptr_ptr,       # *int32, [B+1]
    lse_ptr,             # *float32, [B, H]
    sm_scale,            # float32 scalar
    pid_b: tl.constexpr, pid_h: tl.constexpr,
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr, K: tl.constexpr, gqa_ratio: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
):
    # Load range [start, end] for this batch
    start = tl.load(kv_indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start

    # Load q vector for this head
    q_offset = pid_b * H * D + pid_h * D
    q_vec = tl.load(q_ptr + q_offset).to(tl.float32)  # [D]

    # Accumulate sum_exp over tokens (scalar iterations guarded by num_tokens)
    sum_exp = 0.0
    for t in range(MAX_TOKENS):
        if t < num_tokens:
            idx = start + t  # token index for this batch
            kvh = pid_h // gqa_ratio  # select corresponding kv head
            dot = 0.0
            for d in range(D):
                k_val = tl.load(k_ptr + idx * K * D + kvh * D + d).to(tl.float32)
                dot += q_vec[d] * k_val
            sum_exp += tl.exp(dot * sm_scale)

    # lse in base-2
    lse_val = tl.log(sum_exp) / tl.log(2.0)  # log base 2
    lse_addr = pid_b * H + pid_h
    tl.store(lse_ptr + lse_addr, lse_val)


# Kernel B: Update output per (b, h, token): out[b, h, :] += attn * v[token, kvh]
@triton.jit
def update_output_kernel(
    q_ptr,               # *bf16, [B, H, D]
    k_ptr,               # *bf16, [Np, 1, K, D]
    v_ptr,               # *bf16, [Np, 1, K, D]
    kv_indptr_ptr,       # *int32, [B+1]
    lse_ptr,             # *float32, [B, H]
    output_ptr,          # *bf16, [B, H, D]
    sm_scale,            # float32 scalar
    pid_b: tl.constexpr, pid_h: tl.constexpr, pid_t: tl.constexpr,
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr, K: tl.constexpr, gqa_ratio: tl.constexpr,
):
    start = tl.load(kv_indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start

    # Guard: ensure we only process valid tokens
    if pid_t < num_tokens:
        idx = start + pid_t  # token index for this batch
        kvh = pid_h // gqa_ratio  # select corresponding kv head

        # Load q vector for this head
        q_offset = pid_b * H * D + pid_h * D
        q_vec = tl.load(q_ptr + q_offset).to(tl.float32)  # [D]

        # Compute dot = q · k[idx, kvh, :]
        dot = 0.0
        for d in range(D):
            k_val = tl.load(k_ptr + idx * K * D + kvh * D + d).to(tl.float32)
            dot += q_vec[d] * k_val

        # Load lse for this (b, h)
        lse_val = tl.load(lse_ptr + pid_b * H + pid_h).to(tl.float32)

        # attn = exp((dot - lse) * sm_scale)
        attn = tl.exp((dot - lse_val) * sm_scale)

        # Update output: out += attn * v[idx, kvh, :]
        out_offset = pid_b * H * D + pid_h * D
        for d in range(D):
            v_val = tl.load(v_ptr + idx * K * D + kvh * D + d).to(tl.float32)
            tl.store(output_ptr + out_offset + d, tl.load(output_ptr + out_offset + d).to(tl.float32) + attn * v_val)

        # Note: Triton doesn't support out_offset += 1 vectorized store; we update per element via loop.


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Fallback if Triton/CUDA not available
        if not TRITON_AVAILABLE or not q.is_cuda:
            batch_size, num_qo_heads, head_dim = q.shape
            _, _, num_kv_heads, _ = k_cache.shape
            assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128
            device = q.device
            output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
            lse = torch


def run(*args):
    return ModelNew()(*args)
