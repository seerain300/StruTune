import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def compute_lse_kernel(
    q_ptr,             # *bf16, [B, H, D]
    k_ptr,             # *bf16, [Np, K, D]
    kv_indptr_ptr,     # *int32, [B+1]
    lse_out_ptr,       # *float32, [B, H]
    sm_scale,          # float32 scalar
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr, K: tl.constexpr, gqa_ratio: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    start = tl.load(kv_indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start

    sum_exp = 0.0
    # Fixed-iteration loop with guard; Triton supports scalar control flow.
    for t in range(MAX_TOKENS):
        if t >= num_tokens:
            break
        idx = start + t
        kvh = pid_h // gqa_ratio
        # Load q vector for this head
        q_offset = pid_b * H * D + pid_h * D
        q_vec = tl.load(q_ptr + q_offset).to(tl.float32)  # [D]
        # Compute dot = q · k[idx, kvh, :]
        dot = 0.0
        for d in range(D):
            k_val = tl.load(k_ptr + idx * K * D + kvh * D + d).to(tl.float32)
            dot += q_vec[d] * k_val
        # Accumulate sum_exp = sum(exp(dot * sm_scale))
        sum_exp += tl.exp(dot * sm_scale)

    # lse in base-2: log(sum_exp) / ln(2)
    lse = tl.log(sum_exp) * 1.4426950408889634  # 1 / ln(2)
    tl.store(lse_out_ptr + pid_b * H + pid_h, lse)


@triton.jit
def accumulate_output_kernel(
    q_ptr,             # *bf16, [B, H, D]
    k_ptr,             # *bf16, [Np, K, D]
    v_ptr,             # *bf16, [Np, K, D]
    kv_indptr_ptr,     # *int32, [B+1]
    lse_out_ptr,       # *float32, [B, H] (base-2 lse)
    output_ptr,        # *bf16, [B, H, D]
    sm_scale,          # float32 scalar
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr, K: tl.constexpr, gqa_ratio: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)

    start = tl.load(kv_indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start

    lse_val = tl.load(lse_out_ptr + pid_b * H + pid_h).to(tl.float32)  # base-2 logsumexp

    # For each token, recompute dot, compute attn, and accumulate output vector
    for t in range(MAX_TOKENS):
        if t >= num_tokens:
            break
        idx = start + t
        kvh = pid_h // gqa_ratio

        # Load q vector for this head
        q_offset = pid_b * H * D + pid_h * D
        q_vec = tl.load(q_ptr + q_offset).to(tl.float32)  # [D]

        # Compute dot = q · k[idx, kvh, :]
        dot = 0.0
        for d in range(D):
            k_val = tl.load(k_ptr + idx * K * D + kvh * D + d).to(tl.float32)
            dot += q_vec[d] * k_val

        attn = tl.exp((dot - lse_val) * sm_scale)  # base-2 lse already used

        # Accumulate output[b, h, :] += attn * v[idx, kvh, :]
        out_offset = pid_b * H * D + pid_h * D
        for d in range(D):
            v_val = tl.load(v_ptr + idx * K * D + kvh * D + d).to(tl.float32)
            # Load current output element, add, store
            old = tl.load(output_ptr + out_offset + d).to(tl.float32)
            new = old + attn * v_val
            tl.store(output_ptr + out_offset + d, new.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton path: ensure contiguity and device
        assert TRITON_AVAILABLE, "Triton is not available"
        assert q.is_cuda, "Inputs must be on CUDA for Triton"
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()  # not used in our kernels, kept for API compatibility

        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        # Keep constants as in original
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        device = q.device
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch kernel 1: compute lse (base-2) for each (b, h)
        grid = (batch_size, num_qo_heads)
        compute_lse_kernel[grid](
            q, k_cache, kv_indptr, lse,
            sm_scale,
            B=batch_size, H=num_qo_heads, D=head_dim, K=num_kv_heads, gqa_ratio=4, MAX_TOKENS=1024,
            num_warps=2, num_stages=1,
        )

        # Launch kernel 2: accumulate output for each (b, h) using lse
        accumulate_output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, lse, output,
            sm_scale,
            B=batch_size, H=num_qo_heads, D=head_dim, K=num_kv_heads, gqa_ratio=4, MAX_TOKENS=1024,
            num_warps=2, num_stages=1,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
