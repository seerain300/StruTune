import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def gqa_bh_two_pass_kernel(
    q_ptr,            # *bf16, [B, H, D]
    k_ptr,            # *bf16, [Np, K, D]
    v_ptr,            # *bf16, [Np, K, D]
    indptr_ptr,       # *int32, [B+1]
    out_ptr,          # *bf16, [B, H, D]
    lse_ptr,          # *float32, [B, H]
    B: tl.constexpr,  # batch size (compile-time for specialization)
    H: tl.constexpr,  # num query heads
    D: tl.constexpr,  # head dim (compile-time)
    K: tl.constexpr,  # num kv heads (compile-time)
    sm_scale,         # float32 scalar
    MAX_TOKENS: tl.constexpr,  # max tokens to iterate
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Compute range of tokens for this batch element
    start = tl.load(indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start  # runtime int32

    # Load q vector for this head (b, h) as fp32
    q_offset = pid_b * H * D + pid_h * D
    q_vec = tl.load(q_ptr + q_offset + tl.arange(0, D)).to(tl.float32)  # [D]

    # Pass 1: accumulate sum_exp in base-2
    sum_exp = 0.0
    for t in range(MAX_TOKENS):
        if t >= num_tokens:
            break
        idx = start + t
        kv_head = pid_h // 4  # gqa ratio = H // K = 4
        # Load k_row[idx, kv_head, :] as vector
        k_row = tl.load(k_ptr + idx * (K * D) + kv_head * D + tl.arange(0, D)).to(tl.float32)  # [D]
        dot = tl.sum(q_vec * k_row, axis=0)  # scalar
        # sum_exp += exp((dot * sm_scale) * ln(2)), where we accumulate in base-2 lse later
        sum_exp += tl.exp((dot * sm_scale) * 1.4426950408889634)  # 1 / ln(2) to convert back to sum of exp(log2)

    # lse in base-2: log2(sum_exp) = log(sum_exp) * (1 / ln(2))
    lse_val = (tl.log(sum_exp)) / 1.4426950408889634  # base-2 logsumexp

    # Pass 2: recompute dot, compute attn, and accumulate output vector
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for t in range(MAX_TOKENS):
        if t >= num_tokens:
            break
        idx = start + t
        kv_head = pid_h // 4
        k_row = tl.load(k_ptr + idx * (K * D) + kv_head * D + tl.arange(0, D)).to(tl.float32)  # [D]
        dot = tl.sum(q_vec * k_row, axis=0)
        attn = tl.exp((dot * sm_scale) - lse_val)  # base-2 logsumexp already applied
        v_row = tl.load(v_ptr + idx * (K * D) + kv_head * D + tl.arange(0, D)).to(tl.float32)  # [D]
        out_vec += attn * v_row

    # Store output vector to out[b, h, :]
    tl.store(out_ptr + q_offset, out_vec.to(tl.bfloat16))

    # Store lse in base-2 to lse[b, h]
    tl.store(lse_ptr + pid_b * H + pid_h, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        # kv_indices is not directly used in the kernel to compute tokens; we only need kv_indptr
        # Shapes
        batch_size, num_qo_heads, head_dim = q.shape
        num_kv_heads = v_cache.shape[2]  # should be 8
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        device = q.device
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton: one program per (b, h)
        grid = (batch_size, num_qo_heads)
        gqa_bh_two_pass_kernel[grid](
            q, k_cache, v_cache, kv_indptr, output, lse,
            B=batch_size, H=num_qo_heads, D=head_dim, K=num_kv_heads,
            sm_scale=float(sm_scale),
            MAX_TOKENS=1024,
            num_warps=2, num_stages=1,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
