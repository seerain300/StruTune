import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def fused_gqa_row_simple_kernel(
    q_ptr,           # *bf16, [B, H, D]
    k_ptr,           # *bf16, [Np, 1, K, D] (inputs use Np=1; we don't vectorize across Np)
    v_ptr,           # *bf16, [Np, 1, K, D]
    indptr_ptr,      # *int32, [B+1]
    idx_ptr,         # *int32, [num_kv_indices]
    out_ptr,         # *bf16, [B, H, D]
    lse_ptr,         # *float32, [B, H]
    sm_scale,        # float32 scalar
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr, K: tl.constexpr, gqa_ratio: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Load q vector for this (b, h)
    q_offset = pid_b * H * D + pid_h * D
    q_vec = tl.load(q_ptr + q_offset).to(tl.float32)  # [D]

    # Compute token range for this batch
    start = tl.load(indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start  # int32 scalar

    # Pass 1: compute sum_exp = sum(exp((q·k) * sm_scale)) across tokens
    sum_exp = 0.0
    for t in range(MAX_TOKENS):
        if t < num_tokens:
            idx_val = tl.load(idx_ptr + start + t).to(tl.int32)
            kv_h = pid_h // gqa_ratio
            # k layout: [Np, K, D] with Np=1 in provided inputs
            k_offset = idx_val * K * D + kv_h * D
            k_vec = tl.load(k_ptr + k_offset).to(tl.float32)
            dot = tl.sum(q_vec * k_vec, axis=0)
            sum_exp += tl.exp(dot * sm_scale)

    lse_val = tl.log(sum_exp) / 6.931471805599453  # log2(sum_exp) = log(sum_exp)/ln(2)

    # Accumulate output vector across tokens
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for t in range(MAX_TOKENS):
        if t < num_tokens:
            idx_val = tl.load(idx_ptr + start + t).to(tl.int32)
            kv_h = pid_h // gqa_ratio
            k_offset = idx_val * K * D + kv_h * D
            k_vec = tl.load(k_ptr + k_offset).to(tl.float32)

            v_offset = idx_val * K * D + kv_h * D
            v_vec = tl.load(v_ptr + v_offset).to(tl.float32)

            dot = tl.sum(q_vec * k_vec, axis=0)
            attn = tl.exp((dot - lse_val) * sm_scale)
            out_vec += attn * v_vec

    # Store results
    out_offset = pid_b * H * D + pid_h * D
    tl.store(out_ptr + out_offset, out_vec.to(tl.bfloat16))
    tl.store(lse_ptr + pid_b * H + pid_h, lse_val.to(tl.float32))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape  # Expect shape [num_pages, 1, 8, 128]
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128, "Expected dims: [B, 32, 128], [Np, 1, 8, 128]"

        device = q.device

        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch one Triton program per (b, h)
        grid = (batch_size, num_qo_heads)

        fused_gqa_row_simple_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, output, lse,
            sm_scale,
            B=batch_size, H=num_qo_heads, D=head_dim, K=num_kv_heads, gqa_ratio=4,
            MAX_TOKENS=1024,
            num_warps=2, num_stages=1,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
