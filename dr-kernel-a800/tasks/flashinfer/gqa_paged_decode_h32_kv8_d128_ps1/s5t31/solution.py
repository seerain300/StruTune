import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def gqa_attention_kernel(
    q_ptr,            # *bf16, [B, H, D]
    k_ptr,            # *bf16, [Np, 1, K, D]
    v_ptr,            # *bf16, [Np, 1, K, D]
    indptr_ptr,       # *int32, [B+1]
    indices_ptr,      # *int32, [num_kv_indices]
    out_ptr,          # *bf16, [B, H, D]
    lse_ptr,          # *float32, [B, H]
    sm_scale,         # float32 scalar
    B: tl.constexpr,  # batch size (for grid only)
    H: tl.constexpr,  # num_qo_heads (for grid only)
    D: tl.constexpr,  # head_dim = 128 (for vectorization along D)
    K: tl.constexpr,  # num_kv_heads = 8 (for selecting kv_head)
    MAX_TOKENS: tl.constexpr,  # max iterations over tokens, e.g., 1024
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Load token range for this batch element
    start = tl.load(indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start

    # Load q vector for this (b, h) and cast to fp32
    q_vec = tl.load(q_ptr + pid_b * H * D + pid_h * D + tl.arange(0, D)).to(tl.float32)  # [D]

    # First pass: compute sum_exp = sum(exp((q·k) * sm_scale)) across tokens
    sum_exp = 0.0
    for t in range(MAX_TOKENS):
        if t < num_tokens:
            idx_val = tl.load(indices_ptr + start + t).to(tl.int32)
            kv_h = pid_h // 4  # gqa_ratio = 4
            # k layout: [Np, 1, K, D]; we select k_cache[idx_val, 0, kv_h, :]
            k_vec = tl.load(
                k_ptr + idx_val * 1 * K * D + kv_h * D + tl.arange(0, D)
            ).to(tl.float32)  # [D]
            dot = tl.sum(q_vec * k_vec, axis=0)
            sum_exp += tl.exp(dot * sm_scale)

    # lse in base-2
    ln2 = 0.6931471805599453
    lse_val = tl.log(sum_exp) / ln2
    tl.store(lse_ptr + pid_b * H + pid_h, lse_val)

    # Second pass: accumulate output vector
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for t in range(MAX_TOKENS):
        if t < num_tokens:
            idx_val = tl.load(indices_ptr + start + t).to(tl.int32)
            kv_h = pid_h // 4
            k_vec = tl.load(
                k_ptr + idx_val * 1 * K * D + kv_h * D + tl.arange(0, D)
            ).to(tl.float32)  # [D]
            dot = tl.sum(q_vec * k_vec, axis=0)
            attn = tl.exp((dot - lse_val) * sm_scale)  # scalar
            v_vec = tl.load(
                v_ptr + idx_val * 1 * K * D + kv_h * D + tl.arange(0, D)
            ).to(tl.float32)  # [D]
            out_vec += attn * v_vec

    # Store output as bfloat16
    out_row_ptr = out_ptr + pid_b * H * D + pid_h * D
    tl.store(out_row_ptr + tl.arange(0, D), out_vec.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        # Assertions to match the original model
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        device = q.device

        # Output and lse buffers
        out = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size, num_qo_heads)
        gqa_attention_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, out, lse, sm_scale,
            B=batch_size, H=num_qo_heads, D=head_dim, K=num_kv_heads,
            MAX_TOKENS=1024, num_warps=2, num_stages=1,
        )

        return out, lse


def run(*args):
    return ModelNew()(*args)
