import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def per_bh_attention_kernel(
    q_ptr,            # *bf16, shape [B, H, D]
    k_ptr,            # *bf16, shape [Np, K, D], here Np=1 in inputs
    v_ptr,            # *bf16, shape [Np, K, D], here Np=1 in inputs
    kv_indptr_ptr,    # *int32, shape [B+1]
    kv_indices_ptr,   # *int32, shape [num_kv_indices]
    out_ptr,          # *bf16, shape [B, H, D]
    lse_ptr,          # *float32, shape [B, H]
    sm_scale,         # float32 scalar
    B: tl.constexpr,  # batch size (compile-time for specialization)
    H: tl.constexpr,  # num query heads
    D: tl.constexpr,  # head_dim (128)
    K: tl.constexpr,  # num kv heads (8)
    gqa_ratio: tl.constexpr,  # H//K (4)
    MAX_TOKENS: tl.constexpr, # max number of tokens to iterate (1024)
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Load indptr[start, end] for this batch element
    start = tl.load(kv_indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start

    # Prepare q vector for this head
    q_offset = pid_b * H * D + pid_h * D
    q_vec = tl.load(q_ptr + q_offset + tl.arange(0, D)).to(tl.float32)  # [D]

    # Pass 1: compute sum_exp = sum(exp((q·k) * sm_scale)) across tokens
    sum_exp = 0.0
    for t in range(MAX_TOKENS):
        if t < num_tokens:
            idx = start + t
            k_idx = tl.load(kv_indices_ptr + idx).to(tl.int32)
            kv_h = pid_h // gqa_ratio
            k_row = tl.load(k_ptr + k_idx * D + kv_h * D + tl.arange(0, D)).to(tl.float32)  # [D]
            dot = tl.sum(q_vec * k_row, axis=0)
            sum_exp += tl.exp(dot * sm_scale)
    # Compute lse in base-2
    ln2 = 0.6931471805599453  # log(2)
    lse = tl.log(sum_exp) / ln2

    # Pass 2: compute output vector
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for t in range(MAX_TOKENS):
        if t < num_tokens:
            idx = start + t
            k_idx = tl.load(kv_indices_ptr + idx).to(tl.int32)
            kv_h = pid_h // gqa_ratio
            k_row = tl.load(k_ptr + k_idx * D + kv_h * D + tl.arange(0, D)).to(tl.float32)  # [D]
            v_row = tl.load(v_ptr + k_idx * D + kv_h * D + tl.arange(0, D)).to(tl.float32)  # [D]
            dot = tl.sum(q_vec * k_row, axis=0)
            attn = tl.exp((dot * sm_scale - lse))
            out_vec += attn * v_row

    # Store output vector (bfloat16) and lse (float32)
    out_offset = pid_b * H * D + pid_h * D
    tl.store(out_ptr + out_offset, out_vec.to(tl.bfloat16))
    tl.store(lse_ptr + pid_b * H + pid_h, lse)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity for Triton
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        # Constants from the original code
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128
        gqa_ratio = num_qo_heads // num_kv_heads  # 4

        device = q.device
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch one program per (b, h)
        grid = (batch_size, num_qo_heads)
        per_bh_attention_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, output, lse,
            sm_scale,
            B=batch_size, H=num_qo_heads, D=head_dim, K=num_kv_heads, gqa_ratio=gqa_ratio, MAX_TOKENS=1024,
            num_warps=2, num_stages=1,
        )
        return output, lse


def run(*args):
    return ModelNew()(*args)
