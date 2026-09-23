import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def attn_gqa_row_simple_kernel(
    q_ptr,            # *bf16, shape [B, H, D]
    k_ptr,            # *bf16, shape [Np, 1, K, D] (we index via indptr/indices)
    v_ptr,            # *bf16, shape [Np, 1, K, D]
    indptr_ptr,       # *int32, shape [B+1]
    indices_ptr,      # *int32, shape [num_tokens_total]
    out_ptr,          # *bf16, shape [B, H, D]
    lse_ptr,          # *fp32, shape [B, H]
    sm_scale,         # fp32 scalar
    B: tl.constexpr,
    H: tl.constexpr,      # e.g., 32
    D: tl.constexpr,      # e.g., 128
    K: tl.constexpr,      # e.g., 8
    gqa_ratio: tl.constexpr,  # H // K == 4
    MAX_TOKENS: tl.constexpr, # e.g., 1024
):
    pid_b = tl.program_id(0)  # batch index
    pid_h = tl.program_id(1)  # query head index

    # Load q vector for this (b, h) in fp32
    q_offset = pid_b * (H * D) + pid_h * D
    q_vec = tl.load(q_ptr + q_offset).to(tl.float32)  # [D]

    # Load indptr[start, end] for this batch element
    start = tl.load(indptr_ptr + pid_b).to(tl.int32)
    end = tl.load(indptr_ptr + pid_b + 1).to(tl.int32)
    num_tokens = end - start

    # Pass 1: compute sum_exp = sum(exp((q·k) * sm_scale)) across tokens (scalar iterations guarded by num_tokens)
    sum_exp = 0.0
    for t in range(MAX_TOKENS):
        if t >= num_tokens:
            break
        idx = tl.load(indices_ptr + start + t).to(tl.int32)
        kv_head = pid_h // gqa_ratio  # 0..7

        # Load k_row for this token and head (vector of length D) in fp32
        k_row_ptr = k_ptr + idx * (K * D) + kv_head * D
        k_row = tl.load(k_row_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0).to(tl.float32)  # [D]

        # Dot product q_vec · k_row
        dot = 0.0
        d = 0
        while d < D:
            dot += q_vec[d] * k_row[d]
            d += 1

        sum_exp += tl.exp(dot * sm_scale)

    # Compute lse in base-2
    lse_val = tl.log(sum_exp) / 0.6931471805599453  # 1 / ln(2)

    # Prepare output vector out_vec and fill it in the second pass over tokens
    out_vec = tl.zeros((D,), dtype=tl.float32)

    for t in range(MAX_TOKENS):
        if t >= num_tokens:
            break
        idx = tl.load(indices_ptr + start + t).to(tl.int32)
        kv_head = pid_h // gqa_ratio

        # Load k_row for logits
        k_row_ptr_k = k_ptr + idx * (K * D) + kv_head * D
        k_row_k = tl.load(k_row_ptr_k + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0).to(tl.float32)  # [D]
        dot = 0.0
        d = 0
        while d < D:
            dot += q_vec[d] * k_row_k[d]
            d += 1

        # attn = exp((dot - lse_val) * sm_scale)
        attn = tl.exp((dot - lse_val) * sm_scale)

        # Load v_row and accumulate
        v_row_ptr = v_ptr + idx * (K * D) + kv_head * D
        v_row = tl.load(v_row_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0).to(tl.float32)  # [D]
        d2 = 0
        while d2 < D:
            out_vec[d2] += attn * v_row[d2]
            d2 += 1

    # Store output
    out_offset = pid_b * (H * D) + pid_h * D
    tl.store(out_ptr + out_offset, out_vec.to(tl.bfloat16))

    # Store lse in fp32 (base-2)
    lse_offset = pid_b * H + pid_h
    tl.store(lse_ptr + lse_offset, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure contiguity and device
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Shape checks (same as original assumptions)
        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128
        assert kv_indptr.shape[0] == batch_size + 1
        assert kv_indices.numel() == kv_indptr[-1].item()

        device = q.device

        # Output buffers
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, h)
        grid = (batch_size, num_qo_heads)
        attn_gqa_row_simple_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, output, lse,
            sm_scale,
            B=batch_size, H=num_qo_heads, D=head_dim, K=num_kv_heads, gqa_ratio=4, MAX_TOKENS=1024,
            num_warps=2, num_stages=1,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
