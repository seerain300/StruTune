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
    q_ptr,            # *bf16, [B, H, D]
    k_ptr,            # *bf16, [Np, 1, K, D]
    v_ptr,            # *bf16, [Np, 1, K, D]
    indptr_ptr,       # *int32, [B+1]
    indices_ptr,      # *int32, [num_tokens_total]
    out_ptr,          # *bf16, [B, H, D]
    lse_ptr,          # *fp32, [B, H]
    sm_scale,         # fp32 scalar (e.g., 1.0 / sqrt(D))
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    gqa_ratio: tl.constexpr,  # H // K == 4
    MAX_TOKENS: tl.constexpr, # upper bound for iteration (e.g., 1024)
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

    # GQA mapping
    kv_head = pid_h // gqa_ratio  # 0..7

    # Pass 1: compute sum_exp = sum(exp(logits_scaled)) across tokens
    sum_exp = 0.0  # fp32
    for t in range(MAX_TOKENS):
        if t >= num_tokens:
            break
        idx = tl.load(indices_ptr + start + t).to(tl.int32)
        # Load k row for this token and head in fp32: [D]
        k_row_ptr = k_ptr + idx * (K * D) + kv_head * D
        k_row = tl.load(k_row_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0).to(tl.float32)
        # Dot product q_vec @ k_row
        dot = 0.0
        for d in range(D):
            dot += q_vec[d] * k_row[d]
        # Scale and accumulate sum_exp
        sum_exp += tl.exp(dot * sm_scale)

    # Compute lse in base-2
    ln2 = 0.6931471805599453  # ln(2)
    lse_base2 = tl.log(sum_exp) / ln2  # fp32 per (b,h)

    # Pass 2: compute output vector and store in bf16
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for t in range(MAX_TOKENS):
        if t >= num_tokens:
            break
        idx = tl.load(indices_ptr + start + t).to(tl.int32)
        k_row_ptr = k_ptr + idx * (K * D) + kv_head * D
        k_row = tl.load(k_row_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0).to(tl.float32)
        v_row_ptr = v_ptr + idx * (K * D) + kv_head * D
        v_row = tl.load(v_row_ptr + tl.arange(0, D), mask=tl.arange(0, D) < D, other=0.0).to(tl.float32)

        # Compute dot again
        dot = 0.0
        for d in range(D):
            dot += q_vec[d] * k_row[d]
        # attn = exp((dot - lse_base2) * sm_scale)
        attn = tl.exp((dot - lse_base2) * sm_scale)
        # accumulate output
        for d in range(D):
            out_vec[d] += attn * v_row[d]

    # Store output vector in bfloat16
    out_offset = pid_b * (H * D) + pid_h * D
    tl.store(out_ptr + out_offset, out_vec.to(tl.bfloat16))

    # Store lse in fp32 (already base-2)
    tl.store(lse_ptr + pid_b * H + pid_h, lse_base2)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # If Triton not available or not on CUDA, fallback to PyTorch
        if not TRITON_AVAILABLE or not q.is_cuda:
            batch_size, num_qo_heads, head_dim = q.shape
            _, _, num_kv_heads, _ = k_cache.shape
            assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

            output = torch.zeros((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device)
            lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=q.device)

            gqa_ratio = num_qo_heads // num_kv_heads
            k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]
            v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [num_pages, num_kv_heads, head_dim]

            for b in range(batch_size):
                start = int(kv_indptr[b].item())
                end = int(kv_indptr[b + 1].item())
                if start >= end:
                    continue
                token_indices = kv_indices[start:end].to(torch.long)
                num_tokens = token_indices.shape[0]
                if num_tokens == 0:
                    continue
                q_batch = q[b].to(torch.float32)  # [num_qo_heads, head_dim]
                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio
                    q_head = q_batch[h]  # [head_dim]
                    k_head = k_cache_flat[token_indices, kv_head]  # [num_tokens, head_dim]
                    v_head = v_cache_flat[token_indices, kv_head]  # [num_tokens, head_dim]
                    logits = torch.matmul(q_head, k_head.T)  # [num_tokens]
                    logits_scaled = logits * sm_scale
                    lse[b, h] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
                    attn = torch.softmax(logits_scaled, dim=-1)  # [num_tokens]
                    out_head = torch.matmul(attn, v_head)  # [head_dim]
                    output[b, h] = out_head.to(torch.bfloat16)
            return output, lse

        # Triton path: ensure contiguity
        q = q.contiguous()
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        batch_size, num_qo_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k_cache.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        device = q.device
        output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Grid: one program per (b, h)
        grid = (batch_size, num_qo_heads)
        fused_gqa_row_simple_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, output, lse,
            sm_scale,
            B=batch_size, H=num_qo_heads, D=head_dim, K=num_kv_heads, gqa_ratio=4, MAX_TOKENS=1024,
            num_warps=2, num_stages=1,
        )
        return output, lse


def run(*args):
    return ModelNew()(*args)
