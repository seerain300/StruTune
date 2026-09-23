import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def compute_sumexp_kernel(
    q_ptr,            # *bf16, shape [B, H, D]
    k_ptr,            # *bf16, shape [Np, 1, K, D] but we only index via kv_indices/indptr
    v_ptr,            # *bf16, same shape as k_ptr
    indptr_ptr,       # *int32, shape [B+1]
    idx_ptr,          # *int32, shape [Np] (cached token indices)
    lse_ptr,          # *float32, shape [B, H]
    sm_scale,         # float32 scalar
    B: tl.constexpr,  # batch size
    H: tl.constexpr,  # num query heads
    D: tl.constexpr,  # head_dim (128)
    K: tl.constexpr,  # num kv heads (8)
    gqa_ratio: tl.constexpr,  # H // K (4 here)
    MAX_TOKENS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    start = tl.load(indptr_ptr + b).to(tl.int32)
    end = tl.load(indptr_ptr + b + 1).to(tl.int32)
    num_tokens = end - start

    kv_head = h // gqa_ratio  # in [0, K-1]

    # Load q vector for this head and cast to fp32
    q_offset = b * H * D + h * D
    q_vec = tl.load(q_ptr + q_offset).to(tl.float32)  # [D]

    # Accumulate sum_exp over tokens
    sum_exp = 0.0  # scalar fp32
    for t in range(MAX_TOKENS):
        if t >= num_tokens:
            break
        idx = tl.load(idx_ptr + start + t).to(tl.int32)
        k_offset = idx * (K * D) + kv_head * D
        k_vec = tl.load(k_ptr + k_offset + tl.arange(0, D)).to(tl.float32)  # [D]
        dot = tl.sum(q_vec * k_vec, axis=0)  # scalar
        sum_exp += tl.exp(dot * sm_scale)

    lse_b2 = tl.log(sum_exp) / 0.6931471805599453  # log2(sum_exp) = log(sum_exp) / ln(2)
    tl.store(lse_ptr + b * H + h, lse_b2)


@triton.jit
def accumulate_output_kernel(
    q_ptr,            # *bf16, shape [B, H, D]
    k_ptr,            # *bf16, shape [Np, 1, K, D]
    v_ptr,            # *bf16, shape [Np, 1, K, D]
    indptr_ptr,       # *int32, shape [B+1]
    idx_ptr,          # *int32, shape [Np]
    lse_ptr,          # *float32, shape [B, H]
    out_ptr,          # *bf16, shape [B, H, D]
    sm_scale,         # float32 scalar
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    gqa_ratio: tl.constexpr,
    MAX_TOKENS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    start = tl.load(indptr_ptr + b).to(tl.int32)
    end = tl.load(indptr_ptr + b + 1).to(tl.int32)
    num_tokens = end - start

    kv_head = h // gqa_ratio

    q_offset = b * H * D + h * D
    q_vec = tl.load(q_ptr + q_offset).to(tl.float32)  # [D]

    lse_b2 = tl.load(lse_ptr + b * H + h)  # scalar fp32

    out_offset = b * H * D + h * D
    out_vec = tl.zeros([D], dtype=tl.float32)

    for t in range(MAX_TOKENS):
        if t >= num_tokens:
            break
        idx = tl.load(idx_ptr + start + t).to(tl.int32)
        k_offset = idx * (K * D) + kv_head * D
        v_offset = idx * (K * D) + kv_head * D

        k_vec = tl.load(k_ptr + k_offset + tl.arange(0, D)).to(tl.float32)  # [D]
        v_vec = tl.load(v_ptr + v_offset + tl.arange(0, D)).to(tl.float32)  # [D]

        dot = tl.sum(q_vec * k_vec, axis=0)  # scalar
        attn = tl.exp((dot - lse_b2) * sm_scale)  # scalar
        out_vec += attn * v_vec

    tl.store(out_ptr + out_offset + tl.arange(0, D), out_vec.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # If Triton not available or not CUDA, use pure PyTorch (robustness fallback)
        if (not TRITON_AVAILABLE) or (q.device.type != "cuda"):
            batch_size, num_qo_heads, head_dim = q.shape
            _, _, num_kv_heads, _ = k_cache.shape
            assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

            output = torch.empty((batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device)
            lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q.device)

            gqa_ratio = num_qo_heads // num_kv_heads
            for b in range(batch_size):
                start = int(kv_indptr[b].item())
                end = int(kv_indptr[b + 1].item())
                num_tokens = end - start
                for h in range(num_qo_heads):
                    kv_head = h // gqa_ratio
                    qh = q[b, h]  # [head_dim]
                    # Use only the tokens in this batch
                    token_idx = torch.arange(num_tokens, device=q.device)
                    k_rows = k_cache.squeeze(1)[:, kv_head]  # [num_tokens, head_dim]
                    v_rows = v_cache.squeeze(1)[:, kv_head]  # [num_tokens, head_dim]
                    k_batch = k_rows[token_idx]  # [num_tokens, head_dim]
                    v_batch = v_rows[token_idx]  # [num_tokens, head_dim]
                    logits = torch.matmul(qh, k_batch.transpose(0, 1))  # [num_tokens]
                    logits_scaled = logits * sm_scale
                    lse[b, h] = torch.logsumexp(logits_scaled, dim=0) / math.log(2.0)
                    attn = torch.softmax(logits_scaled, dim=0)  # [num_tokens]
                    out_head = torch.matmul(attn, v_batch.transpose(0, 1))  # [head_dim]
                    output[b, h] = out_head.to(torch.bfloat16)
            return output, lse

        # Triton path
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

        # Launch compute sumexp kernel: grid over (b, h)
        grid = (batch_size * num_qo_heads,)
        compute_sumexp_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, lse, sm_scale,
            B=batch_size, H=num_qo_heads, D=head_dim, K=num_kv_heads,
            gqa_ratio=num_qo_heads // num_kv_heads, MAX_TOKENS=1024,
            num_warps=2, num_stages=2
        )

        # Launch accumulate output kernel
        accumulate_output_kernel[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, lse, output, sm_scale,
            B=batch_size, H=num_qo_heads, D=head_dim, K=num_kv_heads,
            gqa_ratio=num_qo_heads // num_kv_heads, MAX_TOKENS=1024,
            num_warps=2, num_stages=2
        )
        return output, lse


def run(*args):
    return ModelNew()(*args)
