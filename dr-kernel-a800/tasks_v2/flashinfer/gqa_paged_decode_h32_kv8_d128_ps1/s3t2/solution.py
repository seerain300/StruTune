import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_gqa_kernel(
    q_ptr,          # *bfloat16, shape [B, H, D]
    k_ptr,          # *float32, shape [N, D] (squeezed from [P,1,N,D] on host)
    v_ptr,          # *float32, shape [N, D] (squeezed from [P,1,N,D] on host)
    kv_indptr_ptr,  # *int32, shape [B+1]
    kv_indices_ptr, # *int32, shape [num_kv_indices]
    out_ptr,        # *bfloat16, shape [B, H, D]
    lse_ptr,        # *float32, shape [B, H]
    sm_scale,       # float32 scalar
    B, H, D, N,     # int32 scalars
    gqa_ratio,      # int32 scalar = H // N (e.g., 4)
):
    # Grid: (B, H)
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Compute token range for this batch
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    num_tokens = end - start  # number of tokens for batch b

    # GQA mapping from query head to KV head
    kvh = h // gqa_ratio

    # Initialize LSE accumulators
    max_logit = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.full((), 0.0, dtype=tl.float32)

    # First pass: compute max over tokens for numerical stability
    t = 0
    while t < num_tokens:
        tok_idx = tl.load(kv_indices_ptr + start + t).to(tl.int32)

        # Load q vector for this (b, h): q layout [B, H, D] contiguous
        q_base = q_ptr + b * (H * D)
        q_vec = tl.load(q_base + h * D, mask=(h < H), other=0.0).to(tl.float32)  # [D] float32

        # Load k_vec and v_vec for this token and kvh head from k_ptr/v_ptr of shape [N, D]
        # Offsets: kvh * D + tok_idx * D + d where d in [0..D-1]
        d = tl.arange(0, D)  # D=128
        k_offsets = kvh * D + tok_idx * D + d
        v_offsets = kvh * D + tok_idx * D + d
        k_vec = tl.load(k_ptr + k_offsets)  # [D] float32
        v_vec = tl.load(v_ptr + v_offsets)  # [D] float32

        # Compute logits_scaled = q·k * sm_scale
        logits_scalar = tl.dot(q_vec, k_vec) * sm_scale  # scalar float32

        # Update max and sum_exp
        max_logit = tl.maximum(max_logit, logits_scalar)
        sum_exp += tl.exp(logits_scalar - max_logit)
        t += 1

    # Compute LSE: base-2 logsumexp
    lse_val = max_logit + tl.log(sum_exp) / tl.log(2.0)
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute output[b, h, :] = sum_t attn[t] * v[t]
    out_base = out_ptr + b * (H * D) + h * D
    acc = tl.zeros((D,), dtype=tl.float32)

    t = 0
    while t < num_tokens:
        tok_idx = tl.load(kv_indices_ptr + start + t).to(tl.int32)
        q_base = q_ptr + b * (H * D)
        q_vec = tl.load(q_base + h * D, mask=(h < H), other=0.0).to(tl.float32)  # [D]
        d = tl.arange(0, D)
        k_offsets = kvh * D + tok_idx * D + d
        v_offsets = kvh * D + tok_idx * D + d
        k_vec = tl.load(k_ptr + k_offsets)
        v_vec = tl.load(v_ptr + v_offsets)

        logits_scalar = tl.dot(q_vec, k_vec) * sm_scale
        attn = tl.exp(logits_scalar - max_logit) / sum_exp
        acc += attn * v_vec
        t += 1

    # Store output vector for this head
    tl.store(out_base, acc.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All inputs must be CUDA tensors for Triton."
        assert q.dtype == torch.bfloat16 and k_cache.dtype == torch.bfloat16 and v_cache.dtype == torch.bfloat16, \
            "q, k_cache, v_cache must be bfloat16 tensors."

        B, H, D = q.shape
        # GQA config: H=32, N=8
        assert H == 32, "num_qo_heads must be 32"
        assert D == 128, "head_dim must be 128"

        # Squeeze dim-1 (P) to [N, D] and convert to float32 for Triton
        k_squeezed = k_cache.squeeze(1).contiguous().to(torch.float32)  # [N, D]
        v_squeezed = v_cache.squeeze(1).contiguous().to(torch.float32)  # [N, D]

        # Output and LSE allocation
        output = torch.empty((B, H, D), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        gqa_ratio = H // 8  # 4

        attention_gqa_kernel[grid](
            q, k_squeezed, v_squeezed, kv_indptr, kv_indices, output, lse, sm_scale,
            B, H, D, 8, gqa_ratio,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
