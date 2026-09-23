import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_gqa_kernel_per_bh(
    q_ptr,          # *bfloat16, [B, H, D]
    k_ptr,          # *float32,  [N, D] (k_cache.squeeze(1).to(float32))
    v_ptr,          # *float32,  [N, D] (v_cache.squeeze(1).to(float32))
    kv_indptr_ptr,  # *int32,    [B+1]
    kv_indices_ptr, # *int32,    [num_kv_indices]
    output_ptr,     # *bfloat16, [B, H, D]
    lse_ptr,        # *float32,  [B, H]
    sm_scale,       # float32 scalar
    B: tl.constexpr,       # batch size (meta)
    H: tl.constexpr,       # num query heads (meta)
    D: tl.constexpr,       # head dim (meta)
    N: tl.constexpr,       # num kv heads (meta, e.g., 8)
    gqa_ratio: tl.constexpr,  # H // N (meta, e.g., 4)
):
    # One program per (b, h)
    b = tl.program_id(0)
    h = tl.program_id(1)

    # GQA mapping: each query head h uses KV head kvh = h // gqa_ratio
    kvh = h // gqa_ratio  # 0..N-1

    # Read start/end indices for this batch b: token range is [start, end)
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    num_tokens = end - start

    # Initialize LSE accumulators
    max_logit = tl.full((), -float("inf"), tl.float32)
    sum_exp = tl.full((), 0.0, tl.float32)

    # First pass: compute max over tokens and sum of exp(logits_scaled) for numerical stability
    t = 0
    while t < num_tokens:
        tok_idx = tl.load(kv_indices_ptr + start + t).to(tl.int32)
        # Load q vector for this (b, h): q is [B, H, D] contiguous; offset = b*(H*D) + h*D
        q_base = q_ptr + b * (H * D) + h * D
        q_vec = tl.load(q_base, mask=(h < H), other=0.0).to(tl.float32)  # [D]

        # Load k_vec and v_vec for kvh and tok_idx from k_ptr and v_ptr (shape [N, D])
        # k_ptr[kvh, tok_idx, :] -> since k_ptr is [N, D], access via kvh * D + tok_idx * D + d
        k_vec = tl.zeros((D,), dtype=tl.float32)
        v_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_off = kvh * D + tok_idx * D + d
            v_off = kvh * D + tok_idx * D + d
            k_vec[d] = tl.load(k_ptr + k_off)
            v_vec[d] = tl.load(v_ptr + v_off)

        # Compute logits_scaled = dot(q_vec, k_vec) * sm_scale
        logits_scaled = tl.sum(q_vec * k_vec, axis=0) * sm_scale  # scalar
        # Update max and sum_exp
        max_logit = tl.maximum(max_logit, logits_scaled)
        # sum_exp += exp(logits_scaled - max_logit) to avoid overflow
        sum_exp += tl.exp(logits_scaled - max_logit)
        t += 1

    # Compute LSE in base-2: lse = max + log(sum_exp) / ln(2)
    ln2 = 0.6931471805599453
    lse_val = max_logit + tl.log(sum_exp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute attention weights and accumulate output
    acc = tl.zeros((D,), dtype=tl.float32)
    t = 0
    while t < num_tokens:
        tok_idx = tl.load(kv_indices_ptr + start + t).to(tl.int32)
        q_base = q_ptr + b * (H * D) + h * D
        q_vec = tl.load(q_base, mask=(h < H), other=0.0).to(tl.float32)  # [D]

        k_vec = tl.zeros((D,), dtype=tl.float32)
        v_vec = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D):
            k_off = kvh * D + tok_idx * D + d
            v_off = kvh * D + tok_idx * D + d
            k_vec[d] = tl.load(k_ptr + k_off)
            v_vec[d] = tl.load(v_ptr + v_off)

        logits_scaled = tl.sum(q_vec * k_vec, axis=0) * sm_scale  # scalar
        attn = tl.exp(logits_scaled - max_logit) / sum_exp
        acc += attn * v_vec
        t += 1

    # Store acc to output[b, h, :]
    out_base = output_ptr + b * (H * D) + h * D
    tl.store(out_base, acc.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        q: [B, H, D], bfloat16
        k_cache, v_cache: [P, 1, N, D], bfloat16 in inputs; squeezed and cast to float32 for compute
        kv_indptr: [B+1], int32
        kv_indices: [num_kv_indices], int32
        sm_scale: float32 scalar
        Returns: output [B, H, D], bfloat16, and lse [B, H], float32 (base-2 logsumexp)
        """
        assert q.is_cuda, "Tensors must be on CUDA device for Triton kernels"
        B, H, D = q.shape
        N = v_cache.shape[2]  # num_kv_heads, expected 8
        assert H % N == 0, "H must be divisible by N (num_kv_heads)"
        gqa_ratio = H // N  # 4 for H=32, N=8

        # Prepare k and v as [N, D] float32 for kernel (squeeze dim=1 from [P, 1, N, D] to [N, D])
        k_squeezed = k_cache.squeeze(1).to(torch.float32).contiguous()  # [N, D]
        v_squeezed = v_cache.squeeze(1).to(torch.float32).contiguous()  # [N, D]

        # Ensure inputs are contiguous and on device
        q = q.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Output and lse buffers
        output = torch.empty((B, H, D), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        attention_gqa_kernel_per_bh[grid](
            q, k_squeezed, v_squeezed, kv_indptr, kv_indices, output, lse, sm_scale,
            B=B, H=H, D=D, N=N, gqa_ratio=gqa_ratio,
            num_warps=4, num_stages=2,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
