import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute logits_scaled[t, h, k] = (q_nope[t, h] · Kc_cache[tok_idx, :]) + (q_pe[t, h] · Kp_cache[tok_idx, :])
# We will not precompute Kc_all/Kp_all; instead, for each k, we decode tok_idx and load from the original caches.
@triton.jit
def _compute_logits_from_indices_kernel(
    q_nope_ptr,         # *float32 [num_tokens, num_qo_heads, head_dim_ckv]
    q_pe_ptr,           # *float32 [num_tokens, num_qo_heads, head_dim_kpe]
    ckv_cache_ptr,      # *float32 [num_pages, page_size, head_dim_ckv]
    kpe_cache_ptr,      # *float32 [num_pages, page_size, head_dim_kpe]
    sparse_indices_ptr, # *int32   [num_tokens, topk]
    out_ptr,            # *float32 [num_tokens, num_qo_heads, topk]
    # dimensions
    num_tokens,         # int32
    num_qo_heads,       # int32
    head_dim_ckv,       # int32 (e.g., 512)
    head_dim_kpe,       # int32 (e.g., 64)
    topk,               # int32 (e.g., 2048)
    # reduction block sizes
    BLOCK_Q: tl.constexpr,   # 128
    BLOCK_KC: tl.constexpr,  # 128
    BLOCK_KP: tl.constexpr,  # 64
):
    # Grid: (num_tokens * num_qo_heads, topk)
    pid_m = tl.program_id(0)  # over tokens * heads
    k = tl.program_id(1)      # over topk indices

    # Map pid_m -> (t, h)
    t = pid_m // num_qo_heads
    h = pid_m % num_qo_heads
    if t >= num_tokens:
        return

    # Load idx from sparse_indices[t, k]
    idx = tl.load(sparse_indices_ptr + t * topk + k)  # int32
    # Valid if idx != -1
    is_valid = idx != -1

    # Load q vectors
    qn = tl.load(q_nope_ptr + t * (num_qo_heads * head_dim_ckv) + h * head_dim_ckv + tl.arange(0, BLOCK_Q),
                 mask=tl.arange(0, BLOCK_Q) < head_dim_ckv, other=0.0).to(tl.float32)
    qp = tl.load(q_pe_ptr + t * (num_qo_heads * head_dim_kpe) + h * head_dim_kpe + tl.arange(0, BLOCK_KP),
                 mask=tl.arange(0, BLOCK_KP) < head_dim_kpe, other=0.0).to(tl.float32)

    # Compute tok_idx for ckv/kpe: tok_idx = idx if valid else 0 (dummy)
    tok_idx = tl.where(is_valid, idx, 0)

    # Compute dot products only if valid; else contribute 0
    contrib1 = 0.0
    contrib2 = 0.0

    # Dot q_nope[t, h, :] · ckv_cache[tok_idx, :]
    for off in range(0, head_dim_ckv, BLOCK_Q):
        qn_chunk = tl.load(q_nope_ptr + t * (num_qo_heads * head_dim_ckv) + h * head_dim_ckv + off + tl.arange(0, BLOCK_Q),
                           mask=tl.arange(0, BLOCK_Q) < (head_dim_ckv - off), other=0.0).to(tl.float32)
        Kc_chunk = tl.load(ckv_cache_ptr + tok_idx * (page_size * head_dim_ckv) + off + tl.arange(0, BLOCK_Q),
                           mask=tl.arange(0, BLOCK_Q) < (head_dim_ckv - off), other=0.0).to(tl.float32)
        # Zero out when invalid
        qn_chunk = tl.where(is_valid, qn_chunk, 0.0)
        Kc_chunk = tl.where(is_valid, Kc_chunk, 0.0)
        contrib1 += tl.sum(qn_chunk * Kc_chunk, axis=0)

    # Dot q_pe[t, h, :] · kpe_cache[tok_idx, :]
    for off in range(0, head_dim_kpe, BLOCK_KP):
        qp_chunk = tl.load(q_pe_ptr + t * (num_qo_heads * head_dim_kpe) + h * head_dim_kpe + off + tl.arange(0, BLOCK_KP),
                           mask=tl.arange(0, BLOCK_KP) < (head_dim_kpe - off), other=0.0).to(tl.float32)
        Kp_chunk = tl.load(kpe_cache_ptr + tok_idx * (page_size * head_dim_kpe) + off + tl.arange(0, BLOCK_KP),
                           mask=tl.arange(0, BLOCK_KP) < (head_dim_kpe - off), other=0.0).to(tl.float32)
        # Zero out when invalid
        qp_chunk = tl.where(is_valid, qp_chunk, 0.0)
        Kp_chunk = tl.where(is_valid, Kp_chunk, 0.0)
        contrib2 += tl.sum(qp_chunk * Kp_chunk, axis=0)

    # Store logits_scaled[t, h, k] only if valid; else store 0.0
    out_val = contrib1 + contrib2
    out_val = tl.where(is_valid, out_val, 0.0)
    tl.store(out_ptr + t * (num_qo_heads * topk) + h * topk + k, out_val, mask=is_valid)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        """
        Inputs:
          q_nope: [num_tokens, num_qo_heads, head_dim_ckv], bfloat16
          q_pe:   [num_tokens, num_qo_heads, head_dim_kpe], bfloat16
          ckv_cache: [num_pages, page_size, head_dim_ckv], bfloat16
          kpe_cache: [num_pages, page_size, head_dim_kpe], bfloat16
          sparse_indices: [num_tokens, topk], int32 (can contain -1 as padding)
          sm_scale: float32 scalar
        Outputs:
          output: [num_tokens, num_qo_heads, head_dim_ckv], bfloat16
          lse:    [num_tokens, num_qo_heads], float32
        """
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All inputs must be CUDA tensors"
        device = q_nope.device

        # Ensure dtypes and contiguity for Triton
        q_nope_f32 = q_nope.contiguous().to(torch.float32)
        q_pe_f32 = q_pe.contiguous().to(torch.float32)
        ckv_cache_f32 = ckv_cache.contiguous().to(torch.float32)
        kpe_cache_f32 = kpe_cache.contiguous().to(torch.float32)
        sparse_indices_i32 = sparse_indices.contiguous().to(torch.int32)

        num_tokens, num_qo_heads, head_dim_ckv = q_nope_f32.shape
        num_pages, page_size, _ = ckv_cache_f32.shape
        assert num_qo_heads == 16 and head_dim_ckv == 512 and head_dim_kpe == 64 and page_size == 64, "Fixed dims expected"

        # Allocate logits buffer [num_tokens, num_qo


def run(*args):
    return ModelNew()(*args)
