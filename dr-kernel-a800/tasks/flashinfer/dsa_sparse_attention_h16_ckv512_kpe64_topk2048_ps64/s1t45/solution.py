import torch
import math
import triton
import triton.language as tl


# Kernel: compute logits_scaled[t, h, k] for valid k only, write to out_ptr
# We vectorize across token/head (M = num_tokens * num_qo_heads) and across chunks of topk.
@triton.jit
def _compute_logits_kernel(
    q_nope_ptr,          # *float32, [num_tokens, num_qo_heads, head_dim_ckv]
    q_pe_ptr,            # *float32, [num_tokens, num_qo_heads, head_dim_kpe]
    Kc_all_ptr,          # *float32, [total_kv_tokens, head_dim_ckv]
    Kp_all_ptr,          # *float32, [total_kv_tokens, head_dim_kpe]
    indices_ptr,         # *int32,   [num_tokens, topk]
    out_ptr,             # *float32, [num_tokens, num_qo_heads, topk]
    num_tokens, num_qo_heads, topk,
    head_dim_ckv, head_dim_kpe,
    BLOCK_K: tl.constexpr,
):
    # Grid: (M, ceil(topk / BLOCK_K))
    pid_m = tl.program_id(0)  # over tokens*heads
    pid_k = tl.program_id(1)  # over chunks of topk

    # Derive token and head
    t = pid_m // num_qo_heads
    h = pid_m % num_qo_heads

    # Compute offsets for this chunk
    k0 = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k0 < topk

    # Load indices for this token chunk
    indices = tl.load(indices_ptr + t * topk + k0, mask=mask_k, other=-1).to(tl.int32)
    valid_mask = indices != -1
    # Compute K row offsets in flattened K arrays
    Kc_offsets = indices * head_dim_ckv
    Kp_offsets = indices * head_dim_kpe

    # Load q_nope row for this token and head
    qn_row_ptr = q_nope_ptr + t * (num_qo_heads * head_dim_ckv) + h * head_dim_ckv
    qn = tl.load(qn_row_ptr, mask=mask_k, other=0.0)  # [BLOCK_K], float32

    # Load q_pe row for this token and head
    qp_row_ptr = q_pe_ptr + t * (num_qo_heads * head_dim_kpe) + h * head_dim_kpe
    qp = tl.load(qp_row_ptr, mask=mask_k, other=0.0)  # [BLOCK_K], float32

    # Prepare accumulators
    contrib1 = tl.zeros([BLOCK_K], dtype=tl.float32)
    contrib2 = tl.zeros([BLOCK_K], dtype=tl.float32)

    # Loop over d in [0, head_dim_ckv) in chunks for better vectorization
    for d0 in range(0, head_dim_ckv, 32):
        d_off = d0 + tl.arange(0, 32)
        d_mask = d_off < head_dim_ckv
        qn_chunk = qn[:, None] * tl.load(Kc_all_ptr + Kc_offsets[:, None] + d_off[None, :], mask=valid_mask[:, None] & d_mask[None, :], other=0.0)
        contrib1 += tl.sum(qn_chunk, axis=1)

    # Loop over e in [0, head_dim_kpe) in chunks for better vectorization
    for e0 in range(0, head_dim_kpe, 32):
        e_off = e0 + tl.arange(0, 32)
        e_mask = e_off < head_dim_kpe
        qp_chunk = qp[:, None] * tl.load(Kp_all_ptr + Kp_offsets[:, None] + e_off[None, :], mask=valid_mask[:, None] & e_mask[None, :], other=0.0)
        contrib2 += tl.sum(qp_chunk, axis=1)

    # Compute logits_scaled: (contrib1 + contrib2) * sm_scale, but store 0 for invalid
    sm_scale = 1.0  # scalar, matches original default
    logits = contrib1 + contrib2
    out_vals = tl.where(valid_mask, logits * sm_scale, 0.0)

    # Store to out_ptr
    tl.store(out_ptr + t * (num_qo_heads * topk) + h * topk + k0, out_vals, mask=mask_k)


# Kernel: compute lse[t, h] = log(sum_k exp(logits_scaled[t,h,k] * sm_scale)) / ln(2)
# We use a two-pass reduction over chunks of topk: first compute max, then sum exp(x - max).
@triton.jit
def _compute_lse_kernel(
    logits_ptr,          # *float32, [num_tokens, num_qo_heads, topk]
    lse_ptr,             # *float32, [num_tokens, num_qo_heads]
    num_tokens, num_qo_heads, topk,
    BLOCK_K: tl.constexpr,
):
    # Grid: (num_tokens, num_qo_heads)
    t = tl.program_id(0)
    h = tl.program_id(1)

    # Pass 1: compute max over valid entries
    max_val = tl.full((), -float("inf"), tl.float32)
    for k0 in range(0, topk, BLOCK_K):
        idx = k0 + tl.arange(0, BLOCK_K)
        mask = idx < topk
        vals = tl.load(logits_ptr + t * (num_qo_heads * topk) + h * topk + idx, mask=mask, other=-float("inf"))
        # max over this chunk
        max_chunk = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, max_chunk)

    # Pass 2: compute sum of exp(vals - max_val)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, topk, BLOCK_K):
        idx = k0 + tl.arange(0, BLOCK_K)
        mask = idx < topk
        vals = tl.load(logits_ptr + t * (num_qo_heads * topk) + h * topk + idx, mask=mask, other=-float("inf"))
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(exp_vals, axis=0)

    lse = tl.log(sum_exp) / math.log(2.0)
    tl.store(lse_ptr + t * num_qo_heads + h, lse)


# Kernel: compute output[t, h, :] from logits and lse, and Kc rows via indices
# Output accumulates [head_dim_ckv] vector per (t,h).
@triton.jit
def _compute_output_kernel(
    logits_ptr,          # *float32, [num_tokens, num_qo_heads, topk]
    lse_ptr,             # *float32, [num_tokens, num_qo_heads]
    Kc_all_ptr,          # *float32, [total_kv_tokens, head_dim_ckv]
    indices_ptr,         # *int32,   [num_tokens, topk]
    output_ptr,          # *bfloat16, [num_tokens, num_qo_heads, head_dim_ckv]
    num_tokens, num_qo_heads, topk,
    head_dim_ckv,        # 512
    BLOCK_K: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    # Load lse for this (t,h)
    lse_t_h = tl.load(lse_ptr + t * num_qo_heads + h)

    # Accumulator for output
    out_accum = tl.zeros([head_dim_ckv], dtype=tl.float32)

    # Loop over k in chunks
    for k0 in range(0, topk, BLOCK_K):
        idx = k0 + tl.arange(0, BLOCK_K)
        mask_k = idx < topk
        indices = tl.load(indices_ptr + t * topk + idx, mask=mask_k, other=-1).to(tl.int32)
        valid_mask = indices != -1

        # Load logits_scaled[t, h, idx]
        logits_vec = tl.load(logits_ptr + t * (num_qo_heads * topk) + h * topk + idx, mask=mask_k, other=0.0)

        # Compute attn = exp(logits_scaled - lse) for valid entries
        attn = tl.exp(logits_vec * 1.0 - lse_t_h)  # sm_scale default 1.0
        attn = tl.where(valid_mask, attn, 0.0)

        # For each valid k, accumulate attn[k] * Kc_all[indices[k], :]
        for j in range(0, BLOCK_K):
            # Scalar checks
            if mask_k[j]:
                k = idx[j]
                row_idx = indices[j]
                # if invalid, skip
                if valid_mask[j]:
                    # add attn[k] * Kc_all[row_idx, :]
                    for d0 in range(0, head_dim_ckv, 64):
                        d_off = d0 + tl.arange(0, 64)
                        d_mask = d_off < head_dim_ckv
                        Kc_chunk = tl.load(Kc_all_ptr + row_idx * head_dim_ckv + d_off, mask=d_mask, other=0.0)
                        out_accum += attn[j] * Kc_chunk

    # Store final output vector (bfloat16)
    tl.store(output_ptr + t * (num_qo_heads * head_dim_ckv) + h * head_dim_ckv, out_accum.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tunable meta-parameters
        self.BLOCK_K_logits = 256  # chunk size for logits computation
        self.BLOCK_K_reduce = 1024  # chunk size for reductions/loop
        self.BLOCK_K_out = 256  # chunk size for output accumulation

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        """
        Triton-only implementation:
          - Computes logits_scaled per (token, head, valid k) in Triton
          - Computes lse per (token, head) in Triton
          - Computes final output in Triton
        Inputs:
          q_nope: [num_tokens, num_qo_heads, 512], bfloat16
          q_pe:   [num_tokens, num_qo_heads, 64], bfloat16
          ckv_cache: [num_pages, 64, 512], bfloat16
          kpe_cache: [num_pages, 64, 64], bfloat16
          sparse_indices: [num_tokens, 2048], int32 (can contain -1 as padding)
          sm_scale: float32 scalar (default 1.0 in original)
        Outputs:
          output: [num_tokens, num_qo_heads, 512], bfloat16
          lse: [num_tokens, num_qo_heads], float32
        """
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All inputs must be CUDA tensors"
        device = q_nope.device

        # Ensure contiguity and dtypes
        q_nope_f32 = q_nope.contiguous().to(torch.float32)
        q_pe_f32 = q_pe.contiguous().to(torch.float32)
        ckv_cache_f32 = ckv_cache.contiguous().to(torch.float32)
        kpe_cache_f32 = kpe_cache.contiguous().to(torch.float32)
        sparse_indices_i32 = sparse_indices.contiguous().to(torch.int32)

        num_tokens, num_qo_heads, head_dim_ckv = q_nope_f32.shape
        num_pages, page_size, _ = ckv_cache_f32.shape
        # Reference asserts: num_qo_heads == 16, head_dim_ckv == 512, head_dim_kpe == 64, page_size == 64, topk == 2048
        assert num_qo_heads == 16 and head_dim_ckv == 512 and kpe_cache_f32.shape[-1] == 64 and page_size == 64
        topk = sparse_indices_i32.shape[-1]
        assert topk == 2048, "topk must be 2048"

        # Flatten K caches to [total_kv_tokens, dim]
        total_kv_tokens = num_pages * page_size
        Kc_all = ckv_cache_f32.reshape(-1, head_dim_ckv)  # [total_kv_tokens, 512]
        Kp_all = kpe_cache_f32.reshape(-1, 64)           # [total_kv_tokens, 64]

        # Output tensors
        logits = torch.empty((num_tokens, num_qo_heads, topk), dtype=torch.float32, device=device)
        lse = torch.empty((num_tokens, num_qo_heads), dtype=torch.float32, device=device)
        output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)

        # Launch 1: compute logits_scaled
        grid_logits = (num_tokens * num_qo_heads, triton.cdiv(topk, self.BLOCK_K_logits))
        _compute_logits_kernel[grid_logits](
            q_nope_f32, q_pe_f32, Kc_all, Kp_all, sparse_indices_i32, logits,
            num_tokens, num_qo_heads, topk, head_dim_ckv, 64,
            BLOCK_K=self.BLOCK_K_logits,
        )

        # Launch 2: compute lse per (t, h)
        grid_lse = (num_tokens, num_qo_heads)
        _compute_lse_kernel[grid_lse](
            logits, lse, num_tokens, num_qo_heads, topk,
            BLOCK_K=self.BLOCK_K_reduce,
        )

        # Launch 3: compute final output
        grid_out = (num_tokens, num_qo_heads)
        _compute_output_kernel[grid_out](
            logits, lse, Kc_all, sparse_indices_i32, output,
            num_tokens, num_qo_heads, topk, head_dim_ckv,
            BLOCK_K=self.BLOCK_K_out,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
