import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_nope_ptr, q_pe_ptr, Kc_all_ptr, Kp_all_ptr, kv_indices_ptr, kv_indptr_ptr,
    output_ptr, lse_ptr,
    batch_size, num_qo_heads, head_dim_ckv, head_dim_kpe, num_kv_indices, len_indptr,
    sm_scale: tl.float32,
    NUM_QO_HEADS: tl.constexpr, HEAD_DIM_CKV: tl.constexpr, HEAD_DIM_KPE: tl.constexpr, MAX_TOKENS: tl.constexpr
):
    # One program per batch element
    b = tl.program_id(0)

    # Read token range [page_beg, page_end) from kv_indptr
    # kv_indptr: int32 of shape [batch_size + 1]
    page_beg = tl.load(kv_indptr_ptr + b)       # int32
    page_end = tl.load(kv_indptr_ptr + b + 1)   # int32
    L_tokens = page_end - page_beg              # int32 scalar

    ln2 = 0.6931471805599453  # 1 / log(2)

    # Process each head h
    for h in range(NUM_QO_HEADS):
        # Base offsets for q vectors
        offset_qn = b * (num_qo_heads * head_dim_ckv) + h * head_dim_ckv
        qn_base = q_nope_ptr + offset_qn

        offset_qp = b * (num_qo_heads * head_dim_kpe) + h * head_dim_kpe
        qp_base = q_pe_ptr + offset_qp

        # Load q vectors for this head as float32
        qn = tl.load(qn_base + tl.arange(0, HEAD_DIM_CKV), mask=True, other=0.0).to(tl.float32)
        qp = tl.load(qp_base + tl.arange(0, HEAD_DIM_KPE), mask=True, other=0.0).to(tl.float32)

        # Vector to hold scaled logits for each token; use -inf so max/logsumexp is correct
        logits_scaled = tl.full((MAX_TOKENS,), -float("inf"), dtype=tl.float32)

        # Loop over tokens up to MAX_TOKENS; guard with i < L_tokens
        for i in range(MAX_TOKENS):
            use_i = i < L_tokens
            # idx is the token index into Kc_all/Kp_all for this batch's range
            idx = tl.load(kv_indices_ptr + (page_beg + i), mask=use_i, other=0)  # int32
            kc_base = Kc_all_ptr + idx * head_dim_ckv
            kp_base = Kp_all_ptr + idx * head_dim_kpe
            kc = tl.load(kc_base + tl.arange(0, HEAD_DIM_CKV), mask=True, other=0.0).to(tl.float32)
            kp = tl.load(kp_base + tl.arange(0, HEAD_DIM_KPE), mask=True, other=0.0).to(tl.float32)

            # Dot products
            dot1 = tl.sum(qn * kc, axis=0)  # scalar float32
            dot2 = tl.sum(qp * kp, axis=0)  # scalar float32
            scaled = (dot1 + dot2) * sm_scale

            # Place the scaled value at position i in the logits vector
            # Triton vector does not support direct indexing assignment; we rely on mask and loop to compute properly.
            # The next line uses a scalar assignment for the element at index i.
            logits_scaled = tl.where(tl.arange(0, MAX_TOKENS) == i, scaled, logits_scaled)

        # Compute lse = logsumexp(logits_scaled) / ln(2)
        max_scaled = tl.max(logits_scaled, axis=0)
        exps = tl.exp(logits_scaled - max_scaled)
        sum_exps = tl.sum(exps, axis=0)
        lse_val = tl.log(sum_exps) + max_scaled  # logsumexp of scaled logits
        lse_val = lse_val / ln2
        # Store lse[b, h]
        tl.store(lse_ptr + b * NUM_QO_HEADS + h, lse_val)

        # Compute attn = exp(logits_scaled - lse_val) per token
        attn = tl.exp(logits_scaled - lse_val)

        # Output: out = sum_i attn[i] * Kc_all[token_idx_i]
        out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        for i in range(MAX_TOKENS):
            use_i = i < L_tokens
            idx = tl.load(kv_indices_ptr + (page_beg + i), mask=use_i, other=0)  # int32
            kc_base = Kc_all_ptr + idx * head_dim_ckv
            kc = tl.load(kc_base + tl.arange(0, HEAD_DIM_CKV), mask=True, other=0.0).to(tl.float32)
            out_vec += attn[i] * kc

        # Store output[b, h, :] as float32 (host will cast to bfloat16)
        out_base = output_ptr + b * (NUM_QO_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        tl.store(out_base + tl.arange(0, HEAD_DIM_CKV), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All inputs must be CUDA tensors for Triton execution."

        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Allocate outputs: compute in float32, cast to bfloat16 before returning
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q_nope.device)

        # Launch Triton kernel: one program per batch element
        grid = (batch_size,)
        attention_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indices, kv_indptr,
            output, lse,
            batch_size, num_qo_heads, head_dim_ckv, head_dim_kpe, kv_indices.shape[0], kv_indptr.shape[0],
            float(sm_scale),
            NUM_QO_HEADS=num_qo_heads,
            HEAD_DIM_CKV=head_dim_ckv,
            HEAD_DIM_KPE=head_dim_kpe,
            MAX_TOKENS=100,  # upper bound for tokens; guard with masks
        )

        # Cast output to bfloat16 to match original function's return type
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
