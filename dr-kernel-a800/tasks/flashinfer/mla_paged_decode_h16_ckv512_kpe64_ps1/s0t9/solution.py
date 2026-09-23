import math
import torch

import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_nope_ptr, q_pe_ptr, ckv_cache_ptr, kpe_cache_ptr, kv_indices_ptr, kv_indptr_ptr,
    output_ptr, lse_ptr,
    batch_size, num_qo_heads, head_dim_ckv, head_dim_kpe,
    len_indptr, num_kv_indices, sm_scale,
    NUM_QO_HEADS: tl.constexpr, HEAD_DIM_CKV: tl.constexpr, HEAD_DIM_KPE: tl.constexpr, MAX_TOKENS: tl.constexpr
):
    b = tl.program_id(axis=0)  # one program per batch element

    # Read token range for this batch element
    base = tl.load(kv_indptr_ptr + b)  # int32
    end = tl.load(kv_indptr_ptr + b + 1)  # int32
    L_tokens = end - base  # number of tokens for this batch element

    # Compute ln(2) as a scalar
    ln2 = 0.6931471805599453  # 1 / log(2)

    # Process each head h
    for h in range(NUM_QO_HEADS):
        # Base offsets for q vectors
        offset_qn = b * (NUM_QO_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        qn_base = q_nope_ptr + offset_qn  # pointer to q_nope[b, h, :]
        qn = tl.load(qn_base + tl.arange(0, HEAD_DIM_CKV), mask=True, other=0.0).to(tl.float32)

        offset_qp = b * (NUM_QO_HEADS * HEAD_DIM_KPE) + h * HEAD_DIM_KPE
        qp_base = q_pe_ptr + offset_qp  # pointer to q_pe[b, h, :]
        qp = tl.load(qp_base + tl.arange(0, HEAD_DIM_KPE), mask=True, other=0.0).to(tl.float32)

        # Vector to hold scaled logits for each token; initialize to -inf for stability
        logits_scaled = tl.full((MAX_TOKENS,), -float("inf"), dtype=tl.float32)

        # Loop over tokens up to MAX_TOKENS; guard with i < L_tokens
        for i in range(MAX_TOKENS):
            use_i = i < L_tokens
            # idx is the token index into Kc_all/Kp_all for this batch's range
            idx = tl.load(kv_indices_ptr + (base + i), mask=use_i, other=0)  # int32 scalar

            # Compute pointers to the key vectors
            kc_base = ckv_cache_ptr + idx * HEAD_DIM_CKV
            kp_base = kpe_cache_ptr + idx * HEAD_DIM_KPE

            # Load key vectors for this token position
            kc = tl.load(kc_base + tl.arange(0, HEAD_DIM_CKV), mask=True, other=0.0).to(tl.float32)
            kp = tl.load(kp_base + tl.arange(0, HEAD_DIM_KPE), mask=True, other=0.0).to(tl.float32)

            # Dot products
            dot1 = tl.sum(qn * kc, axis=0)  # scalar float32
            dot2 = tl.sum(qp * kp, axis=0)  # scalar float32
            scaled = (dot1 + dot2) * sm_scale

            # Place the scaled value at position i in the logits vector
            # Triton does not allow direct vector indexing; we reconstruct the vector
            logits_scaled = tl.where(tl.arange(0, MAX_TOKENS) == i, scaled, logits_scaled)

        # Compute lse = logsumexp(logits_scaled) / ln(2)
        max_scaled = tl.max(logits_scaled, axis=0)
        exps = tl.exp(logits_scaled - max_scaled)
        sum_exps = tl.sum(exps, axis=0)
        lse_val = tl.log(sum_exps) + max_scaled  # logsumexp(scaled_logits)
        lse_val = lse_val / ln2  # divide by ln(2), matching PyTorch's run
        tl.store(lse_ptr + b * NUM_QO_HEADS + h, lse_val)

        # Compute attn = exp(scaled_logits - lse_val)
        attn = tl.exp(logits_scaled - lse_val)  # vector [MAX_TOKENS] with mask applied implicitly by logits_scaled

        # Output: out = sum_i attn[i] * Kc_all[i, :]
        out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        for i in range(MAX_TOKENS):
            use_i = i < L_tokens
            idx = tl.load(kv_indices_ptr + (base + i), mask=use_i, other=0)  # int32 scalar
            kc_base = ckv_cache_ptr + idx * HEAD_DIM_CKV
            kc = tl.load(kc_base + tl.arange(0, HEAD_DIM_CKV), mask=True, other=0.0).to(tl.float32)
            contrib = attn[i] * kc  # elementwise multiply
            out_vec += contrib

        # Store output[b, h, :]
        out_base = output_ptr + b * (NUM_QO_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        tl.store(out_base + tl.arange(0, HEAD_DIM_CKV), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self, num_qo_heads=16, head_dim_ckv=512, head_dim_kpe=64, max_tokens=1024):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.head_dim_ckv = head_dim_ckv
        self.head_dim_kpe = head_dim_kpe
        self.max_tokens = max_tokens

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "Inputs must be CUDA tensors for Triton execution."

        batch_size = q_nope.shape[0]
        num_qo_heads = self.num_qo_heads
        head_dim_ckv = self.head_dim_ckv
        head_dim_kpe = self.head_dim_kpe

        # Output is float32 in-kernel; cast to bfloat16 after to match original function's return
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)

        # lse buffer: [batch_size * num_qo_heads] float32
        lse = torch.empty(batch_size * num_qo_heads, dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per batch element
        grid = (batch_size,)
        attention_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indices, kv_indptr,
            output, lse,
            batch_size, num_qo_heads, head_dim_ckv, head_dim_kpe,
            kv_indptr.shape[0], kv_indices.shape[0], float(sm_scale),
            NUM_QO_HEADS=num_qo_heads,
            HEAD_DIM_CKV=head_dim_ckv,
            HEAD_DIM_KPE=head_dim_kpe,
            MAX_TOKENS=self.max_tokens,
        )

        # Cast output to bfloat16 to match original function's return type
        output_bf16 = output.to(torch.bfloat16)
        # Reshape lse to [batch_size, num_qo_heads]
        lse = lse.view(batch_size, num_qo_heads)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
