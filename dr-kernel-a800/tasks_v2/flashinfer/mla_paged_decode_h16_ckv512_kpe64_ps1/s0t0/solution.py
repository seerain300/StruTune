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

    # Read token range for this batch
    page_beg = tl.load(kv_indptr_ptr + b)         # int32
    page_end = tl.load(kv_indptr_ptr + b + 1)     # int32
    L_tokens = page_end - page_beg

    ln2 = 0.6931471805599453  # 1 / log(2)

    # Process each head
    for h in range(NUM_QO_HEADS):
        # Base offsets for q vectors
        qn_base = q_nope_ptr + b * num_qo_heads * head_dim_ckv + h * head_dim_ckv
        qp_base = q_pe_ptr   + b * num_qo_heads * head_dim_kpe + h * head_dim_kpe

        # Load q vectors for this head as float32
        qn = tl.load(qn_base + tl.arange(0, HEAD_DIM_CKV), mask=True, other=0.0).to(tl.float32)
        qp = tl.load(qp_base + tl.arange(0, HEAD_DIM_KPE), mask=True, other=0.0).to(tl.float32)

        # Initialize vectors for logits_scaled (scaled logits), lse (logsumexp over tokens), and softmax attention
        logits_scaled = tl.full((MAX_TOKENS,), -float("inf"), dtype=tl.float32)
        # We will compute scaled logits per token (q @ K^T) * sm_scale
        for i in range(MAX_TOKENS):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + (page_beg + i))  # token index into Kc_all/Kp_all
                # Load Kc and Kp for this token; keep as float32
                kc_base = Kc_all_ptr + idx * head_dim_ckv
                kp_base = Kp_all_ptr + idx * head_dim_kpe
                kc = tl.load(kc_base + tl.arange(0, HEAD_DIM_CKV), mask=True, other=0.0).to(tl.float32)
                kp = tl.load(kp_base + tl.arange(0, HEAD_DIM_KPE), mask=True, other=0.0).to(tl.float32)
                # Dot products
                dot1 = tl.sum(qn * kc, axis=0)
                dot2 = tl.sum(qp * kp, axis=0)
                logits_scaled[i] = (dot1 + dot2) * sm_scale
            # else: keep -inf

        # Compute lse from scaled logits: logsumexp(scaled_logits) / ln(2)
        max_scaled = tl.max(logits_scaled, axis=0)
        exps = tl.exp(logits_scaled - max_scaled)
        sum_exps = tl.sum(exps, axis=0)
        lse_val = tl.log(sum_exps) + max_scaled  # logsumexp(scaled_logits)
        lse_val = lse_val / ln2  # divide by ln(2), matching PyTorch's run
        tl.store(lse_ptr + b * NUM_QO_HEADS + h, lse_val)

        # Compute softmax on scaled logits
        # attn[i] = exp(scaled_logits[i] - lse_val) since lse_val = log(sum(exp(scaled))) / ln(2)
        # We need sum_exps already; lse_val = log(sum_exps) / ln(2)
        # So exp(scaled_logits[i] - lse_val) = exp(scaled_i - log(sum_exps) / ln2)
        # Implement attn scaling
        # Note: Triton doesn't have vector indexing like x[i]; use exps_scaled and divide by sum_exps
        # Recompute scaled logits relative to lse_val to get proper attention weights:
        # However, we already have sum_exps and lse_val. attn = exp(scaled - lse_val)
        attn = tl.exp(logits_scaled - lse_val)

        # Output: out = attn @ Kc
        out_vec = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
        for i in range(MAX_TOKENS):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + (page_beg + i))
                kc_base = Kc_all_ptr + idx * head_dim_ckv
                kc = tl.load(kc_base + tl.arange(0, HEAD_DIM_CKV), mask=True, other=0.0).to(tl.float32)
                out_vec += attn[i] * kc

        # Store output[b, h, :]
        out_base = output_ptr + b * (NUM_QO_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV
        tl.store(out_base + tl.arange(0, HEAD_DIM_CKV), out_vec)

# Host: ModelNew.forward
class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "Inputs must be CUDA tensors for Triton execution."
        # Optionally assert


def run(*args):
    return ModelNew()(*args)
