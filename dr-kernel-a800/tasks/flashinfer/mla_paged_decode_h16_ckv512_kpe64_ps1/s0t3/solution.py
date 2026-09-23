import torch
import math
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_nope_ptr,     # *bfloat16, [B, H, 512]
    q_pe_ptr,       # *bfloat16, [B, H, 64]
    Kc_all_ptr,     # *bfloat16, [N, 512]
    Kp_all_ptr,     # *bfloat16, [N, 64]
    kv_indices_ptr, # *int32,    [num_kv_indices]
    kv_indptr_ptr,  # *int32,    [len_indptr]
    out_ptr,        # *float32,  [B, H, 512]
    lse_ptr,        # *float32,  [B*H]
    B, H, N512, N64, LEN_IND, NUM_KV_IND,
    sm_scale,       # float32
    NUM_QO_HEADS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,      # 512
    HEAD_DIM_KPE: tl.constexpr,      # 64
    MAX_TOKENS: tl.constexpr,        # e.g., 1024
):
    b = tl.program_id(axis=0)

    # Read token range for this batch element
    base = tl.load(kv_indptr_ptr + b)
    end = tl.load(kv_indptr_ptr + b + 1)
    L_tokens = end - base  # number of tokens for this batch element

    # Process each head
    for h in range(NUM_QO_HEADS):
        # Load query vectors
        offset_qn = b * (H * N512) + h * N512
        qn = tl.load(q_nope_ptr + offset_qn + tl.arange(0, N512), mask=True, other=0.0).to(tl.float32)
        offset_qp = b * (H * N64) + h * N64
        qp = tl.load(q_pe_ptr + offset_qp + tl.arange(0, N64), mask=True, other=0.0).to(tl.float32)

        # Vector of token indices within [0, MAX_TOKENS)
        token_idx = tl.arange(0, MAX_TOKENS)
        token_mask = token_idx < L_tokens

        # Gather token indices for this segment: idx = base + token_idx
        idxs = base + token_idx  # int32

        # Load key blocks for valid tokens
        kc_base = Kc_all_ptr + idxs * N512
        kp_base = Kp_all_ptr + idxs * N64
        kc_block = tl.load(kc_base + tl.arange(0, N512), mask=token_mask, other=0.0).to(tl.float32)  # [MAX_TOKENS, 512]
        kp_block = tl.load(kp_base + tl.arange(0, N64), mask=token_mask, other=0.0).to(tl.float32)  # [MAX_TOKENS, 64]

        # Compute logits_scaled for each token: dot(qn, Kc) + dot(qp, Kp), scaled by sm_scale
        dot1 = tl.sum(qn * kc_block, axis=1)  # [MAX_TOKENS]
        dot2 = tl.sum(qp * kp_block, axis=1)  # [MAX_TOKENS]
        scaled = (dot1 + dot2) * sm_scale
        # Initialize logits_scaled vector; use -inf for masked positions
        logits_scaled = tl.full((MAX_TOKENS,), -float("inf"), dtype=tl.float32)
        logits_scaled = tl.where(token_mask, scaled, logits_scaled)

        # Compute lse = logsumexp(logits_scaled) / ln(2)
        max_scaled = tl.max(logits_scaled, axis=0)
        sum_exps = tl.sum(tl.exp(logits_scaled - max_scaled), axis=0)
        lse_val = tl.log(sum_exps) + max_scaled
        lse_val = lse_val / 0.6931471805599453  # 1 / ln(2)
        tl.store(lse_ptr + b * NUM_QO_HEADS + h, lse_val)

        # Compute attention weights: attn[i] = exp(scaled[i] - lse_val), masked by token_mask
        attn = tl.exp(logits_scaled - lse_val)
        attn = tl.where(token_mask, attn, 0.0)

        # Output: out[b, h, :] = sum_i attn[i] * Kc_all[idxs[i], :]
        out_vec = tl.zeros((N512,), dtype=tl.float32)
        for t in range(MAX_TOKENS):
            use_t = t < L_tokens
            kc = kc_block[t, :]  # [512]
            out_vec += attn[t] * kc

        # Store output vector for this head
        out_base = out_ptr + b * (H * N512) + h * N512
        tl.store(out_base + tl.arange(0, N512), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "Inputs must be CUDA tensors for Triton execution."

        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Output: compute in float32 inside kernel, then cast to bfloat16 to match original
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)

        # lse buffer: [batch_size, num_qo_heads] as flat [batch_size*num_qo_heads]
        lse = torch.empty(batch_size * num_qo_heads, dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per batch element
        grid = (batch_size,)
        attention_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indices, kv_indptr,
            output, lse,
            batch_size, num_qo_heads, head_dim_ckv, head_dim_kpe, kv_indptr.shape[0], kv_indices.shape[0],
            float(sm_scale),
            NUM_QO_HEADS=num_qo_heads,
            HEAD_DIM_CKV=head_dim_ckv,
            HEAD_DIM_KPE=head_dim_kpe,
            MAX_TOKENS=1024,  # upper bound; masks tokens beyond L_tokens
        )

        # Cast output to bfloat16 to match original function's return type
        output_bf16 = output.to(torch.bfloat16)
        # Reshape lse to [batch_size, num_qo_heads]
        lse = lse.view(batch_size, num_qo_heads)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
