import torch
import math
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_nope_ptr, q_pe_ptr, ckv_cache_ptr, kpe_cache_ptr, kv_indices_ptr, kv_indptr_ptr,
    output_ptr, lse_ptr,
    B: tl.constexpr,          # batch_size
    H: tl.constexpr,          # num_qo_heads
    N512: tl.constexpr,       # head_dim_ckv
    N64: tl.constexpr,        # head_dim_kpe
    B_INDPTR: tl.constexpr,   # len(kv_indptr)
    N_INDICES: tl.constexpr,  # num_kv_indices
    sm_scale: tl.float32,
    NUM_QO_HEADS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
):
    # One program per batch element
    b = tl.program_id(axis=0)

    # Read token range for this batch element
    base = tl.load(kv_indptr_ptr + b)  # int32
    end = tl.load(kv_indptr_ptr + b + 1)  # int32
    L_tokens = end - base  # int32

    # Process each head
    for h in range(NUM_QO_HEADS):
        # Load query vectors for this head (as float32)
        offset_qn = b * (H * N512) + h * N512
        qn = tl.load(q_nope_ptr + offset_qn + tl.arange(0, N512), mask=True, other=0.0).to(tl.float32)

        offset_qp = b * (H * N64) + h * N64
        qp = tl.load(q_pe_ptr + offset_qp + tl.arange(0, N64), mask=True, other=0.0).to(tl.float32)

        # Token indices vector [base, base+1, ..., min(end-1, base+MAX_TOKENS-1)]
        token_idx = base + tl.arange(0, MAX_TOKENS)
        mask_tokens = token_idx < end

        # Load token indices (masked beyond end)
        idxs = tl.load(kv_indices_ptr + token_idx, mask=mask_tokens, other=0)  # int32

        # Key blocks: Kc_all [MAX_TOKENS, N512], Kp_all [MAX_TOKENS, N64]
        Kc_blk = tl.zeros((MAX_TOKENS, N512), dtype=tl.float32)
        Kp_blk = tl.zeros((MAX_TOKENS, N64), dtype=tl.float32)

        # Gather Kc_all and Kp_all rows using idxs
        for i in range(MAX_TOKENS):
            mask_i = mask_tokens[i]
            idx = idxs[i]
            kc_row_ptr = ckv_cache_ptr + idx * N512
            kp_row_ptr = kpe_cache_ptr + idx * N64
            Kc_blk[i, :] = tl.load(kc_row_ptr + tl.arange(0, N512), mask=mask_i, other=0.0).to(tl.float32)
            Kp_blk[i, :] = tl.load(kp_row_ptr + tl.arange(0, N64), mask=mask_i, other=0.0).to(tl.float32)

        # Compute logits_scaled per token: (qn @ Kc_blk[i, :]) + (qp @ Kp_blk[i, :])
        logits = tl.zeros((MAX_TOKENS,), dtype=tl.float32)
        for i in range(MAX_TOKENS):
            dot1 = tl.sum(qn * Kc_blk[i, :], axis=0)  # scalar
            dot2 = tl.sum(qp * Kp_blk[i, :], axis=0)  # scalar
            logits[i] = (dot1 + dot2) * sm_scale

        # Per-head logsumexp over valid tokens
        valid = token_idx < end
        logits_valid = tl.where(valid, logits, -float("inf"))
        lse_val = tl.log(tl.sum(tl.exp(logits_valid)))  # logsumexp without 1/ln(2)
        # Store lse per head: lse_ptr is 1D of length B*H
        lse_offset = b * H + h
        tl.store(lse_ptr + lse_offset, lse_val)

        # Softmax attention weights: attn[i] = exp(logits_valid[i] - lse_val)
        attn = tl.exp(logits_valid - lse_val)

        # Final output: out_vec = sum_i attn[i] * Kc_blk[i, :]
        out_vec = tl.zeros((N512,), dtype=tl.float32)
        for i in range(MAX_TOKENS):
            out_vec += attn[i] * Kc_blk[i, :]

        # Store output[b, h, :] (float32 buffer)
        out_base = output_ptr + b * (H * N512) + h * N512
        tl.store(out_base + tl.arange(0, N512), out_vec, mask=True)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors for Triton
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "Inputs must be CUDA tensors for Triton execution."

        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Output buffer as float32 for numerical stability; we cast to bfloat16 later
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        # lse buffer as float32, shape [batch_size * num_qo_heads]
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
            MAX_TOKENS=1024,
        )

        # Cast output to bfloat16 to match original function's return type
        output_bf16 = output.to(torch.bfloat16)
        # Reshape lse to [batch_size, num_qo_heads]
        lse = lse.view(batch_size, num_qo_heads)
        # Divide by ln(2) to match original semantics
        lse = lse / math.log(2.0)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
