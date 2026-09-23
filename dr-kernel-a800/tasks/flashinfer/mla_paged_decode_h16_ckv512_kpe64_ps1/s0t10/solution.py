import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_nope_ptr,   # *bf16 [B, H, N512]
    q_pe_ptr,     # *bf16 [B, H, N64]
    ckv_cache_ptr,  # *bf16 [num_pages, 1, N512]
    kpe_cache_ptr,  # *bf16 [num_pages, 1, N64]
    kv_indices_ptr, # *int32 [num_tokens]
    kv_indptr_ptr,  # *int32 [B+1]
    output_ptr,    # *float32 [B, H, N512]
    lse_ptr,       # *float32 [B, H]
    B,            # int32 batch_size
    H,            # int32 num_qo_heads
    N512,         # int32 head_dim_ckv
    N64,          # int32 head_dim_kpe
    IND_LEN,      # int32 len_indptr (expected to be B+1)
    NUM_TOKENS,   # int32 kv_indices.shape[0]
    sm_scale,     # float32 scale
    MAX_TOKENS: tl.constexpr = 1024,
):
    # One program per batch element
    b = tl.program_id(axis=0)

    # Read token range for this batch element: [base, end) where base = kv_indptr[b], end = kv_indptr[b+1]
    base = tl.load(kv_indptr_ptr + b)       # int32
    end = tl.load(kv_indptr_ptr + b + 1)    # int32
    L_tokens = end - base                    # number of tokens for this batch element

    # Sanity: original code assumes IND_LEN == B + 1 (i.e., kv_indptr[b+1] exists)
    # The provided get_inputs respects this; uncomment if you want to enforce:
    # assert IND_LEN == (b + 1), "kv_indptr length must be batch_size + 1"

    # Process each head h
    for h in range(0, H):
        # Load query vectors for head h: qn = q_nope[b, h, :] (float32), qp = q_pe[b, h, :] (float32)
        offset_qn = b * (H * N512) + h * N512
        qn = tl.load(q_nope_ptr + offset_qn + tl.arange(0, N512), mask=True, other=0.0).to(tl.float32)

        offset_qp = b * (H * N64) + h * N64
        qp = tl.load(q_pe_ptr + offset_qp + tl.arange(0, N64), mask=True, other=0.0).to(tl.float32)

        # Vector to hold logits_scaled for each token, initialized to -inf for stability
        logits_scaled = tl.full((MAX_TOKENS,), -float("inf"), dtype=tl.float32)

        # Iterate tokens up to MAX_TOKENS; guard with i < L_tokens
        for i in range(0, MAX_TOKENS):
            use_i = i < L_tokens
            # idx is the token index into Kc_all/Kp_all for this batch's range
            idx = tl.load(kv_indices_ptr + (base + i), mask=use_i, other=0)  # int32
            kc_base = ckv_cache_ptr + idx * N512
            kp_base = kpe_cache_ptr + idx * N64
            kc = tl.load(kc_base + tl.arange(0, N512), mask=True, other=0.0).to(tl.float32)
            kp = tl.load(kp_base + tl.arange(0, N64), mask=True, other=0.0).to(tl.float32)

            # Dot products: qn @ kc and qp @ kp
            dot1 = tl.sum(qn * kc, axis=0)  # scalar float32
            dot2 = tl.sum(qp * kp, axis=0)  # scalar float32
            scaled = (dot1 + dot2) * sm_scale

            # Update logits_scaled at position i
            idxs = tl.full((MAX_TOKENS,), i, dtype=tl.int32)
            mask_pos = idxs == i
            logits_scaled = tl.where(mask_pos, scaled, logits_scaled)

        # Compute lse = logsumexp(logits_scaled) / ln(2)
        max_scaled = tl.max(logits_scaled, axis=0)
        sum_exps = tl.sum(tl.exp(logits_scaled - max_scaled), axis=0)
        lse_val = tl.log(sum_exps) + max_scaled  # logsumexp(scaled_logits)
        lse_val = lse_val / 0.6931471805599453   # 1 / ln(2)
        tl.store(lse_ptr + b * H + h, lse_val)

        # Compute attention weights: attn = exp(logits_scaled - lse_val)
        attn = tl.exp(logits_scaled - lse_val)

        # Output: out = sum_i attn[i] * Kc_all[i, :]
        out_vec = tl.zeros((N512,), dtype=tl.float32)
        for i in range(0, MAX_TOKENS):
            use_i = i < L_tokens
            idx = tl.load(kv_indices_ptr + (base + i), mask=use_i, other=0)  # int32
            kc_base = ckv_cache_ptr + idx * N512
            kc = tl.load(kc_base + tl.arange(0, N512), mask=True, other=0.0).to(tl.float32)
            out_vec += attn[i] * kc

        # Store output[b, h, :]
        out_base = output_ptr + b * (H * N512) + h * N512
        tl.store(out_base + tl.arange(0, N512), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "Inputs must be CUDA tensors for Triton execution."

        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Output is float32 inside kernel; cast to bfloat16 on host after
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per batch element
        grid = (batch_size,)
        attention_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indices, kv_indptr,
            output, lse,
            batch_size, num_qo_heads, head_dim_ckv, head_dim_kpe,
            kv_indptr.shape[0], kv_indices.shape[0],
            float(sm_scale),
            num_warps=4,  # reasonable default for small vector ops
        )

        # Cast output to bfloat16 to match original function's return type
        output_bf16 = output.to(torch.bfloat16)
        # lse already in float32; shape is [batch_size, num_qo_heads]
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
