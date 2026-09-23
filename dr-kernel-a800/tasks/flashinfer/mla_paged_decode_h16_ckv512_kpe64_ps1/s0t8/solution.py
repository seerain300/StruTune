import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per batch element
# Computes output and lse for each head h.
@triton.jit
def attention_kernel(
    q_nope_ptr,        # *bf16, [B, H, N512]
    q_pe_ptr,          # *bf16, [B, H, N64]
    Kc_all_ptr,        # *bf16, [num_pages, N512]
    Kp_all_ptr,        # *bf16, [num_pages, N64]
    kv_indices_ptr,    # *int32, [num_tokens]
    kv_indptr_ptr,     # *int32, [B+1]
    output_ptr,        # *float32, [B, H, N512]
    lse_ptr,           # *float32, [B*H]
    B, H, N512, N64,   # int scalars
    num_tokens,        # int
    sm_scale,          # float32
    MAX_TOKENS: tl.constexpr,
):
    b = tl.program_id(axis=0)  # program id along batch dimension

    # Read token range for this batch element
    base = tl.load(kv_indptr_ptr + b)         # int32
    end = tl.load(kv_indptr_ptr + b + 1)      # int32
    L_tokens = end - base                      # int32 scalar

    # Precompute constants
    ln2 = 0.6931471805599453  # 1 / log(2)

    # Process each head
    for h in range(H):
        # Load query vectors (cast to float32 for math)
        offset_qn = b * (H * N512) + h * N512
        qn = tl.load(q_nope_ptr + offset_qn + tl.arange(0, N512), mask=True, other=0.0).to(tl.float32)
        offset_qp = b * (H * N64) + h * N64
        qp = tl.load(q_pe_ptr + offset_qp + tl.arange(0, N64), mask=True, other=0.0).to(tl.float32)

        # Prepare token index vector [base, base+1, ..., base+MAX_TOKENS-1], mask i < L_tokens
        token_idx = base + tl.arange(0, MAX_TOKENS)
        mask_tokens = token_idx < end  # boolean mask for valid tokens

        # Load kv_indices for each token position (masked)
        idxs = tl.load(kv_indices_ptr + token_idx, mask=mask_tokens, other=0)  # int32

        # Compute logits_scaled vector for each token position (masked). Initialize to -inf for stability.
        logits_scaled = tl.full((MAX_TOKENS,), -float("inf"), dtype=tl.float32)

        # Loop over token positions; gather keys and compute dot products
        for i in range(MAX_TOKENS):
            valid = i < L_tokens
            idx = tl.load(kv_indices_ptr + (base + i), mask=valid, other=0)  # scalar int32
            kc_base = Kc_all_ptr + idx * N512
            kp_base = Kp_all_ptr + idx * N64
            kc = tl.load(kc_base + tl.arange(0, N512), mask=True, other=0.0).to(tl.float32)
            kp = tl.load(kp_base + tl.arange(0, N64), mask=True, other=0.0).to(tl.float32)

            dot1 = tl.sum(qn * kc, axis=0)  # scalar float32
            dot2 = tl.sum(qp * kp, axis=0)  # scalar float32
            scaled = (dot1 + dot2) * sm_scale  # scalar float32

            # Place scaled value at position i (masked) by setting the whole vector element i with tl.where
            # We use a vector condition to set the i-th element.
            cond = tl.arange(0, MAX_TOKENS) == i
            logits_scaled = tl.where(cond, scaled, logits_scaled)

        # Compute lse = logsumexp(logits_scaled) / ln(2)
        # Triton does not have logsumexp, use max trick
        max_logit = tl.max(logits_scaled, axis=0)
        sum_exp = tl.sum(tl.exp(logits_scaled - max_logit), axis=0)
        lse_val = tl.log(sum_exp) + max_logit
        lse_val = lse_val / ln2

        # Store lse for this head
        tl.store(lse_ptr + b * H + h, lse_val)

        # Compute output vector: out[h] = sum_i attn[i] * Kc_all[base+i, :]
        out_vec = tl.zeros((N512,), dtype=tl.float32)

        # Accumulate out_vec using attn and Kc_all for each token position i < L_tokens
        for i in range(MAX_TOKENS):
            valid = i < L_tokens
            idx = tl.load(kv_indices_ptr + (base + i), mask=valid, other=0)  # scalar int32
            kc_base = Kc_all_ptr + idx * N512
            kc = tl.load(kc_base + tl.arange(0, N512), mask=True, other=0.0).to(tl.float32)

            # attn[i] = exp(logits_scaled[i] - lse_val) for valid, else 0
            scaled_i = tl.load(logits_scaled_ptr + i, mask=valid, other=-float("inf"))  # not used here, use recomputation
            # Since logits_scaled vector is maintained, we can reconstruct attn[i] from vector:
            attn_i = tl.exp((tl.load(logits_scaled_ptr + i, mask=valid, other=-float("inf")) - lse_val))
            # Multiply masked: attn_i where valid, else 0
            attn_i = tl.where(valid, attn_i, 0.0)

            out_vec += attn_i * kc

        # Store output[b, h, :]
        out_base = output_ptr + b * (H * N512) + h * N512
        tl.store(out_base + tl.arange(0, N512), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All inputs must be CUDA tensors for Triton execution."

        # Shapes
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        N512 = q_nope.shape[2]
        N64 = q_pe.shape[2]

        # Output buffer as float32 for numerical stability
        output = torch.empty((B, H, N512), dtype=torch.float32, device=q_nope.device)
        lse_buf = torch.empty(B * H, dtype=torch.float32, device=q_nope.device)

        # Launch Triton kernel: one program per batch element
        grid = (B,)
        attention_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indices, kv_indptr,
            output, lse_buf,
            B, H, N512, N64,
            kv_indices.shape[0],  # num_tokens
            float(sm_scale),
            MAX_TOKENS=1024,  # upper bound for token count; masks out-of-range
        )

        # Cast output to bfloat16 to match original function's return type
        output_bf16 = output.to(torch.bfloat16)
        # Reshape lse to [B, H] and divide by ln(2) to match original run's lse computation
        lse = lse_buf.view(B, H) / math.log(2.0)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
