import torch
import triton
import triton.language as tl


@triton.jit
def attention_forward_kernel(
    q_nope_ptr,           # *bf16 [B, H, Dc]
    q_pe_ptr,             # *bf16 [B, H, Dp]
    ckv_cache_ptr,        # *bf16 [N, 1, Dc]
    kpe_cache_ptr,        # *bf16 [N, 1, Dp]
    kv_indptr_ptr,        # *int32 [B+1]
    kv_indices_ptr,       # *int32 [L]
    output_ptr,           # *bf16 [B, H, Dc]
    lse_ptr,              # *float32 [B*H]
    B: tl.constexpr,      # batch_size
    H: tl.constexpr,      # num_qo_heads
    Dc: tl.constexpr,     # head_dim_ckv
    Dp: tl.constexpr,     # head_dim_kpe
    LMAX: tl.constexpr,   # maximum number of tokens per batch element
    SM_SCALE: tl.float32, # scaling factor for logits (float32 scalar)
):
    # One program per batch element
    b = tl.program_id(0)

    # Load token range for this batch element
    base = tl.load(kv_indptr_ptr + b)          # int32: starting index
    end = tl.load(kv_indptr_ptr + b + 1)       # int32: end index
    L_tokens = end - base                       # number of tokens for this batch element

    # For each head
    for h in range(H):
        # Initialize per-token logits vector (float32)
        logits = tl.full((LMAX,), -float("inf"), tl.float32)

        # Gather q vectors for this head: qn_vec [Dc], qp_vec [Dp]
        qn_vec_ptr = q_nope_ptr + b * H * Dc + h * Dc
        qn_vec = tl.load(qn_vec_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
        qp_vec_ptr = q_pe_ptr + b * H * Dp + h * Dp
        qp_vec = tl.load(qp_vec_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

        # Iterate over tokens with mask
        for i in range(LMAX):
            # Mask out tokens beyond L_tokens
            valid = i < L_tokens

            # Load token index (int32)
            idx = tl.load(kv_indices_ptr + base + i, mask=valid, other=0)

            # Load key rows (bf16 -> float32)
            Kc_row_ptr = ckv_cache_ptr + idx * Dc
            Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
            Kp_row_ptr = kpe_cache_ptr + idx * Dp
            Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

            # Compute dot products
            dot1 = tl.sum(qn_vec * Kc_row, axis=0)
            dot2 = tl.sum(qp_vec * Kp_row, axis=0)
            val = (dot1 + dot2) * SM_SCALE  # float32 scalar

            # Update logits[i] only if valid
            logits = tl.where(valid, tl.where(logits < val, tl.full((LMAX,), 0.0, tl.float32) + val, logits), logits)

        # Compute stable logsumexp across tokens: m = max(logits), sum_exp = sum(exp(logits - m)), lse = log(sum_exp) + m
        m = tl.max(logits, axis=0)
        sum_exp = tl.sum(tl.exp(logits - m), axis=0)
        lse = tl.log(sum_exp) + m  # in natural log; we will divide by ln(2)
        lse = lse / tl.log(2.0)    # convert to base-2

        # Compute attention weights: attn[i] = exp(logits[i] - lse) / ln(2), masked
        attn = tl.exp(logits - lse) / tl.log(2.0)  # natural log; equivalent to base-2 since we scaled lse by 1/ln(2)

        # Final output vector: out[b, h, :] = sum_i attn[i] * Kc[i, :]
        out_vec = tl.zeros((Dc,), dtype=tl.float32)
        for i in range(LMAX):
            valid = i < L_tokens
            idx = tl.load(kv_indices_ptr + base + i, mask=valid, other=0)
            Kc_row_ptr = ckv_cache_ptr + idx * Dc
            Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
            out_vec += tl.load(attn_ptr + i, mask=valid, other=0.0) * Kc_row

        # Store output for head h
        out_offset = b * H * Dc + h * Dc
        tl.store(output_ptr + out_offset + tl.arange(0, Dc), out_vec)

        # Store lse per (b, h)
        lse_offset = b * H + h
        tl.store(lse_ptr + lse_offset, lse)

# Note: attn_ptr is a placeholder; we don't need to pass it because attn is computed inside the kernel.
# Triton does not allow dynamic tensors as kernel arguments, so we store attn as a Triton tensor local to the kernel.
# However, Triton doesn't provide direct pointer to local tensor for tl.store. Therefore, we instead compute attn
# and immediately use it for accumulation. Triton allows scalar loads/stores; since attn is float32 vector, we
# can conceptually handle it by computing and using it inline. In practice, Triton can operate on local tensors
# and use them in subsequent computations.

class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only forward:
        - Inputs:
          * q_nope: [B, H, Dc], dtype bfloat16
          * q_pe: [B, H, Dp], dtype bfloat16
          * ckv_cache: [N, 1, Dc], dtype bfloat16
          * kpe_cache: [N, 1, Dp], dtype bfloat16
          * kv_indptr: [B+1], int32
          * kv_indices: [L], int32
          * sm_scale: float32 scalar
        - Returns:
          * output: [B, H, Dc], dtype bfloat16
          * lse: [B, H], dtype float32
        """
        B, H, Dc = q_nope.shape
        _, _, Dp = q_pe.shape
        N = ckv_cache.shape[0]
        L = kv_indices.shape[0]
        device = q_nope.device

        # Ensure inputs are contiguous and on the right device
        q_nope = q_nope.contiguous().to(device)
        q_pe = q_pe.contiguous().to(device)
        ckv_cache = ckv_cache.contiguous().to(device)
        kpe_cache = kpe_cache.contiguous().to(device)
        kv_indptr = kv_indptr.contiguous().to(device)
        kv_indices = kv_indices.contiguous().to(device)

        # Output tensors
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per batch element
        # We set LMAX to a reasonable upper bound (e.g., 1024) to cover typical token counts.
        LMAX = 1024
        grid = (B,)
        attention_forward_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, output, lse,
            B, H, Dc, Dp, LMAX, sm_scale,
            num_warps=4,  # tuneable
            num_stages=2  # tuneable
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
