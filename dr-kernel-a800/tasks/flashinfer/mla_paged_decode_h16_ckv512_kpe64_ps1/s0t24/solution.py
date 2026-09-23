import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    q_nope_ptr,           # *bf16 [B, H, Dc]
    q_pe_ptr,             # *bf16 [B, H, Dp]
    ckv_cache_ptr,        # *bf16 [N, 1, Dc]
    kpe_cache_ptr,        # *bf16 [N, 1, Dp]
    kv_indptr_ptr,        # *int32 [B+1]
    kv_indices_ptr,       # *int32 [L]
    tmp_ptr,              # *float32 [(B*H), ]
    B: tl.constexpr,      # batch_size
    H: tl.constexpr,      # num_qo_heads
    Dc: tl.constexpr,     # head_dim_ckv
    Dp: tl.constexpr,     # head_dim_kpe
    SM_SCALE: tl.constexpr,  # scaling factor for logits
):
    """
    For each (b, h), compute logits_scaled[h, i] = (qn @ Kc[i]) + (qp @ Kp[i]) * SM_SCALE
    Store into tmp[b*H + h] as a flat vector of length MAX_TOKENS.
    Entries beyond L_tokens are -inf.
    """
    b = tl.program_id(0)  # one program per batch element
    base = tl.load(kv_indptr_ptr + b)            # int32
    end = tl.load(kv_indptr_ptr + b + 1)         # int32
    L_tokens = end - base                         # number of tokens for this batch element

    # Load qn and qp for all heads h; we'll loop h explicitly in the grid or inside here.
    # We need to compute per head, so we do it by h as a separate launch or by looping h in the kernel.
    # Triton program_id only provides b; we'll loop over h in the kernel using for h in range(H).
    # But Triton supports range loops, and we can index q_nope_ptr using h. So:
    for h in range(H):
        # Compute q vectors for this head
        qn = tl.load(q_nope_ptr + b * H * Dc + h * Dc + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
        qp = tl.load(q_pe_ptr + b * H * Dp + h * Dp + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

        # Initialize logits vector with -inf
        logits = tl.full((MAX_TOKENS,), -float("inf"), dtype=tl.float32)

        # Fill valid positions
        for i in range(MAX_TOKENS):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)  # int32
                Kc_row_ptr = ckv_cache_ptr + idx * Dc
                Kp_row_ptr = kpe_cache_ptr + idx * Dp
                Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
                Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

                dot1 = tl.sum(qn * Kc_row, axis=0)
                dot2 = tl.sum(qp * Kp_row, axis=0)
                val = (dot1 + dot2) * SM_SCALE
                logits[i] = val

        # Store logits for this (b, h) into tmp
        tl.store(tmp_ptr + b * H + h, logits)


@triton.jit
def softmax_kernel(
    tmp_ptr,       # *float32 [(B*H), ]
    attn_ptr,      # *float32 [(B*H), MAX_TOKENS]
    B: tl.constexpr,
    H: tl.constexpr,
    SM_SCALE: tl.constexpr,  # not used here, kept for signature symmetry
):
    """
    For each (b, h), apply stable softmax over tmp[b*H + h] (size MAX_TOKENS),
    with invalid positions set to 0 attention. Write to attn_ptr[b*H + h, :].
    """
    b = tl.program_id(0)
    for h in range(H):
        vec = tl.load(tmp_ptr + b * H + h)  # shape [MAX_TOKENS], float32
        max_val = tl.max(vec, axis=0)
        vec_shift = vec - max_val
        exp_vec = tl.exp(vec_shift)
        sum_exp = tl.sum(exp_vec, axis=0)
        attn = exp_vec / sum_exp
        # Store attn
        tl.store(attn_ptr + b * H * MAX_TOKENS + h * MAX_TOKENS + tl.arange(0, MAX_TOKENS), attn)


@triton.jit
def combine_kernel(
    attn_ptr,      # *float32 [(B*H), MAX_TOKENS]
    ckv_cache_ptr, # *bf16 [N, 1, Dc]
    kv_indices_ptr,# *int32 [L]
    output_ptr,    # *bf16 [B, H, Dc]
    B: tl.constexpr,
    H: tl.constexpr,
    Dc: tl.constexpr,
    SM_SCALE: tl.constexpr,  # not used here
):
    """
    For each (b, h), read attn[b*H + h, :], then out[b, h, :] = sum_i attn[i] * Kc[kv_indices[base+i]].
    """
    b = tl.program_id(0)
    base = tl.load(kv_indptr_ptr + b)  # int32
    end = tl.load(kv_indptr_ptr + b + 1)  # int32
    L_tokens = end - base

    for h in range(H):
        attn = tl.load(attn_ptr + b * H * MAX_TOKENS + h * MAX_TOKENS + tl.arange(0, MAX_TOKENS), mask=tl.arange(0, MAX_TOKENS) < L_tokens, other=0.0).to(tl.float32)
        out_vec = tl.zeros((Dc,), dtype=tl.float32)
        # Accumulate over tokens
        for i in range(MAX_TOKENS):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)  # int32
                Kc_row_ptr = ckv_cache_ptr + idx * Dc
                Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
                out_vec += attn[i] * Kc_row
        # Store output for this head
        tl.store(output_ptr + b * H * Dc + h * Dc + tl.arange(0, Dc), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only forward:
        - Inputs:
          * q_nope: [B, H, Dc], dtype bfloat16 (B=1, H=16, Dc=512)
          * q_pe: [B, H, Dp], dtype bfloat16 (Dp=64)
          * ckv_cache: [N, 1, Dc], dtype bfloat16
          * kpe_cache: [N, 1, Dp], dtype bfloat16
          * kv_indptr: [B+1], int32
          * kv_indices: [L], int32
          * sm_scale: float32 scalar
        - Returns:
          * output: [B, H, Dc], dtype bfloat16
          * lse: [B, H], dtype float32
        """
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]
        L = kv_indices.shape[0]
        N = ckv_cache.shape[0]

        # Output and attn buffers
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=q_nope.device)
        attn = torch.empty((B, H, MAX_TOKENS), dtype=torch.float32, device=q_nope.device)
        tmp = torch.empty((B * H,), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Launch compute logits kernel
        compute_logits_kernel[(B,)](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, tmp,
            B=B, H=H, Dc=Dc, Dp=Dp, SM_SCALE=float(sm_scale),
        )

        # Launch softmax kernel (stable)
        # attn[b, h, i] = softmax(tmp[b*H + h][i])
        softmax_kernel[(B,)](
            tmp, attn, B=B, H=H, SM_SCALE=float(sm_scale),
        )

        # Combine attn with Kc to produce output (and compute lse as logsumexp(tmp)/ln(2) for consistency)
        combine_kernel[(B,)](
            attn, ckv_cache, kv_indices, output, B=B, H=H, Dc=Dc, SM_SCALE=float(sm_scale),
        )

        # Compute lse as logsumexp of tmp divided by ln(2). Since attn is softmax(tmp), lse = log(sum(exp(tmp)))/ln(2).
        # However, tmp stores scaled logits; we need the original scaled logits for lse. Re-compute lse for correctness:
        # We can recompute per (b, h) using the same approach as compute_logits_kernel and then lse = logsumexp(vec)/ln(2).
        # To avoid an extra pass, we approximate by using the tmp values (which are scaled logits). The reference uses
        # logits_scaled and divides by ln(2). Since our tmp already includes SM_SCALE, lse = logsumexp(tmp)/ln(2).
        # This matches the original intent: logsumexp over scaled logits.
        for b in range(B):
            for h in range(H):
                vec = tmp[b * H + h]  # shape [MAX_TOKENS]
                max_val = torch.max(vec)
                vec_shift = vec - max_val
                sum_exp = torch.sum(torch.exp(vec_shift))
                lse[b, h] = torch.log(sum_exp) + max_val
                lse[b, h] = lse[b, h] / math.log(2.0)

        return output, lse

# Ensure math is imported
import math


def run(*args):
    return ModelNew()(*args)
