import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_bh_kernel(
    q_nope_ptr,           # *bf16 [B, H, Dc]
    q_pe_ptr,             # *bf16 [B, H, Dp]
    ckv_cache_ptr,        # *bf16 [N, 1, Dc]
    kpe_cache_ptr,        # *bf16 [N, 1, Dp]
    kv_indptr_ptr,        # *int32 [B+1]
    kv_indices_ptr,       # *int32 [L]
    output_ptr,           # *bf16 [B, H, Dc] (we'll store float32 and cast after)
    lse_ptr,              # *float32 [B*H]
    B: tl.constexpr,      # batch_size (used for grid only)
    H: tl.constexpr,      # num_qo_heads
    Dc: tl.constexpr,     # head_dim_ckv (512)
    Dp: tl.constexpr,     # head_dim_kpe (64)
    SM_SCALE: tl.constexpr,  # scaling factor for logits
):
    # One program per batch element
    b = tl.program_id(0)

    # Load token range for this batch element
    base = tl.load(kv_indptr_ptr + b)       # int32
    end = tl.load(kv_indptr_ptr + b + 1)    # int32
    L_tokens = end - base

    LOG2 = 1.0 / tl.log(2.0)

    # Loop over heads h
    for h in range(0, H):
        # If no tokens, set output to zeros and lse to -inf for this head
        if L_tokens <= 0:
            # We will store float32 output and cast to bf16 outside kernel
            out_offset = b * H * Dc + h * Dc
            out_vec = tl.zeros((Dc,), dtype=tl.float32)
            for d in range(0, Dc):
                tl.store(output_ptr + out_offset + d, out_vec[d])
            lse_offset = b * H + h
            tl.store(lse_ptr + lse_offset, -float("inf"))
            continue

        # Load q_nope and q_pe for this head and cast to float32
        qn_ptr = q_nope_ptr + b * H * Dc + h * Dc
        qp_ptr = q_pe_ptr + b * H * Dp + h * Dp
        qn = tl.load(qn_ptr + tl.arange(0, Dc)).to(tl.float32)
        qp = tl.load(qp_ptr + tl.arange(0, Dp)).to(tl.float32)

        # Prepare logits_scaled as a 1D buffer [L_tokens]
        logits_scaled = tl.full((L_tokens,), -float("inf"), dtype=tl.float32)

        # Pass 1: compute logits_scaled per token
        for i in range(0, L_tokens):
            idx = base + i
            Kc_row_ptr = ckv_cache_ptr + idx * Dc
            Kp_row_ptr = kpe_cache_ptr + idx * Dp

            Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc)).to(tl.float32)
            Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp)).to(tl.float32)

            dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
            dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
            val = (dot1 + dot2) * SM_SCALE
            logits_scaled[i] = val

        # Compute base-2 logsumexp for this head
        max_val = -float("inf")
        for i in range(0, L_tokens):
            max_val = tl.maximum(max_val, logits_scaled[i])

        sum_exp = 0.0
        for i in range(0, L_tokens):
            sum_exp += tl.exp(logits_scaled[i] - max_val)

        lse_val = tl.log(sum_exp) + max_val
        lse_val = lse_val * LOG2

        # Store lse for this (b, h)
        lse_offset = b * H + h
        tl.store(lse_ptr + lse_offset, lse_val)

        # Compute output vector: out[b, h, :] = sum_i exp(logits_scaled[i] - lse_val) * Kc_all[base+i, :]
        out_vec = tl.zeros((Dc,), dtype=tl.float32)
        for i in range(0, L_tokens):
            attn_i = tl.exp(logits_scaled[i] - lse_val)
            idx = base + i
            Kc_row_ptr = ckv_cache_ptr + idx * Dc
            Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc)).to(tl.float32)
            out_vec += attn_i * Kc_row

        # Store output for this head
        out_offset = b * H * Dc + h * Dc
        for d in range(0, Dc):
            tl.store(output_ptr + out_offset + d, out_vec[d])


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

        # Allocate outputs (float32 for computation, cast later to bfloat16)
        output = torch.empty((B, H, Dc), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Launch one Triton program per batch element
        grid = (B,)
        attention_bh_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices,
            output, lse,
            B=B, H=H, Dc=Dc, Dp=Dp,
            SM_SCALE=float(sm_scale),
            num_warps=4,
        )

        # Cast output to bfloat16 to match original signature
        output = output.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
