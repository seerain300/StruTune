import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
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
    SM_SCALE: tl.constexpr,  # scaling factor for logits
):
    # One program per batch element
    b = tl.program_id(0)

    # Load token range for this batch element
    base = tl.load(kv_indptr_ptr + b)          # int32
    end = tl.load(kv_indptr_ptr + b + 1)       # int32
    L_tokens = end - base                       # number of tokens for this batch element

    # If no tokens, set lse to -inf and output to zero
    no_tokens = L_tokens == 0
    max_val = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)
    lse_val = tl.full((), -float("inf"), dtype=tl.float32)

    # Loop over heads
    for h in range(H):
        # Compute base offsets for q_nope and q_pe
        qn_offset = b * H * Dc + h * Dc
        qp_offset = b * H * Dp + h * Dp

        # Load q vectors for this head and batch element
        qn = tl.load(q_nope_ptr + qn_offset + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
        qp = tl.load(q_pe_ptr + qp_offset + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

        # Compute logits_scaled per token: initialize with -inf
        logits_scaled = tl.full((Dc,), -float("inf"), dtype=tl.float32)

        # Iterate up to MAX_TOKENS (we use a loop with mask to handle arbitrary L_tokens)
        # Note: Triton supports python-range loops; L_tokens is runtime int.
        for i in range(1024):  # upper bound for tokens; Triton will evaluate this loop
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)  # int32
                # Load Kc row and Kp row
                Kc_row_ptr = ckv_cache_ptr + idx * Dc
                Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
                Kp_row_ptr = kpe_cache_ptr + idx * Dp
                Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

                dot1 = tl.sum(qn * Kc_row, axis=0)
                dot2 = tl.sum(qp * Kp_row, axis=0)
                val = (dot1 + dot2) * SM_SCALE
                # For i < L_tokens, we use val; otherwise logits_scaled remains -inf
                logits_scaled = tl.where(tl.full((Dc,), i < L_tokens, dtype=tl.int1), val, logits_scaled)
            else:
                # do nothing, keep logits_scaled as is (will be -inf beyond L_tokens)
                pass

        # Compute stable logsumexp: max and sum_exp
        max_val = tl.max(logits_scaled, axis=0)
        # sum over positions where i < L_tokens
        # Mask values for i >= L_tokens as -inf so they don't contribute
        mask_i = tl.arange(0, Dc) < (L_tokens)  # boolean vector mask
        masked_logits = tl.where(mask_i, logits_scaled, -float("inf"))
        sum_exp = tl.sum(tl.exp(masked_logits - max_val), axis=0)

        # lse per head
        lse_val = tl.log(sum_exp) + max_val
        # Divide by ln(2) as in original code
        lse_val = lse_val / tl.log(2.0)

        # Final output vector: out[b, h, :] = sum_i attn_i * Kc_all[i, :]
        out_vec = tl.zeros((Dc,), dtype=tl.float32)
        for i in range(1024):
            if i < L_tokens:
                idx = tl.load(kv_indices_ptr + base + i)
                Kc_row_ptr = ckv_cache_ptr + idx * Dc
                Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
                Kp_row_ptr = kpe_cache_ptr + idx * Dp
                Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

                dot1 = tl.sum(qn * Kc_row, axis=0)
                dot2 = tl.sum(qp * Kp_row, axis=0)
                val = (dot1 + dot2) * SM_SCALE
                attn_i = tl.exp(val - lse_val)
                out_vec += attn_i * Kc_row

        # Store output for head h
        out_offset = b * H * Dc + h * Dc
        tl.store(output_ptr + out_offset + tl.arange(0, Dc), out_vec)

        # Store lse per (b, h)
        lse_offset = b * H + h
        tl.store(lse_ptr + lse_offset, lse_val)


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
        # Ensure tensors are on the same device (CUDA required for Triton)
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be CUDA tensors"

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]
        device = q_nope.device

        # Allocate outputs
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per batch element
        grid = (B,)
        attention_kernel[grid](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, output, lse,
            B=B, H=H, Dc=Dc, Dp=Dp, SM_SCALE=sm_scale
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
